import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample_poly

from addse.stft import STFT


@dataclass
class Module1ProbeConfig:
    seed: int = 42
    fs: int = 16000
    segment_seconds: float = 4.0
    audio_path: str = "TIMIT_all_wavs/SA1.WAV"
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512
    vocab_size: int = 1024
    num_layers: int = 4
    report_json: str = "addse/experiments/archive/v6/reports/module1_phase_confidence_probe.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_mono_audio(path: Path, target_sr: int, segment_seconds: float) -> tuple[torch.Tensor, int]:
    try:
        audio, sr = sf.read(str(path), always_2d=False)
    except Exception:
        from scipy.io import wavfile

        sr, audio = wavfile.read(str(path))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        scale = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max(scale, 1.0)
    else:
        audio = audio.astype(np.float32)

    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr

    target_len = int(round(target_sr * segment_seconds))
    if audio.shape[0] < target_len:
        audio = np.pad(audio, (0, target_len - audio.shape[0]))
    else:
        audio = audio[:target_len]

    return torch.from_numpy(audio).float(), sr


def _normalize_per_sample(x: torch.Tensor, dim: int) -> torch.Tensor:
    min_x = x.amin(dim=dim, keepdim=True)
    max_x = x.amax(dim=dim, keepdim=True)
    return (x - min_x) / (max_x - min_x + 1e-8)


def _tensor_stats(x: torch.Tensor) -> dict[str, float | list[int]]:
    x = x.detach()
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "nan_ratio": float(torch.isnan(x).float().mean().item()),
        "inf_ratio": float(torch.isinf(x).float().mean().item()),
    }


def _build_synthetic_logits(frame_energy: torch.Tensor, vocab_size: int, num_layers: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=frame_energy.device)
    generator.manual_seed(seed)
    batch_size, num_frames = frame_energy.shape

    noise = torch.randn((batch_size, num_frames, num_layers, vocab_size), generator=generator, device=frame_energy.device)
    layer_scale = torch.tensor([1.15, 1.05, 0.95, 0.85], device=frame_energy.device, dtype=frame_energy.dtype)
    layer_scale = layer_scale.view(1, 1, num_layers, 1)
    energy_scale = (0.65 + 1.15 * frame_energy.clamp(0.0, 1.0)).unsqueeze(-1).unsqueeze(-1)
    return noise * layer_scale * energy_scale


def _confidence_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    coarse_entropy = entropy[:, :, :3].mean(dim=-1)
    fine_entropy = entropy[:, :, 3]

    coarse_conf = 1.0 - _normalize_per_sample(coarse_entropy, dim=1)
    fine_conf = 1.0 - _normalize_per_sample(fine_entropy, dim=1)
    return coarse_conf.clamp(0.0, 1.0), fine_conf.clamp(0.0, 1.0), entropy


def _build_phase_input(stft: STFT, noisy: torch.Tensor, logits: torch.Tensor) -> dict[str, torch.Tensor]:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft)
    noisy_mag = torch.log1p(noisy_stft.abs())
    noisy_mag = noisy_mag / noisy_mag.amax(dim=1, keepdim=True).clamp(min=1e-6)

    phase_feat = torch.stack((torch.cos(noisy_phase), torch.sin(noisy_phase), noisy_mag), dim=1)
    coarse_conf, fine_conf, entropy = _confidence_from_logits(logits)

    target_frames = noisy_phase.shape[-1]
    freq_bins = noisy_phase.shape[-2]
    coarse_map = F.interpolate(coarse_conf.unsqueeze(1), size=target_frames, mode="linear", align_corners=False)
    fine_map = F.interpolate(fine_conf.unsqueeze(1), size=target_frames, mode="linear", align_corners=False)
    coarse_map = coarse_map.unsqueeze(2).expand(-1, 1, freq_bins, -1)
    fine_map = fine_map.unsqueeze(2).expand(-1, 1, freq_bins, -1)

    phase_input = torch.cat((phase_feat, coarse_map, fine_map), dim=1)
    return {
        "noisy_stft": noisy_stft,
        "noisy_phase": noisy_phase,
        "noisy_mag": noisy_mag,
        "phase_feat": phase_feat,
        "entropy": entropy,
        "coarse_conf": coarse_conf,
        "fine_conf": fine_conf,
        "coarse_map": coarse_map,
        "fine_map": fine_map,
        "phase_input": phase_input,
    }


def _run_probe(cfg: Module1ProbeConfig) -> dict[str, object]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    project_root = _project_root()
    audio_path = project_root / cfg.audio_path
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio sample: {audio_path}")

    audio, sr = _load_mono_audio(audio_path, cfg.fs, cfg.segment_seconds)
    noisy = audio.unsqueeze(0)

    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft)
    noisy_stft = stft(noisy)
    frame_energy = noisy_stft.abs().mean(dim=1)
    frame_energy = _normalize_per_sample(frame_energy, dim=1)

    logits = _build_synthetic_logits(frame_energy, cfg.vocab_size, cfg.num_layers, cfg.seed)
    outputs = _build_phase_input(stft, noisy, logits)

    report: dict[str, object] = {
        "config": asdict(cfg),
        "audio_path": str(audio_path),
        "sample_rate": int(sr),
        "audio_shape": list(noisy.shape),
        "stft_shape": list(noisy_stft.shape),
        "logits_shape": list(logits.shape),
        "phase_input_shape": list(outputs["phase_input"].shape),
        "phase_feat_shape": list(outputs["phase_feat"].shape),
        "entropy_shape": list(outputs["entropy"].shape),
        "coarse_conf_stats": _tensor_stats(outputs["coarse_conf"]),
        "fine_conf_stats": _tensor_stats(outputs["fine_conf"]),
        "coarse_map_stats": _tensor_stats(outputs["coarse_map"]),
        "fine_map_stats": _tensor_stats(outputs["fine_map"]),
        "phase_input_stats": _tensor_stats(outputs["phase_input"]),
        "noisy_mag_stats": _tensor_stats(outputs["noisy_mag"]),
        "entropy_stats": _tensor_stats(outputs["entropy"]),
        "checks": {
            "phase_input_channels": int(outputs["phase_input"].shape[1]),
            "phase_input_matches_expected": bool(outputs["phase_input"].shape[1] == 5),
            "confidence_in_unit_interval": bool(
                outputs["coarse_conf"].min().item() >= -1e-6
                and outputs["coarse_conf"].max().item() <= 1.0 + 1e-6
                and outputs["fine_conf"].min().item() >= -1e-6
                and outputs["fine_conf"].max().item() <= 1.0 + 1e-6
            ),
            "finite_phase_input": bool(torch.isfinite(outputs["phase_input"]).all().item()),
            "frequency_bins": int(outputs["phase_input"].shape[2]),
            "time_frames": int(outputs["phase_input"].shape[3]),
        },
        "summary": {
            "module1_status": "feasible",
            "module1_interface": "[cos(phase), sin(phase), logmag, Conf_s, Conf_a]",
            "core_observation": "The confidence maps align to STFT frames and expand cleanly over frequency bins without shape mismatch.",
        },
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Module1 probe for hierarchical confidence-guided phase input.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    args = parser.parse_args()

    cfg = Module1ProbeConfig()
    if args.config is not None:
        cfg_path = Path(args.config)
        with open(cfg_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        base = asdict(cfg)
        base.update(overrides)
        cfg = Module1ProbeConfig(**base)

    report = _run_probe(cfg)

    project_root = _project_root()
    report_path = project_root / cfg.report_json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Module 1 Confidence Probe ===")
    print(f"audio_path: {report['audio_path']}")
    print(f"sample_rate: {report['sample_rate']}")
    print(f"audio_shape: {report['audio_shape']}")
    print(f"stft_shape: {report['stft_shape']}")
    print(f"logits_shape: {report['logits_shape']}")
    print(f"phase_input_shape: {report['phase_input_shape']}")
    print(f"entropy_shape: {report['entropy_shape']}")
    print(f"coarse_conf_stats: {report['coarse_conf_stats']}")
    print(f"fine_conf_stats: {report['fine_conf_stats']}")
    print(f"phase_input_stats: {report['phase_input_stats']}")
    print(f"checks: {report['checks']}")
    print(f"saved_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())