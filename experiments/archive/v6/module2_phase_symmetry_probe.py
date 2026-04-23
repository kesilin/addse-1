import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from addse.stft import STFT
from module1_phase_confidence_probe import (
    Module1ProbeConfig,
    _build_phase_input,
    _build_synthetic_logits,
    _load_mono_audio,
    _normalize_per_sample,
    _project_root,
)


@dataclass
class Module2ProbeConfig:
    seed: int = 42
    fs: int = 16000
    segment_seconds: float = 4.0
    audio_path: str = "TIMIT_all_wavs/SA1.WAV"
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512
    vocab_size: int = 1024
    num_layers: int = 4
    report_json: str = "addse/experiments/archive/v6/reports/module2_phase_symmetry_probe.json"


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


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels),
            nn.GroupNorm(8, channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.PReLU(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Module2SymmetryProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.PReLU(64),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.coarse_branch = nn.Sequential(
            DepthwiseSeparableBlock(64, kernel_size=5, dilation=1),
            DepthwiseSeparableBlock(64, kernel_size=5, dilation=1),
        )
        self.fine_branch = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.GroupNorm(8, 64),
            nn.PReLU(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, 64),
            nn.PReLU(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(8, 64),
            nn.PReLU(64),
        )
        self.coarse_head = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1), nn.Tanh())
        self.fine_head = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1), nn.Tanh())

    def forward(self, phase_input: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.stem(phase_input)
        uncertainty = self.uncertainty_head(shared)
        coarse_feat = self.coarse_branch(shared)
        fine_feat = self.fine_branch(shared)
        coarse_routed = coarse_feat * (1.0 - uncertainty)
        fine_routed = fine_feat * uncertainty
        coarse_residual = self.coarse_head(coarse_routed) * 0.25
        fine_residual = self.fine_head(fine_routed) * 0.25
        fused_residual = coarse_residual + fine_residual
        return {
            "shared": shared,
            "uncertainty": uncertainty,
            "coarse_feat": coarse_feat,
            "fine_feat": fine_feat,
            "coarse_routed": coarse_routed,
            "fine_routed": fine_routed,
            "coarse_residual": coarse_residual,
            "fine_residual": fine_residual,
            "fused_residual": fused_residual,
        }


def _run_probe(cfg: Module2ProbeConfig) -> dict[str, object]:
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
    frame_energy = _normalize_per_sample(noisy_stft.abs().mean(dim=1), dim=1)

    logits = _build_synthetic_logits(frame_energy, cfg.vocab_size, cfg.num_layers, cfg.seed)
    phase_input_info = _build_phase_input(stft, noisy, logits)
    phase_input = phase_input_info["phase_input"]

    model = Module2SymmetryProbe().eval()
    with torch.no_grad():
        outputs = model(phase_input)

    report: dict[str, object] = {
        "config": asdict(cfg),
        "audio_path": str(audio_path),
        "sample_rate": int(sr),
        "audio_shape": list(noisy.shape),
        "stft_shape": list(noisy_stft.shape),
        "logits_shape": list(logits.shape),
        "phase_input_shape": list(phase_input.shape),
        "shared_shape": list(outputs["shared"].shape),
        "uncertainty_shape": list(outputs["uncertainty"].shape),
        "coarse_feat_shape": list(outputs["coarse_feat"].shape),
        "fine_feat_shape": list(outputs["fine_feat"].shape),
        "coarse_residual_shape": list(outputs["coarse_residual"].shape),
        "fine_residual_shape": list(outputs["fine_residual"].shape),
        "fused_residual_shape": list(outputs["fused_residual"].shape),
        "phase_input_stats": _tensor_stats(phase_input),
        "uncertainty_stats": _tensor_stats(outputs["uncertainty"]),
        "coarse_feat_stats": _tensor_stats(outputs["coarse_feat"]),
        "fine_feat_stats": _tensor_stats(outputs["fine_feat"]),
        "coarse_residual_stats": _tensor_stats(outputs["coarse_residual"]),
        "fine_residual_stats": _tensor_stats(outputs["fine_residual"]),
        "fused_residual_stats": _tensor_stats(outputs["fused_residual"]),
        "checks": {
            "phase_input_channels": int(phase_input.shape[1]),
            "phase_input_matches_expected": bool(phase_input.shape[1] == 5),
            "uncertainty_in_unit_interval": bool(
                outputs["uncertainty"].min().item() >= -1e-6 and outputs["uncertainty"].max().item() <= 1.0 + 1e-6
            ),
            "finite_fused_residual": bool(torch.isfinite(outputs["fused_residual"]).all().item()),
            "coarse_residual_channels": int(outputs["coarse_residual"].shape[1]),
            "fine_residual_channels": int(outputs["fine_residual"].shape[1]),
            "frequency_bins": int(outputs["fused_residual"].shape[2]),
            "time_frames": int(outputs["fused_residual"].shape[3]),
        },
        "summary": {
            "module2_status": "feasible",
            "module2_interface": "[5ch phase_input -> shared(64) -> uncertainty(1) -> coarse/fine residuals(2+2)]",
            "core_observation": "The 5-channel module2 input can be routed into coarse/fine branches without changing the STFT frame or frequency dimensions.",
        },
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Module2 probe for 3+1 symmetric phase residual routing.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    args = parser.parse_args()

    cfg = Module2ProbeConfig()
    if args.config is not None:
        cfg_path = Path(args.config)
        with open(cfg_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        base = asdict(cfg)
        base.update(overrides)
        cfg = Module2ProbeConfig(**base)

    report = _run_probe(cfg)

    project_root = _project_root()
    report_path = project_root / cfg.report_json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Module 2 Symmetry Probe ===")
    print(f"audio_path: {report['audio_path']}")
    print(f"sample_rate: {report['sample_rate']}")
    print(f"audio_shape: {report['audio_shape']}")
    print(f"stft_shape: {report['stft_shape']}")
    print(f"phase_input_shape: {report['phase_input_shape']}")
    print(f"shared_shape: {report['shared_shape']}")
    print(f"uncertainty_shape: {report['uncertainty_shape']}")
    print(f"coarse_residual_shape: {report['coarse_residual_shape']}")
    print(f"fine_residual_shape: {report['fine_residual_shape']}")
    print(f"fused_residual_shape: {report['fused_residual_shape']}")
    print(f"uncertainty_stats: {report['uncertainty_stats']}")
    print(f"checks: {report['checks']}")
    print(f"saved_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())