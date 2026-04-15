import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = EXPERIMENTS_DIR.parents[1]
PROJECT_ROOT = EXPERIMENTS_DIR.parents[2]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase_fusion_scheme1 import build_dset, wrap_to_pi  # type: ignore
from addse.data import AudioStreamingDataLoader
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT


@dataclass
class ProbeConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"
    snr_min: float = 0.0
    snr_max: float = 10.0

    train_steps: int = 40
    train_batch_size: int = 4
    eval_examples: int = 60
    pesq_max_examples: int = 20
    quick_eval_every_steps: int = 20
    num_workers: int = 0

    lr: float = 1.0e-4
    phase_delta_clip: float = 1.2
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512

    baseline_hidden: int = 128
    residual_hidden: int = 96
    hard_threshold: float = 0.55
    route_temperature: float = 0.18
    gate_entropy_weight: float = 0.002
    uncertainty_balance_weight: float = 0.01
    uncertainty_target: float = 0.5

    report_json: str = "experiments/archive/v4/reports/phase_residual_only_probe_report.json"
    weights_dir: str = "experiments/archive/v4/weights/phase_residual"


class BaselineExtractor(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.net(x), None


class CurrentPhaseResidualExtractor(nn.Module):
    """Current uncertainty-routed residual design used for phase residual correction."""

    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 96,
        hard_threshold: float = 0.55,
        route_temperature: float = 0.18,
    ) -> None:
        super().__init__()
        self.hard_threshold = float(hard_threshold)
        self.route_temperature = float(route_temperature)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.uncertainty = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        # Smooth routing avoids hard 0/1 collapse and preserves branch mixing.
        temp = max(self.route_temperature, 1e-3)
        return torch.sigmoid((u - self.hard_threshold) / temp)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        u = self.uncertainty(h)
        m = self._routing_mask(u)
        low = self.low_branch(h)
        high = self.high_branch(h)
        delta = (1.0 - m) * low + m * high
        return delta, u


class BandAwarePhaseResidualExtractor(nn.Module):
    """Innovation: uncertainty-routed residual plus lightweight band-aware gain."""

    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 96,
        hard_threshold: float = 0.55,
        num_bands: int = 3,
        route_temperature: float = 0.18,
    ) -> None:
        super().__init__()
        self.hard_threshold = float(hard_threshold)
        self.num_bands = int(num_bands)
        self.route_temperature = float(route_temperature)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.uncertainty = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

        self.band_gate = nn.Sequential(
            nn.Conv1d(hidden + 1, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, self.num_bands, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.08))

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        temp = max(self.route_temperature, 1e-3)
        return torch.sigmoid((u - self.hard_threshold) / temp)

    def _band_gain_map(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Build band-level gain and upsample back to frequency bins.
        bsz, ch, freq, _ = h.shape
        h_pool = h.mean(dim=-1)  # (B,C,F)
        u_pool = u.mean(dim=-1)  # (B,1,F)
        feat = torch.cat([h_pool, u_pool], dim=1)
        band_w = self.band_gate(feat)  # (B,num_bands,F)

        edges = torch.linspace(0, freq, steps=self.num_bands + 1, device=h.device)
        gain = torch.zeros((bsz, 1, freq, 1), device=h.device, dtype=h.dtype)
        for idx in range(self.num_bands):
            start = int(edges[idx].item())
            end = int(edges[idx + 1].item())
            if end <= start:
                continue
            w = band_w[:, idx : idx + 1, start:end].mean(dim=-1, keepdim=True).unsqueeze(-1)
            gain[:, :, start:end, :] = w
        return gain.clamp(min=0.2, max=1.8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        u = self.uncertainty(h)
        m = self._routing_mask(u)
        low = self.low_branch(h)
        high = self.high_branch(h)

        routed = (1.0 - m) * low + m * high
        gain = self._band_gain_map(h, u)
        scale = torch.clamp(self.residual_scale, min=0.0, max=0.3)
        delta = low + scale * gain * torch.tanh(routed)
        return delta, u


class PhaseResidualStabilityV2Extractor(nn.Module):
    """Optimized v2: uncertainty routing with temporal consistency and bounded residual gain."""

    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 96,
        hard_threshold: float = 0.55,
        route_temperature: float = 0.16,
    ) -> None:
        super().__init__()
        self.hard_threshold = float(hard_threshold)
        self.route_temperature = float(route_temperature)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.uncertainty = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

        self.temporal_refine = nn.Sequential(
            nn.Conv2d(2, hidden // 2, kernel_size=(1, 5), padding=(0, 2)),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )
        self.mix_proj = nn.Sequential(
            nn.Conv2d(3, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.12))

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        temp = max(self.route_temperature, 1e-3)
        return torch.sigmoid((u - self.hard_threshold) / temp)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        u = self.uncertainty(h)
        m = self._routing_mask(u)

        low = self.low_branch(h)
        high = self.high_branch(h)
        routed = (1.0 - m) * low + m * high

        temporal_in = torch.cat([routed, u], dim=1)
        temporal = self.temporal_refine(temporal_in)

        mix_in = torch.cat([low, routed, temporal], dim=1)
        mix = self.mix_proj(mix_in)
        scale = torch.clamp(self.residual_scale, min=0.0, max=0.25)
        delta = low + scale * mix
        return delta, u


class PhaseResidualStabilityV3Extractor(nn.Module):
    """V3: V2-style residual with uncertainty-conditioned tail suppression."""

    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 96,
        hard_threshold: float = 0.52,
        route_temperature: float = 0.14,
    ) -> None:
        super().__init__()
        self.hard_threshold = float(hard_threshold)
        self.route_temperature = float(route_temperature)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.uncertainty = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.temporal_refine = nn.Sequential(
            nn.Conv2d(2, hidden // 2, kernel_size=(1, 5), padding=(0, 2)),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )
        self.mix_proj = nn.Sequential(
            nn.Conv2d(3, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Tanh(),
        )
        self.peak_gate = nn.Sequential(
            nn.Conv2d(2, hidden // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        temp = max(self.route_temperature, 1e-3)
        return torch.sigmoid((u - self.hard_threshold) / temp)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        u = self.uncertainty(h)
        m = self._routing_mask(u)

        low = self.low_branch(h)
        high = self.high_branch(h)
        routed = (1.0 - m) * low + m * high

        temporal = self.temporal_refine(torch.cat([routed, u], dim=1))
        mix = self.mix_proj(torch.cat([low, routed, temporal], dim=1))

        # Suppress extreme residual tails in uncertain regions while keeping fine corrections.
        peak = torch.sigmoid(6.0 * (routed.abs() - 0.35))
        peak_ctrl = self.peak_gate(torch.cat([u, peak], dim=1))
        tail_suppress = 1.0 - 0.45 * peak * peak_ctrl

        scale = torch.clamp(self.residual_scale, min=0.0, max=0.22)
        delta = low + scale * tail_suppress * mix
        return delta, u


def _make_dset_cfg(cfg: ProbeConfig):
    class _Tmp:
        pass

    o = _Tmp()
    o.seed = cfg.seed
    o.fs = cfg.fs
    o.segment_length = cfg.segment_length
    o.speech_dir = cfg.speech_dir
    o.noise_dir = cfg.noise_dir
    o.snr_min = cfg.snr_min
    o.snr_max = cfg.snr_max
    return o


def _make_input_and_target(stft: STFT, noisy: torch.Tensor, clean: torch.Tensor):
    noisy_stft = stft(noisy)
    clean_stft = stft(clean)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    clean_phase = torch.angle(clean_stft[:, 0])
    target_delta = wrap_to_pi(clean_phase - noisy_phase)
    noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
    noisy_mag = noisy_mag / (noisy_mag.abs().amax(dim=1, keepdim=True).clamp(min=1e-6))
    x = torch.stack([torch.cos(noisy_phase), torch.sin(noisy_phase), noisy_mag], dim=1)
    return x, noisy_phase, target_delta, noisy_stft


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    den = (x.norm() * y.norm()).clamp(min=1e-8)
    return float((x @ y / den).item())


def _evaluate_model(model: nn.Module, cfg: ProbeConfig, stft: STFT, device: torch.device) -> dict[str, float]:
    dset = build_dset(_make_dset_cfg(cfg), length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {
        "pesq": 0.0,
        "estoi": 0.0,
        "sdr": 0.0,
        "phase_rmse": 0.0,
        "phase_mae": 0.0,
        "phase_p95_abs": 0.0,
        "phase_large_error_ratio": 0.0,
        "alignment_cos": 0.0,
        "uncertainty_mean": 0.0,
        "uncertainty_error_corr": 0.0,
        "high_uncertainty_ratio": 0.0,
    }
    n_pesq = 0
    n = 0

    model.eval()
    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            x, noisy_phase, target_delta, noisy_stft = _make_input_and_target(stft, noisy, clean)
            pred_delta, uncertainty = model(x)
            pred_delta = pred_delta[:, 0].clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip)
            uncertainty = None if uncertainty is None else uncertainty[:, 0]

            pred_phase = wrap_to_pi(noisy_phase + pred_delta)
            pred_spec = noisy_stft[:, 0].abs() * torch.exp(1j * pred_phase)
            pred_wave = stft.inverse(pred_spec.unsqueeze(1), n=clean.shape[-1])

            phase_err = wrap_to_pi(pred_phase - torch.angle(stft(clean)[:, 0]))
            rmse = torch.sqrt(phase_err.square().mean())
            mae = phase_err.abs().mean()
            p95 = torch.quantile(phase_err.abs().reshape(-1), 0.95)
            large_err_ratio = (phase_err.abs() > (0.5 * torch.pi)).float().mean()
            align_cos = torch.cos(wrap_to_pi(pred_delta - target_delta)).mean()

            if n_pesq < max(0, int(cfg.pesq_max_examples)):
                sums["pesq"] += pesq(pred_wave[0], clean[0])
                n_pesq += 1
            sums["estoi"] += estoi(pred_wave[0], clean[0])
            sums["sdr"] += sdr(pred_wave[0], clean[0])
            sums["phase_rmse"] += float(rmse.item())
            sums["phase_mae"] += float(mae.item())
            sums["phase_p95_abs"] += float(p95.item())
            sums["phase_large_error_ratio"] += float(large_err_ratio.item())
            sums["alignment_cos"] += float(align_cos.item())

            if uncertainty is not None:
                u = uncertainty.detach()
                abs_err = wrap_to_pi(pred_delta - target_delta).abs().detach()
                sums["uncertainty_mean"] += float(u.mean().item())
                sums["uncertainty_error_corr"] += _corrcoef(u, abs_err)
                sums["high_uncertainty_ratio"] += float((u > 0.5).float().mean().item())

            n += 1
            if n % 20 == 0:
                print(f"eval {n}/{cfg.eval_examples}")
            if n >= cfg.eval_examples:
                break

    denom = max(n, 1)
    out = {k: v / denom for k, v in sums.items()} | {"eval_examples": float(n)}
    out["pesq"] = sums["pesq"] / float(max(n_pesq, 1))
    out["pesq_examples"] = float(n_pesq)
    return out


def _train_one(
    name: str,
    model: nn.Module,
    cfg: ProbeConfig,
    stft: STFT,
    device: torch.device,
    save_path: Path | None = None,
) -> dict[str, object]:
    dset = build_dset(_make_dset_cfg(cfg), length=max(cfg.train_steps * cfg.train_batch_size, 1), reset_rngs=False)
    loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    step = 0
    last_loss = 0.0
    model.train()
    for noisy, clean, _ in loader:
        if step >= cfg.train_steps:
            break
        noisy = noisy.to(device)
        clean = clean.to(device)
        x, _noisy_phase, target_delta, _noisy_stft = _make_input_and_target(stft, noisy, clean)
        pred_delta, uncertainty = model(x)
        pred_delta = pred_delta[:, 0].clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip)
        uncertainty = None if uncertainty is None else uncertainty[:, 0]

        phase_loss = (1.0 - torch.cos(wrap_to_pi(pred_delta - target_delta))).mean()
        loss = phase_loss
        if uncertainty is not None:
            u = uncertainty.clamp(min=1e-6, max=1.0 - 1e-6)
            gate_entropy = -(u * u.log() + (1.0 - u) * (1.0 - u).log()).mean()
            balance = (u.mean() - float(cfg.uncertainty_target)).square()
            loss = loss + cfg.gate_entropy_weight * gate_entropy + cfg.uncertainty_balance_weight * balance

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1
        last_loss = float(loss.item())
        if step == 1 or step % 20 == 0:
            print(f"[{name}] step={step}/{cfg.train_steps} loss={last_loss:.5f}")

        if cfg.quick_eval_every_steps > 0 and step % cfg.quick_eval_every_steps == 0:
            quick_cfg = ProbeConfig(**asdict(cfg))
            quick_cfg.eval_examples = min(20, cfg.eval_examples)
            quick = _evaluate_model(model, quick_cfg, stft, device)
            print(
                f"[{name}] quick_eval@{step} pesq={quick['pesq']:.4f} estoi={quick['estoi']:.4f} "
                f"sdr={quick['sdr']:.4f} prmse={quick['phase_rmse']:.4f} align={quick['alignment_cos']:.4f}"
            )
            model.train()

    final_eval = _evaluate_model(model, cfg, stft, device)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "name": name,
                "config": asdict(cfg),
                "final_eval": final_eval,
            },
            str(save_path),
        )
        print(f"[{name}] saved weights: {save_path}")
    return {
        "name": name,
        "params": _param_count(model),
        "train_steps": cfg.train_steps,
        "last_train_loss": last_loss,
        "eval": final_eval,
    }


def _load_init_weights_if_present(model: nn.Module, weight_path: Path, device: torch.device) -> bool:
    if not weight_path.exists():
        return False
    state = torch.load(str(weight_path), map_location=device)
    if not isinstance(state, dict) or "model" not in state:
        raise RuntimeError(f"Invalid weight file format: {weight_path}")
    model.load_state_dict(state["model"], strict=True)
    return True


def _rank_key(item: dict[str, object]) -> tuple[float, float, float]:
    ev = item["eval"]
    assert isinstance(ev, dict)
    return (float(ev["sdr"]), float(ev["pesq"]), float(ev["estoi"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase residual module probe only (no discrete trunk).")
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-examples", type=int, default=60)
    parser.add_argument("--pesq-max-examples", type=int, default=20)
    parser.add_argument("--quick-eval-every-steps", type=int, default=20)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=10.0)
    parser.add_argument("--hard-threshold", type=float, default=0.55)
    parser.add_argument("--route-temperature", type=float, default=0.18)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.002)
    parser.add_argument("--uncertainty-balance-weight", type=float, default=0.01)
    parser.add_argument("--uncertainty-target", type=float, default=0.5)
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline_extractor,phase_residual_current,phase_residual_stability_v2",
        help="Comma-separated subset of variants to run.",
    )
    parser.add_argument("--report-json", type=str, default="")
    parser.add_argument("--weights-dir", type=str, default="")
    parser.add_argument(
        "--resume-from-weights-dir",
        type=str,
        default="",
        help="Optional directory containing {variant}.pt files for warm-start/resume.",
    )
    args = parser.parse_args()

    cfg = ProbeConfig(
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        eval_examples=args.eval_examples,
        pesq_max_examples=args.pesq_max_examples,
        quick_eval_every_steps=args.quick_eval_every_steps,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        hard_threshold=args.hard_threshold,
        route_temperature=args.route_temperature,
        gate_entropy_weight=args.gate_entropy_weight,
        uncertainty_balance_weight=args.uncertainty_balance_weight,
        uncertainty_target=args.uncertainty_target,
    )
    if args.report_json.strip():
        cfg.report_json = args.report_json
    if args.weights_dir.strip():
        cfg.weights_dir = args.weights_dir

    os.chdir(PROJECT_ROOT)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)

    variants: list[tuple[str, nn.Module]] = [
        ("baseline_extractor", BaselineExtractor(hidden=cfg.baseline_hidden)),
        (
            "phase_residual_current",
            CurrentPhaseResidualExtractor(
                hidden=cfg.residual_hidden,
                hard_threshold=cfg.hard_threshold,
                route_temperature=cfg.route_temperature,
            ),
        ),
        (
            "phase_residual_stability_v2",
            PhaseResidualStabilityV2Extractor(
                hidden=cfg.residual_hidden,
                hard_threshold=cfg.hard_threshold,
                route_temperature=cfg.route_temperature,
            ),
        ),
        (
            "phase_residual_stability_v3",
            PhaseResidualStabilityV3Extractor(
                hidden=cfg.residual_hidden,
                hard_threshold=cfg.hard_threshold,
                route_temperature=max(0.12, cfg.route_temperature - 0.02),
            ),
        ),
        (
            "phase_residual_bandaware_innov",
            BandAwarePhaseResidualExtractor(
                hidden=cfg.residual_hidden,
                hard_threshold=cfg.hard_threshold,
                num_bands=3,
                route_temperature=cfg.route_temperature,
            ),
        ),
    ]

    if args.variants.strip():
        wanted = {name.strip() for name in args.variants.split(",") if name.strip()}
        variants = [item for item in variants if item[0] in wanted]
        if "baseline_extractor" not in {name for name, _ in variants}:
            variants.insert(0, ("baseline_extractor", BaselineExtractor(hidden=cfg.baseline_hidden)))

    outputs: dict[str, dict[str, object]] = {}
    weights_root = Path(cfg.weights_dir)
    resume_root = Path(args.resume_from_weights_dir) if args.resume_from_weights_dir.strip() else None
    for name, model in variants:
        if resume_root is not None:
            init_path = resume_root / f"{name}.pt"
            if _load_init_weights_if_present(model, init_path, device):
                print(f"[{name}] loaded init weights: {init_path}")
        outputs[name] = _train_one(name, model, cfg, stft, device, save_path=weights_root / f"{name}.pt")

    baseline_out = outputs["baseline_extractor"]
    baseline_eval = baseline_out["eval"]
    assert isinstance(baseline_eval, dict)

    delta: dict[str, dict[str, float]] = {}
    for name, cur in outputs.items():
        if name == "baseline_extractor":
            continue
        cur_eval = cur["eval"]
        assert isinstance(cur_eval, dict)
        delta[name] = {
            "pesq": float(cur_eval["pesq"]) - float(baseline_eval["pesq"]),
            "estoi": float(cur_eval["estoi"]) - float(baseline_eval["estoi"]),
            "sdr": float(cur_eval["sdr"]) - float(baseline_eval["sdr"]),
            "phase_rmse": float(cur_eval["phase_rmse"]) - float(baseline_eval["phase_rmse"]),
            "phase_mae": float(cur_eval["phase_mae"]) - float(baseline_eval["phase_mae"]),
            "phase_p95_abs": float(cur_eval["phase_p95_abs"]) - float(baseline_eval["phase_p95_abs"]),
            "phase_large_error_ratio": float(cur_eval["phase_large_error_ratio"])
            - float(baseline_eval["phase_large_error_ratio"]),
            "alignment_cos": float(cur_eval["alignment_cos"]) - float(baseline_eval["alignment_cos"]),
            "params_ratio": float(cur["params"]) / max(float(baseline_out["params"]), 1.0),
        }

    ranking = sorted(outputs.values(), key=_rank_key, reverse=True)
    ranking_names = [str(r["name"]) for r in ranking]
    report = {
        "config": asdict(cfg),
        "results": outputs,
        "delta_vs_baseline": delta,
        "ranking_by_performance": ranking_names,
    }

    report_path = Path(cfg.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== phase residual only probe report ===")
    print(json.dumps(report["ranking_by_performance"], ensure_ascii=False, indent=2))
    print(json.dumps(report["delta_vs_baseline"], ensure_ascii=False, indent=2))
    print(f"saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
