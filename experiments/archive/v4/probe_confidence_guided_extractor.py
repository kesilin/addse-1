import argparse
import json
import math
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

    train_steps: int = 120
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
    confidence_hidden: int = 128
    high_hidden_mult: float = 1.5
    hard_threshold: float = 0.55
    topk_ratio: float = 0.25
    gate_entropy_weight: float = 0.002

    report_json: str = "experiments/archive/v4/reports/confidence_extractor_probe_report.json"
    weights_dir: str = "experiments/archive/v4/weights/confidence_extractor"


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
        delta = self.net(x)
        return delta, None


class ConfidenceGuidedExtractor(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 128,
        routing_mode: str = "soft",
        hard_threshold: float = 0.55,
        topk_ratio: float = 0.25,
        high_hidden_mult: float = 1.5,
        high_depth: int = 3,
        high_depthwise: bool = False,
    ) -> None:
        super().__init__()
        if routing_mode not in {"soft", "hard", "topk"}:
            raise ValueError(f"Unsupported routing_mode={routing_mode}")
        self.routing_mode = routing_mode
        self.hard_threshold = float(hard_threshold)
        self.topk_ratio = float(topk_ratio)
        self.high_depth = max(1, int(high_depth))
        self.high_depthwise = bool(high_depthwise)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.low_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

        high_hidden = max(hidden, int(hidden * high_hidden_mult))
        high_blocks: list[nn.Module] = [
            nn.Conv2d(hidden, high_hidden, kernel_size=3, padding=1, dilation=1),
            nn.SiLU(),
        ]
        if self.high_depth >= 2:
            groups_2 = high_hidden if self.high_depthwise else 1
            high_blocks.extend(
                [
                    nn.Conv2d(high_hidden, high_hidden, kernel_size=3, padding=2, dilation=2, groups=groups_2),
                    nn.SiLU(),
                ]
            )
        if self.high_depth >= 3:
            groups_3 = high_hidden if self.high_depthwise else 1
            high_blocks.extend(
                [
                    nn.Conv2d(high_hidden, high_hidden, kernel_size=3, padding=4, dilation=4, groups=groups_3),
                    nn.SiLU(),
                ]
            )
        high_blocks.append(nn.Conv2d(high_hidden, 1, kernel_size=1))
        self.high_branch = nn.Sequential(*high_blocks)

    @staticmethod
    def _topk_mask(u: torch.Tensor, ratio: float) -> torch.Tensor:
        # u: (B,1,F,T), returns hard mask with same shape
        bsz = u.shape[0]
        flat = u.reshape(bsz, -1)
        n = flat.shape[1]
        k = max(1, int(round(ratio * n)))
        # largest top-k uncertain positions
        topk_idx = torch.topk(flat, k=k, dim=1, largest=True).indices
        mask = torch.zeros_like(flat)
        mask.scatter_(1, topk_idx, 1.0)
        return mask.view_as(u)

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        if self.routing_mode == "soft":
            return u
        if self.routing_mode == "hard":
            hard = (u > self.hard_threshold).float()
            # Straight-through: hard in forward, soft in backward.
            return hard + (u - u.detach())
        topk = self._topk_mask(u, self.topk_ratio)
        return topk + (u - u.detach())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        u = self.uncertainty_head(h)
        low = self.low_branch(h)
        high = self.high_branch(h)
        m = self._routing_mask(u)
        delta = (1.0 - m) * low + m * high
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

    x = torch.stack([
        torch.cos(noisy_phase),
        torch.sin(noisy_phase),
        noisy_mag,
    ], dim=1)
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


def _evaluate_model(
    model: nn.Module,
    cfg: ProbeConfig,
    stft: STFT,
    device: torch.device,
) -> dict[str, float]:
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
            pred_delta = pred_delta[:, 0]
            uncertainty = None if uncertainty is None else uncertainty[:, 0]
            pred_delta = pred_delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip)
            pred_phase = wrap_to_pi(noisy_phase + pred_delta)

            pred_spec = noisy_stft[:, 0].abs() * torch.exp(1j * pred_phase)
            pred_wave = stft.inverse(pred_spec.unsqueeze(1), n=clean.shape[-1])

            phase_err = wrap_to_pi(pred_phase - torch.angle(stft(clean)[:, 0]))
            rmse = torch.sqrt((phase_err.square()).mean())
            align_cos = torch.cos(wrap_to_pi(pred_delta - target_delta)).mean()

            if n_pesq < max(0, int(cfg.pesq_max_examples)):
                sums["pesq"] += pesq(pred_wave[0], clean[0])
                n_pesq += 1
            sums["estoi"] += estoi(pred_wave[0], clean[0])
            sums["sdr"] += sdr(pred_wave[0], clean[0])
            sums["phase_rmse"] += float(rmse.item())
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
    if n_pesq > 0:
        out["pesq"] = sums["pesq"] / float(n_pesq)
        out["pesq_examples"] = float(n_pesq)
    else:
        out["pesq"] = float("nan")
        out["pesq_examples"] = 0.0
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

    model.train()
    step = 0
    last_loss = 0.0
    for noisy, clean, _ in loader:
        if step >= cfg.train_steps:
            break
        noisy = noisy.to(device)
        clean = clean.to(device)

        x, _noisy_phase, target_delta, _noisy_stft = _make_input_and_target(stft, noisy, clean)
        pred_delta, uncertainty = model(x)
        pred_delta = pred_delta[:, 0]
        uncertainty = None if uncertainty is None else uncertainty[:, 0]
        pred_delta = pred_delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip)

        phase_loss = (1.0 - torch.cos(wrap_to_pi(pred_delta - target_delta))).mean()
        loss = phase_loss

        if uncertainty is not None:
            u = uncertainty.clamp(min=1e-6, max=1.0 - 1e-6)
            gate_entropy = -(u * u.log() + (1.0 - u) * (1.0 - u).log()).mean()
            loss = loss + cfg.gate_entropy_weight * gate_entropy

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


def _rank_key(item: dict[str, object]) -> tuple[float, float, float]:
    ev = item["eval"]
    assert isinstance(ev, dict)
    # Performance-first: SDR, then PESQ, then ESTOI.
    return (float(ev["sdr"]), float(ev["pesq"]), float(ev["estoi"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone confidence-guided extractor probe (no discrete trunk).")
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-examples", type=int, default=60)
    parser.add_argument("--pesq-max-examples", type=int, default=20)
    parser.add_argument("--quick-eval-every-steps", type=int, default=20)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--baseline-hidden", type=int, default=128)
    parser.add_argument("--confidence-hidden", type=int, default=128)
    parser.add_argument("--high-hidden-mult", type=float, default=1.5)
    parser.add_argument("--hard-threshold", type=float, default=0.55)
    parser.add_argument("--topk-ratio", type=float, default=0.25)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.002)
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated subset of variants to run, e.g. baseline_extractor,confidence_soft_lite.",
    )
    parser.add_argument("--report-json", type=str, default="")
    parser.add_argument("--weights-dir", type=str, default="")
    parser.add_argument("--lite-only", action="store_true")
    args = parser.parse_args()

    cfg = ProbeConfig(
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        eval_examples=args.eval_examples,
        pesq_max_examples=args.pesq_max_examples,
        quick_eval_every_steps=args.quick_eval_every_steps,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        lr=args.lr,
        baseline_hidden=args.baseline_hidden,
        confidence_hidden=args.confidence_hidden,
        high_hidden_mult=args.high_hidden_mult,
        hard_threshold=args.hard_threshold,
        topk_ratio=args.topk_ratio,
        gate_entropy_weight=args.gate_entropy_weight,
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
            "confidence_soft_ref",
            ConfidenceGuidedExtractor(
                hidden=cfg.confidence_hidden,
                routing_mode="soft",
                hard_threshold=cfg.hard_threshold,
                topk_ratio=cfg.topk_ratio,
                high_hidden_mult=cfg.high_hidden_mult,
                high_depth=3,
                high_depthwise=False,
            ),
        ),
        (
            "confidence_soft_lite",
            ConfidenceGuidedExtractor(
                hidden=max(64, cfg.confidence_hidden - 32),
                routing_mode="soft",
                hard_threshold=cfg.hard_threshold,
                topk_ratio=cfg.topk_ratio,
                high_hidden_mult=1.0,
                high_depth=2,
                high_depthwise=True,
            ),
        ),
        (
            "confidence_hard_lite",
            ConfidenceGuidedExtractor(
                hidden=max(64, cfg.confidence_hidden - 32),
                routing_mode="hard",
                hard_threshold=max(0.5, cfg.hard_threshold),
                topk_ratio=cfg.topk_ratio,
                high_hidden_mult=1.0,
                high_depth=2,
                high_depthwise=True,
            ),
        ),
        (
            "confidence_topk_tiny",
            ConfidenceGuidedExtractor(
                hidden=max(48, cfg.confidence_hidden - 48),
                routing_mode="topk",
                hard_threshold=cfg.hard_threshold,
                topk_ratio=min(0.2, cfg.topk_ratio),
                high_hidden_mult=1.0,
                high_depth=1,
                high_depthwise=True,
            ),
        ),
    ]

    if args.lite_only:
        variants = [
            ("baseline_extractor", BaselineExtractor(hidden=cfg.baseline_hidden)),
            variants[2],
            variants[3],
            variants[4],
        ]

    if args.variants.strip():
        wanted = {name.strip() for name in args.variants.split(",") if name.strip()}
        variants = [item for item in variants if item[0] in wanted]
        if "baseline_extractor" not in wanted:
            variants.insert(0, ("baseline_extractor", BaselineExtractor(hidden=cfg.baseline_hidden)))

    outputs: dict[str, dict[str, object]] = {}
    weights_root = Path(cfg.weights_dir)
    for name, model in variants:
        outputs[name] = _train_one(name, model, cfg, stft, device, save_path=weights_root / f"{name}.pt")

    baseline_out = outputs["baseline_extractor"]
    baseline_eval = baseline_out["eval"]
    assert isinstance(baseline_eval, dict)

    delta: dict[str, dict[str, float]] = {}
    for name in outputs:
        if name == "baseline_extractor":
            continue
        cur = outputs[name]
        cur_eval = cur["eval"]
        assert isinstance(cur_eval, dict)
        delta[name] = {
            "pesq": float(cur_eval["pesq"]) - float(baseline_eval["pesq"]),
            "estoi": float(cur_eval["estoi"]) - float(baseline_eval["estoi"]),
            "sdr": float(cur_eval["sdr"]) - float(baseline_eval["sdr"]),
            "phase_rmse": float(cur_eval["phase_rmse"]) - float(baseline_eval["phase_rmse"]),
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

    print("=== confidence extractor probe report ===")
    print(json.dumps(report["ranking_by_performance"], ensure_ascii=False, indent=2))
    print(json.dumps(report["delta_vs_baseline"], ensure_ascii=False, indent=2))
    print(f"saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
