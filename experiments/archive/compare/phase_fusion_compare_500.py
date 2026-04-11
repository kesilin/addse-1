import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase_fusion_scheme1 import (  # type: ignore
    FusionConfig as BaseFusionConfig,
    PhaseBranchCNN,
    PreDecoderFusionAdapter,
    build_dset,
    fused_decode,
    load_addse_lm,
    load_phase_cnn,
    run_discrete_branch,
    wrap_to_pi,
)

from addse.data import AudioStreamingDataLoader
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT


@dataclass
class CompareConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"
    addse_cfg: str = "configs/addse-s-mydata-eval-3metrics.yaml"
    addse_ckpt: str = "logs/addse-edbase-quick/checkpoints/addse-s.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"
    original_adapter_ckpt: str = "experiments/phase_fusion_scheme1_v2/weights/adapter.pt"
    balanced_init_adapter_ckpt: str = "experiments/phase_fusion_scheme1_v2/weights/adapter.pt"
    balanced_adapter_ckpt: str = "experiments/phase_fusion_scheme1_v3/weights/adapter.pt"
    report_json: str = "experiments/phase_fusion_scheme1_v3/reports/report_500.json"
    train_steps: int = 60
    train_batch_size: int = 1
    eval_examples: int = 500
    num_workers: int = 0
    lr: float = 1.2e-4
    lambda_phase: float = 0.35
    band_loss_weight: float = 0.45
    fusion_scale_original: float = 0.25
    fusion_scale_balanced: float = 0.16
    adapter_hidden: int = 512
    use_dilation: bool = True
    dilation_rates: tuple[int, int, int] = (1, 2, 4)
    phase_delta_clip: float = 1.2
    band_edges: tuple[int, int] = (64, 128)
    band_weights: tuple[float, float, float] = (0.6, 1.0, 1.4)
    snr_min: float = 0.0
    snr_max: float = 10.0
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512


def load_cfg(path: str) -> CompareConfig:
    base = asdict(CompareConfig())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    base.update(updates)
    return CompareConfig(**base)


def _band_phase_loss(phase_diff: torch.Tensor, band_edges: tuple[int, int], band_weights: tuple[float, float, float]) -> torch.Tensor:
    bins = phase_diff.shape[1]
    edges = (0, min(band_edges[0], bins), min(band_edges[1], bins), bins)
    total = phase_diff.new_zeros(())
    weight_sum = 0.0
    for (start, end), weight in zip(zip(edges[:-1], edges[1:]), band_weights, strict=True):
        if end <= start:
            continue
        band = phase_diff[:, start:end, :]
        total = total + float(weight) * (1.0 - torch.cos(band)).mean()
        weight_sum += float(weight)
    return total / max(weight_sum, 1e-8)


def _load_adapter(
    cfg: CompareConfig,
    in_channels: int,
    emb_channels: int,
    device: torch.device,
    ckpt_path: str,
) -> PreDecoderFusionAdapter:
    adapter = PreDecoderFusionAdapter(
        in_channels=in_channels,
        emb_channels=emb_channels,
        hidden=cfg.adapter_hidden,
        use_dilation=cfg.use_dilation,
        dilation_rates=cfg.dilation_rates,
    ).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    adapter.load_state_dict(state["model"], strict=True)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad = False
    return adapter


def _maybe_clone_cfg(cfg: CompareConfig, fusion_scale: float) -> BaseFusionConfig:
    base = BaseFusionConfig()
    merged = asdict(base)
    merged.update(
        {
            "seed": cfg.seed,
            "fs": cfg.fs,
            "segment_length": cfg.segment_length,
            "speech_dir": cfg.speech_dir,
            "noise_dir": cfg.noise_dir,
            "addse_cfg": cfg.addse_cfg,
            "addse_ckpt": cfg.addse_ckpt,
            "phase_cnn_ckpt": cfg.phase_cnn_ckpt,
            "adapter_ckpt": cfg.balanced_adapter_ckpt,
            "report_json": cfg.report_json,
            "train_steps": cfg.train_steps,
            "train_batch_size": cfg.train_batch_size,
            "eval_examples": cfg.eval_examples,
            "num_workers": cfg.num_workers,
            "lr": cfg.lr,
            "lambda_phase": cfg.lambda_phase,
            "fusion_scale": fusion_scale,
            "adapter_hidden": cfg.adapter_hidden,
            "use_dilation": cfg.use_dilation,
            "dilation_rates": cfg.dilation_rates,
            "phase_delta_clip": cfg.phase_delta_clip,
            "snr_min": cfg.snr_min,
            "snr_max": cfg.snr_max,
            "frame_length": cfg.frame_length,
            "hop_length": cfg.hop_length,
            "n_fft": cfg.n_fft,
        }
    )
    return BaseFusionConfig(**merged)


def _train_balanced_adapter(cfg: CompareConfig, lm, phase_cnn, adapter, stft: STFT, device: torch.device) -> None:
    dset = build_dset(_maybe_clone_cfg(cfg, cfg.fusion_scale_balanced), length=max(cfg.train_steps, 1), reset_rngs=False)
    loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr)

    adapter.train()
    for step, (noisy, clean, _) in enumerate(loader, start=1):
        if step > cfg.train_steps:
            break
        noisy = noisy.to(device)
        clean = clean.to(device)

        with torch.no_grad():
            _, y_q_sum, z_noisy = run_discrete_branch(lm, noisy)

        noisy_stft = stft(noisy)
        clean_stft = stft(clean)
        noisy_phase = torch.angle(noisy_stft[:, 0])
        clean_phase = torch.angle(clean_stft[:, 0])

        phase_feat = torch.cat([torch.cos(noisy_phase), torch.sin(noisy_phase)], dim=1)
        latent_corr = adapter(phase_feat, latent_frames=y_q_sum.shape[-1])
        y_q_fused = y_q_sum + cfg.fusion_scale_balanced * latent_corr
        y_fused = lm.nac.decoder(y_q_fused)

        fused_stft = stft(y_fused)
        fused_phase = torch.angle(fused_stft[:, 0])
        phase_diff = wrap_to_pi(fused_phase - clean_phase)

        wave_l1 = (y_fused[..., : clean.shape[-1]] - clean).abs().mean()
        phase_loss = (1.0 - torch.cos(phase_diff)).mean()
        band_loss = _band_phase_loss(phase_diff, cfg.band_edges, cfg.band_weights)
        loss = wave_l1 + cfg.lambda_phase * phase_loss + cfg.band_loss_weight * band_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 10 == 0:
            print(
                f"balanced train step={step}/{cfg.train_steps} loss={loss.item():.5f} "
                f"wave={wave_l1.item():.5f} phase={phase_loss.item():.5f} band={band_loss.item():.5f}"
            )

    Path(cfg.balanced_adapter_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": adapter.state_dict(), "config": asdict(cfg)}, cfg.balanced_adapter_ckpt)
    print(f"Saved balanced adapter weights: {cfg.balanced_adapter_ckpt}")


@torch.no_grad()
def _evaluate_variant(
    cfg: CompareConfig,
    lm,
    phase_cnn,
    adapter,
    stft: STFT,
    device: torch.device,
    fusion_scale: float,
) -> dict[str, float]:
    temp_cfg = _maybe_clone_cfg(cfg, fusion_scale)
    dset = build_dset(temp_cfg, length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0}
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        y_base, y_q_sum, z_noisy = run_discrete_branch(lm, noisy)
        y_fused = fused_decode(lm, adapter, phase_cnn, stft, noisy, y_q_sum, z_noisy, temp_cfg)
        _ = y_base  # baseline only used in separate branch

        y_fused = y_fused[..., : clean.shape[-1]]
        clean_stft = stft(clean)
        fused_stft = stft(y_fused)
        fused_phase_rmse = torch.sqrt((wrap_to_pi(torch.angle(fused_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())

        sums["pesq"] += pesq(y_fused[0], clean[0])
        sums["estoi"] += estoi(y_fused[0], clean[0])
        sums["sdr"] += sdr(y_fused[0], clean[0])
        sums["phase_rmse"] += fused_phase_rmse.item()

        n += 1
        if n % 50 == 0:
            print(f"eval {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break

    denom = max(n, 1)
    return {k: v / denom for k, v in sums.items()} | {"eval_examples": float(n)}


@torch.no_grad()
def _evaluate_baseline(cfg: CompareConfig, lm, stft: STFT, device: torch.device) -> dict[str, float]:
    dset = build_dset(_maybe_clone_cfg(cfg, cfg.fusion_scale_original), length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0}
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        y_base, _, _ = run_discrete_branch(lm, noisy)
        y_base = y_base[..., : clean.shape[-1]]
        clean_stft = stft(clean)
        base_stft = stft(y_base)
        base_phase_rmse = torch.sqrt((wrap_to_pi(torch.angle(base_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())

        sums["pesq"] += pesq(y_base[0], clean[0])
        sums["estoi"] += estoi(y_base[0], clean[0])
        sums["sdr"] += sdr(y_base[0], clean[0])
        sums["phase_rmse"] += base_phase_rmse.item()

        n += 1
        if n % 50 == 0:
            print(f"baseline eval {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break

    denom = max(n, 1)
    return {k: v / denom for k, v in sums.items()} | {"eval_examples": float(n)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Three-way comparison: baseline, original v2, and balanced parallel fusion.")
    parser.add_argument("--config", default="configs/phase_fusion_compare_500.yaml")
    parser.add_argument("--mode", choices=["train_eval", "eval_only"], default="train_eval")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lm = load_addse_lm(cfg, device)
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)
    phase_bins = cfg.n_fft // 2 + 1
    in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
    phase_cnn = load_phase_cnn(cfg, in_ch=in_cont_ch, out_ch=phase_bins, device=device)

    original_adapter = _load_adapter(cfg, 2 * phase_bins, lm.nac.decoder.in_conv.conv.in_channels, device, cfg.original_adapter_ckpt)
    original_report = _evaluate_variant(cfg, lm, phase_cnn, original_adapter, stft, device, cfg.fusion_scale_original)

    balanced_adapter = PreDecoderFusionAdapter(
        in_channels=2 * phase_bins,
        emb_channels=lm.nac.decoder.in_conv.conv.in_channels,
        hidden=cfg.adapter_hidden,
        use_dilation=cfg.use_dilation,
        dilation_rates=cfg.dilation_rates,
    ).to(device)
    if args.mode == "train_eval" and os.path.exists(cfg.balanced_init_adapter_ckpt):
        init_state = torch.load(cfg.balanced_init_adapter_ckpt, map_location=device)
        balanced_adapter.load_state_dict(init_state["model"], strict=True)
        print(f"Loaded balanced init adapter: {cfg.balanced_init_adapter_ckpt}")
        _train_balanced_adapter(cfg, lm, phase_cnn, balanced_adapter, stft, device)
    elif args.mode == "eval_only" and os.path.exists(cfg.balanced_adapter_ckpt):
        balanced_state = torch.load(cfg.balanced_adapter_ckpt, map_location=device)
        balanced_adapter.load_state_dict(balanced_state["model"], strict=True)
        print(f"Loaded balanced adapter: {cfg.balanced_adapter_ckpt}")
    else:
        raise FileNotFoundError(
            f"Balanced adapter checkpoint not found for mode={args.mode}: {cfg.balanced_init_adapter_ckpt} / {cfg.balanced_adapter_ckpt}"
        )

    balanced_adapter.eval()
    balanced_report = _evaluate_variant(cfg, lm, phase_cnn, balanced_adapter, stft, device, cfg.fusion_scale_balanced)
    baseline_report = _evaluate_baseline(cfg, lm, stft, device)

    report = {
        "baseline": baseline_report,
        "original_v2": original_report,
        "balanced_v3": balanced_report,
        "delta_original_minus_baseline": {k: original_report[k] - baseline_report[k] for k in ["pesq", "estoi", "sdr", "phase_rmse"]},
        "delta_balanced_minus_baseline": {k: balanced_report[k] - baseline_report[k] for k in ["pesq", "estoi", "sdr", "phase_rmse"]},
        "delta_balanced_minus_original": {k: balanced_report[k] - original_report[k] for k in ["pesq", "estoi", "sdr", "phase_rmse"]},
    }

    Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Three-way Comparison ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {cfg.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
