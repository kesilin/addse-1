import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from addse.data import AudioStreamingDataLoader, AudioStreamingDataset, DynamicMixingDataset
from addse.lightning import load_nac
from addse.stft import STFT


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


class PhaseBranchCNN(nn.Module):
    """Lightweight 1D CNN that predicts STFT phase residual from NAC continuous features."""

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=8),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, z_noisy: torch.Tensor, target_frames: int) -> torch.Tensor:
        # z_noisy: (B, C, Tnac) -> delta_phase: (B, F, Tstft)
        delta = self.net(z_noisy)
        return F.interpolate(delta, size=target_frames, mode="linear", align_corners=False)


@dataclass
class ProbeConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"
    nac_cfg: str = "configs/nac.yaml"
    nac_ckpt: str = "logs/nac/checkpoints/last.ckpt"
    weights_path: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"
    report_path: str = "experiments/phase_branch2/reports/phase_probe_report.json"
    train_steps: int = 200
    train_batch_size: int = 4
    eval_examples: int = 100
    num_workers: int = 0
    lr: float = 3e-4
    snr_min: float = 0.0
    snr_max: float = 10.0
    lambda_group_delay: float = 0.3
    phase_delta_clip: float = 1.2
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512


def load_config(path: str | None) -> ProbeConfig:
    cfg = ProbeConfig()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    base = asdict(cfg)
    base.update(updates)
    return ProbeConfig(**base)


def build_dataset(cfg: ProbeConfig, length: int | float, reset_rngs: bool) -> DynamicMixingDataset:
    speech_dataset = AudioStreamingDataset(
        input_dir=cfg.speech_dir,
        fs=cfg.fs,
        segment_length=cfg.segment_length,
        max_dynamic_range=25.0,
        shuffle=not reset_rngs,
        seed=cfg.seed,
    )
    noise_dataset = AudioStreamingDataset(
        input_dir=cfg.noise_dir,
        fs=cfg.fs,
        segment_length=cfg.segment_length,
        max_dynamic_range=25.0,
        shuffle=not reset_rngs,
        seed=cfg.seed,
    )
    return DynamicMixingDataset(
        speech_dataset=speech_dataset,
        noise_dataset=noise_dataset,
        snr_range=(cfg.snr_min, cfg.snr_max),
        rms_range=(0.0, 0.0),
        length=length,
        resume=False,
        reset_rngs=reset_rngs,
    )


def phase_losses(
    pred_phase: torch.Tensor,
    clean_phase: torch.Tensor,
    cfg: ProbeConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    phase_diff = wrap_to_pi(pred_phase - clean_phase)
    phase_loss = (1.0 - torch.cos(phase_diff)).mean()

    pred_gd = wrap_to_pi(pred_phase[:, 1:, :] - pred_phase[:, :-1, :])
    clean_gd = wrap_to_pi(clean_phase[:, 1:, :] - clean_phase[:, :-1, :])
    gd_loss = (pred_gd - clean_gd).abs().mean()

    total = phase_loss + cfg.lambda_group_delay * gd_loss
    stats = {
        "phase_loss": phase_loss.detach(),
        "group_delay_loss": gd_loss.detach(),
    }
    return total, stats


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, cfg: ProbeConfig) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": asdict(cfg),
    }
    torch.save(ckpt, path)


def load_checkpoint_if_exists(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))


@torch.no_grad()
def evaluate_quantizer_baseline(
    cfg: ProbeConfig,
    nac: nn.Module,
    stft: STFT,
    device: torch.device,
) -> dict[str, float]:
    eval_dset = build_dataset(cfg, length=cfg.eval_examples, reset_rngs=True)
    eval_loader = AudioStreamingDataLoader(eval_dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    phase_rmse_sum = 0.0
    gd_mae_sum = 0.0
    plv_sum = 0.0
    phase_cos_sum = 0.0
    align_feat_sum = 0.0

    n = 0
    for noisy, clean, _ in eval_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        clean_stft = stft(clean)
        clean_phase = torch.angle(clean_stft[:, 0])

        _, q_noisy = nac.encode(noisy, domain="q")
        recon_q = nac.decode(q_noisy, domain="q")
        q_stft = stft(recon_q)
        q_phase = torch.angle(q_stft[:, 0])

        phase_diff = wrap_to_pi(q_phase - clean_phase)
        phase_rmse_sum += torch.sqrt((phase_diff.square()).mean()).item()
        phase_cos_sum += torch.cos(phase_diff).mean().item()

        pred_gd = wrap_to_pi(q_phase[:, 1:, :] - q_phase[:, :-1, :])
        clean_gd = wrap_to_pi(clean_phase[:, 1:, :] - clean_phase[:, :-1, :])
        gd_mae_sum += (pred_gd - clean_gd).abs().mean().item()

        mean_cos = torch.cos(phase_diff).mean()
        mean_sin = torch.sin(phase_diff).mean()
        plv_sum += torch.sqrt(mean_cos.square() + mean_sin.square()).item()

        _, z_clean = nac.encode(clean, domain="x")
        _, z_q_recon = nac.encode(recon_q, domain="x")
        align_feat_sum += F.cosine_similarity(z_q_recon.flatten(1), z_clean.flatten(1), dim=1).mean().item()

        n += 1
        if n % 10 == 0:
            print(f"[quantizer eval] {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break

    denom = max(n, 1)
    return {
        "eval_examples": float(n),
        "phase_rmse": phase_rmse_sum / denom,
        "phase_cos_alignment": phase_cos_sum / denom,
        "group_delay_mae": gd_mae_sum / denom,
        "plv": plv_sum / denom,
        "nac_feature_alignment_to_clean": align_feat_sum / denom,
    }


@torch.no_grad()
def evaluate_cnn_phase_branch(
    cfg: ProbeConfig,
    model: PhaseBranchCNN,
    nac: nn.Module,
    stft: STFT,
    device: torch.device,
) -> dict[str, float]:
    eval_dset = build_dataset(cfg, length=cfg.eval_examples, reset_rngs=True)
    eval_loader = AudioStreamingDataLoader(eval_dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    phase_rmse_sum = 0.0
    gd_mae_sum = 0.0
    plv_sum = 0.0
    phase_cos_sum = 0.0
    align_feat_sum = 0.0

    n = 0
    for noisy, clean, _ in eval_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        noisy_stft = stft(noisy)
        clean_stft = stft(clean)
        noisy_phase = torch.angle(noisy_stft[:, 0])
        clean_phase = torch.angle(clean_stft[:, 0])

        _, z_noisy = nac.encode(noisy, domain="x")
        delta = model(z_noisy, target_frames=noisy_phase.shape[-1])
        pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))

        phase_diff = wrap_to_pi(pred_phase - clean_phase)
        phase_rmse_sum += torch.sqrt((phase_diff.square()).mean()).item()
        phase_cos_sum += torch.cos(phase_diff).mean().item()

        pred_gd = wrap_to_pi(pred_phase[:, 1:, :] - pred_phase[:, :-1, :])
        clean_gd = wrap_to_pi(clean_phase[:, 1:, :] - clean_phase[:, :-1, :])
        gd_mae_sum += (pred_gd - clean_gd).abs().mean().item()

        mean_cos = torch.cos(phase_diff).mean()
        mean_sin = torch.sin(phase_diff).mean()
        plv_sum += torch.sqrt(mean_cos.square() + mean_sin.square()).item()

        phase_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase)], dim=1)
        clean_feat = torch.cat([torch.cos(clean_phase), torch.sin(clean_phase)], dim=1)
        align_feat_sum += F.cosine_similarity(phase_feat.flatten(1), clean_feat.flatten(1), dim=1).mean().item()

        n += 1
        if n % 10 == 0:
            print(f"[cnn eval] {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break

    denom = max(n, 1)
    return {
        "eval_examples": float(n),
        "phase_rmse": phase_rmse_sum / denom,
        "phase_cos_alignment": phase_cos_sum / denom,
        "group_delay_mae": gd_mae_sum / denom,
        "plv": plv_sum / denom,
        "phase_feature_alignment_to_clean": align_feat_sum / denom,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone phase-branch probe without ADDSE discrete diffusion branch.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--init-only", action="store_true", help="Only create an initialized weights file and exit.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nac, _ = load_nac(cfg.nac_cfg, cfg.nac_ckpt)
    nac = nac.to(device)

    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)
    out_bins = cfg.n_fft // 2 + 1
    model = PhaseBranchCNN(in_channels=nac.encoder.out_conv.conv.out_channels, out_channels=out_bins).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    if args.init_only:
        save_checkpoint(cfg.weights_path, model, optimizer, step=0, cfg=cfg)
        print(f"Initialized weights saved: {cfg.weights_path}")
        return 0

    start_step = load_checkpoint_if_exists(cfg.weights_path, model, optimizer)
    if start_step > 0:
        print(f"Resumed phase branch weights from step={start_step}: {cfg.weights_path}")

    if args.mode in ("train", "train_eval"):
        train_length = max(cfg.train_steps * cfg.train_batch_size, cfg.train_batch_size)
        train_dset = build_dataset(cfg, length=train_length, reset_rngs=False)
        train_loader = AudioStreamingDataLoader(
            train_dset,
            batch_size=cfg.train_batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
        )

        model.train()
        step = start_step
        for noisy, clean, _ in train_loader:
            if step >= cfg.train_steps:
                break
            noisy = noisy.to(device)
            clean = clean.to(device)

            with torch.no_grad():
                _, z_noisy = nac.encode(noisy, domain="x")

            noisy_stft = stft(noisy)
            clean_stft = stft(clean)
            noisy_phase = torch.angle(noisy_stft[:, 0])
            clean_phase = torch.angle(clean_stft[:, 0])

            delta = model(z_noisy, target_frames=noisy_phase.shape[-1])
            pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))

            loss, stats = phase_losses(pred_phase, clean_phase, cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            if step % 20 == 0 or step == 1:
                print(
                    f"step={step}/{cfg.train_steps} "
                    f"loss={loss.item():.5f} "
                    f"phase={stats['phase_loss'].item():.5f} "
                    f"gd={stats['group_delay_loss'].item():.5f}"
                )

        save_checkpoint(cfg.weights_path, model, optimizer, step=step, cfg=cfg)
        print(f"Saved phase-branch weights: {cfg.weights_path}")

    if args.mode in ("eval", "train_eval"):
        model.eval()
        quantizer_report = evaluate_quantizer_baseline(cfg, nac, stft, device)
        cnn_report = evaluate_cnn_phase_branch(cfg, model, nac, stft, device)
        report = {
            "quantizer_baseline": quantizer_report,
            "cnn_phase_residual": cnn_report,
            "delta_cnn_minus_quantizer": {
                "phase_rmse": cnn_report["phase_rmse"] - quantizer_report["phase_rmse"],
                "phase_cos_alignment": cnn_report["phase_cos_alignment"] - quantizer_report["phase_cos_alignment"],
                "group_delay_mae": cnn_report["group_delay_mae"] - quantizer_report["group_delay_mae"],
                "plv": cnn_report["plv"] - quantizer_report["plv"],
            },
        }
        Path(cfg.report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("=== Phase Branch 2 Report ===")
        print("[quantizer_baseline]")
        for k, v in quantizer_report.items():
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
        print("[cnn_phase_residual]")
        for k, v in cnn_report.items():
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
        print("[delta_cnn_minus_quantizer]")
        for k, v in report["delta_cnn_minus_quantizer"].items():
            print(f"{k}: {v:+.6f}")
        print(f"Saved report: {cfg.report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
