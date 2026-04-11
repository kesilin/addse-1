import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase_fusion_scheme1 import (  # type: ignore
    PhaseBranchCNN,
    build_dset,
    load_phase_cnn,
    wrap_to_pi,
)

from stagea_logit_modules import GatingModule, LogitOffsetHead, alpha_polarization_penalty

from addse.data import AudioStreamingDataLoader
from addse.lightning import ADDSELightningModule
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT
from addse.utils import load_hydra_config


@dataclass
class StageAConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"

    layered_cfg: str = "configs/addse-s-mydata-layered-ft-pesq20.yaml"
    layered_ckpt: str = "logs/ft-pesq20-3layer/checkpoints/epoch=09-val_pesq=1.494.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"

    use_scheme_b: bool = True
    offset_start_layer: int = 1
    alpha_init_value: float = 0.1
    acoustic_lr_scale: float = 5.0

    train_batch_size: int = 4
    max_epochs: int = 5
    train_batches_per_epoch: int = 100
    val_examples_per_epoch: int = 60
    eval_examples: int = 60
    solve_steps: int = 128
    num_workers: int = 0

    lr: float = 1.0e-4
    logits_scale: float = 0.28
    lambda_ce_branch: float = 0.15
    lambda_alpha_polar: float = 0.05
    entropy_gate_power: float = 1.0
    alpha_max: float = 0.35

    low_rank_dim: int = 128
    gate_hidden: int = 128

    snr_min: float = 0.0
    snr_max: float = 10.0

    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512

    offset_head_last_ckpt: str = "experiments/phase_fusion_stagea_schemeb/weights/offset_head_last.pt"
    gate_last_ckpt: str = "experiments/phase_fusion_stagea_schemeb/weights/gate_last.pt"
    offset_head_best_ckpt: str = "experiments/phase_fusion_stagea_schemeb/weights/offset_head_best_pesq.pt"
    gate_best_ckpt: str = "experiments/phase_fusion_stagea_schemeb/weights/gate_best_pesq.pt"
    report_json: str = "experiments/phase_fusion_stagea_schemeb/reports/report_60_stagea.json"


def load_cfg(path: str) -> StageAConfig:
    base = asdict(StageAConfig())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    base.update(updates)
    return StageAConfig(**base)


def _make_ds_cfg(cfg: StageAConfig) -> SimpleNamespace:
    return SimpleNamespace(
        seed=cfg.seed,
        fs=cfg.fs,
        segment_length=cfg.segment_length,
        speech_dir=cfg.speech_dir,
        noise_dir=cfg.noise_dir,
        addse_cfg=cfg.layered_cfg,
        addse_ckpt=cfg.layered_ckpt,
        phase_cnn_ckpt=cfg.phase_cnn_ckpt,
        train_steps=cfg.train_batches_per_epoch,
        train_batch_size=cfg.train_batch_size,
        eval_examples=cfg.eval_examples,
        num_workers=cfg.num_workers,
        lr=cfg.lr,
        lambda_phase=0.0,
        fusion_scale=0.0,
        adapter_hidden=128,
        use_dilation=False,
        dilation_rates=(1, 2, 4),
        phase_delta_clip=1.2,
        snr_min=cfg.snr_min,
        snr_max=cfg.snr_max,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        n_fft=cfg.n_fft,
        adapter_ckpt="",
        report_json=cfg.report_json,
    )


def _load_lm(cfg_path: str, ckpt_path: str, device: torch.device) -> ADDSELightningModule:
    hydra_cfg, _ = load_hydra_config(cfg_path, overrides=None)
    lm = instantiate(hydra_cfg.lm)
    if not isinstance(lm, ADDSELightningModule):
        raise TypeError("Expected ADDSELightningModule.")
    state = torch.load(ckpt_path, map_location=device)
    lm.load_state_dict(state["state_dict"], strict=False)
    lm = lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad = False
    return lm


def _run_base(lm: ADDSELightningModule, noisy: torch.Tensor, solve_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pad = (lm.nac.downsampling_factor - noisy.shape[-1]) % lm.nac.downsampling_factor
    x_pad = F.pad(noisy, (0, n_pad))
    x_tok, x_q = lm.nac.encode(x_pad, no_sum=True, domain="q")
    _, z_noisy = lm.nac.encode(x_pad, domain="x")
    y_tok = lm.solve(x_tok, x_q, solve_steps)
    if not isinstance(y_tok, torch.Tensor):
        raise TypeError("Unexpected output from solve.")
    y_q_books = lm.nac.quantizer.decode(y_tok, output_no_sum=True, domain="code")
    y_q_sum = y_q_books.sum(dim=2)
    return x_q, y_tok, y_q_books, y_q_sum, z_noisy


def _extract_logits_features(noisy: torch.Tensor, z_noisy: torch.Tensor, stft: STFT, phase_cnn: PhaseBranchCNN) -> torch.Tensor:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
    noisy_mag = noisy_mag / (noisy_mag.abs().amax(dim=1, keepdim=True).clamp(min=1e-6))

    delta = phase_cnn(z_noisy, target_frames=noisy_phase.shape[-1])
    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-1.2, 1.2))
    return torch.cat([torch.cos(pred_phase), torch.sin(pred_phase), noisy_mag], dim=1)


def _layer_mask(num_books: int, start_layer: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, num_books, 1, 1), device=device)
    mask[:, max(0, int(start_layer)) :, :, :] = 1.0
    return mask


def _forward_stagea(
    cfg: StageAConfig,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    offset_head: LogitOffsetHead,
    gate: GatingModule,
    stft: STFT,
    noisy: torch.Tensor,
    solve_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, solve_steps)
    logits_feat = _extract_logits_features(noisy, z_noisy, stft, phase_cnn)

    base_log_prob = lm.log_score(y_q_books, x_q)
    base_detached = base_log_prob.detach()

    delta_logits = offset_head(logits_feat, latent_frames=y_tok.shape[-1])
    with torch.no_grad():
        base_prob = base_detached.exp().clamp(min=1e-8)
        entropy = -(base_prob * base_detached).sum(dim=-1)
        entropy = (entropy / math.log(base_prob.shape[-1])).clamp(0.0, 1.0)
        entropy = entropy.pow(cfg.entropy_gate_power)

    alpha = cfg.alpha_max * gate(logits_feat, entropy, latent_frames=y_tok.shape[-1])
    start_layer = max(0, int(cfg.offset_start_layer))
    layer_mask = _layer_mask(num_books=base_log_prob.shape[1], start_layer=start_layer, device=noisy.device)

    final_logits = base_detached.clone()
    pred_tok = y_tok.clone()
    if cfg.use_scheme_b and start_layer < base_log_prob.shape[1]:
        injected_logits = base_detached[:, start_layer:, :, :] + cfg.logits_scale * alpha[:, start_layer:, :, :] * delta_logits[:, start_layer:, :, :]
        final_logits[:, start_layer:, :, :] = injected_logits
        pred_tok[:, start_layer:, :] = injected_logits.argmax(dim=-1)

    final_log_prob = final_logits.log_softmax(dim=-1)

    y_q_books_ref = lm.nac.quantizer.decode(pred_tok, output_no_sum=True, domain="code")
    y_q_sum_ref = y_q_books_ref.sum(dim=2)
    y_hat = lm.nac.decoder(y_q_sum_ref)

    aux = {
        "y_tok_base": y_tok,
        "base_detached": base_detached,
        "delta_logits": delta_logits,
        "alpha": alpha,
        "layer_mask": layer_mask,
        "final_logits": final_logits,
        "final_log_prob": final_log_prob,
        "pred_tok": pred_tok,
        "y_q_books": y_q_books,
    }
    return y_hat, aux


def _encode_clean_tokens(lm: ADDSELightningModule, clean: torch.Tensor) -> torch.Tensor:
    n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
    clean_pad = F.pad(clean, (0, n_pad))
    clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")
    return clean_tok


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, start_layer: int) -> torch.Tensor:
    logits_sel = logits[:, max(0, int(start_layer)) :, :, :]
    target_sel = target[:, max(0, int(start_layer)) :, :]
    vocab = logits_sel.shape[-1]
    return F.cross_entropy(logits_sel.reshape(-1, vocab), target_sel.reshape(-1))


def _train(
    cfg: StageAConfig,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    offset_head: LogitOffsetHead,
    gate: GatingModule,
    stft: STFT,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], int]:
    for p in offset_head.parameters():
        p.requires_grad = True
    for p in gate.parameters():
        p.requires_grad = True

    lr_scaled = cfg.lr * cfg.acoustic_lr_scale
    opt = torch.optim.AdamW(
        [
            {"params": offset_head.parameters(), "lr": lr_scaled},
            {"params": gate.parameters(), "lr": lr_scaled},
        ]
    )

    train_stats: dict[str, float] = {}
    best_val_stats = {"pesq": float("nan"), "estoi": float("nan"), "sdr": float("nan"), "phase_rmse": float("nan"), "eval_examples": 0.0}
    best_pesq = float("-inf")
    total_steps = max(1, int(cfg.max_epochs)) * max(1, int(cfg.train_batches_per_epoch))
    global_step = 0

    for epoch in range(max(1, int(cfg.max_epochs))):
        dset = build_dset(_make_ds_cfg(cfg), length=cfg.train_batches_per_epoch * cfg.train_batch_size, reset_rngs=False)
        loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)

        offset_head.train()
        gate.train()
        for step, (noisy, clean, _) in enumerate(loader, start=1):
            if step > int(cfg.train_batches_per_epoch):
                break
            noisy = noisy.to(device)
            clean = clean.to(device)

            y_hat, aux = _forward_stagea(cfg, lm, phase_cnn, offset_head, gate, stft, noisy, cfg.solve_steps)
            y_hat = y_hat[..., : clean.shape[-1]]

            clean_tok = _encode_clean_tokens(lm, clean)
            ce_final = _masked_ce(aux["final_logits"], clean_tok, cfg.offset_start_layer)
            ce_branch = _masked_ce(cfg.logits_scale * aux["delta_logits"], clean_tok, cfg.offset_start_layer)
            alpha_pen = alpha_polarization_penalty(aux["alpha"] * aux["layer_mask"])

            loss = ce_final + cfg.lambda_ce_branch * ce_branch + cfg.lambda_alpha_polar * alpha_pen

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            train_stats = {
                "loss": float(loss.item()),
                "ce_final": float(ce_final.item()),
                "ce_branch": float(ce_branch.item()),
                "alpha_pen": float(alpha_pen.item()),
                "alpha_mean": float(aux["alpha"].mean().item()),
                "delta_mean": float(aux["delta_logits"].abs().mean().item()),
            }
            if global_step == 1 or global_step % 20 == 0:
                print(
                    f"stageA train epoch={epoch + 1}/{cfg.max_epochs} step={step}/{cfg.train_batches_per_epoch} "
                    f"global={global_step}/{total_steps} loss={loss.item():.5f} ce_final={ce_final.item():.5f} "
                    f"ce_branch={ce_branch.item():.5f} alpha={aux['alpha'].mean().item():.4f} "
                    f"delta={aux['delta_logits'].abs().mean().item():.5f}"
                )

        current_eval = cfg.eval_examples
        cfg.eval_examples = int(cfg.val_examples_per_epoch)
        offset_head.eval()
        gate.eval()
        val_stats = _evaluate(cfg, lm, phase_cnn, offset_head, gate, stft, device)
        cfg.eval_examples = current_eval
        print(f"stageA val epoch={epoch + 1}/{cfg.max_epochs} pesq={val_stats['pesq']:.5f} estoi={val_stats['estoi']:.5f} sdr={val_stats['sdr']:.5f}")

        if float(val_stats["pesq"]) > best_pesq:
            best_pesq = float(val_stats["pesq"])
            best_val_stats = val_stats
            Path(cfg.offset_head_best_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": offset_head.state_dict(), "config": asdict(cfg), "best_val_pesq": best_pesq, "epoch": epoch + 1}, cfg.offset_head_best_ckpt)
            torch.save({"model": gate.state_dict(), "config": asdict(cfg), "best_val_pesq": best_pesq, "epoch": epoch + 1}, cfg.gate_best_ckpt)
            print("Saved stageA best-by-PESQ checkpoints.")

    Path(cfg.offset_head_last_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": offset_head.state_dict(), "config": asdict(cfg)}, cfg.offset_head_last_ckpt)
    torch.save({"model": gate.state_dict(), "config": asdict(cfg)}, cfg.gate_last_ckpt)
    print("Saved stageA last checkpoints.")

    return train_stats, best_val_stats, global_step


@torch.no_grad()
def _evaluate(
    cfg: StageAConfig,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    offset_head: LogitOffsetHead,
    gate: GatingModule,
    stft: STFT,
    device: torch.device,
) -> dict[str, float]:
    dset = build_dset(_make_ds_cfg(cfg), length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0}
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        y_hat, _ = _forward_stagea(cfg, lm, phase_cnn, offset_head, gate, stft, noisy, cfg.solve_steps)
        y_hat = y_hat[..., : clean.shape[-1]]

        hat_stft = stft(y_hat)
        clean_stft = stft(clean)
        phase_rmse = torch.sqrt((wrap_to_pi(torch.angle(hat_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())

        sums["pesq"] += pesq(y_hat[0], clean[0])
        sums["estoi"] += estoi(y_hat[0], clean[0])
        sums["sdr"] += sdr(y_hat[0], clean[0])
        sums["phase_rmse"] += float(phase_rmse.item())

        n += 1
        if n % 20 == 0:
            print(f"stageA eval {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break

    denom = max(n, 1)
    return {k: v / denom for k, v in sums.items()} | {"eval_examples": float(n)}


def _try_resume_last(cfg: StageAConfig, offset_head: LogitOffsetHead, gate: GatingModule, device: torch.device) -> bool:
    offset_last = Path(cfg.offset_head_last_ckpt)
    gate_last = Path(cfg.gate_last_ckpt)
    if not (offset_last.exists() and gate_last.exists()):
        return False
    offset_state = torch.load(str(offset_last), map_location=device)
    gate_state = torch.load(str(gate_last), map_location=device)
    offset_head.load_state_dict(offset_state["model"], strict=True)
    gate.load_state_dict(gate_state["model"], strict=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-A scheme-B: detached base logits + gated logit offsets.")
    parser.add_argument("--config", default="configs/phase_fusion_stagea_schemeb_60.yaml")
    parser.add_argument("--mode", choices=["train_eval", "eval_only"], default="train_eval")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)

    phase_bins = cfg.n_fft // 2 + 1
    in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
    phase_cnn = load_phase_cnn(_make_ds_cfg(cfg), in_ch=in_cont_ch, out_ch=phase_bins, device=device)
    for p in phase_cnn.parameters():
        p.requires_grad = False
    phase_cnn.eval()

    vocab_size = lm.nac.quantizer.codebooks[0].codebook.weight.shape[0]  # type: ignore[attr-defined]
    num_books = len(lm.nac.quantizer.codebooks)  # type: ignore[attr-defined]
    offset_head = LogitOffsetHead(3 * phase_bins, num_books, vocab_size, rank=cfg.low_rank_dim).to(device)
    gate = GatingModule(3 * phase_bins, num_books, hidden=cfg.gate_hidden, alpha_init_value=cfg.alpha_init_value).to(device)

    if args.mode == "train_eval" and (not args.no_resume):
        resumed = _try_resume_last(cfg, offset_head, gate, device)
        if resumed:
            print("Resumed stageA from last checkpoints.")

    if args.mode == "train_eval":
        train_stats, best_val_stats, total_steps = _train(cfg, lm, phase_cnn, offset_head, gate, stft, device)
        if Path(cfg.offset_head_best_ckpt).exists() and Path(cfg.gate_best_ckpt).exists():
            offset_state = torch.load(cfg.offset_head_best_ckpt, map_location=device)
            gate_state = torch.load(cfg.gate_best_ckpt, map_location=device)
            offset_head.load_state_dict(offset_state["model"], strict=True)
            gate.load_state_dict(gate_state["model"], strict=True)
            print("Loaded stageA best-by-PESQ checkpoints for final eval.")
    else:
        if not (Path(cfg.offset_head_best_ckpt).exists() and Path(cfg.gate_best_ckpt).exists()):
            raise FileNotFoundError("Best checkpoints not found for eval_only mode.")
        offset_state = torch.load(cfg.offset_head_best_ckpt, map_location=device)
        gate_state = torch.load(cfg.gate_best_ckpt, map_location=device)
        offset_head.load_state_dict(offset_state["model"], strict=True)
        gate.load_state_dict(gate_state["model"], strict=True)
        train_stats = {
            "loss": float("nan"),
            "ce_final": float("nan"),
            "ce_branch": float("nan"),
            "alpha_pen": float("nan"),
            "alpha_mean": float("nan"),
            "delta_mean": float("nan"),
        }
        best_val_stats = {"pesq": float("nan"), "estoi": float("nan"), "sdr": float("nan"), "phase_rmse": float("nan"), "eval_examples": 0.0}
        total_steps = 0

    offset_head.eval()
    gate.eval()
    eval_stats = _evaluate(cfg, lm, phase_cnn, offset_head, gate, stft, device)

    report = {
        "train": train_stats,
        "best_val": best_val_stats,
        "total_steps": total_steps,
        "eval": eval_stats,
        "config": asdict(cfg),
    }

    Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Stage-A Report ===")
    print(json.dumps(report["train"], ensure_ascii=False, indent=2))
    print(json.dumps(report["eval"], ensure_ascii=False, indent=2))
    print(f"Saved report: {cfg.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
