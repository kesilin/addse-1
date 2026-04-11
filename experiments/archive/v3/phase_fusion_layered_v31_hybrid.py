import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase_fusion_scheme1 import (  # type: ignore
    PhaseBranchCNN,
    PreDecoderFusionAdapter,
    build_dset,
    load_phase_cnn,
    run_discrete_branch,
    wrap_to_pi,
)

from addse.data import AudioStreamingDataLoader
from addse.lightning import ADDSELightningModule
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT
from addse.utils import load_hydra_config


class FineBookLogitsHead(nn.Module):
    """Predicts a residual delta-logit map for the finest codebook (book index 3)."""

    def __init__(self, in_channels: int, vocab_size: int, hidden: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=8),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, vocab_size, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        # feat: (B, C, T_stft) -> (B, L_latent, V)
        out = self.net(feat)
        out = F.interpolate(out, size=latent_frames, mode="linear", align_corners=False)
        return out.transpose(1, 2)


class CrossGatedRouter(nn.Module):
    """Lightweight cross-gated routing from latent hint to phase/logit features."""

    def __init__(self, q_in: int, kv_in: int, out: int, hidden: int = 128) -> None:
        super().__init__()
        self.q_proj = nn.Conv1d(q_in, hidden, kernel_size=1)
        self.k_proj = nn.Conv1d(kv_in, hidden, kernel_size=1)
        self.v_proj = nn.Conv1d(kv_in, hidden, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv1d(hidden, out, kernel_size=1)

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor, out_frames: int) -> torch.Tensor:
        q_feat = F.interpolate(q_feat, size=out_frames, mode="linear", align_corners=False)
        kv = F.interpolate(kv_feat, size=out_frames, mode="linear", align_corners=False)
        q = self.q_proj(q_feat)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        score = torch.softmax((q * k).sum(dim=1, keepdim=True) / math.sqrt(q.shape[1]), dim=-1)
        g = self.gate(torch.cat([q, k], dim=1))
        routed = v * g * score
        return self.out(routed)


@dataclass
class HybridConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"

    layered_cfg: str = "configs/addse-s-mydata-layered-ft-pesq20.yaml"
    layered_ckpt: str = "logs/ft-pesq20-3layer/checkpoints/epoch=09-val_pesq=1.494.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"
    init_latent_adapter_ckpt: str = "experiments/phase_fusion_scheme1_v2/weights/adapter.pt"

    latent_adapter_last_ckpt: str = "experiments/phase_fusion_layered_v31_hybrid/weights/latent_adapter_last.pt"
    logits_head_last_ckpt: str = "experiments/phase_fusion_layered_v31_hybrid/weights/logits_head_last.pt"
    latent_adapter_best_ckpt: str = "experiments/phase_fusion_layered_v31_hybrid/weights/latent_adapter_best_pesq.pt"
    logits_head_best_ckpt: str = "experiments/phase_fusion_layered_v31_hybrid/weights/logits_head_best_pesq.pt"
    report_json: str = "experiments/phase_fusion_layered_v31_hybrid/reports/report_500.json"

    train_batch_size: int = 4
    max_epochs: int = 10
    train_batches_per_epoch: int = 120
    val_examples_per_epoch: int = 100
    eval_examples: int = 500
    solve_steps: int = 128
    num_workers: int = 0

    lr_latent_adapter: float = 1.0e-4
    lr_logits_head: float = 1.0e-4
    lambda_phase: float = 0.30
    lambda_band: float = 0.45
    lambda_mag: float = 0.35
    lambda_gd: float = 0.20
    lambda_ce_fine: float = 0.9
    lambda_kl_fine: float = 0.05
    use_router: bool = False
    use_gd_loss: bool = False
    use_fine_ce: bool = True
    use_fine_kl: bool = False

    fusion_scale: float = 0.18
    logits_scale: float = 0.28
    entropy_gate_power: float = 1.0

    adapter_hidden: int = 512
    logits_hidden: int = 192
    router_hidden: int = 96
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
    enable_plots: bool = True
    plot_dir: str = "experiments/phase_fusion_layered_v31_hybrid_run500/plots"


def load_cfg(path: str) -> HybridConfig:
    base = asdict(HybridConfig())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    base.update(updates)
    return HybridConfig(**base)


def _make_ds_cfg(cfg: HybridConfig) -> SimpleNamespace:
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
        lr=cfg.lr_latent_adapter,
        lambda_phase=cfg.lambda_phase,
        fusion_scale=cfg.fusion_scale,
        adapter_hidden=cfg.adapter_hidden,
        use_dilation=cfg.use_dilation,
        dilation_rates=cfg.dilation_rates,
        phase_delta_clip=cfg.phase_delta_clip,
        snr_min=cfg.snr_min,
        snr_max=cfg.snr_max,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        n_fft=cfg.n_fft,
        adapter_ckpt=cfg.latent_adapter_last_ckpt,
        report_json=cfg.report_json,
    )


def _band_phase_loss(phase_diff: torch.Tensor, band_edges: tuple[int, int], band_weights: tuple[float, float, float]) -> torch.Tensor:
    bins = phase_diff.shape[1]
    edges = (0, min(band_edges[0], bins), min(band_edges[1], bins), bins)
    total = phase_diff.new_zeros(())
    weight_sum = 0.0
    for (start, end), weight in zip(zip(edges[:-1], edges[1:]), band_weights, strict=True):
        if end <= start:
            continue
        total = total + float(weight) * (1.0 - torch.cos(phase_diff[:, start:end, :])).mean()
        weight_sum += float(weight)
    return total / max(weight_sum, 1e-8)


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


def _build_modules(cfg: HybridConfig, lm: ADDSELightningModule, device: torch.device) -> tuple[PhaseBranchCNN, PreDecoderFusionAdapter, FineBookLogitsHead, CrossGatedRouter]:
    phase_bins = cfg.n_fft // 2 + 1
    in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
    phase_cnn = load_phase_cnn(_make_ds_cfg(cfg), in_ch=in_cont_ch, out_ch=phase_bins, device=device)

    latent_adapter = PreDecoderFusionAdapter(
        in_channels=2 * phase_bins,
        emb_channels=lm.nac.decoder.in_conv.conv.in_channels,
        hidden=cfg.adapter_hidden,
        use_dilation=cfg.use_dilation,
        dilation_rates=cfg.dilation_rates,
    ).to(device)
    if Path(cfg.init_latent_adapter_ckpt).exists():
        state = torch.load(cfg.init_latent_adapter_ckpt, map_location=device)
        latent_adapter.load_state_dict(state["model"], strict=True)

    vocab_size = lm.nac.quantizer.codebooks[0].codebook.weight.shape[0]  # type: ignore[attr-defined]
    logits_head = FineBookLogitsHead(in_channels=3 * phase_bins, vocab_size=vocab_size, hidden=cfg.logits_hidden).to(device)
    router = CrossGatedRouter(
        q_in=lm.nac.decoder.in_conv.conv.in_channels,
        kv_in=3 * phase_bins,
        out=3 * phase_bins,
        hidden=cfg.router_hidden,
    ).to(device)

    return phase_cnn, latent_adapter, logits_head, router


def _try_resume_last(cfg: HybridConfig, latent_adapter: PreDecoderFusionAdapter, logits_head: FineBookLogitsHead, router: CrossGatedRouter, device: torch.device) -> bool:
    latent_last = Path(cfg.latent_adapter_last_ckpt)
    logits_last = Path(cfg.logits_head_last_ckpt)
    router_last = Path(cfg.logits_head_last_ckpt).with_name("router_last.pt")
    if not (latent_last.exists() and logits_last.exists() and router_last.exists()):
        return False

    latent_state = torch.load(str(latent_last), map_location=device)
    logits_state = torch.load(str(logits_last), map_location=device)
    router_state = torch.load(str(router_last), map_location=device)
    latent_adapter.load_state_dict(latent_state["model"], strict=True)
    logits_head.load_state_dict(logits_state["model"], strict=True)
    router.load_state_dict(router_state["model"], strict=True)
    return True


def _extract_noisy_inputs(noisy: torch.Tensor, z_noisy: torch.Tensor, stft: STFT, cfg: HybridConfig, phase_cnn: PhaseBranchCNN) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
    noisy_mag = noisy_mag / (noisy_mag.abs().amax(dim=1, keepdim=True).clamp(min=1e-6))

    delta = phase_cnn(z_noisy, target_frames=noisy_phase.shape[-1])
    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))
    phase_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase)], dim=1)
    logits_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase), noisy_mag], dim=1)
    return pred_phase, phase_feat, logits_feat


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


def _refine_fine_tokens(
    lm: ADDSELightningModule,
    logits_head: FineBookLogitsHead,
    logits_feat: torch.Tensor,
    x_q: torch.Tensor,
    y_tok: torch.Tensor,
    y_q_books: torch.Tensor,
    cfg: HybridConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # base log-score from frozen discrete model
    base_log_prob = lm.log_score(y_q_books, x_q)  # (B, K, L, V)
    base_fine_log_prob = base_log_prob[:, 3]  # (B, L, V)

    delta_logits = logits_head(logits_feat, latent_frames=y_tok.shape[-1])

    with torch.no_grad():
        base_prob = base_fine_log_prob.exp().clamp(min=1e-8)
        entropy = -(base_prob * base_fine_log_prob).sum(dim=-1)
        entropy_gate = (entropy / math.log(base_prob.shape[-1])).clamp(0.0, 1.0)
        entropy_gate = entropy_gate.pow(cfg.entropy_gate_power)

    refined_logits = base_fine_log_prob + cfg.logits_scale * entropy_gate.unsqueeze(-1) * delta_logits
    refined_log_prob = refined_logits.log_softmax(dim=-1)
    refined_tok_fine = refined_log_prob.argmax(dim=-1)

    y_tok_ref = y_tok.clone()
    y_tok_ref[:, 3, :] = refined_tok_fine
    y_q_books_ref = lm.nac.quantizer.decode(y_tok_ref, output_no_sum=True, domain="code")
    y_q_sum_ref = y_q_books_ref.sum(dim=2)

    return y_tok_ref, y_q_books_ref, y_q_sum_ref, refined_log_prob


def _forward_hybrid(
    cfg: HybridConfig,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    latent_adapter: PreDecoderFusionAdapter,
    logits_head: FineBookLogitsHead,
    router: CrossGatedRouter,
    stft: STFT,
    noisy: torch.Tensor,
    solve_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x_q, y_tok, y_q_books, y_q_sum, z_noisy = _run_base(lm, noisy, solve_steps)
    pred_phase, phase_feat, logits_feat = _extract_noisy_inputs(noisy, z_noisy, stft, cfg, phase_cnn)

    if cfg.use_router:
        routed_logits_feat = logits_feat + router(y_q_sum.detach(), logits_feat, out_frames=logits_feat.shape[-1])
    else:
        routed_logits_feat = logits_feat

    y_tok_ref, y_q_books_ref, y_q_sum_ref, refined_log_prob = _refine_fine_tokens(
        lm, logits_head, routed_logits_feat, x_q, y_tok, y_q_books, cfg
    )

    latent_corr = latent_adapter(phase_feat, latent_frames=y_q_sum_ref.shape[-1])
    # Inject only on fine codebook (book index 3) for detail-focused enhancement.
    y_q_books_fused = y_q_books_ref.clone()
    y_q_books_fused[:, :, 3, :] = y_q_books_fused[:, :, 3, :] + cfg.fusion_scale * latent_corr
    y_q_fused = y_q_books_fused.sum(dim=2)
    y_fused = lm.nac.decoder(y_q_fused)

    aux = {
        "x_q": x_q,
        "y_tok_base": y_tok,
        "y_tok_ref": y_tok_ref,
        "y_q_books_base": y_q_books,
        "y_q_books_ref": y_q_books_ref,
        "y_q_books_fused": y_q_books_fused,
        "y_q_sum_base": y_q_sum,
        "y_q_sum_ref": y_q_sum_ref,
        "refined_log_prob": refined_log_prob,
        "pred_phase": pred_phase,
    }
    return y_fused, aux


def _group_delay_loss(pred_phase: torch.Tensor, clean_phase: torch.Tensor) -> torch.Tensor:
    pred_gd = wrap_to_pi(pred_phase[:, 1:, :] - pred_phase[:, :-1, :])
    clean_gd = wrap_to_pi(clean_phase[:, 1:, :] - clean_phase[:, :-1, :])
    return (pred_gd - clean_gd).abs().mean()


def _save_plots(plot_dir: str, clean_stft: torch.Tensor, base_stft: torch.Tensor, fused_stft: torch.Tensor) -> None:
    if plt is None:
        return
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    clean_mag = torch.log1p(clean_stft.abs()).cpu().numpy()
    base_mag = torch.log1p(base_stft.abs()).cpu().numpy()
    fused_mag = torch.log1p(fused_stft.abs()).cpu().numpy()
    err_base = (clean_mag - base_mag)
    err_fused = (clean_mag - fused_mag)

    clean_phase = torch.angle(clean_stft)
    base_phase = torch.angle(base_stft)
    fused_phase = torch.angle(fused_stft)
    phase_err_base = wrap_to_pi(base_phase - clean_phase).flatten().cpu().numpy()
    phase_err_fused = wrap_to_pi(fused_phase - clean_phase).flatten().cpu().numpy()

    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 4))
    im0 = ax1[0].imshow(err_base, aspect="auto", origin="lower", cmap="magma")
    ax1[0].set_title("Error Spectrogram (Base)")
    fig1.colorbar(im0, ax=ax1[0], fraction=0.046, pad=0.04)
    im1 = ax1[1].imshow(err_fused, aspect="auto", origin="lower", cmap="magma")
    ax1[1].set_title("Error Spectrogram (Fused)")
    fig1.colorbar(im1, ax=ax1[1], fraction=0.046, pad=0.04)
    fig1.tight_layout()
    fig1.savefig(Path(plot_dir) / "error_spectrogram_base_vs_fused.png", dpi=140)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax2.hist(phase_err_base, bins=120, alpha=0.55, label="base", density=True)
    ax2.hist(phase_err_fused, bins=120, alpha=0.55, label="fused", density=True)
    ax2.set_title("Phase Error Histogram")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(Path(plot_dir) / "phase_error_histogram_base_vs_fused.png", dpi=140)
    plt.close(fig2)


def _encode_clean_tokens(lm: ADDSELightningModule, clean: torch.Tensor) -> torch.Tensor:
    n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
    clean_pad = F.pad(clean, (0, n_pad))
    clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")
    return clean_tok


def _train(
    cfg: HybridConfig,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    latent_adapter: PreDecoderFusionAdapter,
    logits_head: FineBookLogitsHead,
    router: CrossGatedRouter,
    stft: STFT,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], int]:
    for p in latent_adapter.parameters():
        p.requires_grad = True
    for p in logits_head.parameters():
        p.requires_grad = True
    for p in router.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(
        [
            {"params": latent_adapter.parameters(), "lr": cfg.lr_latent_adapter},
            {"params": logits_head.parameters(), "lr": cfg.lr_logits_head},
            {"params": router.parameters(), "lr": cfg.lr_logits_head},
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

        latent_adapter.train()
        logits_head.train()
        router.train()
        for step, (noisy, clean, _) in enumerate(loader, start=1):
            if step > int(cfg.train_batches_per_epoch):
                break
            noisy = noisy.to(device)
            clean = clean.to(device)

            y_fused, aux = _forward_hybrid(cfg, lm, phase_cnn, latent_adapter, logits_head, router, stft, noisy, cfg.solve_steps)
            y_fused = y_fused[..., : clean.shape[-1]]

            clean_tok = _encode_clean_tokens(lm, clean)
            clean_fine_tok = clean_tok[:, 3, :]

            clean_stft = stft(clean)
            fused_stft = stft(y_fused)
            phase_diff = wrap_to_pi(torch.angle(fused_stft[:, 0]) - torch.angle(clean_stft[:, 0]))
            fused_mag = torch.log1p(fused_stft[:, 0].abs())
            clean_mag = torch.log1p(clean_stft[:, 0].abs())

            wave_l1 = (y_fused - clean).abs().mean()
            phase_loss = (1.0 - torch.cos(phase_diff)).mean()
            band_loss = _band_phase_loss(phase_diff, cfg.band_edges, cfg.band_weights)
            mag_loss = F.l1_loss(fused_mag, clean_mag)
            gd_loss = _group_delay_loss(torch.angle(fused_stft[:, 0]), torch.angle(clean_stft[:, 0]))

            ce_fine = F.nll_loss(aux["refined_log_prob"].transpose(1, 2), clean_fine_tok) if cfg.use_fine_ce else wave_l1.detach() * 0.0

            with torch.no_grad():
                base_log_prob = lm.log_score(aux["y_q_books_ref"], aux["x_q"])[:, 3]
                base_prob = base_log_prob.exp()
            kl_fine = F.kl_div(aux["refined_log_prob"], base_prob, reduction="batchmean")
            kl_fine = kl_fine / max(aux["refined_log_prob"].shape[1] * aux["refined_log_prob"].shape[2], 1)
            if not cfg.use_fine_kl:
                kl_fine = wave_l1.detach() * 0.0

            if not cfg.use_gd_loss:
                gd_loss = wave_l1.detach() * 0.0

            loss = (
                wave_l1
                + cfg.lambda_phase * phase_loss
                + cfg.lambda_band * band_loss
                + cfg.lambda_mag * mag_loss
                + (cfg.lambda_gd * gd_loss if cfg.use_gd_loss else 0.0)
                + (cfg.lambda_ce_fine * ce_fine if cfg.use_fine_ce else 0.0)
                + (cfg.lambda_kl_fine * kl_fine if cfg.use_fine_kl else 0.0)
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            train_stats = {
                "loss": float(loss.item()),
                "wave_l1": float(wave_l1.item()),
                "phase_loss": float(phase_loss.item()),
                "band_loss": float(band_loss.item()),
                "mag_loss": float(mag_loss.item()),
                "gd_loss": float(gd_loss.item()),
                "ce_fine": float(ce_fine.item()),
                "kl_fine": float(kl_fine.item()),
            }
            if global_step == 1 or global_step % 20 == 0:
                print(
                    f"v31 train epoch={epoch + 1}/{cfg.max_epochs} step={step}/{cfg.train_batches_per_epoch} "
                    f"global={global_step}/{total_steps} loss={loss.item():.5f} wave={wave_l1.item():.5f} "
                    f"phase={phase_loss.item():.5f} gd={gd_loss.item():.5f} mag={mag_loss.item():.5f} ce4={ce_fine.item():.5f}"
                )

        # validation for best-by-PESQ
        current_eval = cfg.eval_examples
        cfg.eval_examples = int(cfg.val_examples_per_epoch)
        latent_adapter.eval()
        logits_head.eval()
        router.eval()
        val_stats = _evaluate(cfg, lm, phase_cnn, latent_adapter, logits_head, router, stft, device)
        cfg.eval_examples = current_eval
        print(f"v31 val epoch={epoch + 1}/{cfg.max_epochs} pesq={val_stats['pesq']:.5f} estoi={val_stats['estoi']:.5f} sdr={val_stats['sdr']:.5f}")

        if float(val_stats["pesq"]) > best_pesq:
            best_pesq = float(val_stats["pesq"])
            best_val_stats = val_stats
            Path(cfg.latent_adapter_best_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": latent_adapter.state_dict(), "config": asdict(cfg), "best_val_pesq": best_pesq, "epoch": epoch + 1}, cfg.latent_adapter_best_ckpt)
            torch.save({"model": logits_head.state_dict(), "config": asdict(cfg), "best_val_pesq": best_pesq, "epoch": epoch + 1}, cfg.logits_head_best_ckpt)
            torch.save({"model": router.state_dict(), "config": asdict(cfg), "best_val_pesq": best_pesq, "epoch": epoch + 1}, str(Path(cfg.logits_head_best_ckpt).with_name("router_best_pesq.pt")))
            print("Saved v31 best-by-PESQ checkpoints.")

    Path(cfg.latent_adapter_last_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": latent_adapter.state_dict(), "config": asdict(cfg)}, cfg.latent_adapter_last_ckpt)
    torch.save({"model": logits_head.state_dict(), "config": asdict(cfg)}, cfg.logits_head_last_ckpt)
    torch.save({"model": router.state_dict(), "config": asdict(cfg)}, str(Path(cfg.logits_head_last_ckpt).with_name("router_last.pt")))
    print("Saved v31 last checkpoints.")

    return train_stats, best_val_stats, global_step


@torch.no_grad()
def _evaluate(
    cfg: HybridConfig,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    latent_adapter: PreDecoderFusionAdapter,
    logits_head: FineBookLogitsHead,
    router: CrossGatedRouter,
    stft: STFT,
    device: torch.device,
) -> dict[str, float]:
    dset = build_dset(_make_ds_cfg(cfg), length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0}
    plotted = False
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        y_fused, aux = _forward_hybrid(cfg, lm, phase_cnn, latent_adapter, logits_head, router, stft, noisy, cfg.solve_steps)
        y_fused = y_fused[..., : clean.shape[-1]]

        fused_stft = stft(y_fused)
        clean_stft = stft(clean)
        phase_rmse = torch.sqrt((wrap_to_pi(torch.angle(fused_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())

        sums["pesq"] += pesq(y_fused[0], clean[0])
        sums["estoi"] += estoi(y_fused[0], clean[0])
        sums["sdr"] += sdr(y_fused[0], clean[0])
        sums["phase_rmse"] += phase_rmse.item()

        if cfg.enable_plots and not plotted:
            y_base = lm.nac.decoder(aux["y_q_sum_base"])[..., : clean.shape[-1]]
            base_stft = stft(y_base)
            _save_plots(cfg.plot_dir, clean_stft[:, 0], base_stft[:, 0], fused_stft[:, 0])
            plotted = True

        n += 1
        if n % 20 == 0:
            print(f"v31 eval {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break

    denom = max(n, 1)
    return {k: v / denom for k, v in sums.items()} | {"eval_examples": float(n)}


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.1 hybrid: fine-book logits refinement + latent fusion.")
    parser.add_argument("--config", default="configs/phase_fusion_layered_v31_hybrid_500.yaml")
    parser.add_argument("--mode", choices=["train_eval", "eval_only"], default="train_eval")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--train-batches", type=int, default=None)
    parser.add_argument("--eval-examples", type=int, default=None)
    parser.add_argument("--val-examples", type=int, default=None)
    parser.add_argument("--solve-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.train_batches is not None:
        cfg.train_batches_per_epoch = args.train_batches
    if args.eval_examples is not None:
        cfg.eval_examples = args.eval_examples
    if args.val_examples is not None:
        cfg.val_examples_per_epoch = args.val_examples
    if args.solve_steps is not None:
        cfg.solve_steps = args.solve_steps

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)
    phase_cnn, latent_adapter, logits_head, router = _build_modules(cfg, lm, device)

    if args.mode == "train_eval" and (not args.no_resume):
        resumed = _try_resume_last(cfg, latent_adapter, logits_head, router, device)
        if resumed:
            print("Resumed v31 from last checkpoints.")

    if args.mode == "train_eval":
        train_stats, best_val_stats, total_steps = _train(cfg, lm, phase_cnn, latent_adapter, logits_head, router, stft, device)
        if Path(cfg.latent_adapter_best_ckpt).exists() and Path(cfg.logits_head_best_ckpt).exists():
            latent_state = torch.load(cfg.latent_adapter_best_ckpt, map_location=device)
            logits_state = torch.load(cfg.logits_head_best_ckpt, map_location=device)
            latent_adapter.load_state_dict(latent_state["model"], strict=True)
            logits_head.load_state_dict(logits_state["model"], strict=True)
            router_best = Path(cfg.logits_head_best_ckpt).with_name("router_best_pesq.pt")
            if router_best.exists():
                router_state = torch.load(router_best, map_location=device)
                router.load_state_dict(router_state["model"], strict=True)
            print("Loaded v31 best-by-PESQ checkpoints for final eval.")
    else:
        if not (Path(cfg.latent_adapter_best_ckpt).exists() and Path(cfg.logits_head_best_ckpt).exists()):
            raise FileNotFoundError("Best checkpoints not found for eval_only mode.")
        latent_state = torch.load(cfg.latent_adapter_best_ckpt, map_location=device)
        logits_state = torch.load(cfg.logits_head_best_ckpt, map_location=device)
        latent_adapter.load_state_dict(latent_state["model"], strict=True)
        logits_head.load_state_dict(logits_state["model"], strict=True)
        router_best = Path(cfg.logits_head_best_ckpt).with_name("router_best_pesq.pt")
        if router_best.exists():
            router_state = torch.load(router_best, map_location=device)
            router.load_state_dict(router_state["model"], strict=True)
        train_stats = {"loss": float("nan"), "wave_l1": float("nan"), "phase_loss": float("nan"), "band_loss": float("nan"), "mag_loss": float("nan"), "gd_loss": float("nan"), "ce_fine": float("nan"), "kl_fine": float("nan")}
        best_val_stats = {"pesq": float("nan"), "estoi": float("nan"), "sdr": float("nan"), "phase_rmse": float("nan"), "eval_examples": 0.0}
        total_steps = 0

    latent_adapter.eval()
    logits_head.eval()
    router.eval()
    eval_stats = _evaluate(cfg, lm, phase_cnn, latent_adapter, logits_head, router, stft, device)

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

    print("=== V3.1 Hybrid Report ===")
    print(json.dumps(report["train"], ensure_ascii=False, indent=2))
    print(json.dumps(report["eval"], ensure_ascii=False, indent=2))
    print(f"Saved report: {cfg.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
