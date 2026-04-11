import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from addse.data import AudioStreamingDataLoader
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT
from phase_fusion_scheme1 import build_dset, load_phase_cnn, wrap_to_pi
from phaseadapter_v36_probe import (
    PhaseAdapter,
    _PhaseAdapterInjector,
    _ce_from_logits,
    _freeze_backbone,
    _load_lm,
    _make_ds_cfg,
    _run_base,
    _select_blocks,
)


@dataclass
class ProbeVariant:
    name: str
    inject_mode: str = "adaln_hybrid"
    inject_num_blocks: int = 3
    inject_stride: int = 2
    adapter_scale: float = 0.05
    unfreeze_last_n_blocks: int = 0


@dataclass
class ProbeConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 1.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"

    layered_cfg: str = "configs/addse-s-mydata-layered-ft-pesq20.yaml"
    layered_ckpt: str = "logs/ft-pesq20-3layer/checkpoints/epoch=09-val_pesq=1.494.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"

    train_batch_size: int = 2
    train_steps: int = 12
    eval_examples: int = 12
    solve_steps: int = 8
    num_workers: int = 0

    lr: float = 1.2e-4
    weight_decay: float = 1.0e-4

    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512
    snr_min: float = 0.0
    snr_max: float = 10.0
    adapter_hidden: int = 256

    ce_weight: float = 1.0
    phase_weight: float = 0.15
    if_weight: float = 0.10

    report_json: str = "experiments/phaseadapter_v36_probe/reports/conditioning_probe_report.json"

    variants: list[ProbeVariant] | None = None


def load_cfg(path: str) -> ProbeConfig:
    base = asdict(ProbeConfig())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}

    raw_variants = updates.pop("variants", None)
    base.update(updates)
    cfg = ProbeConfig(**base)
    if raw_variants:
        cfg.variants = [ProbeVariant(**item) for item in raw_variants]
    if not cfg.variants:
        cfg.variants = [
            ProbeVariant(name="alt3_hybrid", inject_mode="adaln_hybrid", inject_num_blocks=3, inject_stride=2),
            ProbeVariant(name="last4_hybrid", inject_mode="adaln_hybrid", inject_num_blocks=4, inject_stride=1),
            ProbeVariant(name="last4_gated", inject_mode="adaln_gated", inject_num_blocks=4, inject_stride=1),
        ]
    return cfg


def _extract_phase_feat(
    noisy: torch.Tensor,
    z_noisy: torch.Tensor,
    stft: STFT,
    phase_cnn: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
    noisy_mag = noisy_mag / (noisy_mag.abs().amax(dim=1, keepdim=True).clamp(min=1e-6))
    delta = phase_cnn(z_noisy, target_frames=noisy_phase.shape[-1])
    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-1.2, 1.2))
    feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase), noisy_mag], dim=1)
    return feat, pred_phase


def _soft_decode_waveform_from_log_prob(lm, log_prob: torch.Tensor) -> torch.Tensor:
    # log_prob: (B, K, L, V)
    probs = log_prob.exp()
    quantized_sum = None
    for k, codebook in enumerate(lm.nac.quantizer.codebooks):
        p = probs[:, k]  # (B, L, V)
        emb = codebook.codebook.weight  # (V, C_proj)
        q_proj = torch.matmul(p, emb).transpose(1, 2).contiguous()  # (B, C_proj, L)
        q = codebook.out_conv(q_proj)  # (B, C_emb, L)
        quantized_sum = q if quantized_sum is None else (quantized_sum + q)
    assert quantized_sum is not None
    return lm.nac.decoder(quantized_sum)


def _phase_if_losses(stft: STFT, y_hat: torch.Tensor, y_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y_hat_stft = stft(y_hat)
    y_ref_stft = stft(y_ref)
    phase_hat = torch.angle(y_hat_stft[:, 0])
    phase_ref = torch.angle(y_ref_stft[:, 0])
    phase_err = wrap_to_pi(phase_hat - phase_ref)
    loss_phase = phase_err.abs().mean()

    if_hat = wrap_to_pi(phase_hat[..., 1:] - phase_hat[..., :-1])
    if_ref = wrap_to_pi(phase_ref[..., 1:] - phase_ref[..., :-1])
    loss_if = wrap_to_pi(if_hat - if_ref).abs().mean()
    return loss_phase, loss_if


def _phase_rmse(stft: STFT, y_hat: torch.Tensor, y_ref: torch.Tensor) -> float:
    y_hat_stft = stft(y_hat)
    y_ref_stft = stft(y_ref)
    diff = wrap_to_pi(torch.angle(y_hat_stft[:, 0]) - torch.angle(y_ref_stft[:, 0]))
    return float(torch.sqrt(diff.square().mean()).item())


def _evaluate_adapter(cfg: ProbeConfig, lm, stft: STFT, phase_cnn, adapter, injector, device: torch.device, use_adapter: bool) -> dict:
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

        x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
        phase_feat, _ = _extract_phase_feat(noisy, z_noisy, stft, phase_cnn)

        if use_adapter:
            cond_tokens = adapter(phase_feat, target_len=y_tok.shape[-1])
            cond_map = injector.build_cond_map(cond_tokens)
            injector.set_cond(cond_map)
            try:
                log_prob = lm.log_score(y_q_books, x_q)
            finally:
                injector.clear()
        else:
            log_prob = lm.log_score(y_q_books, x_q)

        pred_tok = log_prob.argmax(dim=-1)
        y_hat = lm.nac.decode(pred_tok, domain="code")[..., : clean.shape[-1]]

        sums["pesq"] += pesq(y_hat[0], clean[0])
        sums["estoi"] += estoi(y_hat[0], clean[0])
        sums["sdr"] += sdr(y_hat[0], clean[0])
        sums["phase_rmse"] += _phase_rmse(stft, y_hat, clean)

        n += 1
        if n >= cfg.eval_examples:
            break

    d = max(n, 1)
    return {k: v / d for k, v in sums.items()} | {"eval_examples": float(n)}


def _unfreeze_tail_blocks(lm, n: int) -> int:
    n = max(0, int(n))
    if n == 0:
        return 0
    blocks = list(lm.model.time_dit.blocks)
    target = blocks[-min(n, len(blocks)) :]
    cnt = 0
    for block in target:
        for p in block.parameters():
            p.requires_grad = True
            cnt += p.numel()
    return cnt


def _count_trainable_params(params: list[torch.nn.Parameter]) -> int:
    seen = set()
    total = 0
    for p in params:
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        total += p.numel()
    return total


def run_probe(cfg: ProbeConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []

    for variant in cfg.variants or []:
        lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
        stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)

        phase_bins = cfg.n_fft // 2 + 1
        in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
        phase_cnn = load_phase_cnn(_make_ds_cfg(cfg), in_ch=in_cont_ch, out_ch=phase_bins, device=device)
        for p in phase_cnn.parameters():
            p.requires_grad = False
        phase_cnn.eval()

        _freeze_backbone(lm)
        unfrozen_main = _unfreeze_tail_blocks(lm, variant.unfreeze_last_n_blocks)

        all_blocks = list(lm.model.time_dit.blocks)
        block_indices, blocks = _select_blocks(lm, variant.inject_num_blocks, variant.inject_stride)
        model_dim = lm.model.time_dit.blocks[0].norm_1.normalized_shape[0]

        adapter = PhaseAdapter(in_ch=3 * phase_bins, model_dim=model_dim, hidden=cfg.adapter_hidden, mode=variant.inject_mode).to(device)
        injector = _PhaseAdapterInjector(
            all_blocks=all_blocks,
            block_indices=block_indices,
            blocks=blocks,
            adapter=adapter,
            mode=variant.inject_mode,
            adapter_scale=variant.adapter_scale,
            model_dim=model_dim,
        )

        trainable_params = list(adapter.parameters())
        if injector.cross_attn is not None:
            injector.cross_attn = injector.cross_attn.to(device)
            trainable_params.extend(list(injector.cross_attn.parameters()))
        if injector.adaln_bridge is not None:
            injector.adaln_bridge = injector.adaln_bridge.to(device)
            trainable_params.extend(list(injector.adaln_bridge.parameters()))
        injector.cond_norms = injector.cond_norms.to(device)
        injector.cond_proj = injector.cond_proj.to(device)
        trainable_params.extend(list(injector.cond_norms.parameters()))
        trainable_params.extend(list(injector.cond_proj.parameters()))

        if unfrozen_main > 0:
            for p in lm.parameters():
                if p.requires_grad:
                    trainable_params.append(p)

        opt = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        dset = build_dset(_make_ds_cfg(cfg), length=cfg.train_steps * cfg.train_batch_size, reset_rngs=True)
        loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)

        zero_baseline = _evaluate_adapter(cfg, lm, stft, phase_cnn, adapter, injector, device, use_adapter=False)
        zero_adapter = _evaluate_adapter(cfg, lm, stft, phase_cnn, adapter, injector, device, use_adapter=True)

        losses = []
        grad_log = []
        step_times = []

        step = 0
        for noisy, clean, _ in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            t0 = time.perf_counter()

            x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
            phase_feat, _ = _extract_phase_feat(noisy, z_noisy, stft, phase_cnn)

            n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
            clean_pad = F.pad(clean, (0, n_pad))
            clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")

            cond_tokens = adapter(phase_feat, target_len=y_tok.shape[-1])
            cond_map = injector.build_cond_map(cond_tokens)
            injector.set_cond(cond_map)
            try:
                log_prob = lm.log_score(y_q_books, x_q)
            finally:
                injector.clear()

            ce_loss = _ce_from_logits(log_prob, clean_tok)
            y_soft = _soft_decode_waveform_from_log_prob(lm, log_prob)[..., : clean.shape[-1]]
            phase_loss, if_loss = _phase_if_losses(stft, y_soft, clean)
            total_loss = cfg.ce_weight * ce_loss + cfg.phase_weight * phase_loss + cfg.if_weight * if_loss

            opt.zero_grad(set_to_none=True)
            total_loss.backward()

            grad_items = []
            for idx, layer in zip(block_indices, injector.cond_proj):
                g = layer.weight.grad
                if g is not None:
                    grad_items.append((idx, float(g.norm().item())))
            grad_items.sort(key=lambda x: x[0], reverse=True)

            opt.step()
            step_times.append(time.perf_counter() - t0)

            losses.append(
                {
                    "step": int(step + 1),
                    "ce": float(ce_loss.item()),
                    "phase": float(phase_loss.item()),
                    "if": float(if_loss.item()),
                    "total": float(total_loss.item()),
                }
            )
            grad_log.append({"step": int(step + 1), "cond_proj_grad_norm_by_block": grad_items})

            step += 1
            if step >= cfg.train_steps:
                break

        after_adapter = _evaluate_adapter(cfg, lm, stft, phase_cnn, adapter, injector, device, use_adapter=True)

        row = {
            "variant": asdict(variant),
            "inject_block_indices": [int(i) for i in block_indices],
            "trainable_params": _count_trainable_params(trainable_params),
            "unfrozen_main_params": int(unfrozen_main),
            "avg_step_seconds": float(sum(step_times) / max(len(step_times), 1)),
            "zero_step_baseline": zero_baseline,
            "zero_step_adapter": zero_adapter,
            "after_probe_adapter": after_adapter,
            "phase_correction_gain_vs_zero_adapter": float(
                zero_adapter["phase_rmse"] - after_adapter["phase_rmse"]
            ),
            "zero_step_pesq_drop_vs_baseline": float(zero_adapter["pesq"] - zero_baseline["pesq"]),
            "losses": losses,
            "grad_probe": grad_log,
        }
        rows.append(row)

        injector.restore()

    summary = {
        "meta": {
            "device": str(device),
            "config": asdict(cfg),
            "note": "Short-term conditioning probe for injection location and objective alignment.",
        },
        "rows": rows,
    }

    Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved conditioning probe report: {cfg.report_json}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.6 short-term conditioning probe")
    parser.add_argument("--config", default="configs/phaseadapter_v36_conditioning_probe_tiny_seed42.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    run_probe(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
