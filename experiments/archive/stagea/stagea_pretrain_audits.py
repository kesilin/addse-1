import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from phase_fusion_scheme1 import build_dset, load_phase_cnn, wrap_to_pi
from phase_fusion_stagea_schemeb import (
    StageAConfig,
    _extract_logits_features,
    _load_lm,
    _make_ds_cfg,
    _run_base,
    load_cfg,
)
from stagea_logit_modules import GatingModule, LogitOffsetHead, alpha_polarization_penalty

from addse.data import AudioStreamingDataLoader
from addse.stft import STFT


def _sample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
    if x.shape[0] <= max_rows:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:max_rows]
    return x[idx]


def _bin_feature_1d(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    lo = float(x.min().item())
    hi = float(x.max().item())
    if hi <= lo + 1e-12:
        return torch.zeros_like(x, dtype=torch.long)
    edges = torch.linspace(lo, hi, steps=n_bins + 1, device=x.device)
    # bucketize in [0, n_bins-1]
    b = torch.bucketize(x, edges[1:-1], right=False)
    return b.long().clamp(0, n_bins - 1)


def _mutual_info_discrete(x: torch.Tensor, y: torch.Tensor, x_bins: int, y_bins: int) -> float:
    # x, y: int64 tensors with values in [0, bins-1]
    x = x.reshape(-1).long()
    y = y.reshape(-1).long()
    joint_index = x * y_bins + y
    joint = torch.bincount(joint_index, minlength=x_bins * y_bins).float().reshape(x_bins, y_bins)
    total = joint.sum().clamp(min=1.0)
    pxy = joint / total
    px = pxy.sum(dim=1, keepdim=True)
    py = pxy.sum(dim=0, keepdim=True)
    mask = pxy > 0
    mi = (pxy[mask] * (torch.log(pxy[mask]) - torch.log((px @ py)[mask]))).sum()
    return float(mi.item())


def _grad_norm(module: torch.nn.Module) -> float:
    sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        sq += float((p.grad.detach().float().norm() ** 2).item())
    return math.sqrt(max(sq, 0.0))


def _cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-1])
    c = F.cosine_similarity(a2, b2, dim=-1)
    return float(c.mean().item())


def _temporal_tv_stats(x: torch.Tensor) -> dict[str, float]:
    # x: [N, T] or [N, C, T]
    if x.ndim == 2:
        d = x[:, 1:] - x[:, :-1]
    elif x.ndim == 3:
        d = x[:, :, 1:] - x[:, :, :-1]
    else:
        raise ValueError("Unsupported tensor rank for temporal stats")
    ad = d.abs().reshape(-1)
    if ad.numel() == 0:
        return {"mean_abs_diff": 0.0, "p95_abs_diff": 0.0}
    if ad.numel() > 1_000_000:
        ad = _sample_rows(ad.unsqueeze(1), 1_000_000).squeeze(1)
    return {
        "mean_abs_diff": float(ad.mean().item()),
        "p95_abs_diff": float(torch.quantile(ad, 0.95).item()),
    }


@torch.no_grad()
def feature_alignment_check(
    cfg: StageAConfig,
    lm,
    phase_cnn,
    stft,
    examples: int,
    max_tokens: int,
    feature_channels: int,
    bins: int,
    start_layer: int,
    device: torch.device,
) -> dict:
    dset = build_dset(_make_ds_cfg(cfg), length=examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    feat_rows = []
    clean_tok_rows = []
    fix_rows = []
    y_tok_rows = []

    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
        base_logits = lm.log_score(y_q_books, x_q)
        feat = _extract_logits_features(noisy, z_noisy, stft, phase_cnn)

        n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
        clean_pad = F.pad(clean, (0, n_pad))
        clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")

        t_lat = y_tok.shape[-1]
        layers_sel = y_tok[:, start_layer:, :].shape[1]
        feat_lat = feat[:, :, :t_lat].transpose(1, 2).reshape(-1, feat.shape[1])
        feat_rep = feat_lat.repeat_interleave(layers_sel, dim=0)

        clean_tok_sel = clean_tok[:, start_layer:, :].permute(0, 2, 1).reshape(-1)
        y_tok_sel = y_tok[:, start_layer:, :].permute(0, 2, 1).reshape(-1)
        fix_need = (clean_tok[:, start_layer:, :] != y_tok[:, start_layer:, :]).permute(0, 2, 1).reshape(-1).long()

        feat_rows.append(feat_rep)
        clean_tok_rows.append(clean_tok_sel)
        fix_rows.append(fix_need)
        y_tok_rows.append(y_tok_sel)

        n += 1
        if n >= examples:
            break

    feat_all = torch.cat(feat_rows, dim=0)
    clean_tok_all = torch.cat(clean_tok_rows, dim=0)
    fix_all = torch.cat(fix_rows, dim=0)
    y_tok_all = torch.cat(y_tok_rows, dim=0)

    if feat_all.shape[0] > max_tokens:
        idx = torch.randperm(feat_all.shape[0], device=feat_all.device)[:max_tokens]
        feat_all = feat_all[idx]
        clean_tok_all = clean_tok_all[idx]
        fix_all = fix_all[idx]
        y_tok_all = y_tok_all[idx]

    total_ch = feat_all.shape[1]
    ch_take = min(feature_channels, total_ch)
    ch_idx = torch.randperm(total_ch, device=feat_all.device)[:ch_take]

    vocab = int(lm.nac.quantizer.codebooks[0].codebook.weight.shape[0])  # type: ignore[attr-defined]

    mi_to_clean = []
    mi_to_fix = []
    mi_to_base = []
    mi_to_clean_shuffled = []

    perm = torch.randperm(clean_tok_all.shape[0], device=clean_tok_all.device)
    clean_shuffled = clean_tok_all[perm]

    for c in ch_idx.tolist():
        xb = _bin_feature_1d(feat_all[:, c], bins)
        mi_to_clean.append(_mutual_info_discrete(xb, clean_tok_all, bins, vocab))
        mi_to_fix.append(_mutual_info_discrete(xb, fix_all, bins, 2))
        mi_to_base.append(_mutual_info_discrete(xb, y_tok_all, bins, vocab))
        mi_to_clean_shuffled.append(_mutual_info_discrete(xb, clean_shuffled, bins, vocab))

    return {
        "examples": n,
        "tokens_used": int(feat_all.shape[0]),
        "feature_channels_sampled": int(ch_take),
        "mi_feat_to_clean_token": {
            "mean": float(torch.tensor(mi_to_clean).mean().item()),
            "std": float(torch.tensor(mi_to_clean).std(unbiased=False).item()),
        },
        "mi_feat_to_need_fix_binary": {
            "mean": float(torch.tensor(mi_to_fix).mean().item()),
            "std": float(torch.tensor(mi_to_fix).std(unbiased=False).item()),
        },
        "mi_feat_to_base_token": {
            "mean": float(torch.tensor(mi_to_base).mean().item()),
            "std": float(torch.tensor(mi_to_base).std(unbiased=False).item()),
        },
        "mi_feat_to_clean_token_shuffled_baseline": {
            "mean": float(torch.tensor(mi_to_clean_shuffled).mean().item()),
            "std": float(torch.tensor(mi_to_clean_shuffled).std(unbiased=False).item()),
        },
    }


def _build_modules_from_ckpt(cfg: StageAConfig, lm, device: torch.device, ckpt_mode: str):
    phase_bins = cfg.n_fft // 2 + 1
    vocab_size = lm.nac.quantizer.codebooks[0].codebook.weight.shape[0]  # type: ignore[attr-defined]
    num_books = len(lm.nac.quantizer.codebooks)  # type: ignore[attr-defined]

    offset_head = LogitOffsetHead(3 * phase_bins, num_books, vocab_size, rank=cfg.low_rank_dim).to(device)
    gate = GatingModule(3 * phase_bins, num_books, hidden=cfg.gate_hidden, alpha_init_value=cfg.alpha_init_value).to(device)

    if ckpt_mode == "best":
        offset_state = torch.load(cfg.offset_head_best_ckpt, map_location=device)
        gate_state = torch.load(cfg.gate_best_ckpt, map_location=device)
    else:
        offset_state = torch.load(cfg.offset_head_last_ckpt, map_location=device)
        gate_state = torch.load(cfg.gate_last_ckpt, map_location=device)
    offset_head.load_state_dict(offset_state["model"], strict=True)
    gate.load_state_dict(gate_state["model"], strict=True)
    return offset_head, gate


def gradient_flow_audit(
    cfg: StageAConfig,
    lm,
    phase_cnn,
    stft,
    examples: int,
    grad_steps: int,
    tau: float,
    ckpt_mode: str,
    device: torch.device,
) -> dict:
    dset = build_dset(_make_ds_cfg(cfg), length=max(examples, grad_steps * cfg.train_batch_size), reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)
    batches = []
    for noisy, clean, _ in loader:
        batches.append((noisy.to(device), clean.to(device)))
        if len(batches) >= grad_steps:
            break

    start_layer = max(0, int(cfg.offset_start_layer))

    def run_mode(mode: str) -> dict:
        offset_head, gate = _build_modules_from_ckpt(cfg, lm, device, ckpt_mode)
        for p in offset_head.parameters():
            p.requires_grad = True
        for p in gate.parameters():
            p.requires_grad = True

        opt = torch.optim.AdamW(
            [
                {"params": offset_head.parameters(), "lr": cfg.lr * cfg.acoustic_lr_scale},
                {"params": gate.parameters(), "lr": cfg.lr * cfg.acoustic_lr_scale},
            ]
        )

        loss_hist = []
        grad_off_hist = []
        grad_gate_hist = []
        cos_hist = []

        for step, (noisy, clean) in enumerate(batches, start=1):
            x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
            base_logits = lm.log_score(y_q_books, x_q).detach()
            feat = _extract_logits_features(noisy, z_noisy, stft, phase_cnn)

            delta_logits = offset_head(feat, latent_frames=y_tok.shape[-1])
            base_prob = base_logits.exp().clamp(min=1e-8)
            entropy = -(base_prob * base_logits).sum(dim=-1)
            entropy = (entropy / math.log(base_prob.shape[-1])).clamp(0.0, 1.0).pow(cfg.entropy_gate_power)
            alpha = cfg.alpha_max * gate(feat, entropy, latent_frames=y_tok.shape[-1])

            final_logits = base_logits.clone()
            injected = cfg.logits_scale * alpha * delta_logits
            if start_layer < final_logits.shape[1]:
                final_logits[:, start_layer:, :, :] = base_logits[:, start_layer:, :, :] + injected[:, start_layer:, :, :]

            n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
            clean_pad = F.pad(clean, (0, n_pad))
            clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")
            clean_tok_sel = clean_tok[:, start_layer:, :]
            vocab = final_logits.shape[-1]

            if mode == "hard_ce":
                ce_final = F.cross_entropy(final_logits[:, start_layer:, :, :].reshape(-1, vocab), clean_tok_sel.reshape(-1))
            elif mode == "gumbel_soft":
                probs = F.gumbel_softmax(final_logits[:, start_layer:, :, :], tau=tau, hard=False, dim=-1)
                target = F.one_hot(clean_tok_sel, num_classes=vocab).float()
                ce_final = -(target * probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
            else:
                raise ValueError(f"Unknown mode: {mode}")

            ce_branch = F.cross_entropy((cfg.logits_scale * delta_logits[:, start_layer:, :, :]).reshape(-1, vocab), clean_tok_sel.reshape(-1))
            alpha_pen = alpha_polarization_penalty(alpha[:, start_layer:, :, :])
            loss = ce_final + cfg.lambda_ce_branch * ce_branch + cfg.lambda_alpha_polar * alpha_pen

            base_prob_sel = base_logits[:, start_layer:, :, :].softmax(dim=-1)
            ideal = F.one_hot(clean_tok_sel, num_classes=vocab).float() - base_prob_sel
            cos_hist.append(_cosine_mean(delta_logits[:, start_layer:, :, :], ideal))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_off_hist.append(_grad_norm(offset_head))
            grad_gate_hist.append(_grad_norm(gate))
            opt.step()

            loss_hist.append(float(loss.item()))
            if step >= grad_steps:
                break

        return {
            "steps": len(loss_hist),
            "loss_start": float(loss_hist[0]) if loss_hist else float("nan"),
            "loss_end": float(loss_hist[-1]) if loss_hist else float("nan"),
            "grad_norm_offset_mean": float(torch.tensor(grad_off_hist).mean().item()) if grad_off_hist else float("nan"),
            "grad_norm_gate_mean": float(torch.tensor(grad_gate_hist).mean().item()) if grad_gate_hist else float("nan"),
            "cos_delta_to_ideal_start": float(cos_hist[0]) if cos_hist else float("nan"),
            "cos_delta_to_ideal_end": float(cos_hist[-1]) if cos_hist else float("nan"),
            "cos_delta_to_ideal_mean": float(torch.tensor(cos_hist).mean().item()) if cos_hist else float("nan"),
        }

    return {
        "hard_ce": run_mode("hard_ce"),
        "gumbel_soft": run_mode("gumbel_soft"),
        "temperature": float(tau),
    }


@torch.no_grad()
def phase_consistency_audit(
    cfg: StageAConfig,
    lm,
    phase_cnn,
    stft,
    offset_head,
    gate,
    examples: int,
    start_layer: int,
    device: torch.device,
) -> dict:
    dset = build_dset(_make_ds_cfg(cfg), length=examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    pred_phase_tv = []
    ideal_phase_tv = []
    alpha_tv = []
    delta_norm_tv = []

    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
        base_logits = lm.log_score(y_q_books, x_q)

        noisy_stft = stft(noisy)
        clean_stft = stft(clean)
        noisy_phase = torch.angle(noisy_stft[:, 0])
        clean_phase = torch.angle(clean_stft[:, 0])
        ideal_delta_phase = wrap_to_pi(clean_phase - noisy_phase)

        phase_bins = noisy_phase.shape[1]
        delta_phase = phase_cnn(z_noisy, target_frames=noisy_phase.shape[-1]).clamp(-1.2, 1.2)
        pred_phase = wrap_to_pi(noisy_phase + delta_phase)

        logits_feat = torch.cat(
            [torch.cos(pred_phase), torch.sin(pred_phase), torch.log1p(noisy_stft[:, 0].abs())],
            dim=1,
        )
        logits_feat[:, 2 * phase_bins :, :] = logits_feat[:, 2 * phase_bins :, :] / logits_feat[:, 2 * phase_bins :, :].abs().amax(dim=1, keepdim=True).clamp(min=1e-6)

        delta_logits = offset_head(logits_feat, latent_frames=y_tok.shape[-1])
        base_prob = base_logits.exp().clamp(min=1e-8)
        entropy = -(base_prob * base_logits).sum(dim=-1)
        entropy = (entropy / math.log(base_prob.shape[-1])).clamp(0.0, 1.0).pow(cfg.entropy_gate_power)
        alpha = cfg.alpha_max * gate(logits_feat, entropy, latent_frames=y_tok.shape[-1])

        pred_phase_tv.append(_temporal_tv_stats(pred_phase.reshape(-1, pred_phase.shape[-1])))
        ideal_phase_tv.append(_temporal_tv_stats(ideal_delta_phase.reshape(-1, ideal_delta_phase.shape[-1])))

        alpha_sel = alpha[:, start_layer:, :, :].squeeze(-1).reshape(-1, alpha.shape[-2])
        alpha_tv.append(_temporal_tv_stats(alpha_sel))

        delta_norm = delta_logits[:, start_layer:, :, :].norm(dim=-1).reshape(-1, delta_logits.shape[2])
        delta_norm_tv.append(_temporal_tv_stats(delta_norm))

        n += 1
        if n >= examples:
            break

    def _merge(stats_list: list[dict[str, float]]) -> dict[str, float]:
        if not stats_list:
            return {"mean_abs_diff": float("nan"), "p95_abs_diff": float("nan")}
        return {
            "mean_abs_diff": float(sum(s["mean_abs_diff"] for s in stats_list) / len(stats_list)),
            "p95_abs_diff": float(sum(s["p95_abs_diff"] for s in stats_list) / len(stats_list)),
        }

    pred_tv = _merge(pred_phase_tv)
    ideal_tv = _merge(ideal_phase_tv)
    ratio = float(pred_tv["mean_abs_diff"] / max(ideal_tv["mean_abs_diff"], 1e-8))

    return {
        "examples": n,
        "pred_phase_temporal_tv": pred_tv,
        "ideal_phase_delta_temporal_tv": ideal_tv,
        "pred_vs_ideal_tv_ratio": ratio,
        "alpha_temporal_tv": _merge(alpha_tv),
        "delta_norm_temporal_tv": _merge(delta_norm_tv),
    }


def run_pretrain_audits(
    cfg: StageAConfig,
    ckpt_mode: str,
    examples: int,
    grad_steps: int,
    gumbel_tau: float,
    max_tokens: int,
    feature_channels: int,
    bins: int,
    report_json: str,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)

    phase_bins = cfg.n_fft // 2 + 1
    in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
    phase_cnn = load_phase_cnn(_make_ds_cfg(cfg), in_ch=in_cont_ch, out_ch=phase_bins, device=device)
    for p in phase_cnn.parameters():
        p.requires_grad = False
    phase_cnn.eval()

    start_layer = max(0, int(cfg.offset_start_layer))

    print("[A] Running feature alignment check...")
    exp_a = feature_alignment_check(
        cfg=cfg,
        lm=lm,
        phase_cnn=phase_cnn,
        stft=stft,
        examples=examples,
        max_tokens=max_tokens,
        feature_channels=feature_channels,
        bins=bins,
        start_layer=start_layer,
        device=device,
    )

    print("[B] Running gradient flow audit (hard vs gumbel)...")
    exp_b = gradient_flow_audit(
        cfg=cfg,
        lm=lm,
        phase_cnn=phase_cnn,
        stft=stft,
        examples=examples,
        grad_steps=grad_steps,
        tau=gumbel_tau,
        ckpt_mode=ckpt_mode,
        device=device,
    )

    offset_head, gate = _build_modules_from_ckpt(cfg, lm, device, ckpt_mode)
    offset_head.eval()
    gate.eval()

    print("[C] Running phase consistency audit...")
    exp_c = phase_consistency_audit(
        cfg=cfg,
        lm=lm,
        phase_cnn=phase_cnn,
        stft=stft,
        offset_head=offset_head,
        gate=gate,
        examples=examples,
        start_layer=start_layer,
        device=device,
    )

    report = {
        "meta": {
            "device": str(device),
            "ckpt_mode": ckpt_mode,
            "examples": examples,
            "grad_steps": grad_steps,
            "gumbel_tau": gumbel_tau,
            "config": asdict(cfg),
        },
        "audit_a_feature_alignment": exp_a,
        "audit_b_gradient_flow": exp_b,
        "audit_c_phase_consistency": exp_c,
    }

    out = Path(report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved pretrain audit report: {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-A pretrain audits: feature alignment, gradient flow, phase consistency.")
    parser.add_argument("--config", default="configs/phase_fusion_stagea_schemeb_60_pilot.yaml")
    parser.add_argument("--examples", type=int, default=40)
    parser.add_argument("--grad-steps", type=int, default=30)
    parser.add_argument("--gumbel-tau", type=float, default=2.5)
    parser.add_argument("--max-tokens", type=int, default=120000)
    parser.add_argument("--feature-channels", type=int, default=64)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--ckpt-mode", choices=["best", "last"], default="best")
    parser.add_argument(
        "--report-json",
        default="experiments/phase_fusion_stagea_schemeb_pilot/reports/pretrain_audits_report.json",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.seed)

    run_pretrain_audits(
        cfg=cfg,
        ckpt_mode=args.ckpt_mode,
        examples=int(args.examples),
        grad_steps=int(args.grad_steps),
        gumbel_tau=float(args.gumbel_tau),
        max_tokens=int(args.max_tokens),
        feature_channels=int(args.feature_channels),
        bins=int(args.bins),
        report_json=args.report_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
