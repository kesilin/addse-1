import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from phase_fusion_scheme1 import build_dset, load_phase_cnn
from phase_fusion_stagea_schemeb import (
    StageAConfig,
    _extract_logits_features,
    _load_lm,
    _make_ds_cfg,
    _run_base,
    load_cfg,
)
from stagea_logit_modules import GeoLogitHead, proximity_soft_targets

from addse.data import AudioStreamingDataLoader
from addse.stft import STFT


def _cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-1])
    c = F.cosine_similarity(a2, b2, dim=-1)
    return float(c.mean().item())


def _apply_candidate_lock(delta_sel: torch.Tensor, candidate_idx: torch.Tensor | None, lock_fill: float) -> torch.Tensor:
    if candidate_idx is None:
        return delta_sel
    locked = torch.full_like(delta_sel, float(lock_fill))
    locked.scatter_(-1, candidate_idx, delta_sel.gather(-1, candidate_idx))
    return locked


def _build_candidate_idx(
    base_logits_sel: torch.Tensor,
    topk_lock: int,
    mask_mode: str,
    oracle_tok_sel: torch.Tensor | None,
) -> torch.Tensor | None:
    if mask_mode == "none" or topk_lock <= 0:
        return None

    vocab = base_logits_sel.shape[-1]
    k = min(int(topk_lock), int(vocab))
    top_idx = torch.topk(base_logits_sel, k=k, dim=-1).indices

    if mask_mode == "topk":
        return top_idx

    if mask_mode == "topk_oracle_union":
        if oracle_tok_sel is None:
            raise ValueError("oracle_tok_sel is required for topk_oracle_union mode")
        return torch.cat([top_idx, oracle_tok_sel.unsqueeze(-1)], dim=-1)

    raise ValueError(f"Unsupported mask mode: {mask_mode}")


def _build_eval_topk_with_lowconf_expand(
    base_logits_sel: torch.Tensor,
    topk_lock: int,
    lowconf_expand_k: int,
    lowconf_margin: float,
) -> tuple[torch.Tensor | None, float]:
    if topk_lock <= 0:
        return None, 0.0

    vocab = base_logits_sel.shape[-1]
    k = min(int(topk_lock), int(vocab))
    base_topk = torch.topk(base_logits_sel, k=k, dim=-1).indices

    expand = max(0, int(lowconf_expand_k))
    if expand == 0:
        return base_topk, 0.0

    k2 = min(int(vocab), k + expand)
    if k2 <= k:
        return base_topk, 0.0

    top2 = torch.topk(base_logits_sel, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    low_mask = margin < float(lowconf_margin)
    low_ratio = float(low_mask.float().mean().item())

    expanded = torch.topk(base_logits_sel, k=k2, dim=-1).indices
    out = expanded.clone()
    if (~low_mask).any():
        pad_val = base_topk[..., -1:]
        out[..., k:] = torch.where(
            low_mask.unsqueeze(-1),
            out[..., k:],
            pad_val.expand_as(out[..., k:]),
        )
    return out, low_ratio


def _base_fused_distance_stats(
    base_tok_sel: torch.Tensor,
    fused_tok_sel: torch.Tensor,
    codebook_weights: list[torch.Tensor],
    start_layer: int,
) -> dict:
    changed_dists = []
    total_changed = 0
    total = int(base_tok_sel.numel())

    num_sel_books = base_tok_sel.shape[1]
    for j in range(num_sel_books):
        k = start_layer + j
        w = codebook_weights[k]
        b = base_tok_sel[:, j, :]
        f = fused_tok_sel[:, j, :]
        diff = b != f
        if diff.any():
            bd = w[b[diff]]
            fd = w[f[diff]]
            dist = (fd - bd).norm(dim=-1)
            changed_dists.append(dist)
            total_changed += int(dist.numel())

    if len(changed_dists) == 0:
        return {
            "flip_rate": 0.0,
            "changed": 0,
            "distance_mean": 0.0,
            "distance_p95": 0.0,
        }

    d = torch.cat(changed_dists)
    return {
        "flip_rate": float(total_changed / max(total, 1)),
        "changed": int(total_changed),
        "distance_mean": float(d.mean().item()),
        "distance_p95": float(torch.quantile(d, 0.95).item()),
    }


@torch.no_grad()
def _evaluate_probe(
    cfg: StageAConfig,
    lm,
    phase_cnn,
    stft,
    head: GeoLogitHead,
    examples: int,
    start_layer: int,
    solve_steps: int,
    topk_lock: int,
    lock_fill: float,
    eval_mask_mode: str,
    eval_lowconf_expand_k: int,
    eval_lowconf_margin: float,
    codebook_weights: list[torch.Tensor],
    device: torch.device,
) -> dict:
    dset = build_dset(_make_ds_cfg(cfg), length=examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    cos_list = []
    flip_total = 0
    token_total = 0
    dist_stats = []
    lowconf_ratios = []

    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, solve_steps)
        base_logits = lm.log_score(y_q_books, x_q)
        feat = _extract_logits_features(noisy, z_noisy, stft, phase_cnn)

        delta_logits = head(feat, latent_frames=y_tok.shape[-1])

        n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
        clean_pad = F.pad(clean, (0, n_pad))
        clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")

        base_sel = base_logits[:, start_layer:, :, :]
        delta_sel = delta_logits[:, start_layer:, :, :]
        if eval_mask_mode == "topk" and (eval_lowconf_expand_k > 0):
            eval_idx, low_ratio = _build_eval_topk_with_lowconf_expand(
                base_logits_sel=base_sel,
                topk_lock=topk_lock,
                lowconf_expand_k=eval_lowconf_expand_k,
                lowconf_margin=eval_lowconf_margin,
            )
            lowconf_ratios.append(low_ratio)
        else:
            eval_idx = _build_candidate_idx(
                base_logits_sel=base_sel,
                topk_lock=topk_lock,
                mask_mode=eval_mask_mode,
                oracle_tok_sel=None,
            )
            lowconf_ratios.append(0.0)
        delta_locked = _apply_candidate_lock(delta_sel, eval_idx, lock_fill=lock_fill)

        base_prob_sel = base_sel.softmax(dim=-1)
        clean_tok_sel = clean_tok[:, start_layer:, :]
        ideal = F.one_hot(clean_tok_sel, num_classes=base_logits.shape[-1]).float() - base_prob_sel
        cos_list.append(_cosine_mean(delta_locked, ideal))

        final = base_logits.clone()
        final[:, start_layer:, :, :] = base_sel + cfg.logits_scale * delta_locked
        pred_tok = y_tok.clone()
        pred_tok[:, start_layer:, :] = final[:, start_layer:, :, :].argmax(dim=-1)

        diff = pred_tok[:, start_layer:, :] != y_tok[:, start_layer:, :]
        flip_total += int(diff.sum().item())
        token_total += int(diff.numel())
        dist_stats.append(
            _base_fused_distance_stats(
                base_tok_sel=y_tok[:, start_layer:, :],
                fused_tok_sel=pred_tok[:, start_layer:, :],
                codebook_weights=codebook_weights,
                start_layer=start_layer,
            )
        )

    if len(dist_stats) == 0:
        dist_agg = {"flip_rate": 0.0, "changed": 0, "distance_mean": 0.0, "distance_p95": 0.0}
    else:
        dist_agg = {
            "flip_rate": float(sum(x["flip_rate"] for x in dist_stats) / len(dist_stats)),
            "changed": int(sum(x["changed"] for x in dist_stats)),
            "distance_mean": float(sum(x["distance_mean"] for x in dist_stats) / len(dist_stats)),
            "distance_p95": float(sum(x["distance_p95"] for x in dist_stats) / len(dist_stats)),
        }

    return {
        "examples": int(examples),
        "cos_delta_to_ideal_mean": float(sum(cos_list) / max(len(cos_list), 1)),
        "flip_rate_selected_layers": float(flip_total / max(token_total, 1)),
        "gentle_flip_audit": dist_agg,
        "eval_lowconf_ratio": float(sum(lowconf_ratios) / max(len(lowconf_ratios), 1)),
    }


def run_probe(
    cfg: StageAConfig,
    mode: str,
    start_layer: int,
    probe_examples: int,
    train_steps: int,
    batch_size: int,
    solve_steps: int,
    lr: float,
    proximity_temp: float,
    proximity_weight: float,
    ce_weight: float,
    direction_weight: float,
    dist_weight: float,
    sim_scale: float,
    topk_lock: int,
    train_lock_fill: float,
    eval_lock_fill: float,
    train_mask_mode: str,
    eval_mask_mode: str,
    eval_lowconf_expand_k: int,
    eval_lowconf_margin: float,
    dist_margin: float,
    dist_flip_only: bool,
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

    codebook_weights = [cb.codebook.weight for cb in lm.nac.quantizer.codebooks]  # type: ignore[attr-defined]
    head = GeoLogitHead(
        in_channels=3 * phase_bins,
        codebook_weights=codebook_weights,
        hidden=cfg.low_rank_dim,
        sim_scale=sim_scale,
        mode=mode,
    ).to(device)

    before = _evaluate_probe(
        cfg=cfg,
        lm=lm,
        phase_cnn=phase_cnn,
        stft=stft,
        head=head,
        examples=probe_examples,
        start_layer=start_layer,
        solve_steps=solve_steps,
        topk_lock=topk_lock,
        lock_fill=eval_lock_fill,
        eval_mask_mode=eval_mask_mode,
        eval_lowconf_expand_k=eval_lowconf_expand_k,
        eval_lowconf_margin=eval_lowconf_margin,
        codebook_weights=codebook_weights,
        device=device,
    )

    dset = build_dset(_make_ds_cfg(cfg), length=max(1, train_steps * batch_size), reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=batch_size, num_workers=cfg.num_workers, shuffle=True)

    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    step_logs = []

    head.train()
    step = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, solve_steps)
        base_logits = lm.log_score(y_q_books, x_q).detach()
        feat = _extract_logits_features(noisy, z_noisy, stft, phase_cnn)
        delta_logits = head(feat, latent_frames=y_tok.shape[-1])

        n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
        clean_pad = F.pad(clean, (0, n_pad))
        clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")
        clean_tok_sel = clean_tok[:, start_layer:, :]

        vocab = base_logits.shape[-1]
        base_sel = base_logits[:, start_layer:, :, :]
        delta_sel = delta_logits[:, start_layer:, :, :]
        train_idx = _build_candidate_idx(
            base_logits_sel=base_sel,
            topk_lock=topk_lock,
            mask_mode=train_mask_mode,
            oracle_tok_sel=clean_tok_sel,
        )
        delta_locked = _apply_candidate_lock(delta_sel, train_idx, lock_fill=train_lock_fill)
        final_sel = base_sel + cfg.logits_scale * delta_locked

        ce_hard = F.cross_entropy(final_sel.reshape(-1, vocab), clean_tok_sel.reshape(-1))

        soft_targets = proximity_soft_targets(codebook_weights, clean_tok, temperature=proximity_temp)
        soft_sel = soft_targets[:, start_layer:, :, :]
        logp = final_sel.log_softmax(dim=-1)
        loss_prox = -(soft_sel * logp).sum(dim=-1).mean()

        base_prob_sel = base_sel.softmax(dim=-1)
        ideal = F.one_hot(clean_tok_sel, num_classes=vocab).float() - base_prob_sel
        cos_val = _cosine_mean(delta_locked, ideal)

        # Directional alignment: encourage delta to point toward ideal correction direction.
        dir_loss = 1.0 - F.cosine_similarity(
            delta_locked.reshape(-1, vocab),
            ideal.reshape(-1, vocab),
            dim=-1,
        ).mean()

        fused_tok_sel = final_sel.argmax(dim=-1)
        dist_reg_terms = []
        for j in range(fused_tok_sel.shape[1]):
            k = start_layer + j
            w = codebook_weights[k]
            emb_fused = w[fused_tok_sel[:, j, :].reshape(-1)]
            emb_base = w[y_tok[:, k, :].reshape(-1)]
            d = (emb_fused - emb_base).norm(dim=-1)
            if dist_flip_only:
                flip_mask = (fused_tok_sel[:, j, :].reshape(-1) != y_tok[:, k, :].reshape(-1))
                if flip_mask.any():
                    d = d[flip_mask]
                else:
                    continue
            dist_reg_terms.append(F.relu(d - float(dist_margin)).square().mean())
        loss_dist = torch.stack(dist_reg_terms).mean() if len(dist_reg_terms) > 0 else torch.tensor(0.0, device=device)

        loss = ce_weight * ce_hard + proximity_weight * loss_prox + direction_weight * dir_loss + dist_weight * loss_dist

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1
        if step == 1 or step % 10 == 0 or step == train_steps:
            print(
                f"geo probe step={step}/{train_steps} loss={loss.item():.5f} "
                f"ce={ce_hard.item():.5f} prox={loss_prox.item():.5f} dir={dir_loss.item():.5f} "
                f"dist={loss_dist.item():.5f} cos={cos_val:.5f}"
            )
            step_logs.append(
                {
                    "step": int(step),
                    "loss": float(loss.item()),
                    "ce_hard": float(ce_hard.item()),
                    "loss_proximity": float(loss_prox.item()),
                    "loss_direction": float(dir_loss.item()),
                    "loss_dist": float(loss_dist.item()),
                    "cos_delta_to_ideal": float(cos_val),
                }
            )

        if step >= train_steps:
            break

    head.eval()
    after = _evaluate_probe(
        cfg=cfg,
        lm=lm,
        phase_cnn=phase_cnn,
        stft=stft,
        head=head,
        examples=probe_examples,
        start_layer=start_layer,
        solve_steps=solve_steps,
        topk_lock=topk_lock,
        lock_fill=eval_lock_fill,
        eval_mask_mode=eval_mask_mode,
        eval_lowconf_expand_k=eval_lowconf_expand_k,
        eval_lowconf_margin=eval_lowconf_margin,
        codebook_weights=codebook_weights,
        device=device,
    )

    report = {
        "meta": {
            "device": str(device),
            "mode": mode,
            "start_layer_zero_based": int(start_layer),
            "train_steps": int(train_steps),
            "batch_size": int(batch_size),
            "probe_examples": int(probe_examples),
            "solve_steps": int(solve_steps),
            "lr": float(lr),
            "proximity_temp": float(proximity_temp),
            "proximity_weight": float(proximity_weight),
            "ce_weight": float(ce_weight),
            "direction_weight": float(direction_weight),
            "dist_weight": float(dist_weight),
            "sim_scale": float(sim_scale),
            "topk_lock": int(topk_lock),
            "train_lock_fill": float(train_lock_fill),
            "eval_lock_fill": float(eval_lock_fill),
            "train_mask_mode": str(train_mask_mode),
            "eval_mask_mode": str(eval_mask_mode),
            "eval_lowconf_expand_k": int(eval_lowconf_expand_k),
            "eval_lowconf_margin": float(eval_lowconf_margin),
            "dist_margin": float(dist_margin),
            "dist_flip_only": bool(dist_flip_only),
            "logits_scale": float(cfg.logits_scale),
            "config": asdict(cfg),
        },
        "before": before,
        "after": after,
        "delta": {
            "cos_delta_to_ideal_mean": float(after["cos_delta_to_ideal_mean"] - before["cos_delta_to_ideal_mean"]),
            "flip_rate_selected_layers": float(after["flip_rate_selected_layers"] - before["flip_rate_selected_layers"]),
            "distance_mean": float(after["gentle_flip_audit"]["distance_mean"] - before["gentle_flip_audit"]["distance_mean"]),
        },
        "train_log": step_logs,
    }

    out = Path(report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved geologit probe report: {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.4 LocalGeo probe: Top-K reranking + geometry alignment (no long training).")
    parser.add_argument("--config", default="configs/phase_fusion_stagea_schemeb_60_pilot.yaml")
    parser.add_argument("--mode", choices=["cosine", "suppressive"], default="suppressive")
    parser.add_argument("--start-layer", type=int, default=2, help="Zero-based RVQ layer start; 2 means only layer-3/4 are editable.")
    parser.add_argument("--probe-examples", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--solve-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--proximity-temp", type=float, default=0.35)
    parser.add_argument("--proximity-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--direction-weight", type=float, default=0.5)
    parser.add_argument("--dist-weight", type=float, default=0.1)
    parser.add_argument("--sim-scale", type=float, default=6.0)
    parser.add_argument("--topk-lock", type=int, default=10)
    parser.add_argument("--train-lock-fill", type=float, default=-40.0)
    parser.add_argument("--eval-lock-fill", type=float, default=-12.0)
    parser.add_argument("--train-mask-mode", choices=["none", "topk", "topk_oracle_union"], default="topk_oracle_union")
    parser.add_argument("--eval-mask-mode", choices=["none", "topk"], default="topk")
    parser.add_argument("--eval-lowconf-expand-k", type=int, default=10)
    parser.add_argument("--eval-lowconf-margin", type=float, default=0.12)
    parser.add_argument("--dist-margin", type=float, default=4.0)
    parser.add_argument("--dist-flip-only", action="store_true")
    parser.add_argument(
        "--report-json",
        default="experiments/phase_fusion_stagea_schemeb_pilot/reports/geologit_probe_report.json",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.seed)

    run_probe(
        cfg=cfg,
        mode=args.mode,
        start_layer=int(args.start_layer),
        probe_examples=int(args.probe_examples),
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        solve_steps=int(args.solve_steps),
        lr=float(args.lr),
        proximity_temp=float(args.proximity_temp),
        proximity_weight=float(args.proximity_weight),
        ce_weight=float(args.ce_weight),
        direction_weight=float(args.direction_weight),
        dist_weight=float(args.dist_weight),
        sim_scale=float(args.sim_scale),
        topk_lock=int(args.topk_lock),
        train_lock_fill=float(args.train_lock_fill),
        eval_lock_fill=float(args.eval_lock_fill),
        train_mask_mode=str(args.train_mask_mode),
        eval_mask_mode=str(args.eval_mask_mode),
        eval_lowconf_expand_k=int(args.eval_lowconf_expand_k),
        eval_lowconf_margin=float(args.eval_lowconf_margin),
        dist_margin=float(args.dist_margin),
        dist_flip_only=bool(args.dist_flip_only),
        report_json=args.report_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
