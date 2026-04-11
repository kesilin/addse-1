import argparse
import json
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
from stagea_logit_modules import GatingModule, LogitOffsetHead

from addse.data import AudioStreamingDataLoader
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT


def _decode_tokens(lm, tokens: torch.Tensor, clean_len: int) -> torch.Tensor:
    q_books = lm.nac.quantizer.decode(tokens, output_no_sum=True, domain="code")
    q_sum = q_books.sum(dim=2)
    y = lm.nac.decoder(q_sum)
    return y[..., :clean_len]


def _encode_clean_tokens(lm, clean: torch.Tensor) -> torch.Tensor:
    n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
    clean_pad = F.pad(clean, (0, n_pad))
    clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")
    return clean_tok


def _margin_stats(base_logits: torch.Tensor) -> dict[str, float]:
    top2 = torch.topk(base_logits, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    return {
        "margin_mean": float(margin.mean().item()),
        "margin_std": float(margin.std(unbiased=False).item()),
        "margin_p95": float(torch.quantile(margin.flatten(), 0.95).item()),
    }


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat_abs = x.abs().flatten()
    if flat_abs.numel() > 1_000_000:
        idx = torch.randperm(flat_abs.numel(), device=flat_abs.device)[:1_000_000]
        flat_abs = flat_abs[idx]
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "abs_mean": float(x.abs().mean().item()),
        "p95_abs": float(torch.quantile(flat_abs, 0.95).item()),
    }


def _cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-1])
    c = F.cosine_similarity(a2, b2, dim=-1)
    return float(c.mean().item())


@torch.no_grad()
def run_diagnostics(cfg: StageAConfig, ckpt_mode: str, examples: int, report_json: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if ckpt_mode == "best":
        offset_state = torch.load(cfg.offset_head_best_ckpt, map_location=device)
        gate_state = torch.load(cfg.gate_best_ckpt, map_location=device)
    else:
        offset_state = torch.load(cfg.offset_head_last_ckpt, map_location=device)
        gate_state = torch.load(cfg.gate_last_ckpt, map_location=device)
    offset_head.load_state_dict(offset_state["model"], strict=True)
    gate.load_state_dict(gate_state["model"], strict=True)
    offset_head.eval()
    gate.eval()

    dset = build_dset(_make_ds_cfg(cfg), length=examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    start_layer = max(0, int(cfg.offset_start_layer))

    # Exp1 accumulators
    margin_vals = []
    offset_vals = []

    # Exp2 accumulators
    flip_total = 0
    token_total = 0
    flip_by_layer = [0 for _ in range(num_books)]
    total_by_layer = [0 for _ in range(num_books)]
    changed_distances = [[] for _ in range(num_books)]

    # Exp3 metrics
    m_base = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0}
    m_l4 = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0}
    m_l3 = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0}
    m_l34 = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0}
    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    # Exp4 accumulators
    cos_delta_to_ideal = []
    cos_weighted_delta_to_ideal = []

    codebook_weights = [cb.codebook.weight for cb in lm.nac.quantizer.codebooks]  # type: ignore[attr-defined]

    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        x_q, y_tok, y_q_books, y_q_sum, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
        base_logits = lm.log_score(y_q_books, x_q)
        logits_feat = _extract_logits_features(noisy, z_noisy, stft, phase_cnn)

        delta_logits = offset_head(logits_feat, latent_frames=y_tok.shape[-1])
        base_prob = base_logits.exp().clamp(min=1e-8)
        entropy = -(base_prob * base_logits).sum(dim=-1)
        entropy = (entropy / torch.log(torch.tensor(float(base_prob.shape[-1]), device=entropy.device))).clamp(0.0, 1.0)
        entropy = entropy.pow(cfg.entropy_gate_power)
        alpha = cfg.alpha_max * gate(logits_feat, entropy, latent_frames=y_tok.shape[-1])

        injected = cfg.logits_scale * alpha * delta_logits
        final_logits = base_logits.clone()
        if start_layer < num_books:
            final_logits[:, start_layer:, :, :] = base_logits[:, start_layer:, :, :] + injected[:, start_layer:, :, :]
        pred_tok = y_tok.clone()
        if start_layer < num_books:
            pred_tok[:, start_layer:, :] = final_logits[:, start_layer:, :, :].argmax(dim=-1)

        # Exp1
        margin_vals.append(base_logits[:, start_layer:, :, :])
        offset_vals.append(injected[:, start_layer:, :, :])

        # Exp2
        diff = pred_tok != y_tok
        flip_total += int(diff.sum().item())
        token_total += int(diff.numel())
        for k in range(num_books):
            dk = diff[:, k, :]
            flip_by_layer[k] += int(dk.sum().item())
            total_by_layer[k] += int(dk.numel())
            if dk.any():
                old_id = y_tok[:, k, :][dk]
                new_id = pred_tok[:, k, :][dk]
                w = codebook_weights[k]
                dist = (w[new_id] - w[old_id]).norm(dim=-1)
                changed_distances[k].append(dist)

        # Exp3
        clean_tok = _encode_clean_tokens(lm, clean)
        y_base = _decode_tokens(lm, y_tok, clean.shape[-1])

        tok_l4 = y_tok.clone()
        tok_l4[:, 3, :] = clean_tok[:, 3, :]
        y_l4 = _decode_tokens(lm, tok_l4, clean.shape[-1])

        tok_l3 = y_tok.clone()
        tok_l3[:, 2, :] = clean_tok[:, 2, :]
        y_l3 = _decode_tokens(lm, tok_l3, clean.shape[-1])

        tok_l34 = y_tok.clone()
        tok_l34[:, 2:, :] = clean_tok[:, 2:, :]
        y_l34 = _decode_tokens(lm, tok_l34, clean.shape[-1])

        m_base["pesq"] += pesq(y_base[0], clean[0])
        m_base["estoi"] += estoi(y_base[0], clean[0])
        m_base["sdr"] += sdr(y_base[0], clean[0])

        m_l4["pesq"] += pesq(y_l4[0], clean[0])
        m_l4["estoi"] += estoi(y_l4[0], clean[0])
        m_l4["sdr"] += sdr(y_l4[0], clean[0])

        m_l3["pesq"] += pesq(y_l3[0], clean[0])
        m_l3["estoi"] += estoi(y_l3[0], clean[0])
        m_l3["sdr"] += sdr(y_l3[0], clean[0])

        m_l34["pesq"] += pesq(y_l34[0], clean[0])
        m_l34["estoi"] += estoi(y_l34[0], clean[0])
        m_l34["sdr"] += sdr(y_l34[0], clean[0])

        # Exp4
        clean_tok_sel = clean_tok[:, start_layer:, :]
        base_prob_sel = base_logits[:, start_layer:, :, :].softmax(dim=-1)
        ideal = F.one_hot(clean_tok_sel, num_classes=vocab_size).float() - base_prob_sel
        delta_sel = delta_logits[:, start_layer:, :, :]
        weighted_delta_sel = injected[:, start_layer:, :, :]
        cos_delta_to_ideal.append(_cosine_mean(delta_sel, ideal))
        cos_weighted_delta_to_ideal.append(_cosine_mean(weighted_delta_sel, ideal))

        n += 1
        if n % 10 == 0:
            print(f"diag progress {n}/{examples}")
        if n >= examples:
            break

    denom = max(n, 1)

    all_margin = torch.cat([x.reshape(-1, x.shape[-1]) for x in margin_vals], dim=0)
    top2 = torch.topk(all_margin, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]

    all_offset = torch.cat([x.reshape(-1, x.shape[-1]) for x in offset_vals], dim=0)

    exp1 = {
        "base_margin": {
            "mean": float(margin.mean().item()),
            "std": float(margin.std(unbiased=False).item()),
            "p95": float(torch.quantile(margin, 0.95).item()),
        },
        "offset_alpha_delta": _tensor_stats(all_offset),
        "diagnosis": (
            "offset_too_small"
            if float(margin.mean().item()) > 3.0 * float(all_offset.abs().mean().item())
            else "offset_potentially_effective"
        ),
    }

    dist_stats = []
    for k in range(num_books):
        if len(changed_distances[k]) == 0:
            dist_stats.append({"layer": k + 1, "changed": 0, "mean": 0.0, "p95": 0.0})
            continue
        d = torch.cat(changed_distances[k])
        dist_stats.append(
            {
                "layer": k + 1,
                "changed": int(d.numel()),
                "mean": float(d.mean().item()),
                "p95": float(torch.quantile(d, 0.95).item()),
            }
        )

    exp2 = {
        "flip_rate": float(flip_total / max(token_total, 1)),
        "flip_rate_by_layer": [float(flip_by_layer[k] / max(total_by_layer[k], 1)) for k in range(num_books)],
        "migration_distance_by_layer": dist_stats,
    }

    def _norm_metrics(m):
        return {k: float(v / denom) for k, v in m.items()}

    base_m = _norm_metrics(m_base)
    l4_m = _norm_metrics(m_l4)
    l3_m = _norm_metrics(m_l3)
    l34_m = _norm_metrics(m_l34)
    exp3 = {
        "base": base_m,
        "oracle_l4": l4_m,
        "oracle_l3": l3_m,
        "oracle_l34": l34_m,
        "delta_l4_minus_base": {k: l4_m[k] - base_m[k] for k in base_m},
        "delta_l3_minus_base": {k: l3_m[k] - base_m[k] for k in base_m},
        "delta_l34_minus_base": {k: l34_m[k] - base_m[k] for k in base_m},
    }

    exp4 = {
        "cos_delta_to_ideal_mean": float(sum(cos_delta_to_ideal) / max(len(cos_delta_to_ideal), 1)),
        "cos_weighted_delta_to_ideal_mean": float(sum(cos_weighted_delta_to_ideal) / max(len(cos_weighted_delta_to_ideal), 1)),
    }

    report = {
        "meta": {
            "examples": n,
            "device": str(device),
            "ckpt_mode": ckpt_mode,
            "config": asdict(cfg),
        },
        "exp1_scale_conflict_audit": exp1,
        "exp2_token_migration_map": exp2,
        "exp3_layer_sensitivity_sweep": exp3,
        "exp4_feature_correlation_check": exp4,
    }

    out = Path(report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved diagnostics report: {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-A diagnostics bundle: scale, migration, sensitivity, correlation.")
    parser.add_argument("--config", default="configs/phase_fusion_stagea_schemeb_60_pilot.yaml")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--ckpt-mode", choices=["best", "last"], default="best")
    parser.add_argument(
        "--report-json",
        default="experiments/phase_fusion_stagea_schemeb_pilot/reports/diagnostics_report.json",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.seed)
    run_diagnostics(cfg, ckpt_mode=args.ckpt_mode, examples=int(args.examples), report_json=args.report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
