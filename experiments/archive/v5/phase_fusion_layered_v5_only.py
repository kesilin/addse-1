import argparse
from dataclasses import replace
import json
import os
import sys
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = EXPERIMENTS_DIR.parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase_fusion_layered_compare_500 import (
    _evaluate_layered_fused,
    _load_lm,
    _make_base_cfg,
    _train_layered_adapter,
    build_adapter_variant,
    count_total_params,
    count_trainable_params,
    load_phase_model,
    load_adapter_checkpoint,
    load_cfg,
)
from addse.stft import STFT


def _summarize_system_checks(train_diag: dict[str, object]) -> dict[str, object]:
    snaps = train_diag.get("diagnostic_snapshots", [])
    if not isinstance(snaps, list) or not snaps:
        return {
            "status": "no-diagnostics",
            "reason": "No diagnostic snapshots were captured.",
        }

    fusion_ratios: list[float] = []
    grad_l2: list[float] = []
    has_nan_or_inf = False
    y_delta_abs_mean: list[float] = []
    for s in snaps:
        if not isinstance(s, dict):
            continue
        fr = s.get("fusion_ratio_l2")
        if isinstance(fr, (int, float)):
            fusion_ratios.append(float(fr))
        g = s.get("grad_l2")
        if isinstance(g, (int, float)):
            grad_l2.append(float(g))
        yd = s.get("y_delta_abs_mean")
        if isinstance(yd, (int, float)):
            y_delta_abs_mean.append(float(yd))
        if bool(s.get("grad_has_nan_or_inf", False)):
            has_nan_or_inf = True
        for key in ["phase_feat", "delta", "y_q_sum", "latent_corr", "y_q_fused", "y_fused", "y_base"]:
            part = s.get(key)
            if isinstance(part, dict):
                if float(part.get("nan_ratio", 0.0)) > 0.0 or float(part.get("inf_ratio", 0.0)) > 0.0:
                    has_nan_or_inf = True

    fr_mean = sum(fusion_ratios) / max(len(fusion_ratios), 1)
    grad_mean = sum(grad_l2) / max(len(grad_l2), 1)
    yd_mean = sum(y_delta_abs_mean) / max(len(y_delta_abs_mean), 1)

    checks = {
        "numerical_finite": (not has_nan_or_inf),
        "fusion_ratio_ok": (0.03 <= fr_mean <= 0.25),
        "gradient_not_too_small": (grad_mean >= 1e-5),
        "wave_delta_reasonable": (yd_mean <= 0.35),
    }
    status = "pass" if all(checks.values()) else "needs-adjustment"
    return {
        "status": status,
        "checks": checks,
        "summary": {
            "fusion_ratio_l2_mean": fr_mean,
            "grad_l2_mean": grad_mean,
            "y_delta_abs_mean": yd_mean,
            "snapshots": len(snaps),
        },
    }


def main() -> int:
    project_root = EXPERIMENTS_DIR.parents[2]
    os.chdir(project_root)

    parser = argparse.ArgumentParser(description="Run V5 layered_fused only with matched budget.")
    parser.add_argument("--config", default="configs/phase_fusion_layered_v5_phasev2_only_smoke.yaml")
    parser.add_argument("--mode", choices=["train_eval", "eval_only", "quick_compare"], default="train_eval")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--variants", default="confidence_agree_lite_v2")
    parser.add_argument("--train-steps", type=int, default=-1)
    parser.add_argument("--train-batch-size", type=int, default=-1)
    parser.add_argument("--eval-examples", type=int, default=-1)
    parser.add_argument("--train-eval-examples", type=int, default=-1)
    parser.add_argument("--train-eval-every-steps", type=int, default=-1)
    parser.add_argument("--pesq-max-examples", type=int, default=-1)
    parser.add_argument("--quick-solve-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--diagnostic-every-steps", type=int, default=-1)
    parser.add_argument("--diagnostic-max-snapshots", type=int, default=-1)
    parser.add_argument("--fusion-train-base-adapter", choices=["on", "off"], default="")
    parser.add_argument("--fusion-base-lr-scale", type=float, default=-1.0)
    parser.add_argument("--fusion-postprocess-mode", choices=["fixed_add", "mag_scale_perturb", "dual_proxy_perturb", "mag_gate_mix", "mag_ortho_softcap", "residual_weight_direct"], default="")
    parser.add_argument("--fusion-soft-limit-ratio", type=float, default=-1.0)
    parser.add_argument("--fusion-mag-power", type=float, default=-1.0)
    parser.add_argument("--fusion-min-scale", type=float, default=-1.0)
    parser.add_argument("--fusion-ortho-mix", type=float, default=-1.0)
    parser.add_argument("--fusion-perturb-min", type=float, default=-1.0)
    parser.add_argument("--fusion-perturb-max", type=float, default=-1.0)
    parser.add_argument("--fusion-mag-proxy-weight", type=float, default=-1.0)
    parser.add_argument("--fusion-step-proxy-weight", type=float, default=-1.0)
    parser.add_argument("--phase-fusion-weight-min", type=float, default=-1.0)
    parser.add_argument("--phase-fusion-weight-max", type=float, default=-1.0)
    parser.add_argument("--phase-train-fusion-weight-head", choices=["on", "off"], default="")
    parser.add_argument("--ra-gate-min", type=float, default=-1.0)
    parser.add_argument("--ra-gate-max", type=float, default=-1.0)
    parser.add_argument("--ra-uncertainty-threshold", type=float, default=-1.0)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.seed >= 0:
        cfg = replace(cfg, seed=args.seed)
    if args.train_steps > 0:
        cfg = replace(cfg, train_steps=args.train_steps)
    if args.train_batch_size > 0:
        cfg = replace(cfg, train_batch_size=args.train_batch_size)
    if args.eval_examples > 0:
        cfg = replace(cfg, eval_examples=args.eval_examples)
    if args.train_eval_examples > 0:
        cfg = replace(cfg, train_eval_examples=args.train_eval_examples)
    if args.train_eval_every_steps >= 0:
        cfg = replace(cfg, train_eval_every_steps=args.train_eval_every_steps)
    if args.pesq_max_examples >= 0:
        cfg = replace(cfg, pesq_max_examples=args.pesq_max_examples)
    if args.diagnostic_every_steps >= 0:
        cfg = replace(cfg, diagnostic_every_steps=args.diagnostic_every_steps)
    if args.diagnostic_max_snapshots > 0:
        cfg = replace(cfg, diagnostic_max_snapshots=args.diagnostic_max_snapshots)
    if args.fusion_train_base_adapter in {"on", "off"}:
        cfg = replace(cfg, fusion_train_base_adapter=(args.fusion_train_base_adapter == "on"))
    if args.fusion_base_lr_scale > 0:
        cfg = replace(cfg, fusion_base_lr_scale=args.fusion_base_lr_scale)
    if args.fusion_postprocess_mode:
        cfg = replace(cfg, fusion_postprocess_mode=args.fusion_postprocess_mode)
    if args.fusion_soft_limit_ratio > 0:
        cfg = replace(cfg, fusion_soft_limit_ratio=args.fusion_soft_limit_ratio)
    if args.fusion_mag_power > 0:
        cfg = replace(cfg, fusion_mag_power=args.fusion_mag_power)
    if args.fusion_min_scale >= 0:
        cfg = replace(cfg, fusion_min_scale=args.fusion_min_scale)
    if args.fusion_ortho_mix >= 0:
        cfg = replace(cfg, fusion_ortho_mix=args.fusion_ortho_mix)
    if args.fusion_perturb_min > 0:
        cfg = replace(cfg, fusion_perturb_min=args.fusion_perturb_min)
    if args.fusion_perturb_max > 0:
        cfg = replace(cfg, fusion_perturb_max=args.fusion_perturb_max)
    if args.fusion_mag_proxy_weight >= 0:
        cfg = replace(cfg, fusion_mag_proxy_weight=args.fusion_mag_proxy_weight)
    if args.fusion_step_proxy_weight >= 0:
        cfg = replace(cfg, fusion_step_proxy_weight=args.fusion_step_proxy_weight)
    if args.phase_fusion_weight_min >= 0:
        cfg = replace(cfg, phase_fusion_weight_min=args.phase_fusion_weight_min)
    if args.phase_fusion_weight_max >= 0:
        cfg = replace(cfg, phase_fusion_weight_max=args.phase_fusion_weight_max)
    if args.phase_train_fusion_weight_head in {"on", "off"}:
        cfg = replace(cfg, phase_train_fusion_weight_head=(args.phase_train_fusion_weight_head == "on"))
    if args.ra_gate_min >= 0:
        cfg = replace(cfg, ra_gate_min=args.ra_gate_min)
    if args.ra_gate_max >= 0:
        cfg = replace(cfg, ra_gate_max=args.ra_gate_max)
    if args.ra_uncertainty_threshold >= 0:
        cfg = replace(cfg, ra_uncertainty_threshold=args.ra_uncertainty_threshold)
    if args.report_json.strip():
        cfg.report_json = args.report_json

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    layered_lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
    if args.mode == "quick_compare" and args.quick_solve_steps > 0:
        layered_lm.num_steps = min(int(layered_lm.num_steps), int(args.quick_solve_steps))
        print(f"Quick compare solve steps: {layered_lm.num_steps}")
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)
    phase_bins = cfg.n_fft // 2 + 1
    in_cont_ch = layered_lm.nac.encoder.out_conv.conv.out_channels
    phase_model = load_phase_model(cfg, in_ch=in_cont_ch, out_ch=phase_bins, stft=stft, device=device)
    phase_model_total_params = int(count_total_params(phase_model))
    phase_model_trainable_params = int(count_trainable_params(phase_model))
    print(
        f"Phase model params total={phase_model_total_params} "
        f"trainable={phase_model_trainable_params} variant={cfg.phase_model_variant}"
    )

    requested_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not requested_variants:
        requested_variants = ["baseline"]

    baseline_ref = build_adapter_variant(
        cfg,
        in_channels=2 * phase_bins,
        emb_channels=layered_lm.nac.decoder.in_conv.conv.in_channels,
        device=device,
        variant="baseline",
        init_ckpt_path=cfg.init_adapter_ckpt,
    )
    baseline_ref_total = int(count_total_params(baseline_ref))
    baseline_ref_trainable = int(baseline_ref_total)

    def _variant_ckpt(base_ckpt: str, variant: str) -> str:
        p = Path(base_ckpt)
        return str(p.with_name(f"{p.stem}_{variant}{p.suffix}"))

    results: dict[str, dict[str, float | int]] = {}
    train_diagnostics: dict[str, dict[str, object]] = {}
    for variant in requested_variants:
        variant_ckpt = cfg.adapter_ckpt if variant == "baseline" else _variant_ckpt(cfg.adapter_ckpt, variant)
        init_ckpt = cfg.init_adapter_ckpt if variant == "baseline" else None
        adapter = build_adapter_variant(
            cfg,
            in_channels=2 * phase_bins,
            emb_channels=layered_lm.nac.decoder.in_conv.conv.in_channels,
            device=device,
            variant=variant,
            init_ckpt_path=init_ckpt,
        )

        if args.mode in {"train_eval", "quick_compare"}:
            run_cfg = replace(cfg, adapter_ckpt=variant_ckpt)
            train_info = _train_layered_adapter(run_cfg, layered_lm, phase_model, adapter, stft, device, log_prefix=variant)
            train_diagnostics[variant] = train_info
        else:
            if not os.path.exists(variant_ckpt):
                raise FileNotFoundError(f"Missing checkpoint for variant '{variant}': {variant_ckpt}")
            load_adapter_checkpoint(adapter, variant_ckpt, device, strict=True)

        adapter.eval()
        ev = _evaluate_layered_fused(cfg, layered_lm, phase_model, adapter, stft, device, log_prefix=f"{variant} eval")
        trainable_params = int(count_trainable_params(adapter))
        total_params = int(count_total_params(adapter))
        results[variant] = ev | {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "phase_model_total_params": phase_model_total_params,
            "phase_model_trainable_params": phase_model_trainable_params,
            "fusion_plus_phase_total_params": int(total_params + phase_model_total_params),
            "fusion_plus_phase_trainable_params": int(trainable_params + phase_model_trainable_params),
            "trainable_ratio": float(trainable_params) / max(float(total_params), 1.0),
            "total_params_vs_baseline_ratio": float(total_params) / max(float(baseline_ref_total), 1.0),
            "trainable_params_vs_baseline_ratio": float(trainable_params) / max(float(baseline_ref_trainable), 1.0),
        }

    baseline_key = "baseline" if "baseline" in results else requested_variants[0]
    baseline = results[baseline_key]
    deltas: dict[str, dict[str, float]] = {}
    for name, ev in results.items():
        if name == baseline_key:
            continue
        deltas[name] = {
            "pesq": float(ev["pesq"]) - float(baseline["pesq"]),
            "estoi": float(ev["estoi"]) - float(baseline["estoi"]),
            "sdr": float(ev["sdr"]) - float(baseline["sdr"]),
            "phase_rmse": float(ev["phase_rmse"]) - float(baseline["phase_rmse"]),
            "trainable_params_ratio": float(ev["trainable_params"]) / max(float(baseline["trainable_params"]), 1.0),
            "total_params_ratio": float(ev["total_params"]) / max(float(baseline["total_params"]), 1.0),
        }

    ranking = sorted(results.keys(), key=lambda k: (float(results[k]["sdr"]), float(results[k]["pesq"]), float(results[k]["estoi"])), reverse=True)

    report = {
        "results": results,
        "train_diagnostics": train_diagnostics,
        "system_checks": {k: _summarize_system_checks(v) for k, v in train_diagnostics.items()},
        "baseline_variant": baseline_key,
        "delta_vs_baseline": deltas,
        "ranking_by_performance": ranking,
        "config": {
            "train_steps": cfg.train_steps,
            "train_batch_size": cfg.train_batch_size,
            "eval_examples": cfg.eval_examples,
            "train_eval_examples": cfg.train_eval_examples,
            "train_eval_every_steps": cfg.train_eval_every_steps,
            "pesq_max_examples": cfg.pesq_max_examples,
            "lr": cfg.lr,
            "lambda_phase": cfg.lambda_phase,
            "band_loss_weight": cfg.band_loss_weight,
            "fusion_scale_layered": cfg.fusion_scale_layered,
            "snr_min": cfg.snr_min,
            "snr_max": cfg.snr_max,
            "phase_model_variant": cfg.phase_model_variant,
            "fusion_train_base_adapter": cfg.fusion_train_base_adapter,
            "fusion_base_lr_scale": cfg.fusion_base_lr_scale,
            "fusion_postprocess_mode": cfg.fusion_postprocess_mode,
            "fusion_soft_limit_ratio": cfg.fusion_soft_limit_ratio,
            "fusion_mag_power": cfg.fusion_mag_power,
            "fusion_min_scale": cfg.fusion_min_scale,
            "fusion_ortho_mix": cfg.fusion_ortho_mix,
            "fusion_perturb_min": cfg.fusion_perturb_min,
            "fusion_perturb_max": cfg.fusion_perturb_max,
            "fusion_mag_proxy_weight": cfg.fusion_mag_proxy_weight,
            "fusion_step_proxy_weight": cfg.fusion_step_proxy_weight,
            "phase_fusion_weight_init": cfg.phase_fusion_weight_init,
            "phase_fusion_weight_min": cfg.phase_fusion_weight_min,
            "phase_fusion_weight_max": cfg.phase_fusion_weight_max,
            "phase_train_fusion_weight_head": cfg.phase_train_fusion_weight_head,
            "ra_gate_min": cfg.ra_gate_min,
            "ra_gate_max": cfg.ra_gate_max,
            "ra_uncertainty_threshold": cfg.ra_uncertainty_threshold,
            "baseline_ref_total_params": baseline_ref_total,
            "baseline_ref_trainable_params": baseline_ref_trainable,
            "variants": requested_variants,
            "mode": args.mode,
            "quick_solve_steps": int(layered_lm.num_steps),
        },
    }

    Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== V4 Residual Module Compare Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {cfg.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())