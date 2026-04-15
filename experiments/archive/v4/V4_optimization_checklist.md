# V4 Optimization Checklist

## Scope and rules
- Keep V3 untouched. All experiments stay in V4 files.
- Use small-budget probes first, then run full training only after passing gates.
- Each probe must report:
  - parameter count
  - alignment metrics
  - quality metrics
  - runtime budget

## Phase 1: Extractor-first probes (current focus)
- [x] Build standalone confidence-guided extractor probe script.
- [ ] Run baseline extractor vs confidence-guided extractor on same data split.
- [ ] Compare lightweight routing variants:
  - soft routing (continuous gate)
  - hard threshold routing (binary gate)
  - top-k uncertain frame routing
- [ ] Compare uncertainty features:
  - entropy-like proxy from phase residual magnitude
  - local temporal variance of phase residual
  - band-wise inconsistency score
- [ ] Gate acceptance criteria:
  - SDR no worse than baseline
  - phase RMSE lower than baseline
  - alignment score improved
  - parameter growth under +25%

## Phase 2: Multi-scale residual extractor (separate script)
- [ ] Implement multi-scale branches (short/mid/dilated).
- [ ] Add optional frequency-group gate.
- [ ] Report quality-params tradeoff curve.
- [ ] Keep best 1-2 variants only.

## Phase 3: Fusion module innovation
- [ ] Add reliability-aware fusion weights from uncertainty map.
- [ ] Add band-group fusion policy (low/mid/high frequency).
- [ ] Test 2-stage training:
  - stage A: conservative fusion cap
  - stage B: release cap for strength learning
- [ ] Compare against current fixed fusion scale.

## Optional extensions (after main path is stable)
- [ ] Dual-domain alignment extractor (latent + acoustic side hint).
- [ ] Frequency-codebook cooperative routing (lightweight version).
- [ ] Local phase consistency auxiliary objective.

## Reporting template per run
- Experiment ID
- Config summary
- Train budget (steps, batch, snr)
- Params and relative growth
- Metrics:
  - PESQ
  - ESTOI
  - SDR
  - phase_rmse
  - alignment_cos
- Decision:
  - promote to next stage / reject

## Immediate next action
- Run confidence-guided standalone probe and collect first comparison report.

## Latest experiment snapshot (2026-04-15)
- Script: experiments/archive/v4/probe_confidence_guided_extractor.py
- Report: experiments/archive/v4/reports/confidence_extractor_probe_report_lite_40x4_eval60.json
- Setup: train_steps=40, batch=4, eval_examples=60, pesq_max_examples=20, snr=[0,10]
- Ranking by performance (SDR > PESQ > ESTOI):
  - confidence_soft_lite
  - confidence_hard_lite
  - baseline_extractor
  - confidence_topk_tiny
- Key delta vs baseline:
  - confidence_soft_lite: SDR +0.00503, PESQ +0.000052, params x1.308
  - confidence_hard_lite: SDR +0.00499, PESQ +0.000051, params x1.308
  - confidence_topk_tiny: SDR -0.00534, PESQ -0.000176, params x0.920
- Decision:
  - Keep confidence_soft_lite as primary candidate.
  - Keep confidence_hard_lite as backup candidate.
  - Drop confidence_topk_tiny for current stage.

## Extractor recheck (2026-04-17)
- Script: experiments/archive/v4/probe_confidence_guided_extractor.py
- Report: experiments/archive/v4/reports/confidence_extractor_probe_report_focus_200x4_eval60_gw1e-3_soft_hard.json
- Setup: train_steps=200, batch=4, eval_examples=60, pesq_max_examples=20, variants=[baseline_extractor, confidence_soft_lite, confidence_hard_lite], gate_entropy_weight=0.001
- Key result vs baseline:
  - confidence_soft_lite: PESQ +0.000196, SDR +0.003358, ESTOI -0.000016, params x1.308
  - confidence_hard_lite: PESQ -0.000005, SDR +0.003052, ESTOI -0.000018, params x1.308
- Decision:
  - Revert extractor to baseline direct extraction for downstream training.
  - Do not promote confidence_soft_lite or confidence_hard_lite as the default extractor path.
  - Treat the confidence extractor as a rejected branch unless a later redesign reduces parameters and yields a clearer gain.

## V4 integration quick-check (2026-04-15)
- Script: experiments/archive/v4/phase_fusion_layered_v4_only.py
- Report: experiments/archive/v4/reports/v4_residual_module_compare_quick_8x4_eval8.json
- Setup: mode=quick_compare, variants=[baseline, confidence_soft_lite, confidence_hard_lite], train_steps=8, batch=4, eval_examples=8, pesq_max_examples=3
- Ranking by performance (SDR > PESQ > ESTOI):
  - baseline
  - confidence_soft_lite
  - confidence_hard_lite
- Key delta vs baseline:
  - confidence_soft_lite: SDR -0.8298, PESQ -0.1385, params x0.282
  - confidence_hard_lite: SDR -1.4978, PESQ -0.1100, params x0.282
- Interpretation:
  - This run is an ultra-short integration sanity check only.
  - The two confidence-lite modules are much smaller than baseline adapter and currently underfit at 8-step budget.
  - Use medium budget (for example 40-60 steps) before final keep/drop decision in V4 mainline.

## V4 redesign quick-check (2026-04-16)
- Design fix:
  - Replace standalone lite adapter with residual-on-top-of-pretrained baseline adapter.
  - Keep pretrained baseline path frozen; train only uncertainty-routed residual branch.
  - Add quick compare solve-step cap in runner to avoid long waiting during module probes.
- Script: experiments/archive/v4/phase_fusion_layered_v4_only.py
- Report: experiments/archive/v4/reports/v4_residual_module_compare_quick_2x2_eval3_v2.json
- Setup: mode=quick_compare, quick_solve_steps=4, train_steps=2, batch=2, eval_examples=3, pesq_max_examples=2
- Ranking by performance (SDR > PESQ > ESTOI):
  - confidence_soft_lite
  - confidence_hard_lite
  - baseline
- Key delta vs baseline:
  - confidence_soft_lite: SDR +0.8632, PESQ +0.1117, params x0.213
  - confidence_hard_lite: SDR +0.4493, PESQ +0.0748, params x0.213
- Interpretation:
  - Previous severe drop was primarily a design/initialization issue, not only parameter size.
  - Conservative residual overlay fixes early-stage collapse tendency.
  - Need a medium-budget confirmation run before final adoption.

## V4 medium-budget 4-variant compare (2026-04-16)
- Script: experiments/archive/v4/phase_fusion_layered_v4_only.py
- Report: experiments/archive/v4/reports/v4_residual_module_compare_medium_20x2_eval8_4variants_v3.json
- Setup: mode=quick_compare, quick_solve_steps=4, train_steps=20, batch=2, eval_examples=8, pesq_max_examples=4
- Variants:
  - baseline
  - confidence_soft_lite
  - confidence_hard_lite
  - confidence_agree_lite (new innovation: agreement-gated residual)
- Ranking by performance (SDR > PESQ > ESTOI):
  - confidence_hard_lite
  - confidence_agree_lite
  - baseline
  - confidence_soft_lite
- Key delta vs baseline:
  - confidence_hard_lite: SDR +0.2866, PESQ -0.0346, ESTOI +0.0332, params x0.213
  - confidence_agree_lite: SDR +0.2143, PESQ +0.0902, ESTOI +0.0111, params x0.231
  - confidence_soft_lite: SDR -0.1095, PESQ -0.0430, ESTOI +0.0069, params x0.213
- Decision for paper-main candidate:
  - Primary: confidence_agree_lite (quality and SDR both improved vs baseline, with low parameter ratio).
  - Secondary: confidence_hard_lite (best SDR/ESTOI but slight PESQ drop).

## ConfidenceAgree consolidation run (2026-04-16)
- Target: baseline vs confidence_agree_lite, 2 rounds
- Common setup: train_steps=200, batch=2, eval_examples=60, pesq_max_examples=20, quick_solve_steps=4, snr=[0,10]
- Reports:
  - experiments/archive/v4/reports/v4_confagree_vs_baseline_seed42_200step_eval60.json
  - experiments/archive/v4/reports/v4_confagree_vs_baseline_seed43_200step_eval60.json
- Delta (agree - baseline):
  - seed42: PESQ -0.0185, ESTOI -0.0065, SDR -0.0601, phase_rmse +0.00132
  - seed43: PESQ -0.0402, ESTOI -0.0041, SDR +0.0746, phase_rmse -0.00024
- Mean delta over 2 rounds:
  - PESQ -0.0293
  - ESTOI -0.0053
  - SDR +0.0072
  - phase_rmse +0.00054
- Interim conclusion:
  - confidence_agree_lite remains highly parameter-efficient (params ratio x0.231).
  - Under this 2x200 setup, quality gains are not yet stable; SDR is roughly on-par with slight positive mean.

## ConfidenceAgree 120-example recheck (2026-04-16)
- Target: baseline vs confidence_agree_lite, 2 rounds
- Common setup: train_steps=200, batch=2, eval_examples=120, pesq_max_examples=20, quick_solve_steps=4, snr=[0,10]
- Reports:
  - experiments/archive/v4/reports/v4_confagree_vs_baseline_seed42_200step_eval120.json
  - experiments/archive/v4/reports/v4_confagree_vs_baseline_seed43_200step_eval120.json
- Delta (agree - baseline):
  - seed42: PESQ -0.0579, ESTOI -0.0057, SDR -0.0951, phase_rmse +0.00024, params x0.238
  - seed43: PESQ -0.0707, ESTOI +0.0012, SDR -0.2030, phase_rmse -0.00094, params x0.238
- Final conclusion:
  - The explicit fusion block is now in place, so the parallel branch matches the intended three-module design more closely.
  - Even with that fix, confidence_agree_lite is still not stable against baseline at the 120-example recheck.
  - Keep it as a strong parameter-efficiency candidate, but do not promote it as the final mainline unless a stronger training schedule or gating rule is added.
