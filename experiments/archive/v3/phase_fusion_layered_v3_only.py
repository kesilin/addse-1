import argparse
import json
import sys
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = EXPERIMENTS_DIR.parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
V4_DIR = EXPERIMENTS_DIR.parent / "v4"
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))

from phase_fusion_layered_compare_500 import (
    _evaluate_layered_fused,
    _load_adapter,
    _load_lm,
    _make_base_cfg,
    _train_layered_adapter,
    count_total_params,
    count_trainable_params,
    load_cfg,
)
from phase_fusion_scheme1 import load_phase_cnn
from addse.stft import STFT


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V3 layered_fused only with matched budget.")
    parser.add_argument("--config", default="configs/phase_fusion_layered_v3_run100_match.yaml")
    parser.add_argument("--mode", choices=["train_eval", "eval_only"], default="train_eval")
    parser.add_argument("--report-json", default="experiments/phase_fusion_layered_v3_run100_match/reports/report_100_v3_only.json")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    cfg.report_json = args.report_json

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    layered_lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)
    phase_bins = cfg.n_fft // 2 + 1
    in_cont_ch = layered_lm.nac.encoder.out_conv.conv.out_channels
    phase_cnn = load_phase_cnn(_make_base_cfg(cfg, cfg.fusion_scale_layered), in_ch=in_cont_ch, out_ch=phase_bins, device=device)
    phase_model_total_params = int(count_total_params(phase_cnn))
    phase_model_trainable_params = int(count_trainable_params(phase_cnn))
    print(
        f"Phase model params total={phase_model_total_params} "
        f"trainable={phase_model_trainable_params} variant=phase_cnn"
    )

    adapter = _load_adapter(cfg, 2 * phase_bins, layered_lm.nac.decoder.in_conv.conv.in_channels, device, cfg.init_adapter_ckpt)
    adapter_total_params = int(count_total_params(adapter))
    adapter_trainable_params = int(count_trainable_params(adapter))
    print(
        f"Adapter params total={adapter_total_params} trainable={adapter_trainable_params}"
    )
    if args.mode == "train_eval":
        _train_layered_adapter(cfg, layered_lm, phase_cnn, adapter, stft, device)

    adapter.eval()
    fused_report = _evaluate_layered_fused(cfg, layered_lm, phase_cnn, adapter, stft, device)

    report = {
        "layered_fused_v3_only": fused_report,
        "param_report": {
            "phase_model_total_params": phase_model_total_params,
            "phase_model_trainable_params": phase_model_trainable_params,
            "adapter_total_params": adapter_total_params,
            "adapter_trainable_params": adapter_trainable_params,
            "fusion_plus_phase_total_params": int(adapter_total_params + phase_model_total_params),
            "fusion_plus_phase_trainable_params": int(adapter_trainable_params + phase_model_trainable_params),
        },
        "config": {
            "train_steps": cfg.train_steps,
            "train_batch_size": cfg.train_batch_size,
            "eval_examples": cfg.eval_examples,
            "lr": cfg.lr,
            "lambda_phase": cfg.lambda_phase,
            "band_loss_weight": cfg.band_loss_weight,
            "fusion_scale_layered": cfg.fusion_scale_layered,
        },
    }

    Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== V3 Only Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {cfg.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
