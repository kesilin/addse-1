import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def _run(cmd: list[str], cwd: Path, tag: str) -> None:
    print(f"\n[{tag}] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"{tag} failed with exit code {proc.returncode}")
    print(f"[{tag}] Done.")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]

    v3_cfg = Path("configs/phase_fusion_layered_v3_vs_v4_long_300data_5x300.yaml")
    v4_cfg = Path("configs/phase_fusion_layered_v4_phasev2_only_long_300data_5x300.yaml")

    v3_report = project_root / "experiments/archive/v3/reports/v3_vs_v4_long_300data_5x300_train_eval.json"
    v4_report = project_root / "experiments/archive/v4/reports/v4_phasev2_only_long_300data_5x300_train_eval.json"
    summary_report = project_root / "experiments/archive/reports/v3_v4_long_300data_5x300_summary.json"

    v3_cmd = [
        sys.executable,
        "experiments/archive/v3/phase_fusion_layered_v3_only.py",
        "--config",
        str(v3_cfg).replace('\\', '/'),
        "--mode",
        "train_eval",
        "--report-json",
        str(v3_report).replace('\\', '/'),
    ]

    v4_cmd = [
        sys.executable,
        "experiments/archive/v4/phase_fusion_layered_v4_only.py",
        "--config",
        str(v4_cfg).replace('\\', '/'),
        "--mode",
        "train_eval",
        "--variants",
        "baseline",
        "--report-json",
        str(v4_report).replace('\\', '/'),
    ]

    # Serial run: V3 fully finishes before V4 starts.
    _run(v3_cmd, project_root, tag="V3")
    _run(v4_cmd, project_root, tag="V4")

    v3 = _load_json(v3_report)
    v4 = _load_json(v4_report)

    v3_metrics = v3["layered_fused_v3_only"]
    v4_metrics = v4["results"]["baseline"]

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "setting": {
            "description": "Serial long run: V3 then V4, same fusion adapter baseline, only phase branch swapped in V4",
            "train_steps": 1500,
            "eval_examples": 300,
            "train_eval_every_steps": 300,
            "train_eval_examples": 300,
        },
        "v3": {
            "report": str(v3_report).replace('\\', '/'),
            "metrics": {
                "pesq": float(v3_metrics["pesq"]),
                "estoi": float(v3_metrics["estoi"]),
                "sdr": float(v3_metrics["sdr"]),
                "phase_rmse": float(v3_metrics["phase_rmse"]),
            },
            "params": v3.get("param_report", {}),
        },
        "v4": {
            "report": str(v4_report).replace('\\', '/'),
            "metrics": {
                "pesq": float(v4_metrics["pesq"]),
                "estoi": float(v4_metrics["estoi"]),
                "sdr": float(v4_metrics["sdr"]),
                "phase_rmse": float(v4_metrics["phase_rmse"]),
            },
            "params": {
                "phase_model_total_params": int(v4_metrics["phase_model_total_params"]),
                "phase_model_trainable_params": int(v4_metrics["phase_model_trainable_params"]),
                "adapter_total_params": int(v4_metrics["total_params"]),
                "adapter_trainable_params": int(v4_metrics["trainable_params"]),
                "fusion_plus_phase_total_params": int(v4_metrics["fusion_plus_phase_total_params"]),
                "fusion_plus_phase_trainable_params": int(v4_metrics["fusion_plus_phase_trainable_params"]),
            },
        },
        "delta_v4_minus_v3": {
            "pesq": float(v4_metrics["pesq"] - v3_metrics["pesq"]),
            "estoi": float(v4_metrics["estoi"] - v3_metrics["estoi"]),
            "sdr": float(v4_metrics["sdr"] - v3_metrics["sdr"]),
            "phase_rmse": float(v4_metrics["phase_rmse"] - v3_metrics["phase_rmse"]),
        },
    }

    summary_report.parent.mkdir(parents=True, exist_ok=True)
    with summary_report.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SERIAL LONG RUN SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary report: {summary_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
