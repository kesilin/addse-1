import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path("f:/ksl/addse")
BASE_CFG = ROOT / "configs/phaseadapter_v361_focus_single_base.yaml"
OUT_CFG_DIR = ROOT / "configs/phaseadapter_v361_focus_generated"
OUT_REPORT_DIR = ROOT / "experiments/phaseadapter_v36_probe/reports"
SCRIPT = ROOT / "experiments/phaseadapter_v361_adalnzero_probe.py"
SUMMARY_PATH = OUT_REPORT_DIR / "v361_focus_summary_5rounds.json"
SEEDS = [42, 43, 44, 45, 46]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def run_one(cfg_path: Path) -> None:
    cmd = [sys.executable, str(SCRIPT), "--config", str(cfg_path)]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def load_report_with_retry(report_path: Path, retries: int = 20, delay_s: float = 3.0) -> dict:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
            row = report.get("after_probe_adapter") or report.get("eval_adapter")
            if row is None and report.get("rows"):
                row = report["rows"][0].get("after_probe_adapter") or report["rows"][0].get("eval_adapter")
            if row is None:
                raise KeyError("missing after_probe_adapter/eval_adapter")
            return report
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(delay_s)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load report: {report_path}")


def report_is_complete(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    try:
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        row = report.get("after_probe_adapter") or report.get("eval_adapter")
        if row is None and report.get("rows"):
            row = report["rows"][0].get("after_probe_adapter") or report["rows"][0].get("eval_adapter")
        avg_step_seconds = report.get("avg_step_seconds")
        if avg_step_seconds is None and report.get("rows"):
            avg_step_seconds = report["rows"][0].get("avg_step_seconds")
        return row is not None and avg_step_seconds is not None
    except Exception:  # noqa: BLE001
        return False


def mean(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def main() -> int:
    base = load_yaml(BASE_CFG)

    rows = []
    for seed in SEEDS:
        cfg = dict(base)
        cfg["seed"] = int(seed)
        cfg["report_json"] = f"experiments/phaseadapter_v36_probe/reports/v361_focus_seed{seed}.json"

        cfg_path = OUT_CFG_DIR / f"v361_focus_seed{seed}.yaml"
        save_yaml(cfg_path, cfg)

        report_path = ROOT / cfg["report_json"]
        if report_is_complete(report_path):
            print(f"[SKIP] seed={seed} existing complete report")
        else:
            if report_path.exists():
                print(f"[RUN] seed={seed} incomplete report, rerunning")
            else:
                print(f"[RUN] seed={seed}")
            run_one(cfg_path)

        rep = load_report_with_retry(report_path)

        adapter_row = rep.get("after_probe_adapter") or rep.get("eval_adapter")
        if adapter_row is None and rep.get("rows"):
            adapter_row = rep["rows"][0].get("after_probe_adapter") or rep["rows"][0].get("eval_adapter")
        baseline_row = rep.get("zero_step_baseline")
        avg_step_seconds = rep.get("avg_step_seconds")
        if avg_step_seconds is None and rep.get("rows"):
            avg_step_seconds = rep["rows"][0].get("avg_step_seconds")

        row = {
            "seed": int(seed),
            "after_probe_adapter": adapter_row,
            "zero_step_baseline": baseline_row,
            "phase_correction_gain_vs_zero_adapter": rep.get("phase_correction_gain_vs_zero_adapter"),
            "zero_step_pesq_drop_vs_baseline": rep.get("zero_step_pesq_drop_vs_baseline"),
            "avg_step_seconds": avg_step_seconds,
        }
        rows.append(row)

    pesq_vals = [r["after_probe_adapter"]["pesq"] for r in rows]
    estoi_vals = [r["after_probe_adapter"]["estoi"] for r in rows]
    sdr_vals = [r["after_probe_adapter"]["sdr"] for r in rows]
    prmse_vals = [r["after_probe_adapter"]["phase_rmse"] for r in rows]
    step_time_vals = [r["avg_step_seconds"] for r in rows]

    summary = {
        "spec": {
            "seeds": SEEDS,
            "rounds": len(SEEDS),
            "train_steps": int(base["train_steps"]),
            "eval_examples": int(base["eval_examples"]),
            "model": "v3.6.1 temporal_adalnzero only",
        },
        "rows": rows,
        "aggregate": {
            "after_probe_adapter_mean": {
                "pesq": mean(pesq_vals),
                "estoi": mean(estoi_vals),
                "sdr": mean(sdr_vals),
                "phase_rmse": mean(prmse_vals),
            },
            "avg_step_seconds_mean": mean(step_time_vals),
        },
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
