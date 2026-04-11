import json
import statistics
import subprocess
from pathlib import Path


def run_one(config_path: str) -> None:
    cmd = [
        r"f:\ksl\.venv312\Scripts\python.exe",
        r"experiments\phaseadapter_v35_probe.py",
        "--config",
        config_path,
    ]
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=r"f:\ksl\addse", capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise SystemExit(proc.returncode)


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    de = d["delta_adapter_minus_baseline"]
    return {
        "report": path,
        "seed": d["meta"]["config"]["seed"],
        "mode": d["meta"]["inject_mode"],
        "delta_pesq": de["pesq"],
        "delta_estoi": de["estoi"],
        "delta_sdr": de["sdr"],
        "train_last_loss": d["train_log"][-1]["loss"],
        "eval_examples": d["eval_adapter"]["eval_examples"],
    }


def summarize_cross(rows: list[dict]) -> dict:
    return {
        "mode": "cross_attn",
        "runs": len(rows),
        "mean_delta_pesq": statistics.mean(r["delta_pesq"] for r in rows),
        "mean_delta_estoi": statistics.mean(r["delta_estoi"] for r in rows),
        "mean_delta_sdr": statistics.mean(r["delta_sdr"] for r in rows),
        "std_delta_pesq": statistics.pstdev(r["delta_pesq"] for r in rows),
        "std_delta_estoi": statistics.pstdev(r["delta_estoi"] for r in rows),
        "std_delta_sdr": statistics.pstdev(r["delta_sdr"] for r in rows),
    }


def main() -> int:
    jobs = [
        (
            r"configs\phaseadapter_v35_stage2_crossattn_seed42.yaml",
            r"experiments\phaseadapter_v35_probe\reports\probe_report_stage2_crossattn_seed42.json",
        ),
        (
            r"configs\phaseadapter_v35_stage2_crossattn_seed43.yaml",
            r"experiments\phaseadapter_v35_probe\reports\probe_report_stage2_crossattn_seed43.json",
        ),
        (
            r"configs\phaseadapter_v35_stage2_crossattn_seed44.yaml",
            r"experiments\phaseadapter_v35_probe\reports\probe_report_stage2_crossattn_seed44.json",
        ),
        (
            r"configs\phaseadapter_v35_stage2_adaln_fix_seed42.yaml",
            r"experiments\phaseadapter_v35_probe\reports\probe_report_stage2_adaln_fix_seed42.json",
        ),
    ]

    for cfg, _ in jobs:
        run_one(cfg)

    rows = [load_report(report) for _, report in jobs]
    cross_rows = [r for r in rows if r["mode"] == "cross_attn"]
    cross_summary = summarize_cross(cross_rows)

    final = {
        "rows": rows,
        "cross_attn_summary": cross_summary,
    }

    print("=== STAGE2 SUMMARY ===")
    print(json.dumps(final, ensure_ascii=False, indent=2))

    out_path = Path(r"f:\ksl\addse\experiments\phaseadapter_v35_probe\reports\stage2_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
