import json
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


def summarize(report_path: str) -> dict:
    with open(report_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {
        "report": report_path,
        "mode": d["meta"]["inject_mode"],
        "delta_pesq": d["delta_adapter_minus_baseline"]["pesq"],
        "delta_estoi": d["delta_adapter_minus_baseline"]["estoi"],
        "delta_sdr": d["delta_adapter_minus_baseline"]["sdr"],
        "train_last_loss": d["train_log"][-1]["loss"],
        "eval_examples": d["eval_adapter"]["eval_examples"],
    }


def main() -> int:
    jobs = [
        (
            r"configs\phaseadapter_v35_stage1_adaln.yaml",
            r"experiments\phaseadapter_v35_probe\reports\probe_report_stage1_adaln.json",
        ),
        (
            r"configs\phaseadapter_v35_stage1_crossattn.yaml",
            r"experiments\phaseadapter_v35_probe\reports\probe_report_stage1_crossattn.json",
        ),
        (
            r"configs\phaseadapter_v35_stage1_inputadd.yaml",
            r"experiments\phaseadapter_v35_probe\reports\probe_report_stage1_inputadd.json",
        ),
    ]

    for cfg, _ in jobs:
        run_one(cfg)

    out = [summarize(report) for _, report in jobs]
    print("=== STAGE1 SUMMARY ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    out_path = Path(r"f:\ksl\addse\experiments\phaseadapter_v35_probe\reports\stage1_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
