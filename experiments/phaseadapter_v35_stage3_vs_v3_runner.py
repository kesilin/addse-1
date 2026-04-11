import json
import os
import statistics
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(r"f:/ksl/addse")
PY = Path(r"f:/ksl/.venv312/Scripts/python.exe")

V35_BASE_CFG = ROOT / "configs/phaseadapter_v35_stage2_crossattn_seed42.yaml"
V3_BASE_CFG = ROOT / "configs/phase_fusion_layered_compare_500.yaml"
OUT_CFG_DIR = ROOT / "configs/phaseadapter_stage3_generated"
OUT_REPORT_DIR = ROOT / "experiments/phaseadapter_v35_probe/reports"


def run_cmd(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "experiments") + (
        (";" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def make_v35_cfg(base: dict, seed: int, train_steps: int, eval_examples: int, train_batch_size: int) -> Path:
    c = deepcopy(base)
    c["seed"] = int(seed)
    c["train_steps"] = int(train_steps)
    c["eval_examples"] = int(eval_examples)
    c["train_batch_size"] = int(train_batch_size)
    c["inject_mode"] = "cross_attn"
    c["adapter_scale"] = 0.05
    c["adapter_weight"] = f"experiments/phaseadapter_v35_probe/weights/phase_adapter_stage3_crossattn_seed{seed}.pt"
    c["report_json"] = f"experiments/phaseadapter_v35_probe/reports/probe_report_stage3_crossattn_seed{seed}.json"

    path = OUT_CFG_DIR / f"v35_stage3_seed{seed}.yaml"
    dump_yaml(path, c)
    return path


def make_v3_cfg(base: dict, seed: int, train_steps: int, eval_examples: int, train_batch_size: int) -> Path:
    c = deepcopy(base)
    c["seed"] = int(seed)
    c["segment_length"] = 1.0
    c["train_steps"] = int(train_steps)
    c["eval_examples"] = int(eval_examples)
    c["train_batch_size"] = int(train_batch_size)
    c["report_json"] = f"experiments/phaseadapter_v35_probe/reports/probe_report_stage3_v3_seed{seed}.json"
    c["adapter_ckpt"] = f"experiments/phaseadapter_v35_probe/weights/phase_adapter_stage3_v3_seed{seed}.pt"

    path = OUT_CFG_DIR / f"v3_stage3_seed{seed}.yaml"
    dump_yaml(path, c)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    seeds = [42, 43, 44, 45, 46, 47]
    train_steps = 200
    eval_examples = 300
    train_batch_size = 2

    base_v35 = load_yaml(V35_BASE_CFG)
    base_v3 = load_yaml(V3_BASE_CFG)

    rows = []
    for seed in seeds:
        v35_cfg = make_v35_cfg(base_v35, seed, train_steps, eval_examples, train_batch_size)
        v3_cfg = make_v3_cfg(base_v3, seed, train_steps, eval_examples, train_batch_size)

        run_cmd([str(PY), "experiments/phaseadapter_v35_probe.py", "--config", str(v35_cfg.relative_to(ROOT)).replace("/", "\\")])
        run_cmd([str(PY), "experiments/archive/fusion/phase_fusion_layered_compare_500.py", "--config", str(v3_cfg.relative_to(ROOT)).replace("/", "\\"), "--mode", "train_eval"])

        v35_report = load_json(ROOT / f"experiments/phaseadapter_v35_probe/reports/probe_report_stage3_crossattn_seed{seed}.json")
        v3_report = load_json(ROOT / f"experiments/phaseadapter_v35_probe/reports/probe_report_stage3_v3_seed{seed}.json")

        row = {
            "seed": seed,
            "v35_delta_vs_layered": {
                "pesq": v35_report["delta_adapter_minus_baseline"]["pesq"],
                "estoi": v35_report["delta_adapter_minus_baseline"]["estoi"],
                "sdr": v35_report["delta_adapter_minus_baseline"]["sdr"],
            },
            "v3_delta_vs_layered": {
                "pesq": v3_report["delta_layered_fused_minus_layered_only"]["pesq"],
                "estoi": v3_report["delta_layered_fused_minus_layered_only"]["estoi"],
                "sdr": v3_report["delta_layered_fused_minus_layered_only"]["sdr"],
            },
            "v35_abs": {
                "pesq": v35_report["eval_adapter"]["pesq"],
                "estoi": v35_report["eval_adapter"]["estoi"],
                "sdr": v35_report["eval_adapter"]["sdr"],
            },
            "v3_abs": {
                "pesq": v3_report["layered_fused_v3"]["pesq"],
                "estoi": v3_report["layered_fused_v3"]["estoi"],
                "sdr": v3_report["layered_fused_v3"]["sdr"],
            },
        }
        rows.append(row)

    v35_p = [r["v35_delta_vs_layered"]["pesq"] for r in rows]
    v35_e = [r["v35_delta_vs_layered"]["estoi"] for r in rows]
    v35_s = [r["v35_delta_vs_layered"]["sdr"] for r in rows]

    v3_p = [r["v3_delta_vs_layered"]["pesq"] for r in rows]
    v3_e = [r["v3_delta_vs_layered"]["estoi"] for r in rows]
    v3_s = [r["v3_delta_vs_layered"]["sdr"] for r in rows]

    summary = {
        "spec": {
            "seeds": seeds,
            "rounds": len(seeds),
            "train_steps": train_steps,
            "eval_examples": eval_examples,
            "train_batch_size": train_batch_size,
            "target": "strict V35 vs V3 matched-budget contrast",
        },
        "rows": rows,
        "aggregate": {
            "v35_delta_vs_layered": {
                "pesq": mean_std(v35_p),
                "estoi": mean_std(v35_e),
                "sdr": mean_std(v35_s),
            },
            "v3_delta_vs_layered": {
                "pesq": mean_std(v3_p),
                "estoi": mean_std(v3_e),
                "sdr": mean_std(v3_s),
            },
            "v35_minus_v3_delta": {
                "pesq": mean_std([a - b for a, b in zip(v35_p, v3_p)]),
                "estoi": mean_std([a - b for a, b in zip(v35_e, v3_e)]),
                "sdr": mean_std([a - b for a, b in zip(v35_s, v3_s)]),
            },
        },
    }

    out = OUT_REPORT_DIR / "stage3_v35_vs_v3_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== STAGE3 V35 VS V3 SUMMARY ===")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
