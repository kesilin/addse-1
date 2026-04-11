import json
import statistics
from pathlib import Path


ROOT = Path(r"f:/ksl/addse")
REPORT_DIR = ROOT / "experiments/phaseadapter_v35_probe/reports"


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    seeds = [42, 43, 44, 45, 46, 47]
    rows = []

    for seed in seeds:
        v35_path = REPORT_DIR / f"probe_report_stage3_crossattn_seed{seed}.json"
        v3_path = REPORT_DIR / f"probe_report_stage3_v3_seed{seed}.json"
        if not v35_path.exists() or not v3_path.exists():
            print(f"Missing report for seed={seed}: v35_exists={v35_path.exists()} v3_exists={v3_path.exists()}")
            return 2

        v35_report = load_json(v35_path)
        v3_report = load_json(v3_path)
        rows.append(
            {
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
        )

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
            "train_steps": 200,
            "eval_examples": 300,
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

    out_path = REPORT_DIR / "stage3_v35_vs_v3_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
