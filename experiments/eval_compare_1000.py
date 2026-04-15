import argparse
import json
import os
import sqlite3
from datetime import datetime

from addse.app.eval import eval as run_eval


def summarize(db_path: str) -> dict[str, float]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    avg = dict(cur.execute("SELECT metric, AVG(value) FROM results GROUP BY metric").fetchall())
    conn.close()
    if rows == 0:
        raise RuntimeError(f"No rows in {db_path}")
    return {"rows": float(rows), **{k: float(v) for k, v in avg.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/addse-s-mydata-eval-3metrics.yaml")
    parser.add_argument("--baseline-ckpt", default="logs/addse-edbase-quick/checkpoints/addse-s.ckpt")
    parser.add_argument("--finetune-ckpt", required=True)
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--num-consumers", type=int, default=4)
    parser.add_argument("--baseline-db", default="eval_baseline_1000_3m.db")
    parser.add_argument("--finetune-db", default="eval_finetune_1000_3m.db")
    parser.add_argument("--summary-json", default="eval_compare_1000_summary.json")
    parser.add_argument("--force-reset", action="store_true")
    args = parser.parse_args()

    if args.force_reset:
        for p in [args.baseline_db, args.finetune_db]:
            if os.path.exists(p):
                os.remove(p)

    summary: dict[str, dict[str, float] | str] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    def has_results(db_path: str) -> bool:
        if not os.path.exists(db_path):
            return False
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
        ok = cur.fetchone() is not None
        complete = False
        if ok:
            per_metric = cur.execute(
                "SELECT metric, COUNT(*) FROM results GROUP BY metric"
            ).fetchall()
            if per_metric:
                complete = all(int(count) >= args.num_examples for _, count in per_metric)
        conn.close()
        return complete

    # Baseline first (same config/dataset setup)
    if has_results(args.baseline_db):
        print(f"[skip] baseline DB already has results: {args.baseline_db}")
    else:
        run_eval(
            args.config,
            args.baseline_ckpt,
            overrides=["+name=baseline-1000"],
            output_db=args.baseline_db,
            num_examples=args.num_examples,
            overwrite=True,
            num_consumers=args.num_consumers,
        )

    b = summarize(args.baseline_db)
    summary["baseline"] = b
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=== BASELINE (saved) ===")
    print(b)

    # Finetuned model second (same config/dataset setup)
    if has_results(args.finetune_db):
        print(f"[skip] finetune DB already has results: {args.finetune_db}")
    else:
        run_eval(
            args.config,
            args.finetune_ckpt,
            overrides=["+name=finetune-1000"],
            output_db=args.finetune_db,
            num_examples=args.num_examples,
            overwrite=True,
            num_consumers=args.num_consumers,
        )

    f = summarize(args.finetune_db)
    summary["finetune"] = f

    delta = {}
    for k in ["pesq", "estoi", "sdr"]:
        delta[k] = f.get(k, float("nan")) - b.get(k, float("nan"))
    summary["delta"] = delta
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    with open(args.summary_json, "w", encoding="utf-8") as fjson:
        json.dump(summary, fjson, ensure_ascii=False, indent=2)

    print("=== FINETUNE (saved) ===")
    print(f)
    print("=== DELTA (finetune - baseline) ===")
    for k in ["pesq", "estoi", "sdr"]:
        print(f"{k}: {delta[k]:+.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
