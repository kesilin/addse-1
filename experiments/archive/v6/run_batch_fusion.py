"""Batch runner for V6 closed-loop fusion smoke.

Usage (example):
  python run_batch_fusion.py --n_examples 3 --steps 20 --out reports/v6_closed_loop_batch_verify.json
"""
from pathlib import Path
import json
import argparse
import importlib.util
import sys
from dataclasses import asdict

HERE = Path(__file__).resolve().parent

def find_project_root(start: Path) -> Path:
    # Walk upwards to find a directory containing TIMIT_all_wavs
    for p in [start] + list(start.parents):
        if (p / "TIMIT_all_wavs").exists():
            return p
    # fallback
    return start.parents[4]

PROJECT_ROOT = find_project_root(HERE)

def load_v6_module():
    path = HERE / "v6_closed_loop_smoke.py"
    spec = importlib.util.spec_from_file_location("v6_closed_loop_smoke", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v6_closed_loop_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


def list_audio_files(root: Path):
    audio_dir = root / "TIMIT_all_wavs"
    files = sorted([p for p in audio_dir.iterdir() if p.suffix.lower() in (".wav", ".flac", ".mp3")])
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--fusion_lr", type=float, default=5e-5)
    parser.add_argument("--train_module2_heads", action="store_true", help="Enable training of Module2 heads")
    parser.add_argument("--module2_lr", type=float, default=1e-5)
    parser.add_argument("--save_weights", action="store_true", help="Save trained weights for each run")
    parser.add_argument("--weights_dir", type=str, default="addse/experiments/archive/v6/weights")
    parser.add_argument("--out", type=str, default="addse/experiments/archive/v6/reports/v6_closed_loop_batch_verify.json")
    args = parser.parse_args()

    mod = load_v6_module()
    files = list_audio_files(PROJECT_ROOT)
    if len(files) == 0:
        raise FileNotFoundError(f"No audio files found under {PROJECT_ROOT / 'TIMIT_all_wavs'}")

    chosen = files[: args.n_examples]
    reports = []
    for i, p in enumerate(chosen):
        cfg = mod.V6ClosedLoopSmokeConfig()
        # override
        cfg.audio_path = str(Path("TIMIT_all_wavs") / p.name)
        cfg.fusion_train_steps = int(args.steps)
        cfg.fusion_lr = float(args.fusion_lr)
        cfg.train_module2_heads = bool(args.train_module2_heads)
        cfg.module2_lr = float(args.module2_lr)
        cfg.save_weights = bool(args.save_weights)
        cfg.weights_dir = str(args.weights_dir)
        cfg.seed = int(args.base_seed + i)
        # per-run report path
        per_report = Path("addse/experiments/archive/v6/reports") / f"v6_closed_loop_{p.stem}_s{cfg.seed}_s{cfg.fusion_train_steps}.json"
        cfg.report_json = str(per_report)
        print(f"Running: file={p.name}, seed={cfg.seed}, steps={cfg.fusion_train_steps}")
        try:
            report = mod._run_closed_loop(cfg)
            report["audio_file"] = str(p)
            reports.append(report)
        except Exception as e:
            # record failure info and continue
            err_report = {
                "config": asdict(cfg),
                "audio_file": str(p),
                "error": repr(e),
            }
            reports.append(err_report)
            print(f"Run failed for {p.name}: {e}")

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"runs": reports}, f, ensure_ascii=False, indent=2)
    print(f"Saved batch report: {out_path}")


if __name__ == "__main__":
    main()
