import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


def _state_dict_from_ckpt(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _mean_abs_model_delta(init_ckpt: str, best_ckpt: str) -> tuple[float, int]:
    init_sd = _state_dict_from_ckpt(init_ckpt)
    best_sd = _state_dict_from_ckpt(best_ckpt)
    keys = [k for k in init_sd.keys() if k.startswith("model.") and k in best_sd and init_sd[k].shape == best_sd[k].shape]
    if not keys:
        return 0.0, 0
    acc = 0.0
    for k in keys:
        acc += (best_sd[k].float() - init_sd[k].float()).abs().mean().item()
    return acc / len(keys), len(keys)


def _pick_best_ckpt(ckpt_dir: Path) -> str:
    cands = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
    if not cands:
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return str(last)
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    # filename pattern contains val_pesq due to config checkpoint_filename
    def score(p: Path) -> float:
        stem = p.stem
        if "-" not in stem:
            return float("-inf")
        tail = stem.split("-")[-1]
        try:
            return float(tail)
        except Exception:
            return float("-inf")

    best = max(cands, key=score)
    return str(best)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/addse-s-mydata-layered-ft-pesq20.yaml")
    parser.add_argument("--init-ckpt", default="logs/addse-edbase-quick/checkpoints/addse-s.ckpt")
    parser.add_argument("--run-name", default="ft-pesq20-3layer")
    args = parser.parse_args()

    if not os.path.exists(args.init_ckpt):
        raise FileNotFoundError(args.init_ckpt)

    train_cmd = [
        str(Path(sys.executable).with_name("addse.exe")),
        "train",
        args.config,
        f"name={args.run_name}",
        "--overwrite",
        "--init-ckpt",
        args.init_ckpt,
    ]
    subprocess.run(train_cmd, check=True)

    ckpt_dir = Path("logs") / args.run_name / "checkpoints"
    best_ckpt = _pick_best_ckpt(ckpt_dir)

    delta, n_keys = _mean_abs_model_delta(args.init_ckpt, best_ckpt)
    print(f"BEST_CKPT={best_ckpt}")
    print(f"MODEL_PARAM_DELTA_MEAN_ABS={delta:.8f}")
    print(f"MODEL_PARAM_DELTA_KEYS={n_keys}")

    if n_keys == 0 or delta < 1e-7:
        print("ERROR: finetune appears ineffective (no meaningful model delta)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
