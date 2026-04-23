import importlib.util
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def main() -> int:
    module_path = EXPERIMENTS_DIR / "v6_closed_loop_smoke.py"
    spec = importlib.util.spec_from_file_location("v6_closed_loop_smoke", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load smoke module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
