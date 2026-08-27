"""Launch one bounded real verl GRPO update for the Learn Track."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    default_model = os.environ.get("ETRL_MODEL")
    default_verl_config = os.environ.get("VERL_CONFIG_PATH")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(default_model) if default_model else None,
        required=default_model is None,
        help="Local Qwen-compatible checkpoint, normally Qwen3-1.7B.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ["ETRL_DATA_DIR"])
        if os.environ.get("ETRL_DATA_DIR")
        else None,
        required=os.environ.get("ETRL_DATA_DIR") is None,
        help="Directory containing the 500-row verl parquet files.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="New empty directory for this smoke run; existing non-empty runs are refused.",
    )
    parser.add_argument(
        "--verl-config-path",
        type=Path,
        default=Path(default_verl_config) if default_verl_config else None,
        help="Directory containing the installed verl trainer configs.",
    )
    parser.add_argument("--run-name", default="learn_grpo_smoke")
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and environment without launching Ray.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("run-name may contain only letters, numbers, '.', '_' and '-'")
    if args.n_gpus < 1:
        raise ValueError("n-gpus must be positive")
    if args.verl_config_path is None:
        raise ValueError("--verl-config-path or VERL_CONFIG_PATH is required")

    model = args.model.resolve()
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    verl_config_path = args.verl_config_path.resolve()
    if not model.exists():
        raise FileNotFoundError(f"model checkpoint does not exist: {model}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    required_data = (
        data_dir / "verl_hotpotqa_train_500.parquet",
        data_dir / "verl_hotpotqa_val_100.parquet",
    )
    missing = [str(path) for path in required_data if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Learn Track data files: " + ", ".join(missing))
    if not verl_config_path.is_dir():
        raise FileNotFoundError(f"verl config directory does not exist: {verl_config_path}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run directory: {run_dir}")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_ppo_m3.py"),
        "--config-name",
        "learn_grpo_smoke",
        f"trainer.experiment_name={args.run_name}",
        f"trainer.default_local_dir={run_dir}",
        f"trainer.rollout_data_dir={run_dir / 'rollouts'}",
        f"trainer.validation_data_dir={run_dir / 'validation'}",
        f"trainer.n_gpus_per_node={args.n_gpus}",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "ETRL_ROOT": str(ROOT),
            "ETRL_MODEL": str(model),
            "ETRL_DATA_DIR": str(data_dir),
            "ETRL_RUN_DIR": str(run_dir.parent),
            "VERL_CONFIG_PATH": str(verl_config_path),
        }
    )
    python_bin = str(Path(sys.executable).resolve().parent)
    environment["PATH"] = os.pathsep.join(
        part for part in (python_bin, environment.get("PATH", "")) if part
    )
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH", "")) if part
    )

    print("This is a real one-update verl GRPO run: 8 prompts × 4 rollouts.")
    print("It reuses the project's task-only reward and may use one GPU.")
    print("Resolved command:")
    print(" ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        print("Dry run: Ray was not started and no output directory was created.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
