from __future__ import annotations

import argparse

from task2_runner import run_task2_until
from demo.task_config import Task2Config


def main() -> int:
    parser = argparse.ArgumentParser(description="Task2 cuSignal Python pipeline")
    parser.add_argument("--stop-after", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--evidence-id", default="pipeline")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = Task2Config.load(args.config)
    run_task2_until(args.stop_after, args.evidence_id, config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
