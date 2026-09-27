from __future__ import annotations

import argparse

from task1_runner import run_task1_until
from task1_common import Task1Config


def main() -> int:
    parser = argparse.ArgumentParser(description="Task1 cuSignal Python pipeline")
    parser.add_argument("--stop-after", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--evidence-id", default="pipeline")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_task1_until(args.stop_after, args.evidence_id, Task1Config.load(args.config), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
