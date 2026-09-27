from __future__ import annotations

import argparse
import gc
import threading

from task1_common import Task1Config, cp, require_cusignal_runtime
from task1_runner import run_task1_until
from memory_metrics import AllocationPeakHook


def _device_used_bytes() -> int:
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return int(total_bytes - free_bytes)


def _run_peak_monitor(stop: threading.Event, samples: list[int], errors: list[str]) -> None:
    try:
        cp.cuda.Device().use()
        while not stop.wait(0.001):
            samples.append(_device_used_bytes())
        samples.append(_device_used_bytes())
    except Exception as error:  # pragma: no cover - ZQ500 runtime path
        errors.append(str(error))


def main() -> int:
    parser = argparse.ArgumentParser(description="Task1 Python memory stability check")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-pool-growth-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    require_cusignal_runtime()
    config = Task1Config.load(args.config)
    pool = cp.get_default_memory_pool()
    for index in range(args.warmup):
        run_task1_until(5, f"memory_warmup_{index + 1}", config, args.output_dir, write_json=False, quiet=True)
    gc.collect()
    cp.cuda.get_current_stream().synchronize()
    allocator_baseline = int(pool.used_bytes())
    allocation_hook = AllocationPeakHook(cp, allocator_baseline)
    baseline_used = _device_used_bytes()
    device_samples = [baseline_used]
    monitor_errors: list[str] = []
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=_run_peak_monitor,
        args=(stop_monitor, device_samples, monitor_errors),
        daemon=True,
    )
    monitor.start()
    baseline = int(pool.total_bytes())
    total_samples: list[int] = []
    used_samples: list[int] = []
    with allocation_hook:
        for index in range(args.iterations):
            run_task1_until(5, f"memory_{index + 1}", config, args.output_dir, write_json=False, quiet=True)
            gc.collect()
            cp.cuda.get_current_stream().synchronize()
            total_samples.append(int(pool.total_bytes()))
            used_samples.append(int(pool.used_bytes()))
    cp.cuda.get_current_stream().synchronize()
    device_samples.append(_device_used_bytes())
    stop_monitor.set()
    monitor.join()
    if monitor_errors:
        raise RuntimeError(f"device peak memory monitor failed: {monitor_errors[0]}")
    growth = max(total_samples, default=baseline) - baseline
    passed = growth <= args.max_pool_growth_bytes and max(used_samples, default=0) == 0
    print(
        f"[TASK1_PYTHON][MEMORY] iterations={args.iterations} baseline_bytes={baseline} "
        f"max_pool_bytes={max(total_samples, default=baseline)} "
        f"max_live_bytes={max(used_samples, default=0)} growth_bytes={growth} "
        f"growth_limit_bytes={args.max_pool_growth_bytes} status={'pass' if passed else 'fail'}"
    )
    peak_used = max(device_samples)
    print(
        f"[TASK][MEMORY] task=Task1 implementation=cusignal_python_gpu "
        f"iterations={args.iterations} baseline_used_bytes={baseline_used} "
        f"peak_used_bytes={peak_used} delta_peak_bytes={max(0, peak_used - baseline_used)} "
        f"allocator_baseline_bytes={allocation_hook.baseline_bytes} "
        f"allocator_peak_live_bytes={allocation_hook.peak_bytes} "
        f"allocator_delta_peak_bytes={allocation_hook.peak_bytes - allocation_hook.baseline_bytes} "
        f"allocator_final_bytes={allocation_hook.current_bytes} "
        f"sampling_interval_ms=1 scope=task-allocator-live-peak "
        f"device_scope=device-global-post-warmup-diagnostic allocator_status=pass "
        f"status={'pass' if passed else 'fail'}"
    )
    pool.free_all_blocks()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
