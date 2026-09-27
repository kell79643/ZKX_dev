from __future__ import annotations

import threading


class AllocationPeakHook:
    """CuPy allocation-event peak tracker that cannot miss short allocations."""

    def __init__(self, cp, baseline_bytes: int = 0) -> None:
        self._hook = self._build_hook(cp)
        self._lock = threading.Lock()
        self._active: dict[int, int] = {}
        self.baseline_bytes = int(baseline_bytes)
        self.current_bytes = int(baseline_bytes)
        self.peak_bytes = int(baseline_bytes)

    def _build_hook(self, cp):
        owner = self

        class Hook(cp.cuda.MemoryHook):
            def malloc_postprocess(self, **kwargs) -> None:
                size = int(kwargs["size"])
                allocation_id = int(kwargs["pmem_id"])
                with owner._lock:
                    previous = owner._active.get(allocation_id, 0)
                    owner._active[allocation_id] = size
                    owner.current_bytes += size - previous
                    owner.peak_bytes = max(owner.peak_bytes, owner.current_bytes)

            def free_postprocess(self, **kwargs) -> None:
                allocation_id = int(kwargs["pmem_id"])
                with owner._lock:
                    size = owner._active.pop(allocation_id, 0)
                    owner.current_bytes -= size

        return Hook()

    def __enter__(self):
        self._hook.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._hook.__exit__(exc_type, exc_value, traceback)

