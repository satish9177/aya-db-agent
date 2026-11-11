"""Durable allocation/peak snapshots, including the OOM path; no GPU work on import."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class GpuMemoryRecorder:
    def __init__(self, path: Path, *, phase: str, cuda=None):
        if cuda is None:
            import torch
            cuda = torch.cuda
        self.path, self.phase, self.cuda = Path(path), phase, cuda

    def reset(self):
        self.cuda.reset_peak_memory_stats()

    def record(self, event: str, *, step=None, **details):
        # No synchronize/empty_cache here: preserve the actual allocator peak and OOM evidence.
        row = {"time_utc": datetime.now(timezone.utc).isoformat(), "phase": self.phase,
               "event": event, "optimizer_step": step, **details}
        try:
            row.update(allocated_bytes=self.cuda.memory_allocated(), reserved_bytes=self.cuda.memory_reserved(),
                       peak_allocated_bytes=self.cuda.max_memory_allocated(),
                       peak_reserved_bytes=self.cuda.max_memory_reserved())
            free, total = self.cuda.mem_get_info()
            row.update(device_free_bytes=free, device_total_bytes=total)
        except Exception as error:
            row["measurement_error"] = type(error).__name__
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
        print(json.dumps({"gpu_memory": row}), flush=True)
        if row.get("measurement_error") and event != "out_of_memory":
            raise RuntimeError("GPU memory measurement failed; see durable event")
        return row


@contextmanager
def record_oom(recorder):
    import torch
    try:
        yield
    except torch.OutOfMemoryError:
        try:
            recorder.record("out_of_memory")
        except Exception as error:
            print(json.dumps({"oom_memory_record_error": type(error).__name__}), flush=True)
        raise  # Never swallow the OOM, empty the cache, truncate or retry.
