from __future__ import annotations

import os
from pathlib import Path

_ORT_THREAD_ENV = "MANGA_ORT_INTRA_OP_THREADS"


def _read_positive_int(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value if value > 0 else None


def _cgroup_cpu_limit() -> int | None:
    """Return the Linux cgroup CPU quota, rounded up to one visible CPU."""
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text(
            encoding="utf-8"
        ).strip().split()
        if len(parts) >= 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if quota > 0 and period > 0:
                return max(1, (quota + period - 1) // period)
    except (OSError, ValueError):
        pass

    quota = _read_positive_int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period = _read_positive_int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota is not None and period is not None:
        return max(1, (quota + period - 1) // period)
    return None


def visible_cpu_count() -> int:
    host = max(1, os.cpu_count() or 2)
    quota = _cgroup_cpu_limit()
    return min(host, quota) if quota is not None else host


def recommended_ort_intra_threads(cpu_count: int | None = None) -> int:
    """Keep model inference fast while reserving CPU for the browser and OS."""
    cpu = max(1, int(cpu_count or visible_cpu_count()))
    if cpu <= 2:
        return 1
    if cpu <= 6:
        return 2
    if cpu <= 9:
        return 3
    return 4


def responsive_process_workers(
    requested: int | None,
    cpu_count: int | None = None,
) -> int:
    """Cap parallel page work so model workers cannot consume every CPU."""
    cpu = max(1, int(cpu_count or visible_cpu_count()))
    try:
        wanted = int(requested) if requested is not None else 2
    except (TypeError, ValueError):
        wanted = 2
    wanted = max(1, wanted)
    if cpu <= 5:
        return 1

    reserve = 2
    model_threads = recommended_ort_intra_threads(cpu)
    compute_budget = max(1, cpu - reserve)
    safe_workers = max(1, compute_budget // model_threads)
    return max(1, min(wanted, 2, safe_workers))


def configure_local_cpu_headroom(cpu_count: int | None = None) -> int:
    """Set a responsive ORT default while preserving a valid explicit override."""
    recommended = recommended_ort_intra_threads(cpu_count)
    raw = os.environ.get(_ORT_THREAD_ENV, "").strip()
    if raw:
        try:
            explicit = int(raw)
        except ValueError:
            explicit = 0
        if explicit > 0:
            return explicit

    os.environ[_ORT_THREAD_ENV] = str(recommended)
    return recommended
