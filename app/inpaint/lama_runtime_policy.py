from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

from app.runtime_responsiveness import visible_cpu_count

_GIB = 1024 ** 3
_MIB = 1024 ** 2

_LAMA_THREADS_ENV = "MANGA_LAMA_INTRA_OP_THREADS"
_LAMA_BLOCK_BASE_ENV = "MANGA_LAMA_DYNAMIC_BLOCK_BASE"
_LAMA_ARENA_ENV = "MANGA_LAMA_CPU_MEM_ARENA"
_LAMA_MEM_PATTERN_ENV = "MANGA_LAMA_MEM_PATTERN"
_LAMA_ARENA_MIN_LIMIT_MIB_ENV = "MANGA_LAMA_ARENA_MIN_LIMIT_MIB"
_LAMA_ARENA_MIN_AVAILABLE_MIB_ENV = "MANGA_LAMA_ARENA_MIN_AVAILABLE_MIB"

_DEFAULT_ARENA_MIN_LIMIT_BYTES = 6 * _GIB
_DEFAULT_ARENA_MIN_AVAILABLE_BYTES = 2 * _GIB


@dataclass(frozen=True)
class LamaRuntimeProfile:
    page_workers: int
    cpu_count: int
    intra_op_threads: int
    dynamic_block_base: int
    enable_cpu_mem_arena: bool
    enable_mem_pattern: bool
    memory_limit_bytes: int | None
    memory_available_bytes: int | None
    arena_reason: str

    def session_signature(self) -> tuple[int, int, bool, bool]:
        return (
            int(self.intra_op_threads),
            int(self.dynamic_block_base),
            bool(self.enable_cpu_mem_arena),
            bool(self.enable_mem_pattern),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def _optional_env_flag(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    return {
        "1": True,
        "true": True,
        "yes": True,
        "on": True,
        "0": False,
        "false": False,
        "no": False,
        "off": False,
    }.get(raw)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    return max(int(minimum), int(value))


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0 or value >= (1 << 60):
        return None
    return value


def _cgroup_memory_status() -> tuple[int | None, int | None]:
    limit = _read_int(Path("/sys/fs/cgroup/memory.max"))
    current = _read_int(Path("/sys/fs/cgroup/memory.current"))
    if limit is not None:
        return limit, current

    limit = _read_int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    current = _read_int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    return limit, current


def detected_memory_status() -> tuple[int | None, int | None]:
    cgroup_limit, cgroup_current = _cgroup_memory_status()
    host_total = host_available = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        host_total = int(vm.total)
        host_available = int(vm.available)
    except Exception:
        pass

    if cgroup_limit is not None:
        available = None
        if cgroup_current is not None:
            available = max(0, int(cgroup_limit) - int(cgroup_current))
        if host_available is not None:
            available = (
                min(int(available), int(host_available))
                if available is not None
                else int(host_available)
            )
        return int(cgroup_limit), available

    return host_total, host_available


def _adaptive_threads(page_workers: int, cpu_count: int) -> int:
    workers = max(1, int(page_workers))
    cpu = max(1, int(cpu_count))
    explicit = _env_int(_LAMA_THREADS_ENV, 0, 0)
    if explicit > 0:
        return explicit
    if cpu <= 2:
        return 1
    if workers <= 1:
        return min(4, cpu)
    return min(2, max(1, cpu // workers))


def select_lama_runtime_profile(
    page_workers: int,
    *,
    cpu_count: int | None = None,
    memory_limit_bytes: int | None = None,
    memory_available_bytes: int | None = None,
) -> LamaRuntimeProfile:
    workers = max(1, int(page_workers or 1))
    cpu = max(1, int(cpu_count or visible_cpu_count()))
    if memory_limit_bytes is None and memory_available_bytes is None:
        memory_limit_bytes, memory_available_bytes = detected_memory_status()

    threads = _adaptive_threads(workers, cpu)
    explicit_block = _env_int(_LAMA_BLOCK_BASE_ENV, -1, -1)
    block_base = explicit_block if explicit_block >= 0 else (4 if threads >= 2 else 0)

    min_limit = _env_int(
        _LAMA_ARENA_MIN_LIMIT_MIB_ENV,
        _DEFAULT_ARENA_MIN_LIMIT_BYTES // _MIB,
        1,
    ) * _MIB
    min_available = _env_int(
        _LAMA_ARENA_MIN_AVAILABLE_MIB_ENV,
        _DEFAULT_ARENA_MIN_AVAILABLE_BYTES // _MIB,
        1,
    ) * _MIB

    arena_override = _optional_env_flag(_LAMA_ARENA_ENV)
    if arena_override is not None:
        arena = arena_override
        arena_reason = "env_override"
    elif memory_limit_bytes is None or memory_available_bytes is None:
        arena = False
        arena_reason = "memory_unknown"
    elif int(memory_limit_bytes) < int(min_limit):
        arena = False
        arena_reason = "memory_limit_below_gate"
    elif int(memory_available_bytes) < int(min_available):
        arena = False
        arena_reason = "memory_headroom_below_gate"
    else:
        arena = True
        arena_reason = "memory_gate_pass"

    mem_pattern_override = _optional_env_flag(_LAMA_MEM_PATTERN_ENV)
    mem_pattern = arena if mem_pattern_override is None else mem_pattern_override

    return LamaRuntimeProfile(
        page_workers=workers,
        cpu_count=cpu,
        intra_op_threads=max(1, int(threads)),
        dynamic_block_base=max(0, int(block_base)),
        enable_cpu_mem_arena=bool(arena),
        enable_mem_pattern=bool(mem_pattern),
        memory_limit_bytes=(
            int(memory_limit_bytes) if memory_limit_bytes is not None else None
        ),
        memory_available_bytes=(
            int(memory_available_bytes) if memory_available_bytes is not None else None
        ),
        arena_reason=arena_reason,
    )
