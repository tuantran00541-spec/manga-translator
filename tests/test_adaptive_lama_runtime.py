from __future__ import annotations

from app.inpaint.lama_runtime_policy import select_lama_runtime_profile


GIB = 1024 ** 3


def test_single_worker_four_cpu_uses_four_lama_threads_and_arena(monkeypatch):
    monkeypatch.delenv("MANGA_LAMA_INTRA_OP_THREADS", raising=False)
    monkeypatch.delenv("MANGA_LAMA_CPU_MEM_ARENA", raising=False)
    monkeypatch.delenv("MANGA_LAMA_MEM_PATTERN", raising=False)
    profile = select_lama_runtime_profile(
        1,
        cpu_count=4,
        memory_limit_bytes=8 * GIB,
        memory_available_bytes=4 * GIB,
    )
    assert profile.intra_op_threads == 4
    assert profile.dynamic_block_base == 4
    assert profile.enable_cpu_mem_arena is True
    assert profile.enable_mem_pattern is True
    assert profile.arena_reason == "memory_gate_pass"


def test_two_page_workers_four_cpu_uses_two_lama_threads(monkeypatch):
    monkeypatch.delenv("MANGA_LAMA_INTRA_OP_THREADS", raising=False)
    profile = select_lama_runtime_profile(
        2,
        cpu_count=4,
        memory_limit_bytes=8 * GIB,
        memory_available_bytes=4 * GIB,
    )
    assert profile.intra_op_threads == 2
    assert profile.dynamic_block_base == 4


def test_four_gib_cgroup_disables_arena(monkeypatch):
    monkeypatch.delenv("MANGA_LAMA_CPU_MEM_ARENA", raising=False)
    monkeypatch.delenv("MANGA_LAMA_MEM_PATTERN", raising=False)
    profile = select_lama_runtime_profile(
        1,
        cpu_count=4,
        memory_limit_bytes=4 * GIB,
        memory_available_bytes=3 * GIB,
    )
    assert profile.enable_cpu_mem_arena is False
    assert profile.enable_mem_pattern is False
    assert profile.arena_reason == "memory_limit_below_gate"


def test_low_memory_headroom_disables_arena(monkeypatch):
    monkeypatch.delenv("MANGA_LAMA_CPU_MEM_ARENA", raising=False)
    profile = select_lama_runtime_profile(
        1,
        cpu_count=4,
        memory_limit_bytes=16 * GIB,
        memory_available_bytes=1 * GIB,
    )
    assert profile.enable_cpu_mem_arena is False
    assert profile.arena_reason == "memory_headroom_below_gate"


def test_runtime_env_overrides_are_respected(monkeypatch):
    monkeypatch.setenv("MANGA_LAMA_INTRA_OP_THREADS", "3")
    monkeypatch.setenv("MANGA_LAMA_DYNAMIC_BLOCK_BASE", "6")
    monkeypatch.setenv("MANGA_LAMA_CPU_MEM_ARENA", "1")
    monkeypatch.setenv("MANGA_LAMA_MEM_PATTERN", "0")
    profile = select_lama_runtime_profile(
        2,
        cpu_count=4,
        memory_limit_bytes=4 * GIB,
        memory_available_bytes=1 * GIB,
    )
    assert profile.intra_op_threads == 3
    assert profile.dynamic_block_base == 6
    assert profile.enable_cpu_mem_arena is True
    assert profile.enable_mem_pattern is False
    assert profile.arena_reason == "env_override"
