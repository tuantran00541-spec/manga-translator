from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime_responsiveness import (  # noqa: E402
    configure_local_cpu_headroom,
    recommended_ort_intra_threads,
    responsive_process_workers,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def runtime_checks() -> None:
    expected_threads = {
        1: 1,
        2: 1,
        4: 2,
        6: 2,
        8: 3,
        9: 3,
        10: 4,
        16: 4,
    }
    for cpu, expected in expected_threads.items():
        check(
            recommended_ort_intra_threads(cpu) == expected,
            f"unexpected ORT thread recommendation for {cpu} CPUs",
        )

    expected_workers = {
        2: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 1,
        8: 2,
        10: 2,
        16: 2,
    }
    for cpu, expected in expected_workers.items():
        workers = responsive_process_workers(8, cpu)
        check(workers == expected, f"unexpected page worker cap for {cpu} CPUs")
        reserve = 2 if cpu >= 6 else 1
        used = workers * recommended_ort_intra_threads(cpu)
        check(
            used <= max(1, cpu - reserve),
            f"CPU headroom contract violated for {cpu} CPUs",
        )

    key = "MANGA_ORT_INTRA_OP_THREADS"
    previous = os.environ.get(key)
    try:
        os.environ.pop(key, None)
        check(
            configure_local_cpu_headroom(8) == 3,
            "responsive ORT default not applied",
        )
        check(os.environ.get(key) == "3", "ORT environment default missing")
        os.environ[key] = "1"
        check(
            configure_local_cpu_headroom(16) == 1,
            "explicit ORT override was replaced",
        )
        os.environ[key] = "invalid"
        check(
            configure_local_cpu_headroom(8) == 3,
            "invalid ORT override was not repaired",
        )
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def source_checks() -> None:
    main_js = (ROOT / "app/static/js/main.js").read_text(encoding="utf-8")
    processing_jobs = (ROOT / "app/page_processing_jobs.py").read_text(encoding="utf-8")

    check(
        "PROCESS_JOB_BATCH_SIZE = 16" in processing_jobs,
        "backend processing batch is not fixed at sixteen pages",
    )
    check(
        "job.page_indices[start : start + self._batch_size]" in processing_jobs,
        "backend processing batch slicing contract missing",
    )
    check(
        "window._processSelectedPagesOnce = async function serverOwnedProcessSelectedPagesOnce" in main_js,
        "server-owned page-processing entrypoint missing",
    )
    check(
        'processingJson("/api/process/chapter"' in main_js,
        "frontend does not submit one complete chapter processing job",
    )
    check(
        "page_indices: indices" in main_js,
        "frontend does not transfer the complete page plan to the backend",
    )
    check(
        "workers: getWorkersSetting()" in main_js,
        "page plan must remain independent from worker concurrency",
    )
    check(
        "monitorProcessingJob(snapshot, chapterId)" in main_js,
        "frontend processing monitor is missing",
    )
    check(
        "reconnectPageProcessingJob" in main_js,
        "processing reconnect contract is missing",
    )
    check(
        'btn.setAttribute("aria-busy", "true")' in main_js,
        "processing busy state missing",
    )
    check("AbortController" not in main_js, "frontend must not fake-cancel backend CPU work")

    dependencies = (ROOT / "app/dependencies.py").read_text(encoding="utf-8")
    configure_at = dependencies.find("configure_local_cpu_headroom()")
    pipeline_import_at = dependencies.find("from app.optimized_pipeline")
    check(configure_at >= 0, "CPU headroom configuration missing")
    check(
        pipeline_import_at >= 0 and configure_at < pipeline_import_at,
        "CPU headroom must be configured before pipeline imports",
    )

    optimized = (ROOT / "app/optimized_pipeline.py").read_text(encoding="utf-8")
    check(
        "workers=responsive_process_workers(workers)" in optimized,
        "optimized pipeline does not apply responsive worker cap",
    )


def main() -> None:
    runtime_checks()
    source_checks()
    print("process responsiveness sanity: PASS")


if __name__ == "__main__":
    main()
