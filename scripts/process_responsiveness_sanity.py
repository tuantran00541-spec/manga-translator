from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime_responsiveness import (  # noqa: E402
    configure_local_cpu_headroom,
    recommended_ort_intra_threads,
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
        "PROCESS_JOB_BATCH_SIZE" not in processing_jobs,
        "legacy fixed-size processing batch still exists",
    )
    check(
        "self._process_plan(" in processing_jobs
        and "list(job.page_indices)" in processing_jobs,
        "backend does not dispatch the complete page queue once",
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
    pipeline_import_at = dependencies.find(
        "from app.processing_pipeline_factory import build_processing_pipeline"
    )
    check(configure_at >= 0, "CPU headroom configuration missing")
    check(
        pipeline_import_at >= 0 and configure_at < pipeline_import_at,
        "CPU headroom must be configured before pipeline factory imports",
    )

    optimized = (ROOT / "app/optimized_pipeline.py").read_text(encoding="utf-8")
    check(
        "requested_workers = max(1, min(8" in optimized
        and "workers=requested_workers" in optimized,
        "optimized pipeline does not preserve the public 1..8 worker contract",
    )

    pipeline = (ROOT / "app/pipeline.py").read_text(encoding="utf-8")
    check(
        "ThreadPoolExecutor(" in pipeline
        and "max_workers=max_workers" in pipeline,
        "pipeline worker pool contract missing",
    )
    check(
        "progress_callback(idx)" in pipeline,
        "full-queue processing does not report per-page progress",
    )


def main() -> None:
    runtime_checks()
    source_checks()
    print("process responsiveness sanity: PASS")


if __name__ == "__main__":
    main()
