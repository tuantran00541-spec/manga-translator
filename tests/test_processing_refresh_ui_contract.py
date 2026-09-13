from pathlib import Path


def test_main_ui_uses_server_owned_processing_job_contract():
    source = Path("app/static/js/main.js").read_text(encoding="utf-8")

    assert 'processingJson("/api/process/chapter"' in source
    assert "/api/process/jobs/" in source
    assert "reconnectPageProcessingJob" in source
    assert "resumeChapterWithProcessingReconnect" in source

    # Regression guard for the F5 bug: main.js must never own a loop that
    # submits successive /api/process_pages batches. A browser refresh would
    # destroy that loop and silently truncate the remaining chapter queue.
    assert 'fetch("/api/process_pages"' not in source
    assert "RESPONSIVE_PROCESS_BATCH_SIZE" not in source
