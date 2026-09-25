from pathlib import Path


def test_main_ui_uses_server_owned_processing_job_contract():
    source = Path("app/static/js/main.js").read_text(encoding="utf-8")

    assert 'processingJson("/api/process/chapter"' in source
    assert "/api/process/jobs/" in source
    assert "reconnectPageProcessingJob" in source
    assert "resumeChapterWithProcessingReconnect" in source

    assert 'fetch("/api/process_pages"' not in source
    assert "RESPONSIVE_PROCESS_BATCH_SIZE" not in source
