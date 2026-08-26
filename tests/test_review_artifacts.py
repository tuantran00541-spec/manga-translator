import os
import time
from pathlib import Path
from unittest.mock import patch

from app.manifest_utils import cleanup_stale_temp_artifacts
from app.pipeline import ChapterPipeline

ROOT = Path(__file__).resolve().parents[1]


def test_review_workspace_is_lazy_not_all_cards_prebuilt():
    review = (ROOT / 'app/static/js/review.js').read_text(encoding='utf-8')
    workspace = (ROOT / 'app/static/js/review-workspace.js').read_text(encoding='utf-8')
    assert 'window.createReviewCard = createReviewCard' in review
    assert 'window.REVIEW_VIRTUALIZED = true' in workspace
    assert 'container.querySelectorAll(":scope > .review-card")' not in workspace
    assert 'createReviewCard(canonicalIndex' in workspace


def test_output_sync_deletes_stale_unrendered_copy_instead_of_copying_clean(tmp_path: Path):
    clean = tmp_path / 'clean.png'
    clean.write_bytes(b'clean-image')
    out_root = tmp_path / 'output'
    out_dir = out_root / 'abc12345'
    out_dir.mkdir(parents=True)
    stale = out_dir / 'page_000.png'
    stale.write_bytes(b'old-render')
    manifest = {
        'pages': [
            {'original': str(clean), 'clean': str(clean), 'rendered': False},
        ]
    }
    with patch('app.config.OUTPUT_DIR', out_root):
        ChapterPipeline._sync_output_dir('abc12345', manifest, [0])
    assert not stale.exists()
    assert list(out_dir.iterdir()) == []


def test_output_sync_keeps_committed_rendered_output(tmp_path: Path):
    out_root = tmp_path / 'output'
    out_dir = out_root / 'abc12345'
    out_dir.mkdir(parents=True)
    rendered = out_dir / 'page_000.png'
    rendered.write_bytes(b'rendered')
    manifest = {'pages': [{'original': str(tmp_path / 'x.png'), 'rendered': True}]}
    with patch('app.config.OUTPUT_DIR', out_root):
        ChapterPipeline._sync_output_dir('abc12345', manifest, [0])
    assert rendered.read_bytes() == b'rendered'


def test_stale_temp_cleanup_removes_only_old_known_artifacts(tmp_path: Path):
    now = time.time()
    old_names = [
        'manifest.json.deadbeef.tmp',
        'clean_page.png.deadbeef.tmp.png',
        'auto_clean_page.png.deadbeef.tmp.png',
        'manual_mask_page.png.deadbeef.tmp.png',
        'page_000.rendering.deadbeef.tmp',
    ]
    for name in old_names:
        p = tmp_path / name
        p.write_bytes(b'tmp')
        os.utime(p, (now - 7200, now - 7200))

    recent = tmp_path / 'clean_page.png.recent.tmp.png'
    recent.write_bytes(b'recent')
    canonical = tmp_path / 'clean_page.png'
    canonical.write_bytes(b'canonical')
    unrelated = tmp_path / 'notes.tmp'
    unrelated.write_bytes(b'keep')

    removed = cleanup_stale_temp_artifacts(tmp_path, max_age_seconds=3600, now=now)
    assert removed == len(old_names)
    assert all(not (tmp_path / name).exists() for name in old_names)
    assert recent.exists()
    assert canonical.exists()
    assert unrelated.exists()
