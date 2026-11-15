import subprocess

from app.agent import gitguard, isolate
from app.agent.checkpoint import Checkpoints, MAX_FILE_BYTES


def test_gitguard_restores_worktree_common_config_and_hooks(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '--allow-empty', '-qm', 'initial'], check=True)
    worktree = tmp_path / 'worktree'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '-qb', 'test-branch', str(worktree)], check=True)
    config = repo / '.git' / 'config'
    original = config.read_bytes()
    hook = repo / '.git' / 'hooks' / 'pre-commit'
    before = gitguard.snapshot(worktree)
    config.write_bytes(b'changed')
    hook.write_text('changed')
    touched = gitguard.restore(worktree, before)
    assert config.read_bytes() == original
    assert not hook.exists()
    assert len(touched) == 2


def test_snapshot_rejects_omitted_large_files_and_symlinks(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    large = root / 'large'
    large.write_bytes(b'x' * (isolate.MAX_FILE_BYTES + 1))
    assert isolate.snapshot(root, tmp_path / 'copy') is None
    large.unlink()
    (root / 'link').symlink_to('missing')
    assert isolate.snapshot(root, tmp_path / 'copy2') is None


def test_undo_reports_large_file_not_saved(tmp_path):
    file = tmp_path / 'large'
    file.write_bytes(b'x' * (MAX_FILE_BYTES + 1))
    cp = Checkpoints()
    cp.begin()
    cp.save(file)
    file.write_text('changed')
    assert cp.undo() == []
    assert cp.last_undo_skipped == [str(file)]
