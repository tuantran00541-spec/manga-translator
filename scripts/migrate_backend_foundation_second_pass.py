from pathlib import Path

# Foundation B01 recovery fixture now has to publish the exact transaction
# identity, because B02 deliberately rejects revision-only proof.
path = Path("scripts/backend_foundation_sanity.py")
text = path.read_text(encoding="utf-8")
old = "n=r/'n.tmp'; n.write_bytes(b'NEW'); mu.atomic_replace(n,live); (r/'manifest.json').write_bytes(mb(1)); check(mu.recover_page_artifact_transactions(r)==1,'commit recovery count'); check(live.read_bytes()==b'NEW','committed artifact rolled back')"
new = "n=r/'n.tmp'; n.write_bytes(b'NEW'); mu.atomic_replace(n,live); committed=json.loads((r/'manifest.json').read_text()); committed_page=committed['pages'][0]; committed_page['clean_revision']=1; tx.mark_manifest_commit(committed_page); (r/'manifest.json').write_text(json.dumps(committed),encoding='utf-8'); check(mu.recover_page_artifact_transactions(r)==1,'commit recovery count'); check(live.read_bytes()==b'NEW','committed artifact rolled back')"
count = text.count(old)
if count != 1:
    raise SystemExit(f"foundation transaction fixture changed; expected one match, found {count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

# The stale-outcome regression exercises ChapterPipeline only. Stub the
# downloader registry before importing app.pipeline so this deterministic test
# does not pull Playwright/browser dependencies that are unrelated to B03.
path = Path("scripts/backend_second_pass_sanity.py")
text = path.read_text(encoding="utf-8")
old = '''def stale_outcome_checks() -> None:
    import app.manifest_utils as mu
    import app.pipeline as pipeline_module
'''
new = '''def stale_outcome_checks() -> None:
    import types
    registry = types.ModuleType("app.downloader.registry")
    registry.download_chapter = lambda *args, **kwargs: []
    sys.modules.setdefault("app.downloader.registry", registry)

    import app.manifest_utils as mu
    import app.pipeline as pipeline_module
'''
count = text.count(old)
if count != 1:
    raise SystemExit(f"second-pass pipeline import fixture changed; expected one match, found {count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

print("foundation identity fixture and second-pass import isolation aligned")
