from pathlib import Path

path = Path("scripts/backend_foundation_sanity.py")
text = path.read_text(encoding="utf-8")
old = "n=r/'n.tmp'; n.write_bytes(b'NEW'); mu.atomic_replace(n,live); (r/'manifest.json').write_bytes(mb(1)); check(mu.recover_page_artifact_transactions(r)==1,'commit recovery count'); check(live.read_bytes()==b'NEW','committed artifact rolled back')"
new = "n=r/'n.tmp'; n.write_bytes(b'NEW'); mu.atomic_replace(n,live); committed=json.loads((r/'manifest.json').read_text()); committed_page=committed['pages'][0]; committed_page['clean_revision']=1; tx.mark_manifest_commit(committed_page); (r/'manifest.json').write_text(json.dumps(committed),encoding='utf-8'); check(mu.recover_page_artifact_transactions(r)==1,'commit recovery count'); check(live.read_bytes()==b'NEW','committed artifact rolled back')"
count = text.count(old)
if count != 1:
    raise SystemExit(f"foundation transaction fixture changed; expected one match, found {count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("foundation transaction fixture aligned with artifact identity")
