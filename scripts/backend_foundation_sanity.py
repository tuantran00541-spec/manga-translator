from __future__ import annotations
import json, os, sys, tempfile
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
def check(c,m):
    if not c: raise AssertionError(m)
def mb(r): return json.dumps({"schema_version":3,"chapter_id":"a1b2c3d4","pages":[{"clean_revision":r}]},separators=(",",":")).encode()
def publication_safety_checks():
    import app.manifest_utils as mu
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); src=r/'n.tmp'; dst=r/'live'; src.write_bytes(b'NEW'); dst.write_bytes(b'OLD'); calls=0
        def locked(a,b):
            nonlocal calls; calls+=1; raise PermissionError('locked')
        with mock.patch.object(mu.os,'replace',side_effect=locked):
            try: mu.atomic_replace(src,dst,max_retries=3,delay=0)
            except PermissionError: pass
            else: raise AssertionError('rename unexpectedly succeeded')
        check(calls==3,'retry count'); check(dst.read_bytes()==b'OLD','old destination changed'); check(src.read_bytes()==b'NEW','temp changed')
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); live=r/'clean.png'; live.write_bytes(b'OLD'); (r/'manifest.json').write_bytes(mb(0)); tx=mu.PageArtifactTransaction(r,0,[live],1); tx.__enter__(); backup=r/str(tx.records[0]['backup']); live.write_bytes(b'NEW'); check(backup.read_bytes()==b'OLD','rollback shares inode'); check(tx.rollback(),'rollback failed'); check(live.read_bytes()==b'OLD','rollback bytes wrong')
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); live=r/'clean.png'; live.write_bytes(b'OLD'); (r/'manifest.json').write_bytes(mb(0)); tx=mu.PageArtifactTransaction(r,0,[live],1); tx.__enter__(); live.write_bytes(b'NEW'); check(mu.recover_page_artifact_transactions(r)==1,'recovery count'); check(live.read_bytes()==b'OLD','precommit crash not restored')
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); live=r/'clean.png'; live.write_bytes(b'OLD'); (r/'manifest.json').write_bytes(mb(0)); tx=mu.PageArtifactTransaction(r,0,[live],1); tx.__enter__(); n=r/'n.tmp'; n.write_bytes(b'NEW'); mu.atomic_replace(n,live); (r/'manifest.json').write_bytes(mb(1)); check(mu.recover_page_artifact_transactions(r)==1,'commit recovery count'); check(live.read_bytes()==b'NEW','committed artifact rolled back')
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); src=r/'render.tmp'; dst=r/'page.png'; src.write_bytes(b'NEW'); dst.write_bytes(b'OLD')
        try: mu.publish_then_commit(src,dst,lambda: (_ for _ in ()).throw(RuntimeError('manifest')))
        except RuntimeError: pass
        else: raise AssertionError('metadata failure succeeded')
        check(dst.read_bytes()==b'OLD','render rollback failed')
    p=(ROOT/'app/pipeline.py').read_text(encoding='utf-8'); rr=(ROOT/'app/routers/render_commit.py').read_text(encoding='utf-8'); ex=(ROOT/'app/routers/export.py').read_text(encoding='utf-8')
    check('atomic_replace(tmp_clean_path, final_clean_path)' in p,'clean publish'); check('atomic_replace(tmp_auto_clean_path, auto_clean_path)' in p,'auto-clean publish'); check('publish_then_commit(tmp_path, final_path, commit_manifest)' in rr,'render publish'); check('atomic_replace(tmp_archive, final_archive)' in ex,'export publish')
def windows_locked_reader_check():
    if os.name!='nt': return
    import ctypes; from ctypes import wintypes; import app.manifest_utils as mu
    k=ctypes.WinDLL('kernel32',use_last_error=True); cf=k.CreateFileW; cf.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]; cf.restype=wintypes.HANDLE; ch=k.CloseHandle
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); src=r/'n.tmp'; dst=r/'live'; src.write_bytes(b'NEW'); dst.write_bytes(b'OLD'); h=cf(str(dst),0x80000000,1,None,3,0,None)
        try:
            try: mu.atomic_replace(src,dst,max_retries=2,delay=0)
            except PermissionError: pass
            else: raise AssertionError('locked Windows rename succeeded')
            check(dst.read_bytes()==b'OLD','Windows old bytes changed'); check(src.read_bytes()==b'NEW','Windows temp changed')
        finally: ch(h)
publication_safety_checks(); windows_locked_reader_check(); print('backend foundation sanity: publication safety PASS')
