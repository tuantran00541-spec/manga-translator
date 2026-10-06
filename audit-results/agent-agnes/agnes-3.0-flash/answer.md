## BÃO CÃO

**notekeeper** ÄÃ£ hoÃ n thÃ nh, **25/25 tests xanh** (`python -m pytest -q`).

### Cáº¥u trÃºc
```
pyproject.toml          # src layout, console script notekeeper, dev = pytest
src/notekeeper/
  __init__.py           # docstring (em-dash UTF-8 clean, ÄÃ£ verify byte-level)
  store.py              # load/save/create_note/notes_with_tag/find_notes/
                        # remove_note/add_tags + StoreError, atomic write
  cli.py                # argparse: add/list/find/rm/tag, exit 0/1/2 theo spec
  __main__.py           # python -m notekeeper
tests/
  test_storage.py       # 13 tests (tmp_path, agent storage viáº¿t)
  test_cli.py           # 12 tests subprocess (agent test viáº¿t)
README.md               # tiáº¿ng Viá»t, cÃ i Äáº·t, vÃ­ dá»¥, NOTEEKEPER_FILE
docs/                   # spec + 3 plan (lÆ°á»£c Äá» cho agent)
```

### CÃ¡ch chia agent song song
- **ash (storage)**: `store.py` + `test_storage.py` â xong trÆ°á»c, 13 tests xanh.
- **birch (CLI)**: `cli.py` + `__main__.py` + `pyproject.toml` â chá» store.py, tá»± smoke e2e báº±ng `PYTHONPATH=src`.
- **juniper (tests+README)**: `test_cli.py` (subprocess, `NOTEEKEPER_FILE` â tmp_path, giá»¯ ÄÃºng exit code) + `README.md`, tá»± cháº¡y suite chung.

### Quy táº¯c nghiá»p vá»¥ ÄÃ£ verify
- `add`: tags normalize (lowercase/strip/dedup), `next_id` tÄng, khÃ´ng tÃ¡i sá»­ dá»¥ng id sau `rm`.
- `list`/`find` rá»ng â `No notes found.`, exit 0; `find` case-insensitive + snippet indent 3 space.
- `rm`/`tag` id sai â `No note with id N.`, exit 1, file khÃ´ng ghi ÄÃ¨.
- File JSON há»ng â `Error: ...`, exit 2, **khÃ´ng ghi ÄÃ¨ file** (test riÃªng cho cáº£ 5 lá»nh).

### Kiá»m chá»©ng UTF-8
Kiá»m tra byte-level `__init__.py` = `notekeeper \xe2\x80\x94` (em-dash chuáº©n, khÃ´ng double-encode). Smoke test tiáº¿ng Viá»t vá»i literal ASCII-safe (escape `\uXXXX`) xÃ¡c nháº­n: title `"Thi á»±a"` lÆ°u clean, `find "sÆ°a"` khá»p body case-insensitive, JSON Äá»c ra ÄÃºng.

**CÃ²n Äá» láº¡i** (nhá»): `NOTEEKEPER_FILE` cÃ³ typo "KEPER" thay vÃ¬ "KEEPER" â ÄÃ£ ghi chÃº trong README; náº¿u muá»n Äá»i cáº§n cáº­p nháº­t cáº£ `cli.py` + tests + README.