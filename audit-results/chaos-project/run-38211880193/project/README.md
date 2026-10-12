# kvstore

A tiny in-memory key/value store written in Python (standard library only).

Features:

- JSON-serializable values, per-key TTL in seconds
- Optional persistence to a JSON file (`save` / `Store.load`)
- A small command line interface

## Usage

```python
from kvstore import Store
import time

store = Store(clock=time.time)   # a custom clock is handy for tests
store.set("answer", 42)
store.set("temp", "hello", ttl=60)
store.get("answer")          # 42
store.get("missing", "x")    # "x"
store.keys()                  # sorted live keys
store.delete("temp")          # True / False

store.save("data.json")
store = Store.load("data.json")
```

Keys whose TTL has elapsed behave as deleted: `get` returns `default`,
`keys()` skips them and `delete` returns `False`.  `Store.load` drops
entries that were already expired at load time.

## CLI

```sh
python -m kvstore --file data.json set KEY VALUE [--ttl N]
python -m kvstore --file data.json get KEY    # prints JSON; exit 1 if missing
python -m kvstore --file data.json del KEY
python -m kvstore --file data.json keys       # one key per line
```

`VALUE` is stored as JSON when it parses (`3` -> number, `[1,2]` -> list),
otherwise it is stored as a plain string.  A missing file is treated as an
empty store; `set` and `del` save the file afterwards.

## Tests

```sh
python -m pytest
```
