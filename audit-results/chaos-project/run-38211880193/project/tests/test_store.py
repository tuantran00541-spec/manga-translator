"""Tests for kvstore: core API, TTL, file persistence and the CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kvstore import Store

ROOT = Path(__file__).resolve().parent.parent


class FakeClock:
    """A clock that only advances when told to."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# -- 1. basic store ---------------------------------------------------------


def test_set_get_roundtrip():
    store = Store()
    store.set("a", 1)
    store.set("b", "text")
    assert store.get("a") == 1
    assert store.get("b") == "text"


def test_get_missing_returns_default():
    store = Store()
    assert store.get("nope") is None
    assert store.get("nope", "fallback") == "fallback"


def test_set_overwrites_value():
    store = Store()
    store.set("k", 1)
    store.set("k", 2)
    assert store.get("k") == 2


def test_keys_sorted_and_unique():
    store = Store()
    for key in ["banana", "apple", "cherry"]:
        store.set(key, key)
    assert store.keys() == ["apple", "banana", "cherry"]


def test_store_rejects_non_serializable_value():
    store = Store()
    with pytest.raises(TypeError):
        store.set("bad", object())


# -- 2. TTL -----------------------------------------------------------------


def test_get_after_ttl_returns_default():
    clock = FakeClock()
    store = Store(clock=clock)
    store.set("k", "v", ttl=10)
    clock.advance(9)
    assert store.get("k") == "v"
    clock.advance(1)  # at exactly the expiry the key is gone
    assert store.get("k") is None
    assert store.get("k", "fallback") == "fallback"


def test_keys_drops_expired_entries():
    clock = FakeClock()
    store = Store(clock=clock)
    store.set("live", 1)
    store.set("short", 2, ttl=5)
    store.set("long", 3, ttl=100)
    assert store.keys() == ["live", "long", "short"]
    clock.advance(5)
    assert store.keys() == ["live", "long"]


def test_delete_live_key_returns_true():
    store = Store()
    store.set("k", "v")
    assert store.delete("k") is True
    assert store.get("k") is None
    assert store.keys() == []


def test_delete_missing_key_returns_false():
    store = Store()
    assert store.delete("nope") is False


def test_delete_expired_key_returns_false():
    clock = FakeClock()
    store = Store(clock=clock)
    store.set("k", "v", ttl=5)
    clock.advance(6)
    assert store.delete("k") is False
    assert store.keys() == []


# -- 3. save / load ---------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "store.json"
    src = Store()
    src.set("a", 1)
    src.set("b", {"nested": [1, 2, 3]})
    src.set("c", None)
    src.save(path)

    loaded = Store.load(path)
    assert loaded.get("a") == 1
    assert loaded.get("b") == {"nested": [1, 2, 3]}
    assert loaded.get("c") is None
    assert loaded.keys() == ["a", "b", "c"]


def test_load_keeps_live_ttl_entries(tmp_path):
    clock = FakeClock()
    path = tmp_path / "store.json"
    src = Store(clock=clock)
    src.set("k", "v", ttl=100)
    src.save(path)

    clock.advance(50)
    loaded = Store.load(path, clock=clock)
    assert loaded.get("k") == "v"
    clock.advance(50)  # expiry was relative to the original clock
    assert loaded.get("k") is None


def test_load_drops_expired_entries(tmp_path):
    clock = FakeClock()
    path = tmp_path / "store.json"
    src = Store(clock=clock)
    src.set("gone", 1, ttl=5)
    src.set("kept", 2, ttl=500)
    src.save(path)

    clock.advance(10)
    loaded = Store.load(path, clock=clock)
    assert loaded.keys() == ["kept"]
    assert loaded.get("gone") is None


# -- 4. CLI -----------------------------------------------------------------


def run_cli(args, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run(
        [sys.executable, "-m", "kvstore", "--file", "data.json", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_set_get_json_value(tmp_path):
    result = run_cli(["set", "num", "3"], cwd=tmp_path)
    assert result.returncode == 0
    assert (tmp_path / "data.json").exists()

    result = run_cli(["get", "num"], cwd=tmp_path)
    assert result.returncode == 0
    assert json.loads(result.stdout) == 3

    raw = json.loads((tmp_path / "data.json").read_text())
    assert raw["keys"]["num"]["value"] == 3


def test_cli_stores_json_types_and_plain_strings(tmp_path):
    run_cli(["set", "nums", "[1,2]"], cwd=tmp_path)
    run_cli(["set", "greeting", "hello"], cwd=tmp_path)
    raw = json.loads((tmp_path / "data.json").read_text())
    assert raw["keys"]["nums"]["value"] == [1, 2]
    assert raw["keys"]["greeting"]["value"] == "hello"


def test_cli_get_missing_key_exits_1_with_stderr(tmp_path):
    run_cli(["set", "a", "1"], cwd=tmp_path)
    result = run_cli(["get", "missing"], cwd=tmp_path)
    assert result.returncode == 1
    assert "missing" in result.stderr
    assert result.stdout.strip() == ""


def test_cli_del_and_keys(tmp_path):
    run_cli(["set", "a", "1"], cwd=tmp_path)
    run_cli(["set", "b", "2"], cwd=tmp_path)

    result = run_cli(["keys"], cwd=tmp_path)
    assert result.stdout.splitlines() == ["a", "b"]

    assert run_cli(["del", "a"], cwd=tmp_path).stdout.strip() == "True"
    assert run_cli(["del", "a"], cwd=tmp_path).stdout.strip() == "False"
    assert run_cli(["del", "never"], cwd=tmp_path).stdout.strip() == "False"

    result = run_cli(["keys"], cwd=tmp_path)
    assert result.stdout.splitlines() == ["b"]


def test_cli_missing_file_is_empty_store(tmp_path):
    result = run_cli(["keys"], cwd=tmp_path)
    assert result.returncode == 0
    assert result.stdout == ""


def test_cli_ttl_value_is_dropped_after_expiry(tmp_path):
    import time as _time

    result = run_cli(["set", "k", "v", "--ttl", "1"], cwd=tmp_path)
    assert result.returncode == 0
    assert run_cli(["get", "k"], cwd=tmp_path).returncode == 0
    _time.sleep(1.1)
    assert run_cli(["get", "k"], cwd=tmp_path).returncode == 1
    assert run_cli(["keys"], cwd=tmp_path).stdout == ""
