"""Hidden acceptance tests for the kvstore task; the agent never sees them."""
import json
import subprocess
import sys

from kvstore import Store


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_basic_api():
    s = Store()
    s.set("b", {"x": [1, 2]})
    s.set("a", 3)
    assert s.get("b") == {"x": [1, 2]} and s.get("a") == 3 and s.get("zz", "d") == "d"
    assert s.keys() == ["a", "b"]
    assert s.delete("a") is True and s.delete("a") is False and s.keys() == ["b"]


def test_ttl():
    c = Clock()
    s = Store(clock=c)
    s.set("t", "v", ttl=10)
    s.set("p", "forever")
    c.now += 9
    assert s.get("t") == "v"
    c.now += 2
    assert s.get("t") is None and s.keys() == ["p"] and s.delete("t") is False


def test_save_and_load(tmp_path):
    c = Clock()
    s = Store(clock=c)
    s.set("keep", [1, "two"])
    s.set("short", 1, ttl=5)
    s.set("long", 2, ttl=500)
    path = tmp_path / "db.json"
    s.save(str(path))
    json.loads(path.read_text())
    c.now += 50
    loaded = Store.load(str(path), clock=c)
    assert loaded.keys() == ["keep", "long"] and loaded.get("keep") == [1, "two"]
    c.now += 1000
    assert loaded.keys() == ["keep"]


def cli(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "kvstore", "--file", str(tmp_path / "data.json"), *args], capture_output=True, text=True, timeout=60)


def test_cli(tmp_path):
    assert cli(tmp_path, "get", "nope").returncode == 1
    assert cli(tmp_path, "set", "n", "3").returncode == 0
    assert cli(tmp_path, "set", "l", "[1,2]").returncode == 0
    assert cli(tmp_path, "set", "s", "hello world").returncode == 0
    assert json.loads(cli(tmp_path, "get", "n").stdout) == 3
    assert json.loads(cli(tmp_path, "get", "l").stdout) == [1, 2]
    assert json.loads(cli(tmp_path, "get", "s").stdout) == "hello world"
    assert cli(tmp_path, "keys").stdout.split() == ["l", "n", "s"]
    assert cli(tmp_path, "del", "n").returncode == 0
    assert cli(tmp_path, "get", "n").returncode == 1
    assert cli(tmp_path, "set", "gone", "1", "--ttl", "0.01").returncode == 0
    import time
    time.sleep(0.2)
    assert cli(tmp_path, "get", "gone").returncode == 1
