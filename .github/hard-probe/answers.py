"""Grader only: the expected answers from SQLite over the same dataset."""
import csv
import json
import os


def cell(value) -> str:
    """One value printed the way the spec says."""
    return "NULL" if value is None else repr(value) if isinstance(value, float) else str(value)


def statements(path: str) -> list[str]:
    return [s.strip() for s in open(path, encoding="utf-8").read().split(";\n") if s.strip().rstrip(";")]


def load_sqlite(data: str):
    """The dataset in an in-memory SQLite database, typed by schema.json, for the expected answers."""
    import sqlite3
    schema = json.load(open(os.path.join(data, "schema.json")))
    db = sqlite3.connect(":memory:")
    for table, cols in schema.items():
        db.execute(f"CREATE TABLE {table} ({', '.join(f'{c} {t}' for c, t in cols)})")
        cast = {"INTEGER": int, "REAL": float, "TEXT": str}
        with open(os.path.join(data, f"{table}.csv"), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[None if v == "" else cast[t](v) for v, (_, t) in zip(row, cols)] for row in reader]
        db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})", rows)
    return db
