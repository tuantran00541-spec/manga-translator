"""Writes examples/NN.sql and the expected examples/NN.out from SQLite, before the agent starts."""
import os
import sys

from answers import cell, load_sqlite, statements

data, queries, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
db = load_sqlite(data)
for i, sql in enumerate(statements(queries), 1):
    rows = db.execute(sql.rstrip(";")).fetchall()
    open(os.path.join(out, f"{i:02d}.sql"), "w", encoding="utf-8").write(sql + "\n")
    open(os.path.join(out, f"{i:02d}.out"), "w", encoding="utf-8").write("".join("\t".join(cell(v) for v in r) + "\n" for r in rows))
print(len(statements(queries)), "examples")
