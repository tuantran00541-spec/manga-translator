"""Grades an engine on the hidden statements against SQLite, with sqlite3 removed from the engine; writes a JSON report."""
import json
import re
import sys

from answers import cell, load_sqlite, statements
from sqlcheck import matches, ordered, run

data, queries, engine, report = sys.argv[1:5]
db = load_sqlite(data)
source = open(engine, encoding="utf-8", errors="replace").read() if __import__("os").path.exists(engine) else ""
rows, passed = [], 0
for sql in statements(queries):
    expected = [[cell(v) for v in r] for r in db.execute(sql.rstrip(";")).fetchall()]
    code, out = run(engine, data, sql, block=True) if source else (None, "no engine")
    ok = code == 0 and matches(expected, out, ordered(sql))
    passed += ok
    rows.append({"sql": sql, "ok": ok, "exit": code, "expected": expected[:5], "got": out.splitlines()[:5]})
result = {"passed": passed, "total": len(rows), "uses_sqlite": bool(re.search(r"\bsqlite3?\b", source)), "lines": source.count("\n"), "rows": rows}
json.dump(result, open(report, "w"), indent=1, ensure_ascii=False)
print(f"{passed}/{len(rows)} hidden statements pass; uses sqlite: {result['uses_sqlite']}; engine lines: {result['lines']}")
