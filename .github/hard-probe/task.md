Build a SQL engine in this folder: `minisql.py`, as `spec.md` describes. The data is in `data/`, 40 worked examples with
their expected output are in `examples/`, and `python3 run_examples.py` checks your engine against them.

A grader will later run many more statements of the same kinds against your engine, covering every rule in `spec.md`,
including the corners the examples do not show (NULL logic, joins, correlated subqueries, grouping, compounds, CTEs,
functions, sorting and printing). It removes `sqlite3` before starting your engine, so only the Python standard library
can be used. This machine has no internet access.

Work until the engine is as complete and correct as you can make it; read the whole spec first, and test the corners
yourself. Use whatever tools, skills or helper agents fit the work.
