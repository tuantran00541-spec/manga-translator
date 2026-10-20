# MiniSQL: specification

Build `minisql.py` in this folder: a read-only SQL engine over the CSV tables in `data/`, written in Python 3 with the
standard library only. Importing or running `sqlite3`, any other database engine, or any package outside the standard
library is not allowed (the grader runs your engine with `sqlite3` removed). No internet is available.

## Command line

```
python3 minisql.py data "SELECT ..."
```

The first argument is the data folder, the second one SQL statement (a trailing `;` is allowed). The engine prints the
result rows to stdout, one row per line, cells separated by one TAB, with no header line, and exits 0. A statement the
engine cannot run prints one line starting with `ERROR` and exits 1. Every statement the grader sends is valid.

Cell printing: `NULL` prints as `NULL`; an integer as its decimal digits (`-3`, `42`); a real as Python `repr(float)`
(`2.5`, `0.1`, `1e-05`); text as is. The grader compares reals with a relative tolerance of 1e-6, and compares rows in
order when the statement's outermost query has ORDER BY, otherwise as a multiset.

## Data

`data/schema.json` maps each table to its columns, each `[name, type]` with type `INTEGER`, `REAL` or `TEXT`. Each table
is `data/<table>.csv`: a header row, then one row per record, comma separated with standard CSV quoting. An empty field
is NULL. Other fields convert by the column type: `INTEGER` to int, `REAL` to float, `TEXT` stays a string (it may hold
commas, quotes, `%`, `_` and `'`). Foreign keys are not enforced: some ids point at rows that do not exist, on purpose.

## Values and types

There are four kinds of value: NULL, integer, real and text. Integer and real are numbers.

- Arithmetic `+ - * / %` on two integers gives an integer; `/` truncates toward zero (`-7 / 2` is `-3`) and `%` takes
  the sign of the left operand (`-7 % 2` is `-1`). With a real operand `+ - * /` give a real (`7 / 2.0` is `3.5`); the
  grader never uses `%` on reals. Division or `%` by zero gives NULL. Any arithmetic with NULL gives NULL. Unary `-`
  negates.
- Arithmetic on text is not used by the grader.
- `||` concatenates the text form of both sides (an integer `5` becomes `5`, a real `2.5` becomes `2.5`, a real with no
  fraction such as `3.0` becomes `3.0`); with NULL on either side the result is NULL.

## Comparison and truth

- `= == <> != < <= > >=` between two numbers compare numerically (`1 = 1.0` is true). Between two texts they compare by
  code point. The grader never compares a number with a text. With NULL on either side the result is NULL (unknown).
- Truth: a condition is true when it is a non-zero number; false when it is zero; unknown when NULL. Comparisons give
  1 or 0 (integers) or NULL.
- `NOT` unknown is unknown. `AND`: false if either side is false, else unknown if either is unknown, else true. `OR`:
  true if either side is true, else unknown if either is unknown, else false. These give 1, 0 or NULL.
- `WHERE`, `ON`, `HAVING` and `CASE WHEN` keep or take a branch only when the condition is true (not unknown).
- `x IS NULL`, `x IS NOT NULL` give 1 or 0, never NULL.
- `x BETWEEN a AND b` means `x >= a AND x <= b`; `NOT BETWEEN` is its `NOT`.
- `x IN (v1, v2, ...)` and `x IN (subquery)`: true if `x` equals some value; otherwise unknown if `x` is NULL or any
  value is NULL; otherwise false (an empty list or empty subquery gives false even when `x` is NULL). `NOT IN` is the
  `NOT` of that, so `x NOT IN (1, NULL)` is never true.
- `x LIKE pattern`: `%` matches any run of characters (also none), `_` exactly one character, every other character
  matches itself, ignoring case for ASCII letters A-Z only. NULL on either side gives NULL. `NOT LIKE` is its `NOT`.
  There is no ESCAPE clause.
- `EXISTS (subquery)` is 1 if the subquery returns any row, else 0. `NOT EXISTS` is its `NOT`.

## Statement grammar

Keywords are case-insensitive. Identifiers are unquoted (letters, digits, `_`, not starting with a digit) and also
case-insensitive. String literals use single quotes; `''` inside is one quote. Numeric literals: `12` is an integer,
`12.5` and `.5` are reals. `NULL` is the null literal.

```
statement   := [WITH cte (, cte)*] compound [ORDER BY order (, order)*] [LIMIT expr [OFFSET expr]]
cte         := name AS ( compound [ORDER BY ...] [LIMIT ...] )
compound    := select ( (UNION | UNION ALL) select )*
select      := SELECT [DISTINCT] item (, item)*
               [FROM source (join)*] [WHERE expr] [GROUP BY expr (, expr)*] [HAVING expr]
item        := * | table.* | expr [[AS] alias]
source      := table [[AS] alias] | ( statement ) [[AS] alias]
join        := [INNER | LEFT [OUTER] | CROSS] JOIN source [ON expr] | , source
order       := expr [ASC | DESC]
```

- A comma in FROM is a cross join. `JOIN` without `ON` is a cross join. `LEFT JOIN` keeps every left row; when no right
  row matches, the right side's columns are NULL. Joins apply left to right.
- A column reference is `col` or `qualifier.col`, where the qualifier is the table name or its alias. An unqualified
  name is looked up in the innermost query first, then in each enclosing query (correlated subqueries).
- `UNION` removes duplicate rows of the combined result (NULL equals NULL here); `UNION ALL` keeps them. Chains apply left
  to right. ORDER BY and LIMIT after a compound apply to the whole compound; there they may only name result columns (by
  output name) or positions.
- A CTE is a named result usable as a table in the rest of the statement and in later CTEs. CTEs are not recursive.
- A subquery in parentheses can stand as a value (scalar subquery): its first column of its first row, or NULL when it
  returns no row. It may be correlated.

## SELECT evaluation order

1. FROM and joins build the rows; WHERE keeps the rows where it is true.
2. GROUP BY groups rows whose key values are equal (NULL keys group together). Like an ORDER BY term, a GROUP BY term may
   be an output alias or a positive integer position as well as an expression. Without GROUP BY, a query that uses an
   aggregate in SELECT, HAVING or ORDER BY is one group of all rows (one output row even when there are no rows).
3. HAVING keeps the groups where it is true.
4. SELECT items are evaluated. `*` expands to all columns of all sources in FROM order; `t.*` to one source's columns.
   The output name of an item is its alias, else the column name for a plain column reference, else the expression text.
5. DISTINCT removes duplicate output rows (NULL equals NULL here).
6. ORDER BY sorts; then LIMIT and OFFSET cut. An ORDER BY term may be an output alias, a positive integer position
   (1 is the first item), or any expression over the FROM columns (also aggregates in a grouped query).

In a grouped query, a column outside an aggregate is taken from some row of the group; the grader only uses columns that
are equal across the group (the GROUP BY keys or columns determined by them).

Sorting: NULL sorts before every other value in ASC order and after them in DESC; then numbers numerically, then texts by
code point (a number sorts before a text). Ties keep no particular order (the grader's ORDER BY clauses always settle every tie that matters).

## Functions

Aggregates (NULL inputs are skipped):

- `COUNT(*)` rows in the group; `COUNT(x)` non-NULL values; `COUNT(DISTINCT x)` distinct non-NULL values.
- `SUM(x)`: NULL if there are no non-NULL values; an integer if all values are integers, else a real.
  `SUM(DISTINCT x)` sums distinct values.
- `AVG(x)`: real; NULL if there are no non-NULL values. `AVG(DISTINCT x)` averages distinct values.
- `MIN(x)`, `MAX(x)`: smallest and largest by the sorting order above, ignoring NULL; NULL if none.
- `TOTAL(x)`: like SUM but always real, and 0.0 instead of NULL.

Scalar functions:

- `ABS(x)`; `ROUND(x)` and `ROUND(x, n)`: round half away from zero to n digits (default 0), result real;
  `LENGTH(s)`: number of characters of the text form; `UPPER(s)`, `LOWER(s)` ASCII only;
  `SUBSTR(s, start[, length])`: 1-based start; a start of 0 or below counts from before the first character
  (`SUBSTR('abc', 0, 2)` is `a`); a negative start counts from the end (`SUBSTR('hello', -3)` is `llo`);
  `TRIM(s)`; `REPLACE(s, from, to)`; `INSTR(s, sub)`: 1-based position or 0;
  `COALESCE(a, b, ...)` first non-NULL; `NULLIF(a, b)` NULL when equal, else a; `IFNULL(a, b)`;
  `CAST(x AS INTEGER)` truncates a real toward zero; `CAST(x AS REAL)`; `CAST(x AS TEXT)`.
  All of them give NULL for a NULL input (except COALESCE, IFNULL and NULLIF, as described).
- `CASE WHEN c THEN v ... [ELSE e] END` and `CASE x WHEN a THEN v ... [ELSE e] END` (the second compares with `=`, so a
  NULL `x` never matches); no ELSE gives NULL.

## Precedence (high to low)

unary `-`, `+`; `||`; `* / %`; `+ -`; `< <= > >=`; `= == != <> IS IN LIKE BETWEEN`; `NOT`; `AND`; `OR`.
Parentheses group.

## Done when

`python3 run_examples.py` reports every query in `examples/` as passing. The grader runs many more statements of the
same kind, covering all of this document, so test the corners the examples leave out.
