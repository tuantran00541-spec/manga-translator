import sys, os, json, csv, math, functools

# ---------- Data loading ----------
DATA_DIR = None
SCHEMA = None
TABLES = {}

def load_data(data_dir):
    global DATA_DIR, SCHEMA, TABLES
    DATA_DIR = data_dir
    SCHEMA = json.load(open(os.path.join(data_dir, "schema.json"), encoding="utf-8"))
    for tname, cols in SCHEMA.items():
        path = os.path.join(data_dir, tname + ".csv")
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for record in reader:
                if record == []:
                    continue
                vals = []
                for (cname, ctype), raw in zip(cols, record):
                    if raw == "":
                        vals.append(None)
                    elif ctype == "INTEGER":
                        vals.append(int(raw))
                    elif ctype == "REAL":
                        vals.append(float(raw))
                    else:
                        vals.append(raw)
                rows.append(tuple(vals))
        TABLES[tname.lower()] = {"name": tname.lower(), "columns": [c[0].lower() for c in cols], "rows": rows}

RESERVED = set("""select from where group by having order asc desc limit offset distinct join inner left outer cross on and or not is null in like between exists case when then else end as union all with cast""".split())

AGG_FUNCS = {"count", "sum", "avg", "min", "max", "total"}

KEYWORDS = set("""select from where group by having order asc desc limit offset distinct join inner left outer cross on and or not is null in like between exists case when then else end as union all with cast""".split()) | {"by"}

# ---------- Tokenizer ----------
def tokenize(sql):
    toks = []
    i = 0
    n = len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if c == "'":
            i += 1
            buf = []
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i+1] == "'":
                        buf.append("'")
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(sql[i])
                i += 1
            toks.append(("STR", "".join(buf)))
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            word = sql[i:j]
            if word.lower() in KEYWORDS:
                toks.append(("KW", word.lower()))
            else:
                toks.append(("IDENT", word.lower()))
            i = j
            continue
        if c.isdigit() or (c == "." and i + 1 < n and sql[i+1].isdigit()):
            j = i
            has_dot = False
            while j < n and (sql[j].isdigit() or sql[j] == "."):
                if sql[j] == ".":
                    has_dot = True
                j += 1
            num = sql[i:j]
            if has_dot:
                toks.append(("NUM", float(num)))
            else:
                toks.append(("NUM", int(num)))
            i = j
            continue
        two = sql[i:i+2]
        if two in ("<=", ">=", "<>", "!=", "==", "||"):
            toks.append(("OP", two))
            i += 2
            continue
        if c in ("=", "<", ">", "+", "-", "*", "/", "%"):
            toks.append(("OP", c))
            i += 1
            continue
        if c in ("(", ")", ",", ".", ";"):
            toks.append(("PUNCT", c))
            i += 1
            continue
        raise ValueError("unexpected char %r" % c)
    toks.append(("EOF", ""))
    return toks

# ---------- AST ----------
class Literal:
    def __init__(self, v): self.v = v
class ColRef:
    def __init__(self, qual, name): self.qual = qual; self.name = name
class Star:
    def __init__(self, qual=None): self.qual = qual
class BinaryOp:
    def __init__(self, op, l, r): self.op = op; self.l = l; self.r = r
class UnaryOp:
    def __init__(self, op, e): self.op = op; self.e = e
class FuncCall:
    def __init__(self, name, args, distinct=False, star=False, cast_type=None):
        self.name = name; self.args = args; self.distinct = distinct; self.star = star; self.cast_type = cast_type
class CaseExpr:
    def __init__(self, base, branches, else_expr):
        self.base = base; self.branches = branches; self.else_expr = else_expr
class SubqueryExpr:
    def __init__(self, stmt): self.stmt = stmt
class ExistsExpr:
    def __init__(self, stmt): self.stmt = stmt
class InExpr:
    def __init__(self, left, items, subquery, neg):
        self.left = left; self.items = items; self.subquery = subquery; self.neg = neg
class BetweenExpr:
    def __init__(self, x, lo, hi, neg):
        self.x = x; self.lo = lo; self.hi = hi; self.neg = neg
class IsNullExpr:
    def __init__(self, x, neg): self.x = x; self.neg = neg
class LikeExpr:
    def __init__(self, x, pat, neg): self.x = x; self.pat = pat; self.neg = neg

class CTE:
    def __init__(self, name, query): self.name = name; self.query = query
class SourceTable:
    def __init__(self, name, alias): self.name = name; self.alias = alias
class SourceSubquery:
    def __init__(self, query, alias): self.query = query; self.alias = alias
class Join:
    def __init__(self, kind, source, on): self.kind = kind; self.source = source; self.on = on
class SelectItem:
    def __init__(self, kind, expr=None, star_qual=None, alias=None):
        self.kind = kind; self.expr = expr; self.star_qual = star_qual; self.alias = alias
class Select:
    def __init__(self, distinct, items, source, joins, where, groupby, having):
        self.distinct = distinct; self.items = items; self.source = source; self.joins = joins
        self.where = where; self.groupby = groupby; self.having = having
class Compound:
    def __init__(self, selects, ops): self.selects = selects; self.ops = ops
class Statement:
    def __init__(self, ctes, compound, order_by, limit, offset):
        self.ctes = ctes; self.compound = compound; self.order_by = order_by
        self.limit = limit; self.offset = offset

# ---------- Parser ----------
class Parser:
    def __init__(self, toks):
        self.toks = toks
        self.pos = 0
    def peek(self, off=0):
        i = self.pos + off
        if i < len(self.toks):
            return self.toks[i]
        return ("EOF", "")
    def advance(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t
    def match_kw(self, kw):
        if self.peek()[0] == "KW" and self.peek()[1] == kw:
            self.advance()
            return True
        return False
    def match_punct(self, ch):
        if self.peek()[0] == "PUNCT" and self.peek()[1] == ch:
            self.advance()
            return True
        return False
    def expect_kw(self, kw):
        if not self.match_kw(kw):
            raise SyntaxError("expected %s" % kw)
    def expect_punct(self, ch):
        if not self.match_punct(ch):
            raise SyntaxError("expected %s" % ch)
    def expect_ident(self):
        t = self.peek()
        if t[0] in ("IDENT", "KW"):
            self.advance()
            return t[1]
        raise SyntaxError("expected ident")
    def parse_statement(self):
        ctes = []
        if self.match_kw("with"):
            while True:
                name = self.expect_ident()
                self.expect_kw("as")
                self.expect_punct("(")
                q = self.parse_statement()
                self.expect_punct(")")
                ctes.append(CTE(name, q))
                if self.peek()[0] == "PUNCT" and self.peek()[1] == ",":
                    self.advance()
                    continue
                break
        compound = self.parse_compound()
        order_by = None
        if self.match_kw("order"):
            self.expect_kw("by")
            order_by = []
            term = self.parse_expr()
            desc = False
            if self.match_kw("desc"):
                desc = True
            elif self.match_kw("asc"):
                desc = False
            order_by.append((term, desc))
            while self.peek()[0] == "PUNCT" and self.peek()[1] == ",":
                self.advance()
                term = self.parse_expr()
                desc = False
                if self.match_kw("desc"):
                    desc = True
                elif self.match_kw("asc"):
                    desc = False
                order_by.append((term, desc))
        limit = None
        offset = None
        if self.match_kw("limit"):
            limit = self.parse_expr()
            if self.match_kw("offset"):
                offset = self.parse_expr()
        if self.peek()[0] == "PUNCT" and self.peek()[1] == ";":
            self.advance()
        return Statement(ctes, compound, order_by, limit, offset)

    def parse_compound(self):
        selects = [self.parse_select()]
        ops = []
        while self.match_kw("union"):
            allf = False
            if self.match_kw("all"):
                allf = True
            selects.append(self.parse_select())
            ops.append(allf)
        return Compound(selects, ops)

    def parse_select(self):
        self.expect_kw("select")
        distinct = False
        if self.match_kw("distinct"):
            distinct = True
        items = [self.parse_select_item()]
        while self.peek()[0] == "PUNCT" and self.peek()[1] == ",":
            self.advance()
            items.append(self.parse_select_item())
        source = None
        joins = []
        if self.match_kw("from"):
            source = self.parse_source()
            while True:
                if self.peek()[0] == "PUNCT" and self.peek()[1] == ",":
                    self.advance()
                    s = self.parse_source()
                    joins.append(Join("CROSS", s, None))
                elif self.peek()[0] == "KW" and self.peek()[1] in ("join", "inner", "left", "cross"):
                    kind = "INNER"
                    if self.match_kw("left"):
                        self.match_kw("outer")
                        kind = "LEFT"
                    elif self.match_kw("inner"):
                        kind = "INNER"
                    elif self.match_kw("cross"):
                        kind = "CROSS"
                    self.expect_kw("join")
                    s = self.parse_source()
                    on = None
                    if self.match_kw("on"):
                        on = self.parse_expr()
                    joins.append(Join(kind, s, on))
                elif self.peek()[0] == "KW" and self.peek()[1] == "join":
                    self.advance()
                    s = self.parse_source()
                    on = None
                    if self.match_kw("on"):
                        on = self.parse_expr()
                    joins.append(Join("INNER", s, on))
                else:
                    break
        where = None
        if self.match_kw("where"):
            where = self.parse_expr()
        groupby = None
        if self.match_kw("group"):
            self.expect_kw("by")
            groupby = [self.parse_expr()]
            while self.peek()[0] == "PUNCT" and self.peek()[1] == ",":
                self.advance()
                groupby.append(self.parse_expr())
        having = None
        if self.match_kw("having"):
            having = self.parse_expr()
        return Select(distinct, items, source, joins, where, groupby, having)

    def parse_source(self):
        if self.peek()[0] == "PUNCT" and self.peek()[1] == "(":
            self.advance()
            stmt = self.parse_statement()
            self.expect_punct(")")
            alias = self.parse_alias()
            return SourceSubquery(stmt, alias)
        name = self.expect_ident()
        alias = self.parse_alias()
        return SourceTable(name, alias)

    def parse_alias(self):
        if self.match_kw("as"):
            return self.expect_ident()
        t = self.peek()
        if t[0] in ("IDENT", "KW") and t[1] not in RESERVED:
            self.advance()
            return t[1]
        return None

    def parse_select_item(self):
        if self.peek()[0] == "OP" and self.peek()[1] == "*":
            self.advance()
            return SelectItem("star", star_qual=None)
        if self.peek()[0] in ("IDENT", "KW") and self.peek(1)[0] == "PUNCT" and self.peek(1)[1] == "." and self.peek(2)[0] == "OP" and self.peek(2)[1] == "*":
            qual = self.advance()[1]
            self.expect_punct(".")
            if not (self.peek()[0] == "OP" and self.peek()[1] == "*"):
                raise SyntaxError("expected *")
            self.advance()
            return SelectItem("star", star_qual=qual)
        expr = self.parse_expr()
        alias = None
        if self.match_kw("as"):
            alias = self.expect_ident()
        else:
            t = self.peek()
            if t[0] in ("IDENT", "KW") and t[1] not in RESERVED:
                alias = self.advance()[1]
        return SelectItem("expr", expr=expr, alias=alias)

    # expressions
    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.match_kw("or"):
            right = self.parse_and()
            left = BinaryOp("OR", left, right)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.match_kw("and"):
            right = self.parse_not()
            left = BinaryOp("AND", left, right)
        return left

    def parse_not(self):
        if self.match_kw("not"):
            return UnaryOp("NOT", self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_relational()
        # equality
        if self.peek()[0] == "OP" and self.peek()[1] in ("=", "==", "!=", "<>"):
            op = self.advance()[1]
            right = self.parse_relational()
            return BinaryOp(op, left, right)
        if self.match_kw("is"):
            neg = False
            if self.match_kw("not"):
                neg = True
            self.expect_kw("null")
            return IsNullExpr(left, neg)
        # optional NOT before in/like/between
        neg = False
        if self.peek()[0] == "KW" and self.peek()[1] == "not":
            # peek next
            nxt = self.peek(1)
            if nxt[0] == "KW" and nxt[1] in ("in", "like", "between"):
                self.advance()
                neg = True
        if self.match_kw("in"):
            self.expect_punct("(")
            if self.peek()[0] == "PUNCT" and self.peek()[1] == "(":
                self.advance()
                stmt = self.parse_statement()
                self.expect_punct(")")
                return InExpr(left, None, stmt, neg)
            if self.peek()[0] == "KW" and self.peek()[1] in ("select", "with"):
                stmt = self.parse_statement()
                self.expect_punct(")")
                return InExpr(left, None, stmt, neg)
            items = []
            if not (self.peek()[0] == "PUNCT" and self.peek()[1] == ")"):
                items.append(self.parse_expr())
                while self.peek()[0] == "PUNCT" and self.peek()[1] == ",":
                    self.advance()
                    items.append(self.parse_expr())
            self.expect_punct(")")
            return InExpr(left, items, None, neg)
        if self.match_kw("like"):
            pat = self.parse_relational()
            e = LikeExpr(left, pat, neg)
            if neg:
                e.neg = True
            # neg already set; ensure
            e.neg = neg
            return e
        if self.match_kw("between"):
            lo = self.parse_relational()
            self.expect_kw("and")
            hi = self.parse_relational()
            e = BetweenExpr(left, lo, hi, neg)
            e.neg = neg
            return e
        # if we consumed not but didn't match, shouldn't happen
        return left

    def parse_relational(self):
        left = self.parse_additive()
        while self.peek()[0] == "OP" and self.peek()[1] in ("<", "<=", ">", ">="):
            op = self.advance()[1]
            right = self.parse_additive()
            left = BinaryOp(op, left, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.peek()[0] == "OP" and self.peek()[1] in ("+", "-"):
            op = self.advance()[1]
            right = self.parse_multiplicative()
            left = BinaryOp(op, left, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_concat()
        while self.peek()[0] == "OP" and self.peek()[1] in ("*", "/", "%"):
            op = self.advance()[1]
            right = self.parse_concat()
            left = BinaryOp(op, left, right)
        return left

    def parse_concat(self):
        left = self.parse_unary()
        while self.peek()[0] == "OP" and self.peek()[1] == "||":
            self.advance()
            right = self.parse_unary()
            left = BinaryOp("||", left, right)
        return left

    def parse_unary(self):
        if self.peek()[0] == "OP" and self.peek()[1] in ("-", "+"):
            op = self.advance()[1]
            return UnaryOp(op, self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        t = self.peek()
        if t[0] == "PUNCT" and t[1] == "(":
            self.advance()
            if self.peek()[0] == "KW" and self.peek()[1] in ("select", "with"):
                stmt = self.parse_statement()
                self.expect_punct(")")
                return SubqueryExpr(stmt)
            if self.peek()[0] == "PUNCT" and self.peek()[1] == "(":
                stmt = self.parse_statement()
                self.expect_punct(")")
                return SubqueryExpr(stmt)
            expr = self.parse_expr()
            self.expect_punct(")")
            return expr
        if t[0] == "NUM":
            self.advance()
            return Literal(t[1])
        if t[0] == "STR":
            self.advance()
            return Literal(t[1])
        if t[0] == "KW" and t[1] == "null":
            self.advance()
            return Literal(None)
        if t[0] == "KW" and t[1] == "case":
            self.advance()
            if self.peek()[0] == "KW" and self.peek()[1] == "when":
                base = None
            else:
                base = self.parse_expr()
            branches = []
            while self.match_kw("when"):
                cond = self.parse_expr()
                self.expect_kw("then")
                val = self.parse_expr()
                branches.append((cond, val))
            else_expr = None
            if self.match_kw("else"):
                else_expr = self.parse_expr()
            self.expect_kw("end")
            return CaseExpr(base, branches, else_expr)
        if t[0] == "KW" and t[1] == "exists":
            self.advance()
            self.expect_punct("(")
            stmt = self.parse_statement()
            self.expect_punct(")")
            return ExistsExpr(stmt)
        if t[0] == "KW" and t[1] == "cast":
            self.advance()
            self.expect_punct("(")
            expr = self.parse_expr()
            self.expect_kw("as")
            typ = self.expect_ident()
            self.expect_punct(")")
            return FuncCall("cast", [expr], cast_type=typ.upper())
        if (t[0] == "IDENT" or t[0] == "KW") and self.peek(1)[0] == "PUNCT" and self.peek(1)[1] == "(":
            name = self.advance()[1]
            self.advance()  # (
            distinct = False
            if self.match_kw("distinct"):
                distinct = True
            if self.peek()[0] == "OP" and self.peek()[1] == "*":
                self.advance()
                self.expect_punct(")")
                return FuncCall(name, [], distinct=distinct, star=True)
            args = []
            if not (self.peek()[0] == "PUNCT" and self.peek()[1] == ")"):
                args.append(self.parse_expr())
                while self.peek()[0] == "PUNCT" and self.peek()[1] == ",":
                    self.advance()
                    args.append(self.parse_expr())
            self.expect_punct(")")
            return FuncCall(name, args, distinct=distinct)
        if t[0] in ("IDENT", "KW"):
            name = self.advance()[1]
            if self.peek()[0] == "PUNCT" and self.peek()[1] == ".":
                self.advance()
                col = self.expect_ident()
                return ColRef(name, col)
            return ColRef(None, name)
        if t[0] == "OP" and t[1] == "*":
            self.advance()
            return Star(None)
        raise SyntaxError("unexpected token %r" % (t,))

# ---------- Context ----------
class Ctx:
    __slots__ = ("fields", "values", "group_rows", "outer", "cte_map")
    def __init__(self, fields, values, group_rows=None, outer=None, cte_map=None):
        self.fields = list(fields)
        self.values = list(values)
        self.group_rows = group_rows
        self.outer = outer
        if cte_map is None:
            cte_map = outer.cte_map if outer is not None else {}
        self.cte_map = cte_map

def text_form(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)

def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)

def ternary_truth(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        return v != 0
    return False

def logic_and(a, b):
    ta = ternary_truth(a)
    tb = ternary_truth(b)
    if ta is False or tb is False:
        return 0
    if ta is None or tb is None:
        return None
    return 1

def logic_or(a, b):
    ta = ternary_truth(a)
    tb = ternary_truth(b)
    if ta is True or tb is True:
        return 1
    if ta is None or tb is None:
        return None
    return 0

def logic_not(v):
    t = ternary_truth(v)
    if t is None:
        return None
    return 0 if t else 1

def cmp_asc(a, b):
    # returns -1/0/1 or None if NULL
    if a is None or b is None:
        return None
    an = is_number(a)
    bn = is_number(b)
    if an and bn:
        if a < b: return -1
        if a > b: return 1
        return 0
    if an and not bn:
        return -1
    if (not an) and bn:
        return 1
    if isinstance(a, str) and isinstance(b, str):
        if a < b: return -1
        if a > b: return 1
        return 0
    return 0

def sort_cmp(a, b):
    if a is None and b is None:
        return 0
    if a is None:
        return -1
    if b is None:
        return 1
    an = is_number(a)
    bn = is_number(b)
    if an and bn:
        if a < b: return -1
        if a > b: return 1
        return 0
    if an and not bn:
        return -1
    if (not an) and bn:
        return 1
    if isinstance(a, str) and isinstance(b, str):
        if a < b: return -1
        if a > b: return 1
        return 0
    return 0

def values_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    an = is_number(a); bn = is_number(b)
    if an and bn:
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return a == b

def row_equal(r1, r2):
    if len(r1) != len(r2):
        return False
    return all(values_equal(a, b) for a, b in zip(r1, r2))

def is_true(v):
    t = ternary_truth(v)
    return t is True

def eval_comparison(op, left, right):
    if left is None or right is None:
        return None
    c = cmp_asc(left, right)
    if c is None:
        c = 0
    if op in ("=", "=="):
        return 1 if c == 0 else 0
    if op in ("!=", "<>"):
        return 0 if c == 0 else 1
    if op == "<":
        return 1 if c < 0 else 0
    if op == "<=":
        return 1 if c <= 0 else 0
    if op == ">":
        return 1 if c > 0 else 0
    if op == ">=":
        return 1 if c >= 0 else 0
    return 0

def resolve_column(ctx, qual, name):
    c = ctx
    while c is not None:
        for i, (q, col) in enumerate(c.fields):
            if col.lower() == name.lower() and (qual is None or q.lower() == qual.lower()):
                return c.values[i]
        c = c.outer
    return None

def contains_aggregate(e):
    if e is None:
        return False
    if isinstance(e, Literal):
        return False
    if isinstance(e, ColRef):
        return False
    if isinstance(e, Star):
        return False
    if isinstance(e, BinaryOp):
        return contains_aggregate(e.l) or contains_aggregate(e.r)
    if isinstance(e, UnaryOp):
        return contains_aggregate(e.e)
    if isinstance(e, FuncCall):
        if e.name.lower() in AGG_FUNCS:
            return True
        return any(contains_aggregate(a) for a in e.args)
    if isinstance(e, CaseExpr):
        if contains_aggregate(e.base):
            return True
        for cond, val in e.branches:
            if contains_aggregate(cond) or contains_aggregate(val):
                return True
        return contains_aggregate(e.else_expr)
    if isinstance(e, BetweenExpr):
        return contains_aggregate(e.x) or contains_aggregate(e.lo) or contains_aggregate(e.hi)
    if isinstance(e, InExpr):
        if contains_aggregate(e.left):
            return True
        if e.items:
            return any(contains_aggregate(x) for x in e.items)
        return False
    if isinstance(e, LikeExpr):
        return contains_aggregate(e.x) or contains_aggregate(e.pat)
    if isinstance(e, IsNullExpr):
        return contains_aggregate(e.x)
    return False

# ---------- Expression evaluation ----------
def eval_expr(e, ctx):
    if isinstance(e, Literal):
        return e.v
    if isinstance(e, ColRef):
        return resolve_column(ctx, e.qual, e.name)
    if isinstance(e, Star):
        return None
    if isinstance(e, UnaryOp):
        v = eval_expr(e.e, ctx)
        if v is None:
            return None
        if e.op == "-":
            if is_number(v):
                return -v
            return None
        if e.op == "+":
            return v
        if e.op == "NOT":
            return logic_not(v)
        return None
    if isinstance(e, BinaryOp):
        if e.op in ("+", "-", "*", "/", "%"):
            l = eval_expr(e.l, ctx)
            r = eval_expr(e.r, ctx)
            if l is None or r is None:
                return None
            if not (is_number(l) and is_number(r)):
                return None
            if e.op == "+":
                if isinstance(l, int) and isinstance(r, int):
                    return l + r
                return float(l) + float(r)
            if e.op == "-":
                if isinstance(l, int) and isinstance(r, int):
                    return l - r
                return float(l) - float(r)
            if e.op == "*":
                if isinstance(l, int) and isinstance(r, int):
                    return l * r
                return float(l) * float(r)
            if e.op == "/":
                if r == 0:
                    return None
                if isinstance(l, int) and isinstance(r, int):
                    # truncate toward zero
                    q = abs(l) // abs(r)
                    if (l < 0) != (r < 0):
                        q = -q
                    return q
                return float(l) / float(r)
            if e.op == "%":
                if r == 0:
                    return None
                if isinstance(l, int) and isinstance(r, int):
                    q = abs(l) // abs(r)
                    if (l < 0) != (r < 0):
                        q = -q
                    return l - q * r
                return float(l) % float(r)
        if e.op == "||":
            l = eval_expr(e.l, ctx)
            r = eval_expr(e.r, ctx)
            if l is None or r is None:
                return None
            return text_form(l) + text_form(r)
        if e.op in ("=", "==", "!=", "<>", "<", "<=", ">", ">="):
            l = eval_expr(e.l, ctx)
            r = eval_expr(e.r, ctx)
            return eval_comparison(e.op, l, r)
        if e.op == "AND":
            l = eval_expr(e.l, ctx)
            r = eval_expr(e.r, ctx)
            return logic_and(l, r)
        if e.op == "OR":
            l = eval_expr(e.l, ctx)
            r = eval_expr(e.r, ctx)
            return logic_or(l, r)
        return None
    if isinstance(e, FuncCall):
        name = e.name.lower()
        if name in AGG_FUNCS:
            rows = ctx.group_rows if ctx.group_rows is not None else [(ctx.fields, ctx.values)]
            # count
            if name == "count":
                if e.star:
                    return len(rows)
                if not e.args:
                    # count(*) represented maybe no args star handled; count() none => 0? 
                    return len(rows)
                vals = []
                for rf, rv in rows:
                    cctx = Ctx(list(rf), list(rv), None, ctx.outer, ctx.cte_map)
                    v = eval_expr(e.args[0], cctx)
                    if v is not None:
                        vals.append(v)
                if e.distinct:
                    uniq = []
                    for v in vals:
                        if not any(values_equal(v, u) for u in uniq):
                            uniq.append(v)
                    return len(uniq)
                return len(vals)
            if name == "sum":
                vals = []
                for rf, rv in rows:
                    cctx = Ctx(list(rf), list(rv), None, ctx.outer, ctx.cte_map)
                    v = eval_expr(e.args[0], cctx)
                    if v is not None:
                        vals.append(v)
                if e.distinct:
                    uniq = []
                    for v in vals:
                        if not any(values_equal(v, u) for u in uniq):
                            uniq.append(v)
                    vals = uniq
                if not vals:
                    return None
                if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                    return sum(vals)
                return sum(float(v) for v in vals)
            if name == "total":
                vals = []
                for rf, rv in rows:
                    cctx = Ctx(list(rf), list(rv), None, ctx.outer, ctx.cte_map)
                    v = eval_expr(e.args[0], cctx)
                    if v is not None:
                        vals.append(v)
                if e.distinct:
                    uniq = []
                    for v in vals:
                        if not any(values_equal(v, u) for u in uniq):
                            uniq.append(v)
                    vals = uniq
                if not vals:
                    return 0.0
                return float(sum(float(v) for v in vals))
            if name == "avg":
                vals = []
                for rf, rv in rows:
                    cctx = Ctx(list(rf), list(rv), None, ctx.outer, ctx.cte_map)
                    v = eval_expr(e.args[0], cctx)
                    if v is not None:
                        vals.append(v)
                if e.distinct:
                    uniq = []
                    for v in vals:
                        if not any(values_equal(v, u) for u in uniq):
                            uniq.append(v)
                    vals = uniq
                if not vals:
                    return None
                return sum(float(v) for v in vals) / len(vals)
            if name == "min":
                vals = []
                for rf, rv in rows:
                    cctx = Ctx(list(rf), list(rv), None, ctx.outer, ctx.cte_map)
                    v = eval_expr(e.args[0], cctx)
                    if v is not None:
                        vals.append(v)
                if e.distinct:
                    uniq = []
                    for v in vals:
                        if not any(values_equal(v, u) for u in uniq):
                            uniq.append(v)
                    vals = uniq
                if not vals:
                    return None
                # min by sorting order
                best = vals[0]
                for v in vals[1:]:
                    if cmp_asc(v, best) is not None and cmp_asc(v, best) < 0:
                        best = v
                return best
            if name == "max":
                vals = []
                for rf, rv in rows:
                    cctx = Ctx(list(rf), list(rv), None, ctx.outer, ctx.cte_map)
                    v = eval_expr(e.args[0], cctx)
                    if v is not None:
                        vals.append(v)
                if e.distinct:
                    uniq = []
                    for v in vals:
                        if not any(values_equal(v, u) for u in uniq):
                            uniq.append(v)
                    vals = uniq
                if not vals:
                    return None
                best = vals[0]
                for v in vals[1:]:
                    if cmp_asc(v, best) is not None and cmp_asc(v, best) > 0:
                        best = v
                return best
        # scalar functions
        if name == "coalesce":
            for a in e.args:
                v = eval_expr(a, ctx)
                if v is not None:
                    return v
            return None
        if name == "ifnull":
            v = eval_expr(e.args[0], ctx)
            if v is not None:
                return v
            return eval_expr(e.args[1], ctx)
        if name == "nullif":
            a = eval_expr(e.args[0], ctx)
            b = eval_expr(e.args[1], ctx)
            # equal?
            if a is None and b is None:
                return a  # NULL
            if a is not None and b is not None and values_equal(a, b):
                return None
            return a
        if name == "abs":
            v = eval_expr(e.args[0], ctx)
            if v is None:
                return None
            if isinstance(v, bool):
                return None
            if is_number(v):
                return abs(v)
            return None
        if name == "round":
            v = eval_expr(e.args[0], ctx)
            if v is None:
                return None
            n = 0
            if len(e.args) > 1:
                nv = eval_expr(e.args[1], ctx)
                if nv is None:
                    return None
                n = int(nv)
            if not is_number(v):
                return None
            from decimal import Decimal, ROUND_HALF_UP
            try:
                d = Decimal(str(v))
                q = Decimal(1).scaleb(-n)
                r = d.quantize(q, rounding=ROUND_HALF_UP)
                return float(r)
            except Exception:
                return None
        if name == "length":
            v = eval_expr(e.args[0], ctx)
            if v is None:
                return None
            return len(text_form(v))
        if name == "upper":
            v = eval_expr(e.args[0], ctx)
            if v is None:
                return None
            s = text_form(v)
            res = []
            for ch in s:
                if "a" <= ch <= "z":
                    res.append(chr(ord(ch) - 32))
                else:
                    res.append(ch)
            return "".join(res)
        if name == "lower":
            v = eval_expr(e.args[0], ctx)
            if v is None:
                return None
            s = text_form(v)
            res = []
            for ch in s:
                if "A" <= ch <= "Z":
                    res.append(chr(ord(ch) + 32))
                else:
                    res.append(ch)
            return "".join(res)
        if name == "substr":
            v = eval_expr(e.args[0], ctx)
            if v is None:
                return None
            s = text_form(v)
            start = eval_expr(e.args[1], ctx)
            if start is None:
                return None
            start = int(start)
            length = None
            if len(e.args) > 2:
                lv = eval_expr(e.args[2], ctx)
                if lv is None:
                    return None
                length = int(lv)
            if length is not None and length <= 0:
                return ""
            if start > 0:
                start_pos = start - 1
            elif start == 0:
                start_pos = -1
            else:
                start_pos = len(s) + start
            if length is None:
                end = len(s)
            else:
                end = start_pos + length
            left = max(0, start_pos)
            if length is None:
                right = len(s)
            else:
                right = min(len(s), end)
            if right < left:
                right = left
            return s[left:right]
        if name == "trim":
            v = eval_expr(e.args[0], ctx)
            if v is None:
                return None
            return text_form(v).strip()
        if name == "replace":
            s = eval_expr(e.args[0], ctx)
            f = eval_expr(e.args[1], ctx)
            t = eval_expr(e.args[2], ctx)
            if s is None or f is None or t is None:
                return None
            return text_form(s).replace(text_form(f), text_form(t))
        if name == "instr":
            s = eval_expr(e.args[0], ctx)
            sub = eval_expr(e.args[1], ctx)
            if s is None or sub is None:
                return None
            ss = text_form(s)
            su = text_form(sub)
            pos = ss.find(su)
            if pos == -1:
                return 0
            return pos + 1
        if name == "cast":
            v = eval_expr(e.args[0], ctx)
            if v is None:
                return None
            typ = e.cast_type
            if typ == "INTEGER":
                if isinstance(v, bool):
                    return None
                if isinstance(v, int):
                    return v
                if isinstance(v, float):
                    # truncate toward zero
                    if v >= 0:
                        return int(v)
                    else:
                        return -int(-v)
                # text
                try:
                    f = float(v)
                    if f >= 0:
                        return int(f)
                    else:
                        return -int(-f)
                except Exception:
                    try:
                        return int(v)
                    except Exception:
                        return 0
            if typ == "REAL":
                if isinstance(v, bool):
                    return None
                if is_number(v):
                    return float(v)
                try:
                    return float(v)
                except Exception:
                    return 0.0
            if typ == "TEXT":
                return text_form(v)
            return v
        # unknown function -> try 0 args? 
        # maybe function name is aggregate already handled
        return None
    if isinstance(e, CaseExpr):
        if e.base is None:
            for cond, val in e.branches:
                c = eval_expr(cond, ctx)
                if is_true(c):
                    return eval_expr(val, ctx)
            if e.else_expr is not None:
                return eval_expr(e.else_expr, ctx)
            return None
        else:
            basev = eval_expr(e.base, ctx)
            for cond, val in e.branches:
                cv = eval_expr(cond, ctx)
                # compare with =
                if basev is not None and cv is not None and values_equal(basev, cv):
                    # but spec: NULL x never matches; values_equal None handled false above
                    return eval_expr(val, ctx)
                if basev is None:
                    # NULL never matches
                    pass
            if e.else_expr is not None:
                return eval_expr(e.else_expr, ctx)
            return None
    if isinstance(e, BetweenExpr):
        x = eval_expr(e.x, ctx)
        lo = eval_expr(e.lo, ctx)
        hi = eval_expr(e.hi, ctx)
        # x >= lo AND x <= hi
        ge = eval_comparison(">=", x, lo) if not (x is None or lo is None) else None
        le = eval_comparison("<=", x, hi) if not (x is None or hi is None) else None
        res = logic_and(ge, le)
        if e.neg:
            res = logic_not(res)
        return res
    if isinstance(e, InExpr):
        x = eval_expr(e.left, ctx)
        if e.subquery is not None:
            from_ctx = ctx
            cols, rows = execute_statement(e.subquery, from_ctx, ctx.cte_map)
            vals = [r[0] for r in rows]
        else:
            vals = [eval_expr(it, ctx) for it in e.items]
        # in logic
        if not vals:
            res = 0
        else:
            found = any(values_equal(x, v) for v in vals)
            if found:
                res = 1
            elif x is None or any(v is None for v in vals):
                res = None
            else:
                res = 0
        if e.neg:
            if res == 1:
                res = 0
            elif res == 0:
                res = 1
            else:
                res = None
        return res
    if isinstance(e, LikeExpr):
        x = eval_expr(e.x, ctx)
        p = eval_expr(e.pat, ctx)
        if x is None or p is None:
            res = None
        else:
            res = 1 if like_match(text_form(x), text_form(p)) else 0
        if e.neg:
            res = logic_not(res)
        return res
    if isinstance(e, IsNullExpr):
        v = eval_expr(e.x, ctx)
        if e.neg:
            return 0 if v is None else 1
        else:
            return 1 if v is None else 0
    if isinstance(e, SubqueryExpr):
        cols, rows = execute_statement(e.stmt, ctx, ctx.cte_map)
        if not rows:
            return None
        return rows[0][0] if rows[0] else None
    if isinstance(e, ExistsExpr):
        cols, rows = execute_statement(e.stmt, ctx, ctx.cte_map)
        return 1 if rows else 0
    return None

def like_match(s, pat):
    # case-insensitive ASCII A-Z only
    def low(ch):
        if "A" <= ch <= "Z":
            return chr(ord(ch) + 32)
        return ch
    s = "".join(low(c) for c in s)
    pat = "".join(low(c) for c in pat)
    # DP for % and _
    # memoized recursion
    from functools import lru_cache
    # iterative DP
    # dp[i][j]: match first i of s, first j of pat
    # but % can match any run
    # use two pointers/backtracking simpler
    si = 0
    pi = 0
    star = -1
    match = 0
    while si < len(s):
        if pi < len(pat) and (pat[pi] == s[si] or pat[pi] == "_"):
            si += 1
            pi += 1
        elif pi < len(pat) and pat[pi] == "%":
            star = pi
            match = si
            pi += 1
        elif star != -1:
            pi = star + 1
            match += 1
            si = match
        else:
            return False
    while pi < len(pat) and pat[pi] == "%":
        pi += 1
    return pi == len(pat)

# ---------- Source resolution ----------
def source_to_table(src, cte_map, outer_ctx):
    if isinstance(src, SourceTable):
        name = src.name.lower()
        if name in cte_map:
            t = cte_map[name]
            alias = src.alias.lower() if src.alias else name
            cols = [(alias, c.lower()) for c in t["columns"]]
            rows = [tuple(r) for r in t["rows"]]
            return cols, rows
        base = TABLES[name]
        alias = (src.alias.lower() if src.alias else name)
        cols = [(alias, c.lower()) for c in base["columns"]]
        rows = base["rows"]
        return cols, rows
    if isinstance(src, SourceSubquery):
        cols, rows = execute_statement(src.query, outer_ctx, cte_map)
        alias = (src.alias.lower() if src.alias else "subquery")
        out_cols = [(alias, c.lower()) for c in cols]
        out_rows = [tuple(r) for r in rows]
        return out_cols, out_rows

def build_combined_fields(sel, cte_map, outer_ctx):
    if sel.source is None:
        return []
    fc, _ = source_to_table(sel.source, cte_map, outer_ctx)
    fields = list(fc)
    for join in sel.joins:
        rc, _ = source_to_table(join.source, cte_map, outer_ctx)
        fields = fields + list(rc)
    return fields

def expr_uses_source(expr, source_fields):
    if expr is None:
        return False
    if isinstance(expr, ColRef):
        for q, c in source_fields:
            if (expr.qual is None or expr.qual.lower() == q.lower()) and expr.name.lower() == c.lower():
                return True
        return False
    if isinstance(expr, (Literal, Star)):
        return False
    if isinstance(expr, BinaryOp):
        return expr_uses_source(expr.l, source_fields) or expr_uses_source(expr.r, source_fields)
    if isinstance(expr, UnaryOp):
        return expr_uses_source(expr.e, source_fields)
    if isinstance(expr, FuncCall):
        return any(expr_uses_source(a, source_fields) for a in expr.args)
    if isinstance(expr, CaseExpr):
        if expr_uses_source(expr.base, source_fields):
            return True
        for cond, val in expr.branches:
            if expr_uses_source(cond, source_fields) or expr_uses_source(val, source_fields):
                return True
        return expr_uses_source(expr.else_expr, source_fields)
    if isinstance(expr, BetweenExpr):
        return (expr_uses_source(expr.x, source_fields) or expr_uses_source(expr.lo, source_fields)
                or expr_uses_source(expr.hi, source_fields))
    if isinstance(expr, InExpr):
        if expr_uses_source(expr.left, source_fields):
            return True
        if expr.items:
            return any(expr_uses_source(x, source_fields) for x in expr.items)
        return False
    if isinstance(expr, LikeExpr):
        return expr_uses_source(expr.x, source_fields) or expr_uses_source(expr.pat, source_fields)
    if isinstance(expr, IsNullExpr):
        return expr_uses_source(expr.x, source_fields)
    if isinstance(expr, SubqueryExpr):
        return False
    if isinstance(expr, ExistsExpr):
        return False
    return False

def find_corr_equality(cond, source_fields):
    if cond is None:
        return None
    if isinstance(cond, BinaryOp) and cond.op in ("=", "=="):
        ls = expr_uses_source(cond.l, source_fields)
        rs = expr_uses_source(cond.r, source_fields)
        if ls and not rs:
            return (cond.l, cond.r)
        if rs and not ls:
            return (cond.r, cond.l)
    if isinstance(cond, BinaryOp) and cond.op == "AND":
        r = find_corr_equality(cond.l, source_fields)
        if r:
            return r
        return find_corr_equality(cond.r, source_fields)
    return None

def execute_select(sel, cte_map, outer_ctx, order_by=None):
    # returns (col_names, rows)
    source_fields = build_combined_fields(sel, cte_map, outer_ctx)
    # build source rows
    if sel.source is None:
        rows = [([], [])]
    else:
        fc, fr = source_to_table(sel.source, cte_map, outer_ctx)
        rows = [(fc, r) for r in fr]
        for join in sel.joins:
            rc, rr = source_to_table(join.source, cte_map, outer_ctx)
            rr = [list(r) for r in rr]
            rc = list(rc)
            new_rows = []
            # simple equality optimization
            eq = None
            if join.on is not None and isinstance(join.on, BinaryOp) and join.on.op in ("=", "==") and isinstance(join.on.l, ColRef) and isinstance(join.on.r, ColRef):
                left_fields = rows[0][0] if rows else []
                def _field_exists(col, fields):
                    for q, c in fields:
                        if (col.qual is None or q.lower() == col.qual.lower()) and c.lower() == col.name.lower():
                            return True
                    return False
                a, b = join.on.l, join.on.r
                if _field_exists(a, left_fields) and _field_exists(b, rc):
                    eq = (a, b)
                elif _field_exists(b, left_fields) and _field_exists(a, rc):
                    eq = (b, a)
                else:
                    eq = None
            if join.kind == "LEFT":
                if join.on is None:
                    for lf, lv in rows:
                        for rv in rr:
                            new_rows.append((list(lf) + list(rc), list(lv) + list(rv)))
                elif eq is not None:
                    lk, rk = eq
                    index = {}
                    for rv in rr:
                        rctx = Ctx(list(rc), list(rv), None, outer_ctx, cte_map)
                        key = eval_expr(rk, rctx)
                        if key is None:
                            continue
                        index.setdefault(key, []).append(rv)
                    for lf, lv in rows:
                        lctx = Ctx(list(lf), list(lv), None, outer_ctx, cte_map)
                        key = eval_expr(lk, lctx)
                        matched = False
                        if key is not None:
                            for rv in index.get(key, []):
                                new_rows.append((list(lf) + list(rc), list(lv) + list(rv)))
                                matched = True
                        if not matched:
                            new_rows.append((list(lf) + list(rc), list(lv) + [None] * len(rc)))
                else:
                    for lf, lv in rows:
                        matched = False
                        for rv in rr:
                            cf = list(lf) + list(rc)
                            cv = list(lv) + list(rv)
                            ctx = Ctx(cf, cv, None, outer_ctx, cte_map)
                            if is_true(eval_expr(join.on, ctx)):
                                new_rows.append((cf, cv))
                                matched = True
                        if not matched:
                            new_rows.append((list(lf) + list(rc), list(lv) + [None] * len(rc)))
            else:
                if join.on is None:
                    for lf, lv in rows:
                        for rv in rr:
                            new_rows.append((list(lf) + list(rc), list(lv) + list(rv)))
                elif eq is not None:
                    lk, rk = eq
                    index = {}
                    for rv in rr:
                        rctx = Ctx(list(rc), list(rv), None, outer_ctx, cte_map)
                        key = eval_expr(rk, rctx)
                        if key is None:
                            continue
                        index.setdefault(key, []).append(rv)
                    for lf, lv in rows:
                        lctx = Ctx(list(lf), list(lv), None, outer_ctx, cte_map)
                        key = eval_expr(lk, lctx)
                        if key is None:
                            continue
                        for rv in index.get(key, []):
                            new_rows.append((list(lf) + list(rc), list(lv) + list(rv)))
                else:
                    for lf, lv in rows:
                        for rv in rr:
                            cf = list(lf) + list(rc)
                            cv = list(lv) + list(rv)
                            ctx = Ctx(cf, cv, None, outer_ctx, cte_map)
                            if is_true(eval_expr(join.on, ctx)):
                                new_rows.append((cf, cv))
            rows = new_rows

    # Correlated equality optimization: pre-filter source rows by an equality
    # between a source column and an outer/constant expression.
    corr = find_corr_equality(sel.where, source_fields)
    if corr is not None and rows and outer_ctx is not None:
        src_expr, outer_expr = corr
        # evaluate outer/constant value
        octx = Ctx([], [], None, outer_ctx, cte_map)
        outer_val = eval_expr(outer_expr, octx)
        # build/cache index on source rows by src_expr value
        static_source = not (isinstance(sel.source, SourceSubquery) or any(isinstance(j.source, SourceSubquery) for j in sel.joins))
        if static_source:
            if not hasattr(sel, "_corr_indexes"):
                sel._corr_indexes = {}
            idx_key = id(src_expr)
            cached = sel._corr_indexes.get(idx_key)
            if cached is None or cached[0] != len(rows):
                index = {}
                for r in rows:
                    rctx = Ctx(list(r[0]), list(r[1]), None, outer_ctx, cte_map)
                    key = eval_expr(src_expr, rctx)
                    if key is None:
                        continue
                    index.setdefault(key, []).append(r)
                sel._corr_indexes[idx_key] = (len(rows), index)
            else:
                index = cached[1]
        else:
            index = {}
            for r in rows:
                rctx = Ctx(list(r[0]), list(r[1]), None, outer_ctx, cte_map)
                key = eval_expr(src_expr, rctx)
                if key is None:
                    continue
                index.setdefault(key, []).append(r)
        if outer_val is None:
            rows = []
        else:
            rows = [r for r in index.get(outer_val, [])]

    # WHERE
    filtered = []
    for f, v in rows:
        ctx = Ctx(list(f), list(v), None, outer_ctx, cte_map)
        if sel.where is None or is_true(eval_expr(sel.where, ctx)):
            filtered.append((list(f), list(v)))

    # has agg / grouped
    has_agg = False
    if contains_aggregate(sel.where):
        has_agg = True
    for item in sel.items:
        if item.kind == "expr" and contains_aggregate(item.expr):
            has_agg = True
    if sel.groupby:
        for g in sel.groupby:
            if contains_aggregate(g):
                has_agg = True
    if sel.having is not None and contains_aggregate(sel.having):
        has_agg = True
    if order_by:
        for term, desc in order_by:
            if contains_aggregate(term):
                has_agg = True
    grouped = bool(sel.groupby) or has_agg or (sel.having is not None)

    # output names
    def output_names_for_source_fields(sf):
        names = []
        for item in sel.items:
            if item.kind == "star":
                qual = item.star_qual
                for (q, c) in sf:
                    if qual is None or q.lower() == qual.lower():
                        names.append(c)
            else:
                if item.alias:
                    names.append(item.alias)
                elif isinstance(item.expr, ColRef):
                    names.append(item.expr.name)
                else:
                    names.append(expr_text(item.expr))
        return names

    # output column expressions (for GROUP BY position/alias resolution)
    expanded_exprs = []
    for item in sel.items:
        if item.kind == "star":
            qual = item.star_qual
            for (q, c) in source_fields:
                if qual is None or q.lower() == qual.lower():
                    expanded_exprs.append(ColRef(q, c))
        else:
            expanded_exprs.append(item.expr)
    col_names = output_names_for_source_fields(source_fields)

    def resolve_groupby_term(gb):
        if isinstance(gb, Literal) and isinstance(gb.v, int) and not isinstance(gb.v, bool):
            pos = gb.v
            if 1 <= pos <= len(expanded_exprs):
                return expanded_exprs[pos - 1]
        if isinstance(gb, ColRef) and gb.qual is None:
            for idx, name in enumerate(col_names):
                if name.lower() == gb.name.lower():
                    if idx < len(expanded_exprs):
                        return expanded_exprs[idx]
        return gb

    # group rows
    if grouped:
        groups = []
        if sel.groupby:
            key_map = {}
            order_keys = []
            for f, v in filtered:
                ctx = Ctx(list(f), list(v), None, outer_ctx, cte_map)
                key = tuple(eval_expr(resolve_groupby_term(g), ctx) for g in sel.groupby)
                if key not in key_map:
                    key_map[key] = []
                    order_keys.append(key)
                key_map[key].append((list(f), list(v)))
            groups = [key_map[k] for k in order_keys]
        else:
            if filtered or has_agg or sel.having is not None:
                groups = [filtered if filtered else []]
            else:
                groups = []
        # empty group for has_agg with no rows: already [] group
        # if no group but has_agg? groups=[[]] handled

        out_with_ctx = []
        for group in groups:
            if group:
                rep_f, rep_v = group[0]
                rep_f = list(rep_f); rep_v = list(rep_v)
            else:
                rep_f = list(source_fields)
                rep_v = [None] * len(source_fields)
            ctx = Ctx(rep_f, rep_v, group, outer_ctx, cte_map)
            if sel.having is not None:
                if not is_true(eval_expr(sel.having, ctx)):
                    continue
            # evaluate items
            out_row = []
            for item in sel.items:
                if item.kind == "star":
                    qual = item.star_qual
                    for (q, c) in source_fields:
                        if qual is None or q.lower() == qual.lower():
                            # value from rep row
                            out_row.append(resolve_column(ctx, q, c))
                else:
                    out_row.append(eval_expr(item.expr, ctx))
            out_with_ctx.append((tuple(out_row), ctx))

        # DISTINCT
        if sel.distinct:
            seen = []
            new = []
            for row, ctx in out_with_ctx:
                if not any(row_equal(row, srow) for srow, _ in seen):
                    seen.append((row, ctx))
                    new.append((row, ctx))
            out_with_ctx = new

        # ORDER BY
        col_names = output_names_for_source_fields(source_fields)
        if order_by:
            def cmp_pair(a, b):
                row1, ctx1 = a
                row2, ctx2 = b
                for term, desc in order_by:
                    v1 = orderby_value(term, row1, ctx1, col_names)
                    v2 = orderby_value(term, row2, ctx2, col_names)
                    c = sort_cmp(v1, v2)
                    if c is None:
                        c = 0
                    if desc:
                        c = -c
                    if c != 0:
                        return c
                return 0
            out_with_ctx.sort(key=functools.cmp_to_key(cmp_pair))

        rows_out = [r for r, _ in out_with_ctx]
        return col_names, rows_out

    # non-grouped
    out_with_ctx = []
    for f, v in filtered:
        ctx = Ctx(list(f), list(v), None, outer_ctx, cte_map)
        out_row = []
        for item in sel.items:
            if item.kind == "star":
                qual = item.star_qual
                for (q, c) in source_fields:
                    if qual is None or q.lower() == qual.lower():
                        out_row.append(resolve_column(ctx, q, c))
            else:
                out_row.append(eval_expr(item.expr, ctx))
        out_with_ctx.append((tuple(out_row), ctx))

    if sel.distinct:
        seen = []
        new = []
        for row, ctx in out_with_ctx:
            if not any(row_equal(row, srow) for srow, _ in seen):
                seen.append((row, ctx))
                new.append((row, ctx))
        out_with_ctx = new

    col_names = output_names_for_source_fields(source_fields)
    if order_by:
        def cmp_pair(a, b):
            row1, ctx1 = a
            row2, ctx2 = b
            for term, desc in order_by:
                v1 = orderby_value(term, row1, ctx1, col_names)
                v2 = orderby_value(term, row2, ctx2, col_names)
                c = sort_cmp(v1, v2)
                if c is None:
                    c = 0
                if desc:
                    c = -c
                if c != 0:
                    return c
            return 0
        out_with_ctx.sort(key=functools.cmp_to_key(cmp_pair))

    rows_out = [r for r, _ in out_with_ctx]
    return col_names, rows_out

def orderby_value(term, row, ctx, col_names):
    # integer position
    if isinstance(term, Literal) and isinstance(term.v, int) and not isinstance(term.v, bool):
        pos = term.v
        if 1 <= pos <= len(row):
            return row[pos - 1]
    if isinstance(term, ColRef):
        # output alias?
        if term.qual is None:
            # find in col_names case-insensitive
            for idx, name in enumerate(col_names):
                if name.lower() == term.name.lower():
                    return row[idx]
        # else evaluate
    return eval_expr(term, ctx)

def expr_text(e):
    # canonical-ish
    if isinstance(e, Literal):
        if isinstance(e.v, str):
            return "'" + e.v.replace("'", "''") + "'"
        if e.v is None:
            return "null"
        if isinstance(e.v, bool):
            return "1" if e.v else "0"
        if isinstance(e.v, float):
            return repr(e.v)
        return str(e.v)
    if isinstance(e, ColRef):
        if e.qual:
            return "%s.%s" % (e.qual, e.name)
        return e.name
    if isinstance(e, Star):
        return "*" if e.qual is None else e.qual + ".*"
    if isinstance(e, BinaryOp):
        return "(%s %s %s)" % (expr_text(e.l), e.op, expr_text(e.r))
    if isinstance(e, UnaryOp):
        return "%s %s" % (e.op, expr_text(e.e))
    if isinstance(e, FuncCall):
        if e.name == "cast":
            return "cast(%s as %s)" % (expr_text(e.args[0]), e.cast_type)
        distinct = "distinct " if e.distinct else ""
        if e.star:
            return "%s(%s*)" % (e.name, distinct)
        return "%s(%s%s)" % (e.name, distinct, ", ".join(expr_text(a) for a in e.args))
    if isinstance(e, CaseExpr):
        return "case ..."
    if isinstance(e, SubqueryExpr):
        return "(subquery)"
    if isinstance(e, ExistsExpr):
        return "exists (...)"
    if isinstance(e, InExpr):
        return "in (...)"
    if isinstance(e, BetweenExpr):
        return "between"
    if isinstance(e, IsNullExpr):
        return "is null"
    if isinstance(e, LikeExpr):
        return "like"
    return "expr"

def execute_compound(compound, cte_map, outer_ctx, order_by=None):
    # order_by only applied if single select? handled by caller; here ignore per-select except if single
    single = len(compound.selects) == 1
    cols = None
    rows = []
    for idx, sel in enumerate(compound.selects):
        c, r = execute_select(sel, cte_map, outer_ctx, order_by=order_by if (single and idx == 0) else None)
        if cols is None:
            cols = c
        if idx == 0:
            rows = list(r)
        else:
            is_all = compound.ops[idx-1]
            if is_all:
                rows = rows + list(r)
            else:
                combined = rows + list(r)
                seen = []
                new = []
                for rr in combined:
                    if not any(row_equal(rr, s) for s in seen):
                        seen.append(rr)
                        new.append(rr)
                rows = new
    return cols, rows

def execute_statement(stmt, outer_ctx=None, cte_map=None):
    if cte_map is None:
        cte_map = {}
    local = dict(cte_map)
    for cte in stmt.ctes:
        ccols, crows = execute_statement(cte.query, outer_ctx, local)
        local[cte.name.lower()] = {"name": cte.name.lower(), "columns": [str(x) for x in ccols], "rows": [tuple(r) for r in crows]}
    single = len(stmt.compound.selects) == 1
    cols, rows = execute_compound(stmt.compound, local, outer_ctx, order_by=stmt.order_by if single else None)
    if not single and stmt.order_by:
        # sort by output values
        def cmp_row(a, b):
            for term, desc in stmt.order_by:
                v1 = orderby_value(term, a, None, cols) if False else None
                # evaluate output-based
                v1 = orderby_value_output(term, a, cols)
                v2 = orderby_value_output(term, b, cols)
                c = sort_cmp(v1, v2)
                if c is None:
                    c = 0
                if desc:
                    c = -c
                if c != 0:
                    return c
            return 0
        rows.sort(key=functools.cmp_to_key(cmp_row))
    # LIMIT/OFFSET
    if stmt.limit is not None or stmt.offset is not None:
        off = 0
        if stmt.offset is not None:
            # evaluate offset; need ctx? use None fields; expression may be constant
            ctx = Ctx([], [], None, outer_ctx, local)
            off = eval_expr(stmt.offset, ctx)
            if off is None:
                off = 0
            off = int(off)
        lim = None
        if stmt.limit is not None:
            ctx = Ctx([], [], None, outer_ctx, local)
            lim = eval_expr(stmt.limit, ctx)
            if lim is not None:
                lim = int(lim)
        if lim is None:
            rows = rows[off:]
        else:
            rows = rows[off:off+lim]
    return cols, rows

def orderby_value_output(term, row, col_names):
    if isinstance(term, Literal) and isinstance(term.v, int) and not isinstance(term.v, bool):
        pos = term.v
        if 1 <= pos <= len(row):
            return row[pos - 1]
    if isinstance(term, ColRef) and term.qual is None:
        for idx, name in enumerate(col_names):
            if name.lower() == term.name.lower():
                return row[idx]
    # expression cannot be evaluated without ctx; return first? 
    return None

def main():
    if len(sys.argv) < 3:
        print("ERROR usage")
        sys.exit(1)
    data_dir = sys.argv[1]
    sql = sys.argv[2]
    load_data(data_dir)
    try:
        toks = tokenize(sql)
        p = Parser(toks)
        stmt = p.parse_statement()
        cols, rows = execute_statement(stmt)
        for row in rows:
            print("\t".join(format_value(v) for v in row))
        sys.exit(0)
    except Exception as e:
        print("ERROR %s" % e)
        sys.exit(1)

def format_value(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)

if __name__ == "__main__":
    main()
