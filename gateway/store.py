from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from gateway.plans import get_plan

JOB_TTL_SECONDS = 6 * 3600
LOGIN_CODE_TTL_SECONDS = 10 * 60
LOGIN_CODE_RESEND_SECONDS = 60
LOGIN_CODE_MAX_ATTEMPTS = 5
LOGIN_CODES_PER_DAY = 10  # with the attempts per code, about 50 guesses a day at a 6-digit code
SESSION_TTL_SECONDS = 90 * 86400
PLAN_PERIOD_SECONDS = 30 * 86400
# A long webtoon chapter makes about 280 A.I calls; this only stops a scripted job token.
JOB_MAX_REQUESTS = 1000
# Per network address and UTC day: login codes asked for, and new accounts made.
LOGIN_STARTS_PER_IP = 30
SIGNUPS_PER_IP = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    plan TEXT NOT NULL,
    plan_expires_at REAL,
    created_at REAL NOT NULL,
    billing_updated_at REAL,
    manage_url TEXT,
    deleted_at REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS login_codes (
    email TEXT PRIMARY KEY,
    code_hash TEXT NOT NULL,
    sent_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    day TEXT NOT NULL DEFAULT '',
    sent_today INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    external_id TEXT NOT NULL,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    plan TEXT NOT NULL,
    amount INTEGER NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    paid_at REAL,
    UNIQUE (provider, external_id)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    token_hash TEXT UNIQUE NOT NULL,
    period TEXT NOT NULL,
    status TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0,
    cost_cap_usd REAL NOT NULL,
    reserved_usd REAL NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    refunded INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_account_period ON jobs(account_id, period);
CREATE TABLE IF NOT EXISTS ip_counts (
    ip TEXT NOT NULL,
    kind TEXT NOT NULL,
    day TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ip, kind, day)
);
"""

# Columns added after the first release, with their definitions.
_MIGRATIONS = {
    "jobs": {"reserved_usd": "REAL NOT NULL DEFAULT 0"},
    "login_codes": {"day": "TEXT NOT NULL DEFAULT ''", "sent_today": "INTEGER NOT NULL DEFAULT 0"},
    "accounts": {"billing_updated_at": "REAL", "manage_url": "TEXT", "deleted_at": "REAL"},
}


class QuotaExceeded(Exception):
    pass


class InvalidToken(Exception):
    pass


class LoginRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def period_of(now: float) -> str:
    return time.strftime("%Y-%m", time.gmtime(now))


def day_of(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now))


def tombstone(email: str) -> str:
    """What a deleted account keeps of its email: enough to stop a reset of the free quota, not the address."""
    return "deleted:" + _hash(email.strip().lower())


class Store:
    def __init__(self, path: Path | str, clock=time.time):
        self._path = str(path)
        self._clock = clock
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
            for table, columns in _MIGRATIONS.items():
                have = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
                for name, ddl in columns.items():
                    if name not in have:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            # Nothing is in flight when the gateway starts.
            db.execute("UPDATE jobs SET reserved_usd = 0 WHERE reserved_usd != 0")

    def now(self) -> float:
        return float(self._clock())

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _write(self):
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    def _account_id_for_email(self, db, email: str) -> str:
        row = db.execute("SELECT id FROM accounts WHERE email = ?", (email,)).fetchone()
        if row is not None:
            return row["id"]
        # A deleted account comes back with its usage, so deleting it cannot reset the free quota.
        old = db.execute("SELECT id FROM accounts WHERE email = ?", (tombstone(email),)).fetchone()
        if old is not None:
            db.execute("UPDATE accounts SET email = ?, deleted_at = NULL WHERE id = ?", (email, old["id"]))
            return old["id"]
        account_id = uuid.uuid4().hex
        db.execute(
            "INSERT INTO accounts (id, email, plan, created_at) VALUES (?, ?, 'free', ?)",
            (account_id, email, self.now()),
        )
        return account_id

    def _new_session(self, db, account_id: str) -> str:
        token = "mc_" + secrets.token_urlsafe(32)
        db.execute(
            "INSERT INTO sessions (token_hash, account_id, created_at) VALUES (?, ?, ?)",
            (_hash(token), account_id, self.now()),
        )
        return token

    def create_account(self, email: str, plan: str = "free") -> tuple[str, str]:
        get_plan(plan)
        with self._write() as db:
            account_id = self._account_id_for_email(db, email)
            db.execute("UPDATE accounts SET plan = ? WHERE id = ?", (plan, account_id))
            return account_id, self._new_session(db, account_id)

    def _count_ip(self, db, ip: str | None, kind: str, limit: int, message: str) -> None:
        if not ip:
            return
        day = day_of(self.now())
        row = db.execute("SELECT count FROM ip_counts WHERE ip = ? AND kind = ? AND day = ?", (ip, kind, day)).fetchone()
        if row is not None and row["count"] >= limit:
            raise LoginRejected(429, message)
        db.execute("INSERT INTO ip_counts (ip, kind, day, count) VALUES (?, ?, ?, 1) "
                   "ON CONFLICT (ip, kind, day) DO UPDATE SET count = count + 1", (ip, kind, day))

    def start_login(self, email: str, ip: str | None = None) -> str:
        now = self.now()
        code = f"{secrets.randbelow(1_000_000):06d}"
        day = day_of(now)
        with self._write() as db:
            self._count_ip(db, ip, "login_start", LOGIN_STARTS_PER_IP,
                           "Mạng này đã xin quá nhiều mã hôm nay, hãy thử lại vào ngày mai")
            row = db.execute("SELECT sent_at, day, sent_today FROM login_codes WHERE email = ?", (email,)).fetchone()
            if row is not None and now - row["sent_at"] < LOGIN_CODE_RESEND_SECONDS:
                raise LoginRejected(429, "Vừa gửi mã, đợi một phút rồi thử lại")
            sent_today = int(row["sent_today"]) if row is not None and row["day"] == day else 0
            if sent_today >= LOGIN_CODES_PER_DAY:
                raise LoginRejected(429, "Đã gửi quá nhiều mã hôm nay, hãy thử lại vào ngày mai")
            db.execute(
                "INSERT OR REPLACE INTO login_codes (email, code_hash, sent_at, expires_at, attempts, day, sent_today) "
                "VALUES (?, ?, ?, ?, 0, ?, ?)",
                (email, _hash(f"{email}:{code}"), now, now + LOGIN_CODE_TTL_SECONDS, day, sent_today + 1),
            )
        return code

    def cancel_login(self, email: str) -> None:
        """Forget an unsent code so the user can retry at once."""
        with self._write() as db:
            db.execute("UPDATE login_codes SET code_hash = '', expires_at = 0, sent_at = 0 WHERE email = ?", (email,))

    def verify_login(self, email: str, code: str, ip: str | None = None) -> tuple[str, str]:
        now = self.now()
        rejected: LoginRejected | None = None
        with self._write() as db:
            row = db.execute("SELECT * FROM login_codes WHERE email = ?", (email,)).fetchone()
            if row is None or row["expires_at"] < now:
                rejected = LoginRejected(400, "Mã đã hết hạn, hãy gửi mã mới")
            elif row["attempts"] >= LOGIN_CODE_MAX_ATTEMPTS:
                rejected = LoginRejected(429, "Nhập sai quá nhiều lần, hãy gửi mã mới")
            elif not hmac.compare_digest(row["code_hash"], _hash(f"{email}:{code.strip()}")):
                db.execute("UPDATE login_codes SET attempts = attempts + 1 WHERE email = ?", (email,))
                rejected = LoginRejected(400, "Mã không đúng")
            else:
                known = db.execute("SELECT 1 FROM accounts WHERE email IN (?, ?)", (email, tombstone(email))).fetchone()
                if known is None:
                    self._count_ip(db, ip, "signup", SIGNUPS_PER_IP,
                                   "Mạng này đã tạo quá nhiều tài khoản hôm nay, hãy thử lại vào ngày mai")
                # The code is spent; the row stays so the day's count holds.
                db.execute("UPDATE login_codes SET code_hash = '', expires_at = 0 WHERE email = ?", (email,))
                account_id = self._account_id_for_email(db, email)
                token = self._new_session(db, account_id)
        if rejected is not None:
            raise rejected
        return account_id, token

    def logout(self, token: str) -> None:
        with self._write() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash(token),))

    def logout_all(self, account_id: str) -> int:
        with self._write() as db:
            return db.execute("DELETE FROM sessions WHERE account_id = ?", (account_id,)).rowcount

    def delete_account(self, account_id: str) -> None:
        """Forget the email and sessions; payments and usage stay for the accounts and the quota."""
        with self._write() as db:
            row = db.execute("SELECT email FROM accounts WHERE id = ?", (account_id,)).fetchone()
            if row is None:
                raise KeyError(account_id)
            db.execute("DELETE FROM sessions WHERE account_id = ?", (account_id,))
            db.execute("DELETE FROM login_codes WHERE email = ?", (row["email"],))
            db.execute("UPDATE jobs SET status = 'cancelled' WHERE account_id = ? AND status = 'active'", (account_id,))
            db.execute("UPDATE accounts SET email = ?, plan = 'free', plan_expires_at = NULL, manage_url = NULL, "
                       "deleted_at = ? WHERE id = ?", (tombstone(row["email"]), self.now(), account_id))

    def payments(self, account_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT provider, plan, amount, currency, status, created_at, paid_at FROM payments "
                "WHERE account_id = ? ORDER BY created_at DESC LIMIT 50", (account_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_plan(self, account_id: str, plan: str, expires_at: float | None = None) -> None:
        get_plan(plan)
        with self._write() as db:
            updated = db.execute(
                "UPDATE accounts SET plan = ?, plan_expires_at = ? WHERE id = ?", (plan, expires_at, account_id),
            ).rowcount
            if updated != 1:
                raise KeyError(account_id)

    def apply_subscription(self, account_id: str, plan: str, expires_at: float | None, updated_at: float,
                           manage_url: str | None) -> bool:
        """Apply a subscription event unless a newer one was already applied; False when it is stale."""
        get_plan(plan)
        with self._write() as db:
            row = db.execute("SELECT billing_updated_at FROM accounts WHERE id = ?", (account_id,)).fetchone()
            if row is None:
                raise KeyError(account_id)
            if row["billing_updated_at"] is not None and updated_at < row["billing_updated_at"]:
                return False
            db.execute("UPDATE accounts SET plan = ?, plan_expires_at = ?, billing_updated_at = ?, "
                       "manage_url = COALESCE(?, manage_url) WHERE id = ?",
                       (plan, expires_at, updated_at, manage_url, account_id))
            return True

    def _effective_plan(self, row) -> str:
        if row["plan"] != "free" and row["plan_expires_at"] is not None and row["plan_expires_at"] < self.now():
            return "free"
        return row["plan"]

    def account(self, account_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        if row is None:
            raise KeyError(account_id)
        data = dict(row)
        data["plan"] = self._effective_plan(row)
        if data["plan"] == "free":
            data["plan_expires_at"] = None
        return data

    def account_by_email(self, email: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT id FROM accounts WHERE email = ?", (email,)).fetchone()
        if row is None:
            raise KeyError(email)
        return self.account(row["id"])

    def account_for_token(self, token: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT account_id, created_at FROM sessions WHERE token_hash = ?",
                             (_hash(token),)).fetchone()
        if row is None or row["created_at"] + SESSION_TTL_SECONDS < self.now():
            raise InvalidToken()
        return self.account(row["account_id"])

    def create_payment(self, provider: str, external_id: str, account_id: str, plan: str,
                       amount: int, currency: str) -> str:
        payment_id = uuid.uuid4().hex
        with self._write() as db:
            db.execute(
                "INSERT INTO payments (id, provider, external_id, account_id, plan, amount, currency, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (payment_id, provider, external_id, account_id, plan, amount, currency, self.now()),
            )
        return payment_id

    def complete_payment(self, provider: str, external_id: str, amount: int) -> dict | None:
        now = self.now()
        with self._write() as db:
            row = db.execute(
                "SELECT * FROM payments WHERE provider = ? AND external_id = ?", (provider, external_id),
            ).fetchone()
            if row is None:
                return None
            if row["status"] == "paid":
                return {"payment_id": row["id"], "applied": False}
            if int(row["amount"]) != int(amount):
                db.execute("UPDATE payments SET status = 'amount_mismatch' WHERE id = ?", (row["id"],))
                return {"payment_id": row["id"], "applied": False, "error": "amount_mismatch"}
            account = db.execute("SELECT * FROM accounts WHERE id = ?", (row["account_id"],)).fetchone()
            current = self._effective_plan(account)
            base = now
            if current == row["plan"] and account["plan_expires_at"]:
                base = max(now, float(account["plan_expires_at"]))
            expires = base + PLAN_PERIOD_SECONDS
            db.execute("UPDATE accounts SET plan = ?, plan_expires_at = ? WHERE id = ?",
                       (row["plan"], expires, row["account_id"]))
            db.execute("UPDATE payments SET status = 'paid', paid_at = ? WHERE id = ?", (now, row["id"]))
            return {"payment_id": row["id"], "applied": True, "plan": row["plan"], "plan_expires_at": expires}

    @staticmethod
    def _used(db, account_id: str, period: str, now: float) -> int:
        return int(db.execute(
            "SELECT COUNT(*) FROM jobs WHERE account_id = ? AND period = ? AND refunded = 0 "
            "AND NOT (status = 'active' AND expires_at < ? AND requests = 0)",
            (account_id, period, now),
        ).fetchone()[0])

    def usage(self, account_id: str) -> dict:
        now = self.now()
        period = period_of(now)
        with self._connect() as db:
            used = self._used(db, account_id, period, now)
            cost = db.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM jobs WHERE account_id = ? AND period = ?",
                (account_id, period),
            ).fetchone()[0]
        return {"period": period, "used": used, "cost_usd": round(float(cost), 6)}

    def reserve_job(self, account_id: str) -> tuple[str, str, float]:
        now = self.now()
        period = period_of(now)
        job_id = uuid.uuid4().hex
        token = "mcj_" + secrets.token_urlsafe(32)
        with self._write() as db:
            account = db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
            plan = get_plan(self._effective_plan(account))
            if self._used(db, account_id, period, now) >= plan.chapters_per_month:
                raise QuotaExceeded(plan.id)
            db.execute(
                "INSERT INTO jobs (id, account_id, token_hash, period, status, cost_cap_usd, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
                (job_id, account_id, _hash(token), period, plan.job_cost_guard_usd, now, now + JOB_TTL_SECONDS),
            )
        return job_id, token, plan.job_cost_guard_usd

    def active_job_for_token(self, token: str) -> sqlite3.Row:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE token_hash = ?", (_hash(token),)).fetchone()
        if row is None or row["status"] != "active" or row["expires_at"] < self.now():
            raise InvalidToken()
        return row

    def begin_request(self, job_id: str, reserve_usd: float = 0.0) -> None:
        """Hold ``reserve_usd`` for a request, so requests in flight together cannot pass the cap."""
        with self._write() as db:
            row = db.execute("SELECT cost_usd, reserved_usd, cost_cap_usd, requests FROM jobs WHERE id = ?",
                             (job_id,)).fetchone()
            if row["cost_usd"] + row["reserved_usd"] + max(0.0, reserve_usd) > row["cost_cap_usd"]:
                raise QuotaExceeded("cost_cap")
            if row["requests"] >= JOB_MAX_REQUESTS:
                raise QuotaExceeded("requests")
            db.execute("UPDATE jobs SET requests = requests + 1, reserved_usd = reserved_usd + ? WHERE id = ?",
                       (max(0.0, reserve_usd), job_id))

    def add_cost(self, job_id: str, cost: float, prompt_tokens: int = 0, completion_tokens: int = 0,
                 released_usd: float = 0.0) -> float:
        with self._write() as db:
            db.execute(
                "UPDATE jobs SET cost_usd = cost_usd + ?, prompt_tokens = prompt_tokens + ?, "
                "completion_tokens = completion_tokens + ?, reserved_usd = MAX(0, reserved_usd - ?) WHERE id = ?",
                (max(0.0, cost), max(0, prompt_tokens), max(0, completion_tokens), max(0.0, released_usd), job_id),
            )
            return float(db.execute("SELECT cost_usd FROM jobs WHERE id = ?", (job_id,)).fetchone()[0])

    def finish_job(self, job_id: str, outcome: str) -> dict:
        with self._write() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            refund = outcome != "completed" and row["requests"] == 0
            db.execute(
                "UPDATE jobs SET status = ?, refunded = ? WHERE id = ?",
                (outcome, int(refund), job_id),
            )
        return {
            "job_id": job_id, "status": outcome, "refunded": refund, "cost_usd": round(float(row["cost_usd"]), 6),
            "requests": int(row["requests"]), "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
        }

    def purge(self) -> dict:
        """Drop spent sessions, codes and counters, and close jobs past their time."""
        now = self.now()
        with self._write() as db:
            counts = {
                "sessions": db.execute("DELETE FROM sessions WHERE created_at < ?",
                                       (now - SESSION_TTL_SECONDS,)).rowcount,
                "login_codes": db.execute("DELETE FROM login_codes WHERE expires_at < ? AND day < ?",
                                          (now, day_of(now - 86400))).rowcount,
                "ip_counts": db.execute("DELETE FROM ip_counts WHERE day < ?", (day_of(now - 86400),)).rowcount,
                # A job that never called the A.I is refunded, as it is while it waits to expire.
                "jobs": db.execute(
                    "UPDATE jobs SET status = 'expired', refunded = CASE WHEN requests = 0 THEN 1 ELSE refunded END "
                    "WHERE status = 'active' AND expires_at < ?", (now,)).rowcount,
            }
        return counts

    def backup(self, dest: Path | str) -> None:
        """Copy the live database to ``dest`` safely while the gateway runs."""
        with self._connect() as db:
            target = sqlite3.connect(str(dest))
            try:
                db.backup(target)
            finally:
                target.close()

    def stats(self) -> dict:
        now = self.now()
        period = period_of(now)
        with self._connect() as db:
            accounts = db.execute("SELECT * FROM accounts WHERE deleted_at IS NULL").fetchall()
            jobs = db.execute(
                "SELECT COUNT(*) AS jobs, COALESCE(SUM(cost_usd), 0) AS cost, COALESCE(SUM(requests), 0) AS requests "
                "FROM jobs WHERE period = ?", (period,)).fetchone()
            revenue = db.execute(
                "SELECT currency, COALESCE(SUM(amount), 0) AS amount, COUNT(*) AS count FROM payments "
                "WHERE status = 'paid' AND paid_at >= ? GROUP BY currency", (now - PLAN_PERIOD_SECONDS,)).fetchall()
        plans: dict[str, int] = {}
        for row in accounts:
            plan = self._effective_plan(row)
            plans[plan] = plans.get(plan, 0) + 1
        return {
            "period": period,
            "accounts": len(accounts),
            "plans": plans,
            "jobs": int(jobs["jobs"]),
            "requests": int(jobs["requests"]),
            "ai_cost_usd": round(float(jobs["cost"]), 4),
            "paid_last_30_days": {row["currency"]: {"amount": int(row["amount"]), "count": int(row["count"])}
                                  for row in revenue},
        }
