from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from gateway.plans import get_plan

JOB_TTL_SECONDS = 6 * 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    plan TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    token_hash TEXT UNIQUE NOT NULL,
    period TEXT NOT NULL,
    status TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0,
    cost_cap_usd REAL NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    refunded INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_account_period ON jobs(account_id, period);
"""


class QuotaExceeded(Exception):
    pass


class InvalidToken(Exception):
    pass


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def period_of(now: float) -> str:
    return time.strftime("%Y-%m", time.gmtime(now))


class Store:
    def __init__(self, path: Path | str, clock=time.time):
        self._path = str(path)
        self._clock = clock
        self._lock = threading.Lock()
        with self._connect() as db:
            db.executescript(_SCHEMA)

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

    def create_account(self, email: str, plan: str = "free") -> tuple[str, str]:
        get_plan(plan)
        account_id = uuid.uuid4().hex
        token = "mc_" + secrets.token_urlsafe(32)
        with self._write() as db:
            db.execute(
                "INSERT INTO accounts (id, email, token_hash, plan, created_at) VALUES (?, ?, ?, ?, ?)",
                (account_id, email, _hash(token), plan, self.now()),
            )
        return account_id, token

    def set_plan(self, account_id: str, plan: str) -> None:
        get_plan(plan)
        with self._write() as db:
            if db.execute("UPDATE accounts SET plan = ? WHERE id = ?", (plan, account_id)).rowcount != 1:
                raise KeyError(account_id)

    def account_for_token(self, token: str) -> sqlite3.Row:
        with self._connect() as db:
            row = db.execute("SELECT * FROM accounts WHERE token_hash = ?", (_hash(token),)).fetchone()
        if row is None:
            raise InvalidToken()
        return row

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
            account = db.execute("SELECT plan FROM accounts WHERE id = ?", (account_id,)).fetchone()
            plan = get_plan(account["plan"])
            if self._used(db, account_id, period, now) >= plan.chapters_per_month:
                raise QuotaExceeded(plan.id)
            db.execute(
                "INSERT INTO jobs (id, account_id, token_hash, period, status, cost_cap_usd, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
                (job_id, account_id, _hash(token), period, plan.max_cost_per_chapter_usd, now, now + JOB_TTL_SECONDS),
            )
        return job_id, token, plan.max_cost_per_chapter_usd

    def active_job_for_token(self, token: str) -> sqlite3.Row:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE token_hash = ?", (_hash(token),)).fetchone()
        if row is None or row["status"] != "active" or row["expires_at"] < self.now():
            raise InvalidToken()
        return row

    def begin_request(self, job_id: str) -> None:
        with self._write() as db:
            row = db.execute("SELECT cost_usd, cost_cap_usd FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row["cost_usd"] >= row["cost_cap_usd"]:
                raise QuotaExceeded("cost_cap")
            db.execute("UPDATE jobs SET requests = requests + 1 WHERE id = ?", (job_id,))

    def add_cost(self, job_id: str, cost: float, prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
        with self._write() as db:
            db.execute(
                "UPDATE jobs SET cost_usd = cost_usd + ?, prompt_tokens = prompt_tokens + ?, "
                "completion_tokens = completion_tokens + ? WHERE id = ?",
                (max(0.0, cost), max(0, prompt_tokens), max(0, completion_tokens), job_id),
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
