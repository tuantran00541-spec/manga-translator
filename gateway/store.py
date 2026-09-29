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

MICROS = 1_000_000  # money is kept in millionths of a US dollar, so sums never drift
JOB_TTL_SECONDS = 6 * 3600
LOGIN_CODE_TTL_SECONDS = 10 * 60
LOGIN_CODE_RESEND_SECONDS = 60
LOGIN_CODE_MAX_ATTEMPTS = 5
LOGIN_CODES_PER_DAY = 10  # with the attempts per code, about 50 guesses a day at a 6-digit code
SESSION_TTL_SECONDS = 90 * 86400
# Most one chapter may spend whatever the balance: a long webtoon chapter costs about $0.25.
JOB_COST_GUARD_USD = 2.0
# A long webtoon chapter makes about 280 A.I calls; this only stops a scripted job token.
JOB_MAX_REQUESTS = 1000
# Per network address and UTC day: login codes asked for, and new accounts made.
LOGIN_STARTS_PER_IP = 30
SIGNUPS_PER_IP = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    created_at REAL NOT NULL,
    balance_micros INTEGER NOT NULL DEFAULT 0,
    held_micros INTEGER NOT NULL DEFAULT 0,
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
    credit_micros INTEGER NOT NULL,
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
    status TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0,
    cost_cap_usd REAL NOT NULL,
    reserved_usd REAL NOT NULL DEFAULT 0,
    charged_micros INTEGER NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_account ON jobs(account_id, created_at);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    kind TEXT NOT NULL,
    delta_micros INTEGER NOT NULL,
    balance_micros INTEGER NOT NULL,
    ref TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_account ON ledger(account_id, id);
CREATE TABLE IF NOT EXISTS ip_counts (
    ip TEXT NOT NULL,
    kind TEXT NOT NULL,
    day TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ip, kind, day)
);
"""


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


def day_of(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now))


def micros(usd: float) -> int:
    return int(round(float(usd) * MICROS))


def usd(value_micros: int) -> float:
    return round(int(value_micros) / MICROS, 6)


def tombstone(email: str) -> str:
    """What a deleted account keeps of its email: enough to return its balance, not the address."""
    return "deleted:" + _hash(email.strip().lower())


class Store:
    def __init__(self, path: Path | str, clock=time.time, fee_rate: float = 0.05):
        self._path = str(path)
        self._clock = clock
        self._lock = threading.Lock()
        # What the account pays on top of the A.I cost, as a share of it.
        self.fee_rate = max(0.0, float(fee_rate))
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            columns = {row[1] for row in db.execute("PRAGMA table_info(accounts)")}
            if "plan" in columns:
                raise RuntimeError(f"{self._path} is from the monthly-plan gateway; start the wallet on a new database")
            db.executescript(_SCHEMA)
            # Nothing is in flight when the gateway starts.
            db.execute("UPDATE jobs SET reserved_usd = 0 WHERE reserved_usd != 0")
            db.execute("UPDATE accounts SET held_micros = 0 WHERE held_micros != 0")

    def now(self) -> float:
        return float(self._clock())

    def price_micros(self, cost_usd: float) -> int:
        """What an account pays for ``cost_usd`` of A.I."""
        return micros(max(0.0, cost_usd) * (1 + self.fee_rate))

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

    # -- accounts and login ----------------------------------------------------
    def _account_id_for_email(self, db, email: str) -> str:
        row = db.execute("SELECT id FROM accounts WHERE email = ?", (email,)).fetchone()
        if row is not None:
            return row["id"]
        # A deleted account comes back with its balance and history.
        old = db.execute("SELECT id FROM accounts WHERE email = ?", (tombstone(email),)).fetchone()
        if old is not None:
            db.execute("UPDATE accounts SET email = ?, deleted_at = NULL WHERE id = ?", (email, old["id"]))
            return old["id"]
        account_id = uuid.uuid4().hex
        db.execute("INSERT INTO accounts (id, email, created_at) VALUES (?, ?, ?)", (account_id, email, self.now()))
        return account_id

    def _new_session(self, db, account_id: str) -> str:
        token = "mc_" + secrets.token_urlsafe(32)
        db.execute(
            "INSERT INTO sessions (token_hash, account_id, created_at) VALUES (?, ?, ?)",
            (_hash(token), account_id, self.now()),
        )
        return token

    def create_account(self, email: str) -> tuple[str, str]:
        with self._write() as db:
            account_id = self._account_id_for_email(db, email)
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
        """Forget the email and sessions; the balance and history wait under a hash of the email."""
        with self._write() as db:
            row = db.execute("SELECT email FROM accounts WHERE id = ?", (account_id,)).fetchone()
            if row is None:
                raise KeyError(account_id)
            db.execute("DELETE FROM sessions WHERE account_id = ?", (account_id,))
            db.execute("DELETE FROM login_codes WHERE email = ?", (row["email"],))
            db.execute("UPDATE jobs SET status = 'cancelled' WHERE account_id = ? AND status = 'active'", (account_id,))
            db.execute("UPDATE accounts SET email = ?, deleted_at = ? WHERE id = ?",
                       (tombstone(row["email"]), self.now(), account_id))

    @staticmethod
    def _public(row) -> dict:
        data = dict(row)
        data["balance_usd"] = usd(row["balance_micros"])
        data["available_usd"] = usd(row["balance_micros"] - row["held_micros"])
        return data

    def account(self, account_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        if row is None:
            raise KeyError(account_id)
        return self._public(row)

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

    # -- wallet ----------------------------------------------------------------
    def _book(self, db, account_id: str, kind: str, delta: int, ref: str = "") -> int:
        """Move ``delta`` micro-dollars on the balance and record it; returns the new balance."""
        db.execute("UPDATE accounts SET balance_micros = balance_micros + ? WHERE id = ?", (delta, account_id))
        balance = int(db.execute("SELECT balance_micros FROM accounts WHERE id = ?", (account_id,)).fetchone()[0])
        db.execute("INSERT INTO ledger (account_id, kind, delta_micros, balance_micros, ref, created_at) "
                   "VALUES (?, ?, ?, ?, ?, ?)", (account_id, kind, delta, balance, ref, self.now()))
        return balance

    def adjust(self, account_id: str, usd_amount: float, note: str) -> float:
        """An admin credit or debit, recorded with its reason."""
        with self._write() as db:
            if db.execute("SELECT 1 FROM accounts WHERE id = ?", (account_id,)).fetchone() is None:
                raise KeyError(account_id)
            return usd(self._book(db, account_id, "adjust", micros(usd_amount), note[:200]))

    def ledger(self, account_id: str, limit: int = 50) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT kind, delta_micros, balance_micros, ref, created_at FROM ledger WHERE account_id = ? "
                "ORDER BY id DESC LIMIT ?", (account_id, limit)).fetchall()
        return [{"kind": row["kind"], "amount_usd": usd(row["delta_micros"]), "balance_usd": usd(row["balance_micros"]),
                 "ref": row["ref"], "created_at": row["created_at"]} for row in rows]

    def create_payment(self, provider: str, external_id: str, account_id: str, credit_usd: float,
                       amount: int, currency: str) -> str:
        payment_id = uuid.uuid4().hex
        with self._write() as db:
            db.execute(
                "INSERT INTO payments (id, provider, external_id, account_id, credit_micros, amount, currency, status, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (payment_id, provider, external_id, account_id, micros(credit_usd), amount, currency, self.now()),
            )
        return payment_id

    def complete_payment(self, provider: str, external_id: str, amount: int) -> dict | None:
        """Credit a paid top-up once; a payment of another amount is refused."""
        with self._write() as db:
            row = db.execute("SELECT * FROM payments WHERE provider = ? AND external_id = ?",
                             (provider, external_id)).fetchone()
            if row is None:
                return None
            if row["status"] != "pending":
                return {"payment_id": row["id"], "applied": False}
            if int(row["amount"]) != int(amount):
                db.execute("UPDATE payments SET status = 'amount_mismatch' WHERE id = ?", (row["id"],))
                return {"payment_id": row["id"], "applied": False, "error": "amount_mismatch"}
            balance = self._book(db, row["account_id"], "topup", int(row["credit_micros"]), f"{provider}:{row['id']}")
            db.execute("UPDATE payments SET status = 'paid', paid_at = ? WHERE id = ?", (self.now(), row["id"]))
            return {"payment_id": row["id"], "applied": True, "balance_usd": usd(balance)}

    def refund_payment(self, provider: str, external_id: str) -> bool:
        """Take back a refunded top-up; the balance may go below zero, which stops new chapters."""
        with self._write() as db:
            row = db.execute("SELECT * FROM payments WHERE provider = ? AND external_id = ?",
                             (provider, external_id)).fetchone()
            if row is None or row["status"] != "paid":
                return False
            self._book(db, row["account_id"], "refund", -int(row["credit_micros"]), f"{provider}:{row['id']}")
            db.execute("UPDATE payments SET status = 'refunded' WHERE id = ?", (row["id"],))
            return True

    def payments(self, account_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT provider, credit_micros, amount, currency, status, created_at, paid_at FROM payments "
                "WHERE account_id = ? ORDER BY created_at DESC LIMIT 50", (account_id,),
            ).fetchall()
        return [{**{k: row[k] for k in ("provider", "amount", "currency", "status", "created_at", "paid_at")},
                 "credit_usd": usd(row["credit_micros"])} for row in rows]

    # -- jobs ------------------------------------------------------------------
    def usage(self, account_id: str, since: float) -> dict:
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) AS chapters, COALESCE(SUM(charged_micros), 0) AS charged FROM jobs "
                "WHERE account_id = ? AND created_at >= ? AND requests > 0", (account_id, since)).fetchone()
        return {"chapters": int(row["chapters"]), "charged_usd": usd(row["charged"])}

    def reserve_job(self, account_id: str, min_usd: float) -> tuple[str, str, float]:
        """Open a job when the available balance covers ``min_usd``; its cap is what that balance can pay for."""
        now = self.now()
        job_id = uuid.uuid4().hex
        token = "mcj_" + secrets.token_urlsafe(32)
        with self._write() as db:
            account = db.execute("SELECT balance_micros, held_micros FROM accounts WHERE id = ?", (account_id,)).fetchone()
            available = int(account["balance_micros"]) - int(account["held_micros"])
            if available < micros(min_usd):
                raise QuotaExceeded("balance")
            cap = min(JOB_COST_GUARD_USD, available / MICROS / (1 + self.fee_rate))
            db.execute(
                "INSERT INTO jobs (id, account_id, token_hash, status, cost_cap_usd, created_at, expires_at) "
                "VALUES (?, ?, ?, 'active', ?, ?, ?)",
                (job_id, account_id, _hash(token), cap, now, now + JOB_TTL_SECONDS),
            )
        return job_id, token, round(cap, 6)

    def active_job_for_token(self, token: str) -> sqlite3.Row:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE token_hash = ?", (_hash(token),)).fetchone()
        if row is None or row["status"] != "active" or row["expires_at"] < self.now():
            raise InvalidToken()
        return row

    def begin_request(self, job_id: str, reserve_usd: float = 0.0) -> None:
        """Hold what a request may cost on the job and the balance, so requests in flight together cannot overspend."""
        reserve_usd = max(0.0, reserve_usd)
        hold = self.price_micros(reserve_usd)
        with self._write() as db:
            row = db.execute("SELECT account_id, cost_usd, reserved_usd, cost_cap_usd, requests FROM jobs WHERE id = ?",
                             (job_id,)).fetchone()
            if row["cost_usd"] + row["reserved_usd"] + reserve_usd > row["cost_cap_usd"]:
                raise QuotaExceeded("cost_cap")
            if row["requests"] >= JOB_MAX_REQUESTS:
                raise QuotaExceeded("requests")
            account = db.execute("SELECT balance_micros, held_micros FROM accounts WHERE id = ?",
                                 (row["account_id"],)).fetchone()
            if int(account["balance_micros"]) - int(account["held_micros"]) < hold:
                raise QuotaExceeded("balance")
            db.execute("UPDATE jobs SET requests = requests + 1, reserved_usd = reserved_usd + ? WHERE id = ?",
                       (reserve_usd, job_id))
            db.execute("UPDATE accounts SET held_micros = held_micros + ? WHERE id = ?", (hold, row["account_id"]))

    def add_cost(self, job_id: str, cost: float, prompt_tokens: int = 0, completion_tokens: int = 0,
                 released_usd: float = 0.0) -> float:
        """Charge a finished request at cost plus the fee and free what it held; returns the job's A.I cost."""
        charge = self.price_micros(cost)
        release = self.price_micros(released_usd)
        with self._write() as db:
            account_id = db.execute("SELECT account_id FROM jobs WHERE id = ?", (job_id,)).fetchone()["account_id"]
            db.execute(
                "UPDATE jobs SET cost_usd = cost_usd + ?, prompt_tokens = prompt_tokens + ?, "
                "completion_tokens = completion_tokens + ?, reserved_usd = MAX(0, reserved_usd - ?), "
                "charged_micros = charged_micros + ? WHERE id = ?",
                (max(0.0, cost), max(0, prompt_tokens), max(0, completion_tokens), max(0.0, released_usd), charge, job_id),
            )
            db.execute("UPDATE accounts SET balance_micros = balance_micros - ?, "
                       "held_micros = MAX(0, held_micros - ?) WHERE id = ?", (charge, release, account_id))
            return float(db.execute("SELECT cost_usd FROM jobs WHERE id = ?", (job_id,)).fetchone()[0])

    def _close_job(self, db, row, status: str) -> None:
        db.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, row["id"]))
        if int(row["charged_micros"]):
            # The balance already moved request by request; the ledger gets one line per chapter.
            balance = int(db.execute("SELECT balance_micros FROM accounts WHERE id = ?",
                                     (row["account_id"],)).fetchone()[0])
            db.execute("INSERT INTO ledger (account_id, kind, delta_micros, balance_micros, ref, created_at) "
                       "VALUES (?, 'chapter', ?, ?, ?, ?)",
                       (row["account_id"], -int(row["charged_micros"]), balance, row["id"], self.now()))

    def finish_job(self, job_id: str, outcome: str) -> dict:
        with self._write() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row["status"] == "active":
                self._close_job(db, row, outcome)
        return {
            "job_id": job_id, "status": outcome, "cost_usd": round(float(row["cost_usd"]), 6),
            "charged_usd": usd(row["charged_micros"]), "requests": int(row["requests"]),
            "prompt_tokens": int(row["prompt_tokens"]), "completion_tokens": int(row["completion_tokens"]),
        }

    # -- operations ------------------------------------------------------------
    def purge(self) -> dict:
        """Drop spent sessions, codes and counters, and close jobs past their time."""
        now = self.now()
        with self._write() as db:
            stale = db.execute("SELECT * FROM jobs WHERE status = 'active' AND expires_at < ?", (now,)).fetchall()
            for row in stale:
                self._close_job(db, row, "expired")
            counts = {
                "sessions": db.execute("DELETE FROM sessions WHERE created_at < ?",
                                       (now - SESSION_TTL_SECONDS,)).rowcount,
                "login_codes": db.execute("DELETE FROM login_codes WHERE expires_at < ? AND day < ?",
                                          (now, day_of(now - 86400))).rowcount,
                "ip_counts": db.execute("DELETE FROM ip_counts WHERE day < ?", (day_of(now - 86400),)).rowcount,
                "jobs": len(stale),
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

    def stats(self, days: int = 30) -> dict:
        """Money in and out over the last ``days``: top-ups, what chapters were charged, and what the A.I cost."""
        since = self.now() - days * 86400
        with self._connect() as db:
            accounts = db.execute("SELECT COUNT(*), COALESCE(SUM(balance_micros), 0) FROM accounts "
                                  "WHERE deleted_at IS NULL").fetchone()
            jobs = db.execute(
                "SELECT COUNT(*) AS jobs, COALESCE(SUM(cost_usd), 0) AS cost, COALESCE(SUM(charged_micros), 0) AS charged, "
                "COALESCE(SUM(requests), 0) AS requests FROM jobs WHERE created_at >= ?", (since,)).fetchone()
            topups = db.execute(
                "SELECT currency, COALESCE(SUM(amount), 0) AS amount, COALESCE(SUM(credit_micros), 0) AS credit, "
                "COUNT(*) AS count FROM payments WHERE status = 'paid' AND paid_at >= ? GROUP BY currency",
                (since,)).fetchall()
        charged, cost = usd(jobs["charged"]), round(float(jobs["cost"]), 6)
        return {
            "days": days,
            "accounts": int(accounts[0]),
            # Balances are owed to users: money taken in but not yet spent.
            "balances_owed_usd": usd(accounts[1]),
            "jobs": int(jobs["jobs"]),
            "requests": int(jobs["requests"]),
            "ai_cost_usd": cost,
            "charged_usd": charged,
            "margin_usd": round(charged - cost, 6),
            "topups": {row["currency"]: {"amount": int(row["amount"]), "credit_usd": usd(row["credit"]),
                                         "count": int(row["count"])} for row in topups},
        }
