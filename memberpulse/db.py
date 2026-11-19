"""Small data-access layer over SQLite.

The agent never writes SQL. It can only call the functions exposed as tools,
and every query here is parameterised.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .data import TODAY


class Store:
    def __init__(self, db_path: str | Path, today: date = TODAY):
        self.db_path = str(db_path)
        self.today = today

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    # ---------- members & plans ----------

    def plans(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM plans ORDER BY tier")

    def plan(self, plan_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM plans WHERE id = ?", (plan_id,))
        return rows[0] if rows else None

    def member(self, member_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM members WHERE id = ?", (member_id,))
        return rows[0] if rows else None

    def search_members(self, name: str | None = None, plan_id: str | None = None,
                       status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        sql = "SELECT * FROM members WHERE 1=1"
        params: list[Any] = []
        if name:
            sql += " AND name LIKE ?"
            params.append(f"%{name}%")
        if plan_id:
            sql += " AND plan_id = ?"
            params.append(plan_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY id LIMIT ?"
        params.append(limit)
        return self.query(sql, tuple(params))

    def member_features(self) -> list[dict[str, Any]]:
        """One row per member with the behavioural features the churn model uses."""
        return self.query(
            """
            SELECT m.id, m.status, m.plan_id, p.tier, p.price_monthly, m.auto_renew,
                   m.has_wallet_pass, m.age, m.payment_method,
                   (m.corporate_account_id IS NOT NULL) AS is_corporate,
                   julianday(COALESCE(m.cancelled_on, ?)) - julianday(m.join_date) AS tenure_days,
                   (SELECT COUNT(*) FROM visits v WHERE v.member_id = m.id) AS visits_90d,
                   (SELECT COUNT(*) FROM visits v WHERE v.member_id = m.id
                        AND v.visit_date >= date(COALESCE(m.cancelled_on, ?), '-30 days')) AS visits_30d,
                   (SELECT MAX(visit_date) FROM visits v WHERE v.member_id = m.id) AS last_visit,
                   (SELECT COUNT(*) FROM payments pa WHERE pa.member_id = m.id
                        AND pa.status = 'failed') AS failed_payments
            FROM members m JOIN plans p ON p.id = m.plan_id
            """,
            (self.today.isoformat(), self.today.isoformat()),
        )

    def visit_count(self, member_id: str, days: int) -> int:
        since = (self.today - timedelta(days=days)).isoformat()
        rows = self.query("SELECT COUNT(*) AS n FROM visits WHERE member_id = ? AND visit_date >= ?",
                          (member_id, since))
        return rows[0]["n"]

    def upcoming_renewals(self, days_ahead: int) -> list[dict[str, Any]]:
        end = (self.today + timedelta(days=days_ahead)).isoformat()
        return self.query(
            "SELECT * FROM members WHERE status = 'active' AND renewal_date >= ? AND renewal_date <= ? "
            "ORDER BY renewal_date",
            (self.today.isoformat(), end),
        )

    def corporate_accounts(self) -> list[dict[str, Any]]:
        return self.query(
            """
            SELECT c.*, p.name AS plan_name,
                   (SELECT COUNT(*) FROM members m WHERE m.corporate_account_id = c.id
                        AND m.status = 'active') AS seats_used
            FROM corporate_accounts c JOIN plans p ON p.id = c.plan_id ORDER BY c.company
            """
        )

    # ---------- analytics ----------

    def monthly_revenue(self, months: int = 6) -> list[dict[str, Any]]:
        since = (self.today.replace(day=1) - timedelta(days=31 * (months - 1))).replace(day=1)
        return self.query(
            """
            SELECT substr(paid_on, 1, 7) AS month,
                   ROUND(SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END), 2) AS collected,
                   ROUND(SUM(CASE WHEN status = 'failed' THEN amount ELSE 0 END), 2) AS failed,
                   COUNT(DISTINCT member_id) AS paying_members
            FROM payments WHERE paid_on >= ? AND paid_on < ?
            GROUP BY month ORDER BY month
            """,
            (since.isoformat(), self.today.isoformat()),
        )

    # ---------- actions (human-approval queue) ----------

    def create_action(self, tool: str, payload: dict[str, Any]) -> int:
        with self.connect() as con:
            cur = con.execute(
                "INSERT INTO actions (tool, payload, status, created_at) VALUES (?, ?, 'pending', ?)",
                (tool, json.dumps(payload), datetime.now().isoformat(timespec="seconds")),
            )
            return int(cur.lastrowid)

    def actions(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self.query("SELECT * FROM actions WHERE status = ? ORDER BY id", (status,))
        else:
            rows = self.query("SELECT * FROM actions ORDER BY id")
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def action(self, action_id: int) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM actions WHERE id = ?", (action_id,))
        if not rows:
            return None
        rows[0]["payload"] = json.loads(rows[0]["payload"])
        return rows[0]

    def set_action_status(self, action_id: int, status: str, result: str | None = None) -> None:
        with self.connect() as con:
            con.execute("UPDATE actions SET status = ?, decided_at = ?, result = ? WHERE id = ?",
                        (status, datetime.now().isoformat(timespec="seconds"), result, action_id))

    # ---------- side effects (only reached after approval) ----------

    def send_message(self, member_id: str, channel: str, message: str) -> int:
        with self.connect() as con:
            cur = con.execute("INSERT INTO outbox (member_id, channel, message, sent_at) VALUES (?,?,?,?)",
                              (member_id, channel, message, datetime.now().isoformat(timespec="seconds")))
            return int(cur.lastrowid)

    def change_plan(self, member_id: str, plan_id: str) -> None:
        with self.connect() as con:
            con.execute("UPDATE members SET plan_id = ? WHERE id = ?", (plan_id, member_id))

    def outbox(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM outbox ORDER BY id")
