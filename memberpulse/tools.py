"""Tools the AI agent can call, with permissions.

Every tool has:
  * a strict JSON schema (Claude's arguments are guaranteed to match it),
  * a permission level: READ tools run immediately, WRITE tools never run
    until a human approves them,
  * business-rule validation in Python, so the merchant is only ever asked to
    approve actions that are actually allowed.

Personal data (emails, phone numbers) is masked before it is sent to the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .churn import ChurnModel
from .db import Store
from .recommender import recommend_plan

MAX_DISCOUNT_PERCENT = 30
CHANNELS = ["sms", "email", "whatsapp", "wallet_push"]
PLAN_IDS = ["basic", "standard", "plus", "premium"]


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"  # needs human approval


class ToolError(Exception):
    """A problem the model should see and can recover from (bad id, rule violated...)."""


def mask_email(email: str) -> str:
    user, _, domain = email.partition("@")
    return f"{user[:1]}***@{domain}"


def mask_phone(phone: str) -> str:
    return phone[:5] + "****" + phone[-2:]


def public_member(m: dict[str, Any]) -> dict[str, Any]:
    """Member record safe to show the model: no raw contact details."""
    return {
        "member_id": m["id"], "name": m["name"], "email": mask_email(m["email"]),
        "phone": mask_phone(m["phone"]), "plan_id": m["plan_id"], "status": m["status"],
        "join_date": m["join_date"], "renewal_date": m["renewal_date"],
        "auto_renew": bool(m["auto_renew"]), "payment_method": m["payment_method"],
        "has_wallet_pass": bool(m["has_wallet_pass"]), "corporate_account_id": m["corporate_account_id"],
    }


@dataclass
class Tool:
    name: str
    description: str
    properties: dict[str, Any]
    permission: Permission
    run: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": self.properties,
                "required": list(self.properties),
                "additionalProperties": False,
            },
        }


class Toolbox:
    def __init__(self, store: Store, churn: ChurnModel):
        self.store = store
        self.churn = churn
        self.tools: dict[str, Tool] = {t.name: t for t in self._build()}

    # ------------------------------------------------------------------ helpers

    def _active_member(self, member_id: str) -> dict[str, Any]:
        m = self.store.member(member_id)
        if not m:
            raise ToolError(f"No member with id {member_id}. Use search_members to find the id.")
        if m["status"] != "active":
            raise ToolError(f"Member {member_id} is not active (status: {m['status']}).")
        return m

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def permission(self, name: str) -> Permission:
        return self.tools[name].permission

    def validate(self, name: str, args: dict[str, Any]) -> None:
        """Business-rule checks for WRITE tools, run *before* asking for approval."""
        validator = getattr(self, f"_validate_{name}", None)
        if validator:
            validator(**args)

    def call(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.tools:
            raise ToolError(f"Unknown tool {name}")
        return json.dumps(self.tools[name].run(**args), default=str)

    # ------------------------------------------------------------------ READ tools

    def business_summary(self) -> dict[str, Any]:
        s = self.store
        counts = {r["status"]: r["n"] for r in s.query("SELECT status, COUNT(*) n FROM members GROUP BY status")}
        mrr = s.query(
            """SELECT ROUND(SUM(p.price_monthly * (1 - COALESCE(c.discount_percent, 0) / 100.0)), 2) AS mrr
               FROM members m JOIN plans p ON p.id = m.plan_id
               LEFT JOIN corporate_accounts c ON c.id = m.corporate_account_id
               WHERE m.status = 'active'""")[0]["mrr"]
        cancelled_90 = s.query("SELECT COUNT(*) n FROM members WHERE cancelled_on >= date(?, '-90 days')",
                               (s.today.isoformat(),))[0]["n"]
        by_plan = s.query("""SELECT p.name AS plan, COUNT(*) AS active_members FROM members m
                             JOIN plans p ON p.id = m.plan_id WHERE m.status = 'active'
                             GROUP BY p.name ORDER BY active_members DESC""")
        scores = self.churn.score_all(s)
        at_risk = [x for x in scores if x.level == "high"]
        return {
            "as_of": s.today.isoformat(),
            "active_members": counts.get("active", 0),
            "monthly_recurring_revenue": mrr,
            "cancellations_last_90_days": cancelled_90,
            "churn_rate_90_days": round(cancelled_90 / max(1, counts.get("active", 0) + cancelled_90), 3),
            "active_members_by_plan": by_plan,
            "high_churn_risk_members": len(at_risk),
            "revenue_at_risk_monthly": round(sum(
                s.plan(s.member(x.member_id)["plan_id"])["price_monthly"] for x in at_risk), 2),
        }

    def revenue_trend(self, months: int) -> dict[str, Any]:
        rows = self.store.monthly_revenue(max(1, min(months, 12)))
        return {"months": rows, "note": "collected = successful payments, failed = declined payments"}

    def search_members(self, name: str, plan_id: str, status: str, limit: int) -> list[dict[str, Any]]:
        rows = self.store.search_members(name or None, plan_id or None, status or None, max(1, min(limit, 50)))
        return [public_member(m) for m in rows]

    def member_profile(self, member_id: str) -> dict[str, Any]:
        m = self.store.member(member_id)
        if not m:
            raise ToolError(f"No member with id {member_id}.")
        score = self.churn.score_member(self.store, member_id)
        out = public_member(m)
        out["visits_last_30_days"] = self.store.visit_count(member_id, 30)
        out["failed_payments"] = self.store.query(
            "SELECT COUNT(*) n FROM payments WHERE member_id = ? AND status = 'failed'", (member_id,))[0]["n"]
        out["churn"] = score.as_dict() if score else None
        return out

    def at_risk_members(self, limit: int, risk_level: str) -> list[dict[str, Any]]:
        scores = [s for s in self.churn.score_all(self.store) if risk_level == "any" or s.level == risk_level]
        out = []
        for s in scores[: max(1, min(limit, 50))]:
            m = self.store.member(s.member_id)
            out.append({**s.as_dict(), "name": m["name"], "plan_id": m["plan_id"],
                        "renewal_date": m["renewal_date"]})
        return out

    def upcoming_renewals(self, days_ahead: int, risk_level: str) -> list[dict[str, Any]]:
        renewals = self.store.upcoming_renewals(max(1, min(days_ahead, 30)))
        scores = {s.member_id: s for s in self.churn.score_all(self.store)}
        out = []
        for m in renewals:
            s = scores.get(m["id"])
            if risk_level != "any" and (s is None or s.level != risk_level):
                continue
            out.append({"member_id": m["id"], "name": m["name"], "plan_id": m["plan_id"],
                        "renewal_date": m["renewal_date"], "auto_renew": bool(m["auto_renew"]),
                        "churn_risk": s.level if s else None,
                        "churn_probability": round(s.probability, 3) if s else None})
        return out

    def plan_recommendation(self, member_id: str) -> dict[str, Any]:
        self._active_member(member_id)
        return recommend_plan(self.store, member_id)

    def list_plans(self) -> list[dict[str, Any]]:
        return self.store.plans()

    def corporate_accounts(self) -> list[dict[str, Any]]:
        rows = self.store.corporate_accounts()
        for r in rows:
            r["contact_email"] = mask_email(r["contact_email"])
            r["seats_available"] = r["seats_purchased"] - r["seats_used"]
        return rows

    # ------------------------------------------------------------------ WRITE tools (approval required)

    def _validate_send_renewal_reminder(self, member_id: str, channel: str, message: str) -> None:
        self._active_member(member_id)
        if not message.strip():
            raise ToolError("Message is empty.")
        if len(message) > 480:
            raise ToolError("Message is longer than 480 characters; shorten it.")

    def send_renewal_reminder(self, member_id: str, channel: str, message: str) -> dict[str, Any]:
        self._validate_send_renewal_reminder(member_id, channel, message)
        outbox_id = self.store.send_message(member_id, channel, message)
        return {"sent": True, "outbox_id": outbox_id, "member_id": member_id, "channel": channel}

    def _validate_offer_discount(self, member_id: str, percent: int, months: int, message: str) -> None:
        self._active_member(member_id)
        if not 5 <= percent <= MAX_DISCOUNT_PERCENT:
            raise ToolError(f"Discount must be between 5% and {MAX_DISCOUNT_PERCENT}%.")
        if not 1 <= months <= 3:
            raise ToolError("Discounts can last 1 to 3 months.")
        if len(message) > 480:
            raise ToolError("Message is longer than 480 characters; shorten it.")

    def offer_discount(self, member_id: str, percent: int, months: int, message: str) -> dict[str, Any]:
        self._validate_offer_discount(member_id, percent, months, message)
        outbox_id = self.store.send_message(member_id, "sms", message)
        return {"offered": True, "member_id": member_id, "percent": percent, "months": months,
                "outbox_id": outbox_id}

    def _validate_change_member_plan(self, member_id: str, new_plan_id: str, reason: str) -> None:
        m = self._active_member(member_id)
        if m["corporate_account_id"]:
            raise ToolError("This member is on a corporate plan; change it through the company account.")
        if m["plan_id"] == new_plan_id:
            raise ToolError(f"Member is already on {new_plan_id}.")

    def change_member_plan(self, member_id: str, new_plan_id: str, reason: str) -> dict[str, Any]:
        self._validate_change_member_plan(member_id, new_plan_id, reason)
        old = self.store.member(member_id)["plan_id"]
        self.store.change_plan(member_id, new_plan_id)
        return {"changed": True, "member_id": member_id, "from": old, "to": new_plan_id}

    def _validate_add_corporate_members(self, account_id: str, employees: list[dict[str, str]]) -> None:
        accounts = {a["id"]: a for a in self.store.corporate_accounts()}
        if account_id not in accounts:
            raise ToolError(f"No corporate account {account_id}.")
        free = accounts[account_id]["seats_purchased"] - accounts[account_id]["seats_used"]
        if len(employees) > free:
            raise ToolError(f"Only {free} seats left on this account; {len(employees)} requested.")
        if not employees:
            raise ToolError("No employees given.")

    def add_corporate_members(self, account_id: str, employees: list[dict[str, str]]) -> dict[str, Any]:
        self._validate_add_corporate_members(account_id, employees)
        acct = [a for a in self.store.corporate_accounts() if a["id"] == account_id][0]
        created = []
        with self.store.connect() as con:
            next_n = con.execute("SELECT COUNT(*) FROM members").fetchone()[0] + 1
            for i, e in enumerate(employees):
                mid = f"M{next_n + i:04d}"
                today = self.store.today.isoformat()
                con.execute(
                    "INSERT INTO members VALUES (?,?,?,?,?,?,?,date(?, '+30 days'),'active',NULL,1,'card',0,30)",
                    (mid, e["name"], e["email"], "+250700000000", acct["plan_id"], account_id, today, today))
                created.append(mid)
        return {"created_member_ids": created, "account_id": account_id}

    # ------------------------------------------------------------------ registry

    def _build(self) -> list[Tool]:
        R, W = Permission.READ, Permission.WRITE
        risk = {"type": "string", "enum": ["high", "medium", "low", "any"]}
        mid = {"type": "string", "description": "Member id, e.g. M0042"}
        msg = {"type": "string", "description": "Message to the member, max 480 characters, friendly tone"}
        return [
            Tool("get_business_summary",
                 "Headline numbers for the business: active members, monthly recurring revenue, "
                 "recent cancellations, churn rate, members per plan and revenue at risk.",
                 {}, R, self.business_summary),
            Tool("get_revenue_trend", "Collected and failed payments per month for the last N months.",
                 {"months": {"type": "integer", "description": "1 to 12"}}, R, self.revenue_trend),
            Tool("search_members",
                 "Find members. Use empty strings for filters you don't need. Contact details are masked.",
                 {"name": {"type": "string", "description": "Part of the name, or empty"},
                  "plan_id": {"type": "string", "enum": PLAN_IDS + [""]},
                  "status": {"type": "string", "enum": ["active", "cancelled", ""]},
                  "limit": {"type": "integer"}}, R, self.search_members),
            Tool("get_member_profile",
                 "Full profile for one member, including visits, failed payments and churn risk with reasons.",
                 {"member_id": mid}, R, self.member_profile),
            Tool("list_at_risk_members",
                 "Active members ranked by predicted probability of cancelling (ML model), with reasons.",
                 {"limit": {"type": "integer"}, "risk_level": risk}, R, self.at_risk_members),
            Tool("list_upcoming_renewals",
                 "Active members whose renewal date falls in the next N days (max 30), with churn risk.",
                 {"days_ahead": {"type": "integer"}, "risk_level": risk}, R, self.upcoming_renewals),
            Tool("recommend_plan",
                 "Recommend keep / upgrade / downgrade for a member based on usage and similar members.",
                 {"member_id": mid}, R, self.plan_recommendation),
            Tool("list_plans", "All membership plans with prices and perks.", {}, R, self.list_plans),
            Tool("list_corporate_accounts",
                 "Corporate (employee benefit) accounts with seats purchased, used and available.",
                 {}, R, self.corporate_accounts),
            Tool("send_renewal_reminder",
                 "Send a renewal reminder to a member. REQUIRES MERCHANT APPROVAL before it is sent.",
                 {"member_id": mid, "channel": {"type": "string", "enum": CHANNELS}, "message": msg},
                 W, self.send_renewal_reminder),
            Tool("offer_discount",
                 f"Offer a retention discount (5-{MAX_DISCOUNT_PERCENT}%, 1-3 months) by SMS. "
                 "REQUIRES MERCHANT APPROVAL.",
                 {"member_id": mid, "percent": {"type": "integer"}, "months": {"type": "integer"},
                  "message": msg}, W, self.offer_discount),
            Tool("change_member_plan", "Move a member to another plan. REQUIRES MERCHANT APPROVAL.",
                 {"member_id": mid, "new_plan_id": {"type": "string", "enum": PLAN_IDS},
                  "reason": {"type": "string"}}, W, self.change_member_plan),
            Tool("add_corporate_members",
                 "Bulk-add employees to a corporate account. REQUIRES MERCHANT APPROVAL.",
                 {"account_id": {"type": "string"},
                  "employees": {"type": "array", "items": {
                      "type": "object",
                      "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                      "required": ["name", "email"], "additionalProperties": False}}},
                 W, self.add_corporate_members),
        ]
