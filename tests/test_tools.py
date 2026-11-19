import json

import pytest

from memberpulse.approval import QueueApprover, execute_approved
from memberpulse.tools import Permission, ToolError


def test_every_schema_is_strict(toolbox):
    for s in toolbox.schemas():
        assert s["strict"] is True
        assert s["input_schema"]["additionalProperties"] is False
        assert set(s["input_schema"]["required"]) == set(s["input_schema"]["properties"])


def test_write_tools_are_flagged(toolbox):
    writes = {n for n, t in toolbox.tools.items() if t.permission is Permission.WRITE}
    assert writes == {"send_renewal_reminder", "offer_discount", "change_member_plan", "add_corporate_members"}


def test_contact_details_are_masked(toolbox):
    rows = json.loads(toolbox.call("search_members", {"name": "", "plan_id": "", "status": "active", "limit": 5}))
    for r in rows:
        assert "***@" in r["email"] and "****" in r["phone"]
    profile = json.loads(toolbox.call("get_member_profile", {"member_id": rows[0]["member_id"]}))
    assert "***@" in profile["email"]


def test_business_summary(toolbox):
    s = json.loads(toolbox.call("get_business_summary", {}))
    assert s["active_members"] > 0 and s["monthly_recurring_revenue"] > 0
    assert 0 < s["churn_rate_90_days"] < 1


def test_unknown_member_is_a_tool_error(toolbox):
    with pytest.raises(ToolError):
        toolbox.call("get_member_profile", {"member_id": "M0000"})


def test_discount_above_cap_is_rejected_before_approval(toolbox, store):
    mid = store.search_members(status="active", limit=1)[0]["id"]
    with pytest.raises(ToolError, match="30%"):
        toolbox.validate("offer_discount", {"member_id": mid, "percent": 50, "months": 1, "message": "hi"})


def test_corporate_seat_limit(toolbox):
    acct = json.loads(toolbox.call("list_corporate_accounts", {}))[0]
    too_many = [{"name": f"E{i}", "email": f"e{i}@co.example"} for i in range(acct["seats_available"] + 1)]
    with pytest.raises(ToolError, match="seats"):
        toolbox.validate("add_corporate_members", {"account_id": acct["id"], "employees": too_many})


def test_queued_action_only_runs_after_approval(toolbox, store):
    mid = store.search_members(status="active", limit=1)[0]["id"]
    decision = QueueApprover(store)("send_renewal_reminder",
                                    {"member_id": mid, "channel": "sms", "message": "See you soon!"})
    assert decision.outcome == "queued"
    assert store.outbox() == []                      # nothing sent yet
    execute_approved(store, toolbox, decision.action_id)
    assert len(store.outbox()) == 1
    assert store.action(decision.action_id)["status"] == "executed"
    with pytest.raises(ValueError):                  # can't run twice
        execute_approved(store, toolbox, decision.action_id)
