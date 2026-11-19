import os

import pytest
from fastapi.testclient import TestClient

from memberpulse import api
from memberpulse.app import App
from memberpulse.retention import choose_offer, run_renewal_campaign
from memberpulse.tools import Toolbox


def test_offer_policy():
    assert choose_offer("high", {"action": "keep"}).discount_percent == 15
    assert choose_offer("high", {"action": "downgrade", "suggested_plan_id": "plus"}).kind == "plan_change"
    assert choose_offer("medium", {"action": "keep"}).discount_percent == 10
    assert choose_offer("low", {"action": "keep"}).kind == "reminder"


def test_campaign_queues_drafts_and_sends_nothing(store, churn):
    out = run_renewal_campaign(store, churn, days_ahead=7, client=None, limit=10)
    assert out["drafts_queued"] == min(10, out["renewals_found"]) > 0
    probs = [d["churn_probability"] for d in out["drafts"]]
    assert probs == sorted(probs, reverse=True)          # riskiest first
    assert all(d["message_source"] == "template" for d in out["drafts"])
    assert len(store.actions("pending")) == out["drafts_queued"]
    assert store.outbox() == []


def test_campaign_falls_back_to_template_when_llm_fails(store, churn):
    class Broken:
        class beta:
            class messages:
                @staticmethod
                def parse(**_):
                    raise RuntimeError("API down")
    out = run_renewal_campaign(store, churn, days_ahead=7, client=Broken(), limit=3)
    assert out["drafts_queued"] == 3
    assert all(d["message_source"] == "template" for d in out["drafts"])


@pytest.fixture
def client(store, churn, monkeypatch):
    monkeypatch.setattr(api, "get_app", lambda: App(store, churn, Toolbox(store, churn)))
    monkeypatch.delenv("MERCHANT_API_KEY", raising=False)
    return TestClient(api.app)


def test_api_health_and_summary(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/summary").json()["active_members"] > 0


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("MERCHANT_API_KEY", "secret")
    assert client.get("/summary").status_code == 401
    assert client.get("/summary", headers={"X-API-Key": "secret"}).status_code == 200


def test_api_campaign_approve_flow(client, store):
    out = client.post("/campaigns/renewals", json={"days_ahead": 7, "limit": 2, "use_llm": False}).json()
    first = out["drafts"][0]["action_id"]
    assert len(client.get("/actions").json()) == 2
    assert client.post(f"/actions/{first}/approve").json()["status"] == "executed"
    assert client.post(f"/actions/{first}/approve").status_code == 409
    assert len(store.outbox()) == 1
    second = out["drafts"][1]["action_id"]
    assert client.post(f"/actions/{second}/reject").json()["status"] == "rejected"
    assert client.get("/actions").json() == []


def test_api_recommendation_404(client):
    assert client.get("/members/NOPE/recommendation").status_code == 404


def test_dashboard_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "MemberPulse AI" in r.text


def test_dashboard_data_endpoints(client):
    assert len(client.get("/revenue?months=6").json()["months"]) == 6
    assert isinstance(client.get("/renewals?days_ahead=7").json(), list)
    assert client.get("/corporate-accounts").json()[0]["seats_available"] >= 0


def test_chat_is_disabled_without_api_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.get("/health").json()["chat_enabled"] is False
    assert client.post("/chat", json={"message": "hi"}).status_code == 503


def test_chat_rate_limit(client, monkeypatch):
    monkeypatch.setattr(api, "CHAT_LIMIT_PER_HOUR", 2)
    api._chat_hits.clear()
    api._rate_limit("1.2.3.4"); api._rate_limit("1.2.3.4")
    with pytest.raises(api.HTTPException):
        api._rate_limit("1.2.3.4")


def test_demo_reset_clears_queue(client):
    client.post("/campaigns/renewals", json={"days_ahead": 7, "limit": 2, "use_llm": False})
    assert client.get("/actions").json()
    client.post("/demo/reset")
    assert client.get("/actions").json() == []
