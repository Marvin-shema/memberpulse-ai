"""REST API: how a merchant dashboard or backend would integrate MemberPulse.

Run with:  uvicorn memberpulse.api:app --reload
Docs at:   http://localhost:8000/docs

If MERCHANT_API_KEY is set, every request must send it in the X-API-Key header.
Write actions proposed by the agent are *queued*; the dashboard approves them
through /actions/{id}/approve.
"""

from __future__ import annotations

import hmac
import os
import time
import uuid
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import MerchantAgent
from .app import App, load_app
from .approval import QueueApprover, execute_approved
from .recommender import recommend_plan
from .retention import run_renewal_campaign

app = FastAPI(title="MemberPulse AI", version="1.0.0",
              description="AI assistant, churn prediction and retention automation for membership businesses.")

STATIC_DIR = Path(__file__).parent / "static"

# In-memory conversation store. A production deployment would use Redis or the database.
_conversations: dict[str, list[dict[str, Any]]] = {}

# Simple per-IP rate limit for the chat endpoint, so a public demo can't run up an API bill.
CHAT_LIMIT_PER_HOUR = int(os.getenv("CHAT_LIMIT_PER_HOUR", "20"))
_chat_hits: dict[str, deque] = defaultdict(deque)


def _rate_limit(ip: str) -> None:
    now, hits = time.time(), _chat_hits[ip]
    while hits and now - hits[0] > 3600:
        hits.popleft()
    if len(hits) >= CHAT_LIMIT_PER_HOUR:
        raise HTTPException(429, "Chat limit reached for this hour. Please try again later.")
    hits.append(now)


@lru_cache
def get_app() -> App:
    return load_app()


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("MERCHANT_API_KEY")
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class CampaignRequest(BaseModel):
    days_ahead: int = Field(default=7, ge=1, le=30)
    limit: int = Field(default=25, ge=1, le=100)
    use_llm: bool = True


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    a = get_app()
    return {"status": "ok", "churn_model_auc": round(a.churn.auc or 0, 3),
            "chat_enabled": bool(os.getenv("ANTHROPIC_API_KEY")),
            "api_key_required": bool(os.getenv("MERCHANT_API_KEY"))}


@app.post("/chat", dependencies=[Depends(require_key)])
def chat(req: ChatRequest, request: Request) -> dict[str, Any]:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "The AI chat is turned off in this demo because no Anthropic API key is set.")
    _rate_limit(request.client.host if request.client else "unknown")
    a = get_app()
    conv_id = req.conversation_id or str(uuid.uuid4())
    history = _conversations.get(conv_id, [])
    try:
        agent = MerchantAgent(a.toolbox, approver=QueueApprover(a.store))
    except Exception as e:
        raise HTTPException(503, f"AI assistant unavailable: {e}") from e
    result = agent.run(req.message, history=history)
    # Keep plain-text turns only, so stored history stays small and model-agnostic.
    _conversations[conv_id] = history + [{"role": "user", "content": req.message},
                                         {"role": "assistant", "content": result.answer}]
    return {"conversation_id": conv_id, **result.as_dict()}


@app.get("/summary", dependencies=[Depends(require_key)])
def summary() -> dict[str, Any]:
    return get_app().toolbox.business_summary()


@app.get("/revenue", dependencies=[Depends(require_key)])
def revenue(months: int = 6) -> dict[str, Any]:
    return get_app().toolbox.revenue_trend(months)


@app.get("/renewals", dependencies=[Depends(require_key)])
def renewals(days_ahead: int = 7, risk_level: str = "any") -> list[dict[str, Any]]:
    return get_app().toolbox.upcoming_renewals(days_ahead, risk_level)


@app.get("/corporate-accounts", dependencies=[Depends(require_key)])
def corporate_accounts() -> list[dict[str, Any]]:
    return get_app().toolbox.corporate_accounts()


@app.post("/demo/reset", dependencies=[Depends(require_key)])
def reset_demo() -> dict[str, Any]:
    """Clear the approval queue and outbox so a fresh visitor sees a clean demo."""
    a = get_app()
    with a.store.connect() as con:
        con.execute("DELETE FROM actions")
        con.execute("DELETE FROM outbox")
    return {"status": "reset"}


@app.get("/members/at-risk", dependencies=[Depends(require_key)])
def at_risk(limit: int = 20, risk_level: str = "high") -> list[dict[str, Any]]:
    return get_app().toolbox.at_risk_members(limit, risk_level)


@app.get("/members/{member_id}/recommendation", dependencies=[Depends(require_key)])
def recommendation(member_id: str) -> dict[str, Any]:
    try:
        return recommend_plan(get_app().store, member_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/campaigns/renewals", dependencies=[Depends(require_key)])
def renewal_campaign(req: CampaignRequest) -> dict[str, Any]:
    a = get_app()
    client = None
    if req.use_llm and os.getenv("ANTHROPIC_API_KEY"):
        from .agent import make_client
        client = make_client()
    return run_renewal_campaign(a.store, a.churn, req.days_ahead, client=client, limit=req.limit)


@app.get("/actions", dependencies=[Depends(require_key)])
def list_actions(status: str | None = "pending") -> list[dict[str, Any]]:
    return get_app().store.actions(status)


@app.post("/actions/{action_id}/approve", dependencies=[Depends(require_key)])
def approve(action_id: int) -> dict[str, Any]:
    a = get_app()
    try:
        return {"action_id": action_id, "status": "executed",
                "result": execute_approved(a.store, a.toolbox, action_id)}
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/actions/{action_id}/reject", dependencies=[Depends(require_key)])
def reject(action_id: int) -> dict[str, Any]:
    a = get_app()
    action = a.store.action(action_id)
    if not action:
        raise HTTPException(404, f"No action {action_id}")
    if action["status"] != "pending":
        raise HTTPException(409, f"Action {action_id} is already {action['status']}")
    a.store.set_action_status(action_id, "rejected")
    return {"action_id": action_id, "status": "rejected"}
