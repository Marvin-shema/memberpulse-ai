"""Automated renewal & retention workflow.

A fixed pipeline (no open-ended agent needed):

    upcoming renewals -> churn score -> plan recommendation -> offer policy
        -> personalised message (Claude structured output, or a template)
        -> queued for merchant approval

Using a deterministic workflow here keeps cost low and behaviour predictable;
the LLM is only used for the part it is good at: writing a personal message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from .agent import DEFAULT_MODEL
from .churn import ChurnModel
from .db import Store
from .recommender import recommend_plan
from .tools import Toolbox


class RenewalMessage(BaseModel):
    """Structured output we ask Claude for. Validated by the SDK."""
    message: str = Field(description="SMS/WhatsApp text to the member, under 320 characters")
    channel: Literal["sms", "email", "whatsapp", "wallet_push"]


@dataclass
class Offer:
    kind: Literal["reminder", "discount", "plan_change"]
    discount_percent: int = 0
    suggested_plan: str | None = None


def choose_offer(risk: str, recommendation: dict[str, Any]) -> Offer:
    """Business policy: who gets what. Kept in plain code so the merchant can audit it."""
    if risk == "high" and recommendation["action"] == "downgrade":
        return Offer("plan_change", suggested_plan=recommendation["suggested_plan_id"])
    if risk == "high":
        return Offer("discount", discount_percent=15)
    if risk == "medium":
        return Offer("discount", discount_percent=10)
    if recommendation["action"] == "upgrade":
        return Offer("plan_change", suggested_plan=recommendation["suggested_plan_id"])
    return Offer("reminder")


def template_message(first_name: str, renewal_date: str, offer: Offer, plan_name: str) -> RenewalMessage:
    if offer.kind == "discount":
        text = (f"Hi {first_name}, your Lion Peak membership renews on {renewal_date}. "
                f"We'd love to keep training with you: renew now and get {offer.discount_percent}% off "
                f"your next month. Reply YES to claim.")
    elif offer.kind == "plan_change":
        text = (f"Hi {first_name}, your membership renews on {renewal_date}. Based on how you use the gym, "
                f"the {plan_name} plan could suit you better. Reply PLAN and we'll switch you.")
    else:
        text = f"Hi {first_name}, a friendly reminder that your Lion Peak membership renews on {renewal_date}. See you soon!"
    return RenewalMessage(message=text, channel="sms")


def llm_message(client, member: dict[str, Any], renewal_date: str, offer: Offer, plan_name: str,
                reasons: list[str], model: str = DEFAULT_MODEL) -> RenewalMessage:
    prompt = (
        "Write one short, warm renewal message from Lion Peak Fitness to a member.\n"
        f"First name: {member['name'].split()[0]}\nRenewal date: {renewal_date}\n"
        f"Offer: {offer.kind}" + (f" {offer.discount_percent}% off next month" if offer.discount_percent else "")
        + (f" (suggest switching to the {plan_name} plan)" if offer.kind == "plan_change" else "") + "\n"
        f"Why they may be drifting (do NOT mention these directly, just use them to set the tone): "
        f"{', '.join(reasons)}\n"
        f"Preferred channel: whatsapp if they use mobile money, otherwise sms. "
        f"Their payment method: {member['payment_method']}.\n"
        "No emojis overload, no pressure tactics, under 320 characters."
    )
    response = client.beta.messages.parse(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        output_format=RenewalMessage,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError("model did not return a message")
    return response.parsed_output


def run_renewal_campaign(store: Store, churn: ChurnModel, days_ahead: int = 7,
                         client=None, limit: int = 25) -> dict[str, Any]:
    """Draft personalised renewal messages and queue them for approval. Sends nothing."""
    toolbox = Toolbox(store, churn)
    scores = {s.member_id: s for s in churn.score_all(store)}
    plans = {p["id"]: p for p in store.plans()}
    drafts = []

    candidates = [m for m in store.upcoming_renewals(days_ahead) if m["id"] in scores]
    # Most at-risk first, so the merchant reviews the important ones at the top.
    candidates.sort(key=lambda m: scores[m["id"]].probability, reverse=True)

    for m in candidates[:limit]:
        score = scores[m["id"]]
        rec = recommend_plan(store, m["id"])
        offer = choose_offer(score.level, rec)
        if offer.kind == "plan_change" and m["corporate_account_id"]:
            offer = Offer("reminder")
        plan_name = plans[offer.suggested_plan or m["plan_id"]]["name"]

        source = "template"
        msg = template_message(m["name"].split()[0], m["renewal_date"], offer, plan_name)
        if client is not None:
            try:
                msg = llm_message(client, m, m["renewal_date"], offer, plan_name, score.top_reasons)
                source = "llm"
            except Exception:
                pass  # fall back to the template so the workflow never blocks on the API

        if offer.kind == "discount":
            tool, args = "offer_discount", {"member_id": m["id"], "percent": offer.discount_percent,
                                            "months": 1, "message": msg.message}
        else:
            tool, args = "send_renewal_reminder", {"member_id": m["id"], "channel": msg.channel,
                                                   "message": msg.message}
        toolbox.validate(tool, args)
        action_id = store.create_action(tool, args)
        drafts.append({"action_id": action_id, "member_id": m["id"], "name": m["name"],
                       "renewal_date": m["renewal_date"], "risk": score.level,
                       "churn_probability": round(score.probability, 3), "offer": offer.kind,
                       "message": msg.message, "message_source": source})

    return {"days_ahead": days_ahead, "renewals_found": len(candidates),
            "drafts_queued": len(drafts), "drafts": drafts}
