"""The merchant assistant: a tool-using agent on the Anthropic Messages API.

The loop is written out by hand (instead of a framework) so every safeguard is
visible in one place:

  1. Claude sees only the tools in the Toolbox, with strict JSON schemas.
  2. READ tools run straight away. WRITE tools are validated against business
     rules, then sent to an Approver; nothing changes without a human "yes".
  3. Tool errors go back to Claude as `is_error` results so it can recover.
  4. A hard cap on turns, plus an audit trail with token usage and estimated cost.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .approval import Approver, DenyAllApprover
from .tools import Permission, ToolError, Toolbox

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
# USD per million tokens for the default model (input, output). Used for a rough cost estimate only.
PRICE_PER_MTOK = (4.00, 20.00)

SYSTEM_PROMPT = """You are MemberPulse, an operations assistant for the owner of Lion Peak Fitness, \
a gym that sells monthly memberships (individual and corporate).

How to work:
- Answer from data. Call the tools to look things up; never guess numbers, names or ids.
- Be concise and practical. Lead with the answer, then the 2-4 facts that support it. \
Use plain language a busy business owner understands; amounts are in USD.
- When you explain churn risk, use the reasons the tools return. The churn score is a \
model estimate, so say "likely" rather than "will".
- Actions that contact members or change accounts (reminders, discounts, plan changes, \
adding corporate employees) need the owner's approval. Propose them with the tool; the \
system will ask the owner. If an action is declined or queued, say so and don't retry it.
- Keep discounts modest (10-15% is typical, 30% is the hard limit) and only for members \
who are actually at risk.
- Member contact details are masked on purpose. Never ask for or invent full emails or phone numbers.
- If a request is outside running this membership business, say what you can help with instead."""


@dataclass
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    permission: str
    outcome: str  # ok | error | approved | rejected | queued
    result_preview: str
    ms: int


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    stop_reason: str = ""
    pending_action_ids: list[int] = field(default_factory=list)

    @property
    def estimated_cost_usd(self) -> float:
        return round(self.input_tokens / 1e6 * PRICE_PER_MTOK[0] + self.output_tokens / 1e6 * PRICE_PER_MTOK[1], 5)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_calls": [r.__dict__ for r in self.tool_calls],
            "pending_action_ids": self.pending_action_ids,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                      "estimated_cost_usd": self.estimated_cost_usd, "turns": self.turns},
            "stop_reason": self.stop_reason,
        }


def make_client():
    import anthropic  # imported lazily so the rest of the package works without it
    return anthropic.Anthropic()


class MerchantAgent:
    def __init__(self, toolbox: Toolbox, approver: Approver | None = None, client=None,
                 model: str = DEFAULT_MODEL, max_turns: int = 10, effort: str = "medium"):
        self.toolbox = toolbox
        self.approver = approver or DenyAllApprover()
        self.client = client or make_client()
        self.model = model
        self.max_turns = max_turns
        self.effort = effort

    # ---------------------------------------------------------------- model call

    def _create(self, messages: list[dict[str, Any]]):
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=self.toolbox.schemas(),
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            # Cache the stable prefix (tools + system prompt) across turns to cut cost and latency.
            cache_control={"type": "ephemeral"},
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    # ---------------------------------------------------------------- tool execution

    def _run_tool(self, name: str, args: dict[str, Any], result: AgentResult) -> dict[str, Any]:
        start = time.perf_counter()
        perm = self.toolbox.permission(name) if name in self.toolbox.tools else Permission.READ
        outcome, content, is_error = "ok", "", False
        try:
            if perm is Permission.WRITE:
                self.toolbox.validate(name, args)          # reject invalid proposals early
                decision = self.approver(name, args)
                outcome = decision.outcome
                if decision.outcome == "approved":
                    content = self.toolbox.call(name, args)
                    if decision.action_id is not None:
                        self.toolbox.store.set_action_status(decision.action_id, "executed", content)
                elif decision.outcome == "queued":
                    result.pending_action_ids.append(decision.action_id)
                    content = json.dumps({"status": "queued_for_approval", "action_id": decision.action_id,
                                          "note": "The owner will review this in the dashboard."})
                else:
                    content = json.dumps({"status": "declined", "note": decision.note})
            else:
                content = self.toolbox.call(name, args)
        except ToolError as e:
            outcome, content, is_error = "error", str(e), True
        except Exception as e:  # unexpected bug: report it, don't crash the conversation
            outcome, content, is_error = "error", f"Internal error in {name}: {type(e).__name__}", True

        result.tool_calls.append(ToolCallRecord(name, args, perm.value, outcome, content[:300],
                                                int((time.perf_counter() - start) * 1000)))
        block = {"type": "tool_result", "content": content}
        if is_error:
            block["is_error"] = True
        return block

    # ---------------------------------------------------------------- loop

    def run(self, user_message: str, history: list[dict[str, Any]] | None = None) -> AgentResult:
        messages = list(history or []) + [{"role": "user", "content": user_message}]
        result = AgentResult(answer="")

        for _ in range(self.max_turns):
            response = self._create(messages)
            result.turns += 1
            result.input_tokens += getattr(response.usage, "input_tokens", 0) or 0
            result.output_tokens += getattr(response.usage, "output_tokens", 0) or 0
            result.stop_reason = response.stop_reason or ""

            if response.stop_reason == "refusal":
                result.answer = "Sorry, I can't help with that request."
                return result

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                result.answer = "\n".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    result.answer += "\n\n[Answer was cut short.]"
                return result

            # Run every tool call from this turn and send all results back in one message.
            tool_results = []
            for tu in tool_uses:
                block = self._run_tool(tu.name, dict(tu.input), result)
                block["tool_use_id"] = tu.id
                tool_results.append(block)
            messages.append({"role": "user", "content": tool_results})

        result.answer = "I stopped after too many steps. Try a more specific question."
        result.stop_reason = "max_turns"
        return result
