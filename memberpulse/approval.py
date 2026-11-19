"""Human-in-the-loop approval for sensitive actions.

The agent can *propose* a write action, but an Approver decides what happens:

* ConsoleApprover - asks the merchant "y/n" in the terminal (CLI chat).
* QueueApprover   - saves the action as 'pending' for the merchant to approve
                    later in the dashboard / API (the production pattern).
* DenyAllApprover - read-only mode; useful for demos and tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from .db import Store

Outcome = Literal["approved", "rejected", "queued"]


@dataclass
class Decision:
    outcome: Outcome
    action_id: int | None = None
    note: str = ""


class Approver(Protocol):
    def __call__(self, tool: str, args: dict[str, Any]) -> Decision: ...


class DenyAllApprover:
    def __call__(self, tool: str, args: dict[str, Any]) -> Decision:
        return Decision("rejected", note="Read-only mode: write actions are disabled.")


class QueueApprover:
    def __init__(self, store: Store):
        self.store = store

    def __call__(self, tool: str, args: dict[str, Any]) -> Decision:
        action_id = self.store.create_action(tool, args)
        return Decision("queued", action_id, "Saved for merchant approval.")


class ConsoleApprover:
    def __init__(self, store: Store, ask: Callable[[str], str] = input):
        self.store = store
        self.ask = ask

    def __call__(self, tool: str, args: dict[str, Any]) -> Decision:
        action_id = self.store.create_action(tool, args)
        print(f"\n  ⚠  The assistant wants to run: {tool}")
        print("     " + json.dumps(args, indent=2).replace("\n", "\n     "))
        answer = self.ask("     Approve? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            self.store.set_action_status(action_id, "approved")
            return Decision("approved", action_id)
        self.store.set_action_status(action_id, "rejected")
        return Decision("rejected", action_id, "Merchant declined.")


def execute_approved(store: Store, toolbox, action_id: int) -> dict[str, Any]:
    """Run a queued action after the merchant approves it (API / dashboard flow)."""
    action = store.action(action_id)
    if not action:
        raise KeyError(f"No action {action_id}")
    if action["status"] != "pending":
        raise ValueError(f"Action {action_id} is already {action['status']}")
    result = toolbox.call(action["tool"], action["payload"])
    store.set_action_status(action_id, "executed", result)
    return json.loads(result)
