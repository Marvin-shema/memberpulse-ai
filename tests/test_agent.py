import json

import httpx2 as httpx  # the Anthropic SDK v2 uses httpx2
import anthropic

from memberpulse.agent import MerchantAgent
from memberpulse.approval import ConsoleApprover, DenyAllApprover, QueueApprover

from conftest import FakeClient, response, text_block, tool_use


def test_agent_answers_using_read_tool(toolbox):
    client = FakeClient([
        response([tool_use("t1", "get_business_summary", {})], "tool_use"),
        response([text_block("You have 508 active members.")], "end_turn"),
    ])
    result = MerchantAgent(toolbox, client=client).run("How is my business doing?")
    assert result.answer == "You have 508 active members."
    assert [t.name for t in result.tool_calls] == ["get_business_summary"]
    # second request carries the tool result back to the model
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "t1"
    assert "active_members" in tool_result["content"]
    assert result.input_tokens == 2000 and result.estimated_cost_usd > 0


def test_parallel_tool_calls_are_returned_in_one_message(toolbox):
    client = FakeClient([
        response([tool_use("a", "list_plans", {}), tool_use("b", "list_corporate_accounts", {})], "tool_use"),
        response([text_block("done")], "end_turn"),
    ])
    MerchantAgent(toolbox, client=client).run("plans and corporates?")
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_tool_errors_are_reported_to_model_not_raised(toolbox):
    client = FakeClient([
        response([tool_use("t1", "get_member_profile", {"member_id": "NOPE"})], "tool_use"),
        response([text_block("I couldn't find that member.")], "end_turn"),
    ])
    result = MerchantAgent(toolbox, client=client).run("profile for NOPE")
    sent = client.requests[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True
    assert result.tool_calls[0].outcome == "error"


def test_write_action_is_blocked_in_read_only_mode(toolbox, store):
    mid = store.search_members(status="active", limit=1)[0]["id"]
    client = FakeClient([
        response([tool_use("t1", "send_renewal_reminder",
                           {"member_id": mid, "channel": "sms", "message": "Renew today!"})], "tool_use"),
        response([text_block("The reminder was not sent.")], "end_turn"),
    ])
    result = MerchantAgent(toolbox, approver=DenyAllApprover(), client=client).run("remind them")
    assert result.tool_calls[0].outcome == "rejected"
    assert store.outbox() == []


def test_write_action_is_queued_for_dashboard_approval(toolbox, store):
    mid = store.search_members(status="active", limit=1)[0]["id"]
    client = FakeClient([
        response([tool_use("t1", "offer_discount",
                           {"member_id": mid, "percent": 10, "months": 1, "message": "10% off!"})], "tool_use"),
        response([text_block("Queued for your approval.")], "end_turn"),
    ])
    result = MerchantAgent(toolbox, approver=QueueApprover(store), client=client).run("offer a discount")
    assert result.pending_action_ids and store.outbox() == []
    assert store.action(result.pending_action_ids[0])["status"] == "pending"


def test_console_approval_yes_executes(toolbox, store):
    mid = store.search_members(status="active", limit=1)[0]["id"]
    client = FakeClient([
        response([tool_use("t1", "send_renewal_reminder",
                           {"member_id": mid, "channel": "whatsapp", "message": "See you!"})], "tool_use"),
        response([text_block("Sent.")], "end_turn"),
    ])
    approver = ConsoleApprover(store, ask=lambda _: "y")
    result = MerchantAgent(toolbox, approver=approver, client=client).run("remind")
    assert result.tool_calls[0].outcome == "approved"
    assert len(store.outbox()) == 1
    assert store.actions()[0]["status"] == "executed"


def test_invalid_write_never_reaches_approver(toolbox, store):
    asked = []
    mid = store.search_members(status="active", limit=1)[0]["id"]
    client = FakeClient([
        response([tool_use("t1", "offer_discount",
                           {"member_id": mid, "percent": 80, "months": 1, "message": "huge sale"})], "tool_use"),
        response([text_block("That discount is above the limit.")], "end_turn"),
    ])
    approver = ConsoleApprover(store, ask=lambda q: asked.append(q) or "y")
    result = MerchantAgent(toolbox, approver=approver, client=client).run("give 80% off")
    assert asked == [] and result.tool_calls[0].outcome == "error"


def test_turn_limit_stops_runaway_loops(toolbox):
    loop = [response([tool_use(f"t{i}", "list_plans", {})], "tool_use") for i in range(3)]
    result = MerchantAgent(toolbox, client=FakeClient(loop), max_turns=3).run("loop forever")
    assert result.stop_reason == "max_turns"


def test_refusal_is_handled(toolbox):
    result = MerchantAgent(toolbox, client=FakeClient([response([], "refusal")])).run("something bad")
    assert result.stop_reason == "refusal" and "can't help" in result.answer


def test_request_is_valid_for_the_real_sdk(toolbox):
    """Run one turn through the real Anthropic SDK with a mocked HTTP layer.

    Proves the request we build (strict tools, adaptive thinking, effort, caching,
    fallbacks) is accepted by the SDK and serialised the way we expect.
    """
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta", "")
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "Hello"}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 2},
        })

    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = MerchantAgent(toolbox, client=client).run("hi")
    assert result.answer == "Hello"
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert all(t["strict"] for t in body["tools"])
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen["beta"]
