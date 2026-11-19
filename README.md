# MemberPulse AI

**An AI operations assistant for membership and subscription businesses** (gyms, salons, clubs, hotels).
A merchant can ask questions about their business in plain language, see which members are likely to
cancel and why, get plan recommendations, and run renewal campaigns, while every action that contacts a
customer or changes an account waits for a human to approve it.

Built with Python, the Anthropic Claude API (tool use / function calling and structured outputs),
scikit-learn, FastAPI, SQLite and Docker. All data is synthetic.

[![CI](https://github.com/Marvin-shema/memberpulse-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Marvin-shema/memberpulse-ai/actions/workflows/ci.yml)

---

## What it does

| Feature | How it works |
|---|---|
| **Merchant assistant (AI agent)** | Claude answers questions like *"Why is revenue down?"* or *"Who should I call this week?"* by calling 13 typed tools over the business data. It never writes SQL and never sees raw contact details. |
| **Churn prediction** | A scikit-learn logistic-regression pipeline trained on past cancellations (ROC AUC ≈ 0.86 on held-out data). Each score comes with plain-language reasons, e.g. *"few visits in the last 30 days; has failed payments"*. |
| **Plan recommendations** | Combines a usage-fit rule (paying for more than you use, or hitting a visit cap) with k-nearest-neighbour lookalikes among long-term members. Returns a structured keep / upgrade / downgrade with the price change. |
| **Renewal & retention workflow** | Finds upcoming renewals, ranks them by churn risk, picks an offer from an auditable policy (reminder, 10–15% discount, or plan change), and has Claude write a personal message as **validated structured output**. Drafts are queued for approval, nothing is sent automatically. |
| **Corporate memberships** | Lists employer accounts and seat usage; bulk-adds employees with seat-limit checks (approval required). |
| **Business analytics** | MRR, churn rate, members per plan, revenue at risk, monthly collected vs failed payments. |
| **REST API** | FastAPI endpoints a dashboard or backend can call, with API-key auth and an approve / reject queue. |

## Architecture

```
                    ┌──────────────────────────────┐
 merchant ──CLI──▶  │  MerchantAgent (agent.py)     │ ◀── REST /chat (api.py)
                    │  Claude tool-use loop         │
                    └──────┬───────────────▲───────┘
                 tool call │               │ tool result (masked PII, JSON)
                    ┌──────▼───────────────┴───────┐
                    │  Toolbox (tools.py)           │
                    │  READ tools ── run directly   │
                    │  WRITE tools ─ validate rules │──▶ Approver (approval.py)
                    └──────┬───────────────────────┘      console y/n │ queue for dashboard
                           │                                          ▼
          ┌────────────────┼──────────────────┐            actions table ─▶ outbox
          ▼                ▼                  ▼
   Store (SQLite)   ChurnModel (sklearn)  recommender (kNN)
```

* `memberpulse/agent.py` - the agent loop, written by hand so every safeguard is visible
* `memberpulse/tools.py` - tool definitions, strict JSON schemas, permissions, business rules, PII masking
* `memberpulse/approval.py` - human-in-the-loop approvers and the approve-then-execute flow
* `memberpulse/churn.py` - feature pipeline, training, scoring and per-member explanations
* `memberpulse/recommender.py` - plan recommendations
* `memberpulse/retention.py` - the renewal campaign workflow
* `memberpulse/api.py` - FastAPI service
* `memberpulse/data.py` - reproducible synthetic data (600 members, visits, payments, 3 corporate accounts)

## Reliability, safety and cost

These are the parts I'd want in production, and they're tested:

* **Human approval for sensitive actions.** Sending messages, discounts, plan changes and bulk adds are
  `WRITE` tools. They are validated against business rules first (e.g. discount ≤ 30%, seat limits,
  member must be active), then go to an approver. Via the API they're queued until someone calls
  `/actions/{id}/approve`; an action can't be executed twice.
* **Least privilege for the model.** The model only gets the tools listed, each with a strict schema, so
  arguments always match. There is no free-form SQL tool.
* **Customer data protection.** Emails and phone numbers are masked before anything is sent to the model.
  Secrets come from environment variables; the API can require an `X-API-Key` (compared in constant time).
* **Graceful failure.** Tool errors go back to Claude as `is_error` results so it can correct itself;
  unexpected exceptions are caught; refusals and `max_tokens` are handled; a turn limit stops runaway loops;
  the renewal workflow falls back to message templates if the API is down.
* **Cost and latency.** The tools + system prompt prefix is prompt-cached across turns; effort is set to
  `medium` for chat and `low` for message writing; the renewal campaign is a fixed workflow (one small LLM
  call per member) rather than an open-ended agent. Every run reports tokens and estimated cost.
* **Observability.** Each tool call is recorded with input, permission level, outcome and latency.
* **Tests.** 34 pytest tests cover data, model quality, recommendations, tools, approval flow, the agent
  loop (with a scripted fake client) and the API. One test sends a real SDK request through a mocked HTTP
  layer to confirm the request shape. Tests need no API key and run in CI on every push.

## Quick start

You need Python 3.11+.

```bash
git clone https://github.com/Marvin-shema/memberpulse-ai.git
cd memberpulse-ai
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m memberpulse seed        # create demo data
python -m memberpulse train       # train the churn model, prints ROC AUC
python -m memberpulse summary     # business snapshot
python -m memberpulse at-risk     # who is likely to cancel, and why
python -m memberpulse renewals --no-llm   # draft a renewal campaign (queued, not sent)
python -m memberpulse actions             # see the approval queue
python -m memberpulse actions --approve 1 # approve one
pytest -q                                 # run the tests
```

None of the commands above need an API key. To use the AI assistant, get a key from
[console.anthropic.com](https://console.anthropic.com) and:

```bash
cp .env.example .env              # then put your key in .env, or:
export ANTHROPIC_API_KEY=sk-ant-...
python -m memberpulse chat -v     # -v shows each tool call
python -m memberpulse renewals    # now Claude writes the messages
```

Things to ask the assistant:

* "How is the business doing this month?"
* "Who is most likely to cancel before their renewal next week, and what should I do?"
* "Should member M0069 be on a different plan?"
* "How many seats does Umoja Bank have left?"
* "Send a friendly renewal reminder to M0013 on WhatsApp." (you'll be asked to approve)

### REST API

```bash
python -m memberpulse serve       # http://127.0.0.1:8000/docs
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status and model AUC |
| POST | `/chat` | `{"message": "...", "conversation_id": null}` → answer, tool calls, pending actions, usage |
| GET | `/summary` | Business snapshot |
| GET | `/members/at-risk?limit=20&risk_level=high` | Ranked churn risk with reasons |
| GET | `/members/{id}/recommendation` | Plan recommendation |
| POST | `/campaigns/renewals` | `{"days_ahead": 7, "limit": 25, "use_llm": true}` → drafts queued |
| GET | `/actions?status=pending` | Approval queue |
| POST | `/actions/{id}/approve` · `/reject` | Decide on an action |

### Docker

```bash
docker build -t memberpulse-ai .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... -e MERCHANT_API_KEY=change-me memberpulse-ai
```

## Design choices

* **Agent vs workflow.** Open questions ("why is revenue down?") need an agent that chooses its own tool
  calls. Renewal campaigns don't: the steps are known, so they're plain code with one LLM call for the
  creative part. That's cheaper, faster and easier to audit.
* **A hand-written loop** instead of a framework, so permission checks, approval and error handling are in
  one readable file. The same `Toolbox` could be exposed through LangChain or an MCP server.
* **A linear churn model** because its per-member explanations are exact and easy for a merchant to trust.
  A gradient-boosted model would be the next experiment, with SHAP for explanations.

## What I'd add next

* Retrieval (RAG) over the merchant's FAQ and policy documents for customer-support answers
* Real notification channels (SMS / WhatsApp / Apple & Google Wallet pass updates) behind the outbox
* An evaluation set of merchant questions with expected tool calls, run in CI to catch prompt regressions
* Postgres and a task queue for scheduled daily renewal runs

## License

MIT. All names and data are fictional.
