"""Command line interface.

    python -m memberpulse seed          # create demo data
    python -m memberpulse train         # train the churn model
    python -m memberpulse summary       # business snapshot (no API key needed)
    python -m memberpulse at-risk       # members most likely to cancel
    python -m memberpulse renewals      # draft renewal campaign, queued for approval
    python -m memberpulse actions       # review / approve queued actions
    python -m memberpulse chat          # talk to the AI assistant (needs ANTHROPIC_API_KEY)
    python -m memberpulse serve         # start the REST API
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .app import DB_PATH, MODEL_PATH, load_app


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_seed(args) -> None:
    from .data import generate
    generate(DB_PATH, n_members=args.members)
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    print(f"Created demo database at {DB_PATH} with {args.members} members.")


def cmd_train(args) -> None:
    from .churn import ChurnModel
    from .db import Store
    if not DB_PATH.exists():
        from .data import generate
        generate(DB_PATH)
    model = ChurnModel()
    auc = model.train(Store(DB_PATH))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Churn model trained. Hold-out ROC AUC = {auc:.3f}. Saved to {MODEL_PATH}.")


def cmd_summary(args) -> None:
    _print(load_app().toolbox.business_summary())


def cmd_at_risk(args) -> None:
    for r in load_app().toolbox.at_risk_members(args.limit, "any"):
        print(f"{r['member_id']}  {r['name']:<24} {r['risk_level']:<6} "
              f"{r['churn_probability']:.0%}  {'; '.join(r['top_reasons'])}")


def cmd_renewals(args) -> None:
    from .retention import run_renewal_campaign
    a = load_app()
    client = None
    if not args.no_llm and os.getenv("ANTHROPIC_API_KEY"):
        from .agent import make_client
        client = make_client()
    out = run_renewal_campaign(a.store, a.churn, args.days, client=client, limit=args.limit)
    print(f"{out['renewals_found']} renewals in the next {args.days} days; "
          f"{out['drafts_queued']} drafts queued for approval.\n")
    for d in out["drafts"]:
        print(f"#{d['action_id']} {d['name']} ({d['risk']} risk, {d['offer']}, {d['message_source']})")
        print(f"    {d['message']}\n")
    print("Review them with:  python -m memberpulse actions")


def cmd_actions(args) -> None:
    from .approval import execute_approved
    a = load_app()
    if args.approve:
        _print(execute_approved(a.store, a.toolbox, args.approve))
        return
    if args.reject:
        a.store.set_action_status(args.reject, "rejected")
        print(f"Action {args.reject} rejected.")
        return
    pending = a.store.actions("pending")
    if not pending:
        print("No pending actions.")
    for act in pending:
        print(f"#{act['id']} {act['tool']} {json.dumps(act['payload'])}")


def cmd_chat(args) -> None:
    from .agent import MerchantAgent
    from .approval import ConsoleApprover
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (see README).")
    a = load_app()
    agent = MerchantAgent(a.toolbox, approver=ConsoleApprover(a.store))
    history: list = []
    print("MemberPulse assistant. Ask about your members, revenue or renewals. Ctrl+C to quit.\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        result = agent.run(q, history=history)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": result.answer}]
        print(f"\nassistant > {result.answer}\n")
        if args.verbose:
            for t in result.tool_calls:
                print(f"   [{t.permission}] {t.name}({json.dumps(t.input)}) -> {t.outcome} ({t.ms} ms)")
        print(f"   ({result.turns} model calls, {result.input_tokens + result.output_tokens} tokens, "
              f"~${result.estimated_cost_usd})\n")


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("memberpulse.api:app", host=args.host, port=args.port)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="memberpulse", description="AI assistant for membership businesses")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed"); s.add_argument("--members", type=int, default=600); s.set_defaults(f=cmd_seed)
    sub.add_parser("train").set_defaults(f=cmd_train)
    sub.add_parser("summary").set_defaults(f=cmd_summary)
    s = sub.add_parser("at-risk"); s.add_argument("--limit", type=int, default=15); s.set_defaults(f=cmd_at_risk)
    s = sub.add_parser("renewals")
    s.add_argument("--days", type=int, default=7); s.add_argument("--limit", type=int, default=10)
    s.add_argument("--no-llm", action="store_true", help="use message templates instead of Claude")
    s.set_defaults(f=cmd_renewals)
    s = sub.add_parser("actions")
    s.add_argument("--approve", type=int); s.add_argument("--reject", type=int); s.set_defaults(f=cmd_actions)
    s = sub.add_parser("chat"); s.add_argument("-v", "--verbose", action="store_true"); s.set_defaults(f=cmd_chat)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.set_defaults(f=cmd_serve)
    args = p.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
