"""Plan recommendations.

Two signals are combined:
1. Usage fit: is the member paying for more (or less) than they use?
2. Lookalikes: what plan do similar members who stayed long-term hold?
   (k-nearest neighbours over behaviour features)

The result is a structured recommendation the agent can explain, not free text.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .db import Store

FEATURES = ["visits_30d", "visits_90d", "age", "tenure_days", "is_corporate"]


def recommend_plan(store: Store, member_id: str, k: int = 15) -> dict[str, Any]:
    rows = store.member_features()
    by_id = {r["id"]: r for r in rows}
    if member_id not in by_id:
        raise KeyError(f"Unknown member {member_id}")
    me = by_id[member_id]
    plans = {p["id"]: p for p in store.plans()}
    current = plans[me["plan_id"]]

    # --- usage fit ---
    monthly_visits = me["visits_30d"]
    action, suggested, reason = "keep", current["id"], "Current plan matches how often they visit."
    if current["visits_per_month"] and monthly_visits >= current["visits_per_month"] - 1:
        action, suggested = "upgrade", "standard"
        reason = (f"Visited {monthly_visits} times last month, at or near the "
                  f"{current['visits_per_month']}-visit cap of {current['name']}.")
    elif current["tier"] >= 3 and monthly_visits <= 3:
        lower = [p for p in plans.values() if p["tier"] == current["tier"] - 1][0]
        action, suggested = "downgrade", lower["id"]
        reason = (f"Only {monthly_visits} visits last month on {current['name']} "
                  f"(${current['price_monthly']:.0f}). A cheaper plan may keep them from cancelling.")

    # --- lookalikes among long-term retained members ---
    retained = [r for r in rows if r["status"] == "active" and r["tenure_days"] > 300 and r["id"] != member_id]
    X = np.array([[float(r[f]) for f in FEATURES] for r in retained])
    scaler = StandardScaler().fit(X)
    nn = NearestNeighbors(n_neighbors=min(k, len(retained))).fit(scaler.transform(X))
    _, idx = nn.kneighbors(scaler.transform([[float(me[f]) for f in FEATURES]]))
    peer_plans = Counter(retained[i]["plan_id"] for i in idx[0])
    peer_plan, peer_count = peer_plans.most_common(1)[0]

    if action == "keep" and peer_plan != current["id"] and plans[peer_plan]["tier"] > current["tier"] \
            and peer_count / len(idx[0]) >= 0.5 and monthly_visits >= 8:
        action, suggested = "upgrade", peer_plan
        reason = (f"{peer_count} of {len(idx[0])} similar long-term members use {plans[peer_plan]['name']}, "
                  f"and this member visits often ({monthly_visits} times last month).")

    target = plans[suggested]
    return {
        "member_id": member_id,
        "current_plan": current["name"],
        "action": action,
        "suggested_plan_id": target["id"],
        "suggested_plan": target["name"],
        "price_change_monthly": round(target["price_monthly"] - current["price_monthly"], 2),
        "reason": reason,
        "similar_members_plan_mix": dict(peer_plans),
    }
