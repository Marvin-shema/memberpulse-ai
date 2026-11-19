from memberpulse.churn import ChurnModel, risk_level
from memberpulse.data import generate
from memberpulse.db import Store
from memberpulse.recommender import recommend_plan


def test_data_is_reproducible(tmp_path):
    a, b = generate(tmp_path / "a.db", n_members=100), generate(tmp_path / "b.db", n_members=100)
    q = "SELECT id, name, plan_id, status FROM members ORDER BY id"
    assert Store(a).query(q) == Store(b).query(q)


def test_dataset_has_both_classes(store):
    counts = {r["status"]: r["n"] for r in store.query("SELECT status, COUNT(*) n FROM members GROUP BY status")}
    assert counts["active"] > 300 and counts["cancelled"] > 30


def test_churn_model_beats_random(churn):
    assert churn.auc > 0.75


def test_scores_are_sorted_probabilities_with_reasons(store, churn):
    scores = churn.score_all(store)
    probs = [s.probability for s in scores]
    assert probs == sorted(probs, reverse=True)
    assert all(0 <= p <= 1 for p in probs)
    assert all(s.top_reasons for s in scores)
    assert {s.level for s in scores} == {"high", "medium", "low"}


def test_low_engagement_scores_higher_than_engaged(store, churn):
    scores = {s.member_id: s.probability for s in churn.score_all(store)}
    feats = {r["id"]: r for r in store.member_features() if r["status"] == "active"}
    idle = [m for m, f in feats.items() if f["visits_30d"] <= 2 and f["failed_payments"] >= 1]
    busy = [m for m, f in feats.items() if f["visits_30d"] >= 12 and f["failed_payments"] == 0]
    assert idle and busy
    avg = lambda ids: sum(scores[i] for i in ids) / len(ids)
    assert avg(idle) > avg(busy) + 0.2


def test_model_save_and_load(tmp_path, store, churn):
    churn.save(tmp_path / "m.joblib")
    loaded = ChurnModel.load(tmp_path / "m.joblib")
    assert loaded.score_all(store)[0].member_id == churn.score_all(store)[0].member_id


def test_risk_levels():
    assert risk_level(0.9) == "high" and risk_level(0.4) == "medium" and risk_level(0.1) == "low"


def test_recommendation_upgrades_basic_member_at_visit_cap(store):
    store.query("SELECT 1")  # warm
    mid = "M9999"
    with store.connect() as con:
        con.execute("INSERT INTO members VALUES (?,?,?,?,?,NULL,'2025-01-01','2026-11-20','active',NULL,1,'card',1,30)",
                    (mid, "Test Person", "t@x.example", "+250700000001", "basic"))
        for d in range(1, 11):
            con.execute("INSERT INTO visits VALUES (?, ?)", (mid, f"2026-10-{d:02d}"))
    rec = recommend_plan(store, mid)
    assert rec["action"] == "upgrade" and rec["suggested_plan_id"] == "standard"
    assert rec["price_change_monthly"] == 15.0


def test_recommendation_downgrades_idle_premium_member(store):
    mid = "M9998"
    with store.connect() as con:
        con.execute("INSERT INTO members VALUES (?,?,?,?,?,NULL,'2025-01-01','2026-11-20','active',NULL,1,'card',1,30)",
                    (mid, "Idle Person", "i@x.example", "+250700000002", "premium"))
    rec = recommend_plan(store, mid)
    assert rec["action"] == "downgrade" and rec["suggested_plan_id"] == "plus"
