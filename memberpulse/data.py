"""Synthetic membership data for a fictional merchant.

Everything here is generated from a fixed random seed, so the demo, the tests
and the README numbers are reproducible. No real customer data is used.

The fictional merchant is "Lion Peak Fitness", a gym with four plans, a few
corporate accounts and members who pay by card, mobile money or crypto.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    price_monthly REAL NOT NULL,
    tier INTEGER NOT NULL,              -- 1 = cheapest, 4 = premium
    visits_per_month INTEGER,           -- NULL = unlimited
    perks TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corporate_accounts (
    id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    plan_id TEXT NOT NULL REFERENCES plans(id),
    seats_purchased INTEGER NOT NULL,
    discount_percent REAL NOT NULL,
    contact_email TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS members (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL,
    plan_id TEXT NOT NULL REFERENCES plans(id),
    corporate_account_id TEXT REFERENCES corporate_accounts(id),
    join_date TEXT NOT NULL,
    renewal_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'cancelled')),
    cancelled_on TEXT,
    auto_renew INTEGER NOT NULL,
    payment_method TEXT NOT NULL,
    has_wallet_pass INTEGER NOT NULL,
    age INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS visits (
    member_id TEXT NOT NULL REFERENCES members(id),
    visit_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    member_id TEXT NOT NULL REFERENCES members(id),
    paid_on TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'failed'))
);

-- Actions the agent proposes. Nothing leaves this table without approval.
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'executed')),
    created_at TEXT NOT NULL,
    decided_at TEXT,
    result TEXT
);

-- Simulated notification outbox (stands in for SMS / email / wallet push).
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    message TEXT NOT NULL,
    sent_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_visits_member ON visits(member_id);
CREATE INDEX IF NOT EXISTS idx_payments_member ON payments(member_id);
"""

PLANS = [
    ("basic", "Basic", 25.0, 1, 8, "Gym floor access, 8 visits per month"),
    ("standard", "Standard", 40.0, 2, None, "Unlimited gym floor access"),
    ("plus", "Plus", 60.0, 3, None, "Unlimited access, group classes, 1 guest pass per month"),
    ("premium", "Premium", 95.0, 4, None, "Everything in Plus, sauna, 2 personal training sessions per month"),
]

CORPORATES = [
    ("corp_kivu", "Kivu Logistics", "standard", 40, 20.0, "hr@kivulogistics.example"),
    ("corp_umoja", "Umoja Bank", "plus", 25, 15.0, "benefits@umojabank.example"),
    ("corp_nyota", "Nyota Tech", "standard", 15, 10.0, "people@nyotatech.example"),
]

FIRST = ["Aline", "Eric", "Grace", "Jean", "Diane", "Patrick", "Claudine", "Olivier", "Sandrine",
         "Kevin", "Esther", "David", "Ange", "Moses", "Ruth", "Samuel", "Joy", "Brian", "Nadia",
         "Felix", "Linda", "Yves", "Chantal", "Ivan", "Sharon", "Didier", "Alice", "Emmanuel"]
LAST = ["Uwase", "Mugisha", "Ingabire", "Habimana", "Mukamana", "Niyonzima", "Iradukunda",
        "Nshuti", "Umutoni", "Ndayisaba", "Kamanzi", "Gasana", "Keza", "Rukundo", "Byiringiro"]
PAYMENT_METHODS = ["card", "mobile_money", "mobile_money", "mobile_money", "cash", "crypto"]

# The date the dataset is "as of". Fixed so results never drift.
TODAY = date(2026, 11, 1)


def _churn_propensity(visits_per_week: float, failed_payments: int, auto_renew: bool,
                      tenure_days: int, tier: int, corporate: bool, wallet: bool) -> float:
    """Hidden 'true' churn logic used only to generate labels. The ML model never sees it."""
    score = -0.6
    score += 1.6 * max(0.0, 1.5 - visits_per_week)       # low engagement is the biggest driver
    score += 0.8 * failed_payments
    score += -0.9 if auto_renew else 0.5
    score += -0.6 if tenure_days > 365 else 0.3
    score += 0.25 * (tier - 2)                            # pricier plans churn a little more
    score += -0.8 if corporate else 0.0
    score += -0.4 if wallet else 0.0
    return 1 / (1 + pow(2.718281828, -score))


def generate(db_path: str | Path, n_members: int = 600, seed: int = 7) -> Path:
    """Create (or recreate) the SQLite database with synthetic data."""
    rng = random.Random(seed)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO plans VALUES (?,?,?,?,?,?)", PLANS)
    con.executemany("INSERT INTO corporate_accounts VALUES (?,?,?,?,?,?)", CORPORATES)

    plan_price = {p[0]: p[2] for p in PLANS}
    plan_tier = {p[0]: p[3] for p in PLANS}
    corp_seats_used = {c[0]: 0 for c in CORPORATES}
    corp_target = {c[0]: int(c[3] * rng.uniform(0.55, 0.85)) for c in CORPORATES}

    for i in range(1, n_members + 1):
        member_id = f"M{i:04d}"
        first, last = rng.choice(FIRST), rng.choice(LAST)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}{i}@mail.example"
        phone = f"+2507{rng.randint(20000000, 99999999)}"

        corporate_id = None
        open_corps = [c for c in CORPORATES if corp_seats_used[c[0]] < corp_target[c[0]]]
        if open_corps and rng.random() < 0.15:
            corp = rng.choice(open_corps)
            corporate_id = corp[0]
            corp_seats_used[corporate_id] += 1
            plan_id = corp[2]
        else:
            plan_id = rng.choices(["basic", "standard", "plus", "premium"], [0.3, 0.35, 0.22, 0.13])[0]

        join = TODAY - timedelta(days=rng.randint(30, 900))
        tenure = (TODAY - join).days
        auto_renew = rng.random() < (0.85 if corporate_id else 0.55)
        wallet = rng.random() < 0.5
        method = "card" if corporate_id else rng.choice(PAYMENT_METHODS)
        visits_per_week = max(0.0, rng.gauss(2.2 + 0.3 * plan_tier[plan_id], 1.3))
        failed = rng.choices([0, 1, 2, 3], [0.72, 0.17, 0.08, 0.03])[0]
        if method == "card":
            failed = max(0, failed - 1)

        p = _churn_propensity(visits_per_week, failed, auto_renew, tenure,
                              plan_tier[plan_id], corporate_id is not None, wallet)
        cancelled = rng.random() < p * 0.55
        cancelled_on = None
        if cancelled:
            cancelled_on = TODAY - timedelta(days=rng.randint(1, min(180, tenure - 1)))
            # cancelled members tend to taper off before they leave
            visits_per_week *= rng.uniform(0.3, 0.8)

        end = cancelled_on or TODAY
        # monthly renewal: next renewal date is the next monthly anniversary after `end`
        months = ((end - join).days // 30) + 1
        renewal = join + timedelta(days=30 * months)

        con.execute(
            "INSERT INTO members VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (member_id, name, email, phone, plan_id, corporate_id, join.isoformat(),
             renewal.isoformat(), "cancelled" if cancelled else "active",
             cancelled_on.isoformat() if cancelled_on else None,
             int(auto_renew), method, int(wallet), rng.randint(18, 64)),
        )

        # Visits over the last 90 days of membership
        window_start = max(join, end - timedelta(days=90))
        days = (end - window_start).days
        n_visits = int(visits_per_week * days / 7)
        visit_rows = [(member_id, (window_start + timedelta(days=rng.randint(0, max(days - 1, 0)))).isoformat())
                      for _ in range(n_visits)]
        con.executemany("INSERT INTO visits VALUES (?,?)", visit_rows)

        # Monthly payments over the whole membership
        price = plan_price[plan_id]
        if corporate_id:
            price *= 1 - dict((c[0], c[4]) for c in CORPORATES)[corporate_id] / 100
        failed_months = set(rng.sample(range(max(months, 1)), min(failed, max(months, 1))))
        pay_rows = []
        for m in range(months):
            paid_on = join + timedelta(days=30 * m)
            if paid_on > end:
                break
            status = "failed" if m in failed_months else "paid"
            pay_rows.append((member_id, paid_on.isoformat(), round(price, 2), status))
        con.executemany("INSERT INTO payments VALUES (?,?,?,?)", pay_rows)

    con.commit()
    con.close()
    return db_path
