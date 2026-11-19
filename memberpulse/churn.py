"""Churn-risk model.

A scikit-learn pipeline trained on members who either cancelled or stayed.
It then scores active members so the merchant (and the agent) can see who is
likely to leave and *why*, in plain language.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .db import Store

NUMERIC = ["visits_30d", "visits_90d", "failed_payments", "tenure_days", "price_monthly",
           "auto_renew", "has_wallet_pass", "is_corporate", "age"]
CATEGORICAL = ["payment_method"]

# Human-readable reasons, shown to merchants instead of raw coefficients.
REASON_TEXT = {
    "visits_30d": ("few visits in the last 30 days", "visits often"),
    "visits_90d": ("low attendance over the last 3 months", "steady attendance"),
    "failed_payments": ("has failed payments", "pays reliably"),
    "tenure_days": ("is a fairly new member", "long-standing member"),
    "price_monthly": ("is on a higher-priced plan", "on an affordable plan"),
    "auto_renew": ("auto-renew is off", "auto-renew is on"),
    "has_wallet_pass": ("hasn't added the wallet pass", "uses the wallet pass"),
    "is_corporate": ("pays individually, not through an employer", "covered by an employer plan"),
    "age": ("age profile", "age profile"),
}


@dataclass
class ChurnScore:
    member_id: str
    probability: float
    level: str
    top_reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"member_id": self.member_id, "churn_probability": round(self.probability, 3),
                "risk_level": self.level, "top_reasons": self.top_reasons}


def risk_level(p: float) -> str:
    if p >= 0.6:
        return "high"
    if p >= 0.35:
        return "medium"
    return "low"


class ChurnModel:
    def __init__(self, pipeline: Pipeline | None = None, auc: float | None = None):
        self.pipeline = pipeline
        self.auc = auc

    @staticmethod
    def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        for col in NUMERIC:
            df[col] = pd.to_numeric(df[col]).fillna(0)
        return df

    def train(self, store: Store, seed: int = 7) -> float:
        df = self._frame(store.member_features())
        y = (df["status"] == "cancelled").astype(int)
        X = df[NUMERIC + CATEGORICAL]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

        pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("num", StandardScaler(), NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ])),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipe.fit(X_train, y_train)
        self.auc = float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))
        # refit on all data for serving
        pipe.fit(X, y)
        self.pipeline = pipe
        return self.auc

    def save(self, path: str | Path) -> None:
        joblib.dump({"pipeline": self.pipeline, "auc": self.auc}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ChurnModel":
        blob = joblib.load(path)
        return cls(blob["pipeline"], blob["auc"])

    def _explain_all(self, X: pd.DataFrame) -> list[list[str]]:
        """Per-member contribution = coefficient x standardised value (exact for a linear model).

        Computed for all rows at once, so scoring 500 members takes milliseconds.
        """
        prep = self.pipeline.named_steps["prep"]
        clf = self.pipeline.named_steps["clf"]
        z = prep.transform(X)
        z = z.toarray() if hasattr(z, "toarray") else np.asarray(z)
        contrib = z[:, : len(NUMERIC)] * clf.coef_[0][: len(NUMERIC)]
        out = []
        for row in contrib:
            reasons = []
            for idx in np.argsort(row)[::-1]:
                if row[idx] <= 0.15 or NUMERIC[idx] == "age":
                    continue
                reasons.append(REASON_TEXT[NUMERIC[idx]][0])
                if len(reasons) == 3:
                    break
            out.append(reasons or ["no strong warning signs"])
        return out

    def score_all(self, store: Store, only_active: bool = True) -> list[ChurnScore]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained. Run `python -m memberpulse train` first.")
        rows = [r for r in store.member_features() if not only_active or r["status"] == "active"]
        if not rows:
            return []
        df = self._frame(rows)
        X = df[NUMERIC + CATEGORICAL]
        probs = self.pipeline.predict_proba(X)[:, 1]
        reasons = self._explain_all(X)
        out = [ChurnScore(df.iloc[i]["id"], float(p), risk_level(float(p)), reasons[i])
               for i, p in enumerate(probs)]
        return sorted(out, key=lambda s: s.probability, reverse=True)

    def score_member(self, store: Store, member_id: str) -> ChurnScore | None:
        for s in self.score_all(store, only_active=False):
            if s.member_id == member_id:
                return s
        return None
