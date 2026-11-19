"""Wires the store, the churn model and the toolbox together from environment config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .churn import ChurnModel
from .data import generate
from .db import Store
from .tools import Toolbox

DB_PATH = Path(os.getenv("MEMBERPULSE_DB", "data/memberpulse.db"))
MODEL_PATH = Path(os.getenv("MEMBERPULSE_MODEL", "data/churn_model.joblib"))


@dataclass
class App:
    store: Store
    churn: ChurnModel
    toolbox: Toolbox


def load_app(db_path: Path = DB_PATH, model_path: Path = MODEL_PATH) -> App:
    """Load the app, creating demo data and training the model on first run."""
    if not db_path.exists():
        generate(db_path)
    store = Store(db_path)
    if model_path.exists():
        churn = ChurnModel.load(model_path)
    else:
        churn = ChurnModel()
        churn.train(store)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        churn.save(model_path)
    return App(store, churn, Toolbox(store, churn))
