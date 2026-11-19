import shutil
from types import SimpleNamespace

import pytest

from memberpulse.churn import ChurnModel
from memberpulse.data import generate
from memberpulse.db import Store
from memberpulse.tools import Toolbox


@pytest.fixture(scope="session")
def _seed(tmp_path_factory):
    path = tmp_path_factory.mktemp("seed") / "seed.db"
    generate(path, n_members=600, seed=7)
    model = ChurnModel()
    model.train(Store(path))
    return path, model


@pytest.fixture
def store(_seed, tmp_path):
    """A fresh copy of the demo database for every test (tests may write to it)."""
    path = tmp_path / "test.db"
    shutil.copy(_seed[0], path)
    return Store(path)


@pytest.fixture
def churn(_seed):
    return _seed[1]


@pytest.fixture
def toolbox(store, churn):
    return Toolbox(store, churn)


# ---------- a scripted stand-in for the Anthropic client ----------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use(id, name, input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def response(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=100))


class FakeClient:
    """Returns pre-scripted responses and records every request it receives."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # copy the message list: the agent keeps appending to the same list
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)
