"""I1: a family is claimed only when a REAL provider answers a typed envelope, the answer is stored, and it is read back
unchanged by a process that never saw it computed.

Every other test in this repository drives the web layer with a fake provider. This one loads the provider an
installation really has, by its entry point, and gives it the shipped example's own data and its own fitted state. It
needs the provider's package and a fitted state; where either is absent the case is SKIPPED BY NAME (and
`M5PHET_E2E_REQUIRE=<area>[,<area>]` turns that skip into a failure, so a host that is meant to prove an area cannot
pass by lacking it).

    unsupervised   feature-eng-hierarchical-regimes   needs FEATURE_ENG_REGIMES_DEMO_DIR (a fitted DEVELOPMENT reference)

Four properties per family: typed output (every question answered or refused by NAME, no bare number), the receipt
(provider and state travel with the answer, `execution_authorized` is false), persistence (a NEW application on the
SAME state directory returns the identical stored message) and reproducibility (the same envelope under another request
id gives the identical answers -- an answer that changes between asks is not an answer).
"""

import json
import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from m5phet.interpret import Interpreter
from m5phet.web.app import create_app
from m5phet.web.engine import Engine


class Mute(Interpreter):
    """No language model: the envelope is explicit, so nothing may be interpreted, and nothing here may need one."""

    def __init__(self):
        super().__init__(command="fixture", model="fixture-v1", environ={})

    @property
    def available(self):
        return False


#: area -> (provider name, environment variable naming its fitted state, title fragment of its shipped example,
#:          the envelope, what each question must come back as)
FAMILIES = {
    "unsupervised": {
        "provider": "feature-eng-hierarchical-regimes",
        "needs_env": "FEATURE_ENG_REGIMES_DEMO_DIR",
        "example": "OHLC",
        "task": {"area": "unsupervised", "state": {},
                 "questions": {"segmentacion": {"type": "clustering", "method": "auto"},
                               "perfil": {"type": "cluster_description", "target_metric": "highest body_pipettes"}}},
        "types": {"segmentacion": "clustering", "perfil": "cluster_description"},
    },
}


def required(area):
    return area in {a.strip() for a in os.environ.get("M5PHET_E2E_REQUIRE", "").split(",") if a.strip()}


def unavailable(area, why):
    if required(area):
        pytest.fail(f"{area} is REQUIRED on this host and cannot be proven: {why}")
    pytest.skip(f"{area}: {why}")


def wait(client, cid, mid, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found = [m for m in client.get(f"/api/chats/{cid}").json()["messages"] if m["id"] == mid]
        if found and found[0]["status"] not in ("QUEUED", "RUNNING"):
            return found[0]
        time.sleep(0.2)
    pytest.fail("the run did not finish")


def make_client(state_dir):
    engine = Engine()
    engine.interpreter = Mute()
    return TestClient(create_app(state_dir, engine=engine), base_url="http://127.0.0.1"), engine


def example_for(engine, family):
    if family["provider"] not in engine.registry.names():
        unavailable(family["area"], f"provider {family['provider']!r} is not installed (entry point group m5phet.providers)")
    if family["needs_env"] not in os.environ:
        unavailable(family["area"], f"{family['needs_env']} does not name a fitted state")
    found = [e for e in engine.catalog()["examples"] if family["example"].lower() in e["title"].lower()
             and e["config"]["provider"] == family["provider"]]
    if not found:
        unavailable(family["area"], f"the provider ships no example whose title contains {family['example']!r}")
    return found[0]


def run(client, example, family, client_id):
    cid = client.post("/api/chats", json={"title": f"e2e {family['area']}"}).json()["id"]
    client.patch(f"/api/chats/{cid}", json={"title": f"e2e {family['area']}", "config": example["config"]})
    data = example["data"] if isinstance(example["data"], str) else json.dumps(example["data"])
    fid = client.post(f"/api/chats/{cid}/files", files={"file": ("rows.json", data.encode(), "application/json")}).json()["id"]
    sent = client.post(f"/api/chats/{cid}/tasks/run",
                       json={"prompt": example["prompt"], "task": family["task"], "file_ids": [fid],
                             "client_id": client_id, "language": "en"})
    assert sent.status_code == 202, sent.text
    return cid, wait(client, cid, sent.json()["message_id"])


@pytest.mark.parametrize("area", sorted(FAMILIES))
def test_a_real_provider_answers_stores_and_is_read_back_unchanged(area, tmp_path):
    family = dict(FAMILIES[area], area=area)
    state = tmp_path / "state"
    client, engine = make_client(state)
    with client:
        example = example_for(engine, family)
        cid, message = run(client, example, family, "e2e-1")
        assert message["status"] == "OK", message["content"]
        detail = message["detail"]
        answers = detail["response"]["answers"]
        # typed output: every question answered under its own name and type, or refused by a NAMED code
        assert sorted(answers) == sorted(family["task"]["questions"])
        for name, expected in family["types"].items():
            assert answers[name].get("type") == expected, answers[name]
            assert "refused" not in answers[name] and answers[name].get("status") != "REFUSED"
        # the receipt: no authority, and who answered
        assert detail["execution_authorized"] is False
        assert family["provider"] in json.dumps(detail)
        first = message
        # reproducibility: the same envelope under another request id gives the identical answers
        cid2, second = run(client, example, family, "e2e-2")
        assert second["status"] == "OK"
        assert second["detail"]["response"]["answers"] == answers
    # persistence: a NEW application on the SAME state directory returns the stored message byte for byte
    again, _ = make_client(state)
    with again:
        stored = [m for m in again.get(f"/api/chats/{cid}").json()["messages"] if m["id"] == first["id"]][0]
        assert json.dumps({k: stored[k] for k in ("status", "content", "detail")}, sort_keys=True) == \
            json.dumps({k: first[k] for k in ("status", "content", "detail")}, sort_keys=True)
        assert stored["detail"]["execution_authorized"] is False
