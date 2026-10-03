"""I1: a family is claimed only when a REAL provider answers a typed envelope, the answer is stored, and it is read back
unchanged by a process that never saw it computed.

Every other test in this repository drives the web layer with a fake provider. This one loads the provider an
installation really has, by its entry point, and gives it the shipped example's own data and its own fitted state. It
needs the provider's package and a fitted state; where either is absent the case is SKIPPED BY NAME (and
`M5PHET_E2E_REQUIRE=<area>[,<area>]` turns that skip into a failure, so a host that is meant to prove an area cannot
pass by lacking it).

    unsupervised   feature-eng-hierarchical-regimes   needs FEATURE_ENG_REGIMES_DEMO_DIR (a fitted DEVELOPMENT reference)
    causal         causal_inference                   needs CAUSAL_INFERENCE_STATE_DIR (retained EconML studies)

The causal case (2026-10-03) names the study by its digest reference, attaches NOTHING (that provider never fits during
inference and refuses an attached dataset), pins the clock, and checks more than the type: every number of the `ate`
answer is compared with the study artifact read from disk independently of the provider, the `cate` question asked of
a study fitted without an effect modifier must come back REFUSED with `NOT_ESTIMABLE` and no number, and a clock before
the study existed must refuse every question.

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
    "causal": {
        "provider": "causal_inference",
        "needs_env": "CAUSAL_INFERENCE_STATE_DIR",
        # the constant-effect synthetic study; its title is written by the provider from the study's own declarations
        "example": "confounded ATE",
        "attach": False,
        "as_of": "2026-10-01T00:00:00+00:00",
        "early_as_of": "2000-01-01T00:00:00+00:00",
        "task": {"area": "causal", "state": {},
                 "questions": {"efecto": {"type": "ate"},
                               "subgrupo": {"type": "cate", "subgroup": "baseline == 1"}}},
        "types": {"efecto": "ate"},
        "refused": {"subgrupo": ("cate", "NOT_ESTIMABLE")},
        # one answered and one refused by name: the workbench says PARTIAL, never OK over a refusal
        "message_status": "PARTIAL",
    },
}


def state_for(family, example):
    """The envelope's state: the regimes case carries none; the causal case names the example's study by digest."""
    if family["area"] == "causal":
        return {"state_ref": example["config"]["state"]}
    return family["task"]["state"]


def check_values(family, answers, example):
    """Values, not only types: each number of a causal answer is the retained artifact's, read here from disk."""
    if family["area"] != "causal":
        return
    ref = example["config"]["state"]
    digest = ref.split(":", 1)[1]
    raw = (__import__("pathlib").Path(os.environ["CAUSAL_INFERENCE_STATE_DIR"]) / f"{digest}.json").read_bytes()
    assert __import__("hashlib").sha256(raw).hexdigest() == digest
    study = json.loads(raw)
    payload = study["result"]["payload"]
    ate = answers["efecto"]
    assert ate["status"] == "OK" and ate["state_ref"] == ref
    assert ate["effect_size"] == payload["estimate"]
    assert ate["confidence_interval"] == list(payload["interval"])
    assert ate["confidence_interval"][0] <= ate["effect_size"] <= ate["confidence_interval"][1]
    assert ate["unit"] == payload["unit"] and ate["population"] == study["population"]
    assert ate["assumptions"] == list(payload["assumptions"]) and ate["assumptions"]
    assert ate["execution_authorized"] is False
    assert "p_value" not in ate or "p_value" in payload
    assert ate["development"] is True        # a synthetic DEVELOPMENT study says so on its answer


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


def envelope(family, example, as_of=None):
    task = dict(family["task"], state=state_for(family, example))
    as_of = as_of or family.get("as_of")
    if as_of:
        task["as_of"] = as_of
    return task


def run(client, example, family, client_id, task=None):
    cid = client.post("/api/chats", json={"title": f"e2e {family['area']}"}).json()["id"]
    client.patch(f"/api/chats/{cid}", json={"title": f"e2e {family['area']}", "config": example["config"]})
    file_ids = []
    if family.get("attach", True):
        data = example["data"] if isinstance(example["data"], str) else json.dumps(example["data"])
        file_ids.append(client.post(f"/api/chats/{cid}/files",
                                    files={"file": ("rows.json", data.encode(), "application/json")}).json()["id"])
    sent = client.post(f"/api/chats/{cid}/tasks/run",
                       json={"prompt": example["prompt"], "task": task or envelope(family, example),
                             "file_ids": file_ids, "client_id": client_id, "language": "en"})
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
        assert message["status"] == family.get("message_status", "OK"), message["content"]
        detail = message["detail"]
        answers = detail["response"]["answers"]
        # typed output: every question answered under its own name and type, or refused by a NAMED code
        assert sorted(answers) == sorted(family["task"]["questions"])
        for name, expected in family["types"].items():
            assert answers[name].get("type") == expected, answers[name]
            assert "refused" not in answers[name] and answers[name].get("status") != "REFUSED"
        # a question the named state cannot answer comes back refused by its NAMED code, carrying no number
        for name, (kind, code) in family.get("refused", {}).items():
            assert answers[name]["status"] == "REFUSED" and answers[name]["refusal"] == code, answers[name]
            assert answers[name]["type"] == kind and answers[name]["why"]
            assert not {"effect_size", "confidence_interval", "estimate"} & set(answers[name])
        check_values(family, answers, example)
        # the receipt: no authority, and who answered
        assert detail["execution_authorized"] is False
        assert family["provider"] in json.dumps(detail)
        first = message
        # reproducibility: the same envelope under another request id gives the identical answers
        cid2, second = run(client, example, family, "e2e-2")
        assert second["status"] == message["status"]
        assert second["detail"]["response"]["answers"] == answers
    # persistence: a NEW application on the SAME state directory returns the stored message byte for byte
    again, _ = make_client(state)
    with again:
        stored = [m for m in again.get(f"/api/chats/{cid}").json()["messages"] if m["id"] == first["id"]][0]
        assert json.dumps({k: stored[k] for k in ("status", "content", "detail")}, sort_keys=True) == \
            json.dumps({k: first[k] for k in ("status", "content", "detail")}, sort_keys=True)
        assert stored["detail"]["execution_authorized"] is False


@pytest.mark.parametrize("area", sorted(a for a in FAMILIES if FAMILIES[a].get("early_as_of")))
def test_a_clock_before_the_state_existed_refuses_every_question(area, tmp_path):
    """Point-in-time: asked at a clock before the fitted state was available, the real provider answers nothing."""
    family = dict(FAMILIES[area], area=area)
    client, engine = make_client(tmp_path / "state")
    with client:
        example = example_for(engine, family)
        _, message = run(client, example, family, "e2e-early", task=envelope(family, example, family["early_as_of"]))
        assert message["status"] == "REFUSED", message["content"]
        answers = message["detail"]["response"]["answers"]
        assert sorted(answers) == sorted(family["task"]["questions"])
        for name, answer in answers.items():
            assert answer["status"] == "REFUSED" and answer["refusal"] == "NOT_ESTIMABLE", answer
            assert "after the requested clock" in answer["why"]
        assert message["detail"]["execution_authorized"] is False
