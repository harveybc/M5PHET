"""CB05: the public start-up and request path, in a FRESH process, positive and negative.

Every case below runs `tests/backend_startup_probe.py` in a child process with nothing inherited but a prepared
environment, a temporary HOME and a temporary state directory. Nothing is monkeypatched inside the product: the
transport is faked where a transport belongs, by putting an `ssh` of our own first on PATH, so the code under test is
the shipped code taking the shipped path.

What must hold, and what each case is a counterexample to:

* the contradiction refuses at START-UP, and no server exists to answer from either backend;
* a declared remote worker that is not bound refuses at start-up, by name;
* a declared remote worker whose transport FAILS lets the product start -- a transport is not a configuration, and a
  worker that was busy for seven seconds must be retried rather than turned into a refusal to exist -- but every
  classification path then refuses BY NAME, and the in-process fixture never answers in its place. That is the
  hidden fallback this file exists to forbid;
* a worker that describes itself as a fixture under a real-weights mode is refused, not served;
* the two unambiguous configurations work, and their receipts record the backend and the checkpoint that answered.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("news_signal")

PROBE = Path(__file__).resolve().parent / "backend_startup_probe.py"
REPO = PROBE.parent.parent

#: never a real host. The operator's worker is only ever the value of $M5PHET_CHAT_LAYA_WORKER in his own environment.
WORKER_PLACEHOLDER = "worker.invalid"
WORKER_COMMAND = "m5phet-cb05-fake-worker"
FAKE_CHECKPOINT = "laya-checkpoint:" + "e" * 64
QUESTION = "Which economy is named in this news?"

#: an `ssh` of our own, first on PATH. It answers `describe` from CB05_FAKE_WORKER and binds an `infer` result to the
#: request digest exactly as the real worker does, so nothing is proven by a transport that could not have passed.
FAKE_SSH = '''#!{python}
import json, os, sys
mode = os.environ.get("CB05_FAKE_WORKER", "ok")
if mode == "fail":
    sys.stderr.write("cb05 fake worker: refusing to connect\\n")
    sys.exit(255)
command = json.load(sys.stdin)
caps = {{"provider": "laya_news", "operations": ["infer"], "families": ["classification"],
        "output_kinds": ["typed_questions"], "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
        "supported": [{{"operation": "infer", "family": "classification", "output_kind": "typed_questions"}}],
        "known_states": [{checkpoint!r}], "question_types": {{"choice": {{"required": ["options"]}}}},
        "tasks": ["news_relevance"], "device": "cuda:0", "backend": "laya", "weights_present": True,
        "quality": "NOT_MEASURED"}}
if mode == "fixture":
    caps = dict(caps, backend="fixture", weights_present=False)
if mode == "other_checkpoint":
    caps = dict(caps, known_states=["laya-checkpoint:" + "f" * 64])
if command.get("action") == "describe":
    print(json.dumps({{"capabilities": caps, "discovery": {{"registered": ["laya_news"], "refused": {{}}}}}}))
    sys.exit(0)
sys.path.insert(0, {repo!r} + "/src")
from m5phet.runtime import request_digest
if command.get("action") == "infer":
    request = command["request"]
    print(json.dumps({{"status": "OK", "request_sha256": request_digest(request), "execution_authorized": False,
                      "outputs": {{"answer": {{"status": "OK", "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES",
                                            "payload": {{"label": "euro_area", "probabilities": {{"euro_area": 0.51}}}}}}}},
                      "fitted_state_ref": caps["known_states"][0]}}))
    sys.exit(0)
print(json.dumps({{"transport_error": "cb05 fake worker: unsupported action"}}))
'''

#: the variables this contract owns. Cleared in every child, so no case can pass because the operator's own shell
#: happened to hold one of them.
OWNED = ("NEWS_SIGNAL_BACKEND", "NEWS_SIGNAL_CHECKPOINT", "NEWS_SIGNAL_MANIFEST", "NEWS_SIGNAL_DEVICE",
         "NEWS_SIGNAL_GPU_UUID", "M5PHET_CHAT_LAYA_WORKER", "M5PHET_CHAT_LAYA_COMMAND", "M5PHET_CLASSIFICATION_MODE",
         "M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT", "M5PHET_CONFIG", "M5PHET_CHAT_TOKEN")


def run_probe(tmp_path, environment, *, ask=None, worker="ok"):
    """One child process, one prepared environment, one JSON line back."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    ssh = fake / "ssh"
    ssh.write_text(FAKE_SSH.format(python=sys.executable, checkpoint=FAKE_CHECKPOINT, repo=str(REPO)))
    ssh.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    child = {key: value for key, value in os.environ.items() if key not in OWNED}
    child.update({"HOME": str(home), "PATH": f"{fake}:{os.environ.get('PATH', '')}",
                  "CB05_FAKE_WORKER": worker, "PYTHONPATH": str(REPO / "src")})
    child.update({key: value for key, value in environment.items() if value is not None})
    argv = [sys.executable, str(PROBE), "--state", str(tmp_path / "state")]
    if ask:
        argv += ["--ask", ask]
    done = subprocess.run(argv, capture_output=True, text=True, env=child, timeout=600, cwd=str(tmp_path))
    line = (done.stdout.strip().splitlines() or [""])[-1]
    try:
        report = json.loads(line)
    except json.JSONDecodeError:
        raise AssertionError(f"probe printed no JSON line\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-4000:]}")
    return done.returncode, report, done.stderr


# --- negative: the configuration that caused this work ---------------------------------------------------------------

def test_a_fixture_selection_with_a_bound_worker_refuses_to_start(tmp_path):
    code, report, _ = run_probe(tmp_path, {"NEWS_SIGNAL_BACKEND": "fixture",
                                           "M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER,
                                           "M5PHET_CHAT_LAYA_COMMAND": WORKER_COMMAND})
    assert code == 3
    assert report["startup"] == "REFUSED"
    assert report["code"] == "CLASSIFICATION_BACKEND_CONTRADICTION"
    assert "NEWS_SIGNAL_BACKEND" in report["detail"] and "M5PHET_CHAT_LAYA_WORKER" in report["detail"]
    assert WORKER_PLACEHOLDER not in json.dumps(report)


def test_a_declared_remote_mode_with_no_worker_bound_refuses_to_start(tmp_path):
    code, report, _ = run_probe(tmp_path, {"M5PHET_CLASSIFICATION_MODE": "remote_worker"})
    assert code == 3 and report["code"] == "CLASSIFICATION_WORKER_NOT_BOUND"


def test_two_real_weights_selections_refuse_to_start_until_one_mode_is_declared(tmp_path):
    binding = {"NEWS_SIGNAL_BACKEND": "laya", "M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER,
               "M5PHET_CHAT_LAYA_COMMAND": WORKER_COMMAND}
    code, report, _ = run_probe(tmp_path, binding)
    assert code == 3 and report["code"] == "CLASSIFICATION_MODE_REQUIRED"


# --- negative: the worker that is there but does not answer ------------------------------------------------------------

def test_a_failed_worker_refuses_the_question_by_name_and_no_fixture_answers(tmp_path):
    code, report, _ = run_probe(tmp_path, {"M5PHET_CLASSIFICATION_MODE": "remote_worker",
                                           "M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER,
                                           "M5PHET_CHAT_LAYA_COMMAND": WORKER_COMMAND},
                                ask=QUESTION, worker="fail")
    assert code == 0 and report["startup"] == "OK"
    effective = report["classification_backend"]
    assert effective["mode"] == "remote_worker"
    assert effective["status"] == "CLASSIFICATION_WORKER_UNREACHABLE"
    # the catalog must not publish the locally installed provider's capabilities in the worker's place
    assert report["providers"]["laya_news"]["refused"].startswith("CLASSIFICATION_WORKER_UNREACHABLE")
    assert report["providers"]["laya_news"]["backend"] is None
    answer = report["answer"]
    assert answer["status"] == "REFUSED"
    assert "CLASSIFICATION_WORKER_UNREACHABLE" in answer["content"]
    assert answer["outputs"] == []                       # nothing was answered, by anything


def test_a_worker_that_describes_itself_as_a_fixture_is_refused_under_a_real_mode(tmp_path):
    code, report, _ = run_probe(tmp_path, {"M5PHET_CLASSIFICATION_MODE": "remote_worker",
                                           "M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER,
                                           "M5PHET_CHAT_LAYA_COMMAND": WORKER_COMMAND},
                                ask=QUESTION, worker="fixture")
    assert code == 0
    assert report["classification_backend"]["status"] == "CLASSIFICATION_BACKEND_MISMATCH"
    assert report["answer"]["status"] == "REFUSED"
    assert "CLASSIFICATION_BACKEND_MISMATCH" in report["answer"]["content"]
    assert report["answer"]["outputs"] == []


def test_a_pinned_checkpoint_other_than_the_one_serving_is_refused(tmp_path):
    code, report, _ = run_probe(tmp_path, {"M5PHET_CLASSIFICATION_MODE": "remote_worker",
                                           "M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER,
                                           "M5PHET_CHAT_LAYA_COMMAND": WORKER_COMMAND,
                                           "M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT": FAKE_CHECKPOINT},
                                ask=QUESTION, worker="other_checkpoint")
    assert code == 0
    assert report["classification_backend"]["status"] == "CLASSIFICATION_CHECKPOINT_MISMATCH"
    assert report["answer"]["status"] == "REFUSED"


# --- positive: the two unambiguous configurations, and what their receipts record --------------------------------------

def test_the_declared_fixture_starts_answers_and_records_a_fixture(tmp_path):
    code, report, _ = run_probe(tmp_path, {"NEWS_SIGNAL_BACKEND": "fixture", "NEWS_SIGNAL_DEVICE": "cpu"},
                                ask=QUESTION)
    assert code == 0 and report["startup"] == "OK"
    effective = report["classification_backend"]
    assert effective["mode"] == "fixture" and effective["status"] == "VALIDATED"
    assert effective["backend"] == "fixture" and effective["weights_present"] is False
    assert report["providers"]["laya_news"]["backend"] == "fixture"
    answer = report["answer"]
    assert answer["status"] == "OK", answer
    assert answer["classification_backend"]["backend"] == "fixture"
    assert answer["classification_backend"]["answered_by"] == "in_process"
    assert answer["classification_backend"]["checkpoint"]
    assert answer["outputs"] == ["answer"]


def test_the_declared_remote_worker_starts_answers_and_records_its_checkpoint(tmp_path):
    code, report, _ = run_probe(tmp_path, {"M5PHET_CLASSIFICATION_MODE": "remote_worker",
                                           "M5PHET_CHAT_LAYA_WORKER": WORKER_PLACEHOLDER,
                                           "M5PHET_CHAT_LAYA_COMMAND": WORKER_COMMAND,
                                           "M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT": FAKE_CHECKPOINT},
                                ask=QUESTION, worker="ok")
    assert code == 0 and report["startup"] == "OK"
    effective = report["classification_backend"]
    assert effective["mode"] == "remote_worker" and effective["status"] == "VALIDATED"
    assert effective["backend"] == "laya" and effective["weights_present"] is True
    assert effective["checkpoint"] == FAKE_CHECKPOINT
    assert effective["checkpoint_pinned"] is True
    answer = report["answer"]
    assert answer["status"] == "OK", answer
    assert answer["classification_backend"]["answered_by"] == "remote_worker"
    assert answer["classification_backend"]["checkpoint"] == FAKE_CHECKPOINT
    assert answer["outputs"] == ["answer"]
    assert WORKER_PLACEHOLDER not in json.dumps(report)
