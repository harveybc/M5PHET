#!/usr/bin/env python3
"""Measure the ROUTER -- the last language-model path in this product with neither a measurement nor a gate.

`tools/measure_interpreter.py` (2026-09-25) scored the interpreter: the model choosing, among values a provider
DECLARED, which target, which horizon, which study an engine should answer about. `orchestrate.route` is the other
path, and it is a different job. There the model is not choosing from a list: it writes a whole envelope -- the area,
the state, the named typed questions -- and what stops a wrong one is `check_proposal`, which refuses an area nobody
serves, a question type no provider declares, a governed value the engine does not have and a column the data lacks.
That refusal bounds the damage. It says nothing at all about how often the model is right, and nobody had measured it.

This harness measures it, on the only labelled corpus that exists: the sentences of `tools/verify_families.py`'s
PROSE list and the prompts of `tools/verify_envelopes.py`'s envelopes, each written against one shipped example whose
correct envelope is therefore known -- the area from the harness, the question types from the envelope the harness
runs, the governed values from the engines' own declared vocabularies. No expectation here comes from a model's
answer; that would make the measurement circular.

Each sentence is routed N times through the running workbench's proposal endpoint
(`POST /api/chats/{id}/tasks/propose`, which builds the envelope and runs NOTHING), and each run is scored as one of:

* `CORRECT` -- it validates AND the area, the question types and every governed value are the expected ones;
* `WRONG_AREA` -- it proposed a different engine. Checked first, because which engine was asked is the primary fact;
* `INVALID_PROPOSAL` -- `check_proposal` refused it. The reason is counted by kind, because "it named a question type
  this area does not answer" and "it named a column the data does not have" are different failures;
* `WRONG_TYPE` -- it validates, and asks question types other than the ones the sentence asks for;
* `WRONG_VALUE` -- it validates and asks the right types, and a governed value is wrong or absent;
* `REFUSED` -- nothing was proposed at all: no interpreter, no JSON, malformed JSON, or the model itself declaring
  that no served area fits.

Two rates are reported and neither is ever quoted alone: `reliability` (correct / runs) and
`reliability_when_it_proposed` (correct / runs that produced a proposal), because a refusal to propose is not a
misreading and collapsing them would overstate one or the other.

    python3 tools/measure_route.py --base http://127.0.0.1:8784 --runs 5 --out route_reliability.json

Nothing is fitted, nothing is trained and no engine runs: `propose` stops at the validated envelope.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_families import call, login                                 # noqa: E402  the same client, read once

REPORT_SCHEMA = "m5phet_route_reliability.v1"

#: What each sentence MUST be routed to. `area` and `types` are read off the harness the sentence belongs to --
#: `verify_families.py` binds each sentence to one shipped example, `verify_envelopes.py` carries the envelope each
#: prompt is written beside -- and `values` holds only the governed fields the SENTENCE names, in the engine's own
#: words or through a phrasing the provider declared as an alias. A field the sentence does not name is not scored:
#: the engine settles it, and a model that leaves it out has not misread anything.
#:
#: `fragment` picks the shipped example whose data is attached, exactly as the other harnesses pick it.
CASES = [
    # --- classification. `laya_news` declares no governed values: the sentence IS the question put to the checkpoint
    {"provider": "laya_news", "fragment": "Noticia", "prompt": "Which economy is named in this news?",
     "area": "classification", "types": ["choice"], "values": {}},
    {"provider": "laya_news", "fragment": "Noticia", "prompt": "¿De qué economía habla esta noticia?",
     "area": "classification", "types": ["choice"], "values": {}},
    {"provider": "laya_news", "fragment": "Noticia", "prompt": "de que economia habla y con que tono",
     "area": "classification", "types": ["choice"], "values": {},
     "why": "the envelope tools/verify_envelopes.py runs for this prompt asks two `choice` questions"},

    # --- forecasting. The household bundle is fitted for exactly one pair; "one hour" is 60 steps of a minute series
    {"provider": "predictor_forecast", "fragment": "household-power", "prompt": "predict household power one hour ahead",
     "area": "forecasting", "types": ["point_forecast"],
     "values": {"target": "Global_active_power", "horizon": 60}},
    {"provider": "predictor_forecast", "fragment": "household-power",
     "prompt": "¿cuánta potencia habrá en la próxima hora?",
     "area": "forecasting", "types": ["point_forecast"],
     "values": {"target": "Global_active_power", "horizon": 60}},
    {"provider": "predictor_forecast", "fragment": "direction_cnn",
     "prompt": "what is the direction_long probability at horizon 1?",
     "area": "forecasting", "types": ["point_forecast"], "values": {"target": "direction_long", "horizon": 1}},
    {"provider": "predictor_forecast", "fragment": "direction_cnn",
     "prompt": "¿cuál es la probabilidad de direction_long a horizonte 1?",
     "area": "forecasting", "types": ["point_forecast"], "values": {"target": "direction_long", "horizon": 1}},
    {"provider": "predictor_forecast", "fragment": "household-power",
     "prompt": "pronostica la potencia y dame un rango",
     "area": "forecasting", "types": ["interval", "point_forecast"],
     "values": {"target": "Global_active_power"},
     "why": "the envelope tools/verify_envelopes.py runs for this prompt asks a point_forecast and an interval; the "
            "sentence names no horizon, so no horizon is scored"},

    # --- unsupervised. WP05: "cuerpo alto" / "a large body" is the declared metric `highest body_pipettes`; a
    # sentence that asks only for an assignment names no metric at all
    {"provider": "feature-eng-hierarchical-regimes", "fragment": "OHLC",
     "prompt": "describe el grupo de velas con cuerpo alto",
     "area": "unsupervised", "types": ["cluster_description"], "values": {"target_metric": "highest body_pipettes"}},
    {"provider": "feature-eng-hierarchical-regimes", "fragment": "OHLC",
     "prompt": "describe the cluster with a large body",
     "area": "unsupervised", "types": ["cluster_description"], "values": {"target_metric": "highest body_pipettes"}},
    {"provider": "feature-eng-hierarchical-regimes", "fragment": "OHLC",
     "prompt": "assign hierarchical regimes to these rows",
     "area": "unsupervised", "types": ["clustering"], "values": {}},
    {"provider": "feature-eng-hierarchical-regimes", "fragment": "OHLC",
     "prompt": "asigna los regímenes jerárquicos a estas filas",
     "area": "unsupervised", "types": ["clustering"], "values": {}},
    {"provider": "feature-eng-hierarchical-regimes", "fragment": "OHLC",
     "prompt": "segmenta estas filas y describe el cluster alto",
     "area": "unsupervised", "types": ["cluster_description", "clustering"], "values": {},
     "why": "the envelope tools/verify_envelopes.py runs for this prompt asks a clustering and a cluster_description; "
            "'el cluster alto' names no declared metric, so none is scored"},

    # --- causal. The sentence names the fitted study in the study's own words
    {"provider": "causal_inference", "fragment": "ATE",
     "prompt": "Report ATE of treatment on outcome, with its uncertainty.",
     "area": "causal", "types": ["ate"], "values": {"study": "ATE of treatment on outcome"}},
    {"provider": "causal_inference", "fragment": "ATE",
     "prompt": "¿Cuál es el ATE of treatment on outcome y su incertidumbre?",
     "area": "causal", "types": ["ate"], "values": {"study": "ATE of treatment on outcome"}},
    {"provider": "causal_inference", "fragment": "ATE", "prompt": "cual fue el efecto del tratamiento y en jovenes",
     "area": "causal", "types": ["ate", "cate"], "values": {},
     "why": "the envelope tools/verify_envelopes.py runs for this prompt asks an ate and a cate; the sentence names "
            "no study in the study's own words, so none is scored"},

    # --- rl. The English sentence names the policy; the Spanish one does not, and the engine has one
    {"provider": "trading_policy", "fragment": "observation vector",
     "prompt": "What action does eth_4h_sac_current_stack_anchor_v1 propose?",
     "area": "rl", "types": ["next_action"], "values": {"policy_id": "eth_4h_sac_current_stack_anchor_v1"}},
    {"provider": "trading_policy", "fragment": "market data",
     "prompt": "¿Qué acción propone la política para estas barras?",
     "area": "rl", "types": ["next_action"], "values": {}},
    {"provider": "trading_policy", "fragment": "market data", "prompt": "que accion propone y que retorno espera",
     "area": "rl", "types": ["next_action", "value_estimation"],
     "values": {"policy_id": "eth_4h_sac_current_stack_anchor_v1"},
     "why": "the envelope tools/verify_envelopes.py runs for this prompt asks a next_action and a value_estimation, "
            "with the policy named in its state"},
]

#: the spellings a governed field may carry in an envelope. `state.target_variable` is the owner's spelling of the
#: forecaster's `target`, and `check_proposal` governs both by the same declared list.
FIELD_SPELLINGS = {"target": ("target", "target_variable"), "horizon": ("horizon",), "study": ("study",),
                   "policy_id": ("policy_id",), "target_metric": ("target_metric",)}

#: the six ways one run can end, kept apart because they are six different facts about the router
CORRECT = "CORRECT"
WRONG_AREA = "WRONG_AREA"
WRONG_TYPE = "WRONG_TYPE"
WRONG_VALUE = "WRONG_VALUE"
INVALID_PROPOSAL = "INVALID_PROPOSAL"
REFUSED = "REFUSED"

VERDICTS = (CORRECT, WRONG_AREA, WRONG_TYPE, WRONG_VALUE, INVALID_PROPOSAL, REFUSED)

#: what `check_proposal` refused, by kind. The text is the framework's own; matching it here keeps the tally readable
#: without asking the checker to grow a code it does not need.
PROBLEM_KINDS = (
    ("UNDECLARED_QUESTION_TYPE", re.compile(r"is not one this area answers")),
    ("VALUE_NOT_DECLARED", re.compile(r"is not one the engine has")),
    ("NOT_A_FITTED_COMBINATION", re.compile(r"is not a fitted combination")),
    ("COLUMN_NOT_IN_DATA", re.compile(r"is not in the attached data")),
    ("DATA_REQUIRED", re.compile(r"needs data attached and none is")),
    ("NO_PROVIDER_FOR_AREA", re.compile(r"no installed provider serves area")),
    ("MALFORMED_ENVELOPE", re.compile(r"UNSUPPORTED_AREA|MALFORMED_QUESTION|STATE_REQUIRED|must be a mapping|"
                                      r"non-empty mapping|needs a non-empty name|is not 'm5phet")),
)


def problem_kind(problem):
    for name, pattern in PROBLEM_KINDS:
        if pattern.search(str(problem)):
            return name
    return "OTHER"


def proposed_questions(outcome):
    """The questions the router proposed, from the validated envelope when there is one and the raw proposal when not."""
    for key in ("task", "proposal"):
        candidate = outcome.get(key)
        if isinstance(candidate, dict) and isinstance(candidate.get("questions"), dict):
            return candidate["questions"]
    return {}


def proposed_area(outcome):
    for key in ("task", "proposal"):
        candidate = outcome.get(key)
        if isinstance(candidate, dict) and candidate.get("area"):
            return candidate["area"]
    return None


def proposed_types(outcome):
    return sorted({str(question.get("type")) for question in proposed_questions(outcome).values()
                   if isinstance(question, dict) and question.get("type")})


def named_values(outcome, fields):
    """`{field: value}` for each governed field the envelope names, in a question or in the state, under any spelling."""
    found = {}
    envelope = outcome.get("task") if isinstance(outcome.get("task"), dict) else outcome.get("proposal")
    if not isinstance(envelope, dict):
        return found
    state = envelope.get("state") if isinstance(envelope.get("state"), dict) else {}
    for field in fields:
        for spelling in FIELD_SPELLINGS.get(field, (field,)):
            for question in proposed_questions(outcome).values():
                if isinstance(question, dict) and spelling in question:
                    found[field] = question[spelling]
                    break
            if field in found:
                break
            if spelling in state:
                found[field] = state[spelling]
                break
    return found


def value_problems(expected, outcome):
    """`{field: "ABSENT"|["got", "wanted"]}` for every governed value the sentence names and the envelope does not."""
    got = named_values(outcome, list(expected))
    problems = {}
    for field, wanted in expected.items():
        if field not in got:
            problems[field] = "ABSENT"
            continue
        value = got[field]
        if isinstance(wanted, int) and not isinstance(wanted, bool):
            try:
                value = int(value)
            except (TypeError, ValueError):
                problems[field] = {"got": got[field], "wanted": wanted}
                continue
        if value != wanted:
            problems[field] = {"got": got[field], "wanted": wanted}
    return problems


def verdict_of(case, outcome):
    """How one routing ended, and why. Pure: a fixture proposal scores here exactly as a live one does.

    The order is the order of the facts. Nothing proposed at all is `REFUSED`. A proposal for another engine is
    `WRONG_AREA` whether or not it would also have failed validation, because which engine was asked is the first
    thing a reader needs. Then `check_proposal`'s own refusal. Then the types, then the governed values."""
    detail = {"area": proposed_area(outcome), "types": proposed_types(outcome),
              "values": named_values(outcome, list(case.get("values") or {}))}
    status = outcome.get("status")
    if status == "REFUSED" or (outcome.get("proposal") is None and outcome.get("task") is None):
        return REFUSED, {**detail, "why": outcome.get("why")}
    if detail["area"] != case["area"]:
        return WRONG_AREA, {**detail, "wanted_area": case["area"]}
    if status == "INVALID_PROPOSAL" or outcome.get("task") is None:
        problems = list(outcome.get("problems") or [])
        return INVALID_PROPOSAL, {**detail, "problems": problems,
                                  "problem_kinds": sorted({problem_kind(p) for p in problems})}
    if detail["types"] != sorted(case["types"]):
        return WRONG_TYPE, {**detail, "wanted_types": sorted(case["types"])}
    problems = value_problems(case.get("values") or {}, outcome)
    if problems:
        return WRONG_VALUE, {**detail, "value_problems": problems}
    return CORRECT, detail


# --- driving the running workbench ------------------------------------------------------------------------------------

def upload(base, cid, example):
    data = example.get("data")
    if not data:
        return []
    suffix = {"json": "json", "csv": "csv"}.get(example["config"].get("input"), "txt")
    blob = data if isinstance(data, str) else json.dumps(data)
    uploaded = call(base, "POST", f"/api/chats/{cid}/files", raw=blob.encode(), filename=f"example.{suffix}")
    return [uploaded["id"]]


def prepare(base, example, prompt):
    """One chat with this example's data attached, reused for every run of the same sentence. Nothing is recorded:
    `propose` is a preview, and the envelope it returns has not run."""
    cid = call(base, "POST", "/api/chats", {"title": prompt[:36]})["id"]
    defaults = call(base, "GET", "/api/catalog")["defaults"]
    call(base, "PATCH", f"/api/chats/{cid}", {"title": prompt[:36], "config": dict(defaults, **example["config"])})
    return cid, upload(base, cid, example)


def one_run(base, cid, file_ids, prompt, timeout):
    started = time.monotonic()
    try:
        out = call(base, "POST", f"/api/chats/{cid}/tasks/propose", {"prompt": prompt, "file_ids": file_ids},
                   timeout=timeout)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        return {"status": "REFUSED", "why": f"HTTP {error.code}: {body[:300]}", "proposal": None, "task": None,
                "problems": [], "seconds": round(time.monotonic() - started, 2)}
    except (urllib.error.URLError, OSError) as error:
        return {"status": "REFUSED", "why": f"{type(error).__name__}: {error}", "proposal": None, "task": None,
                "problems": [], "seconds": round(time.monotonic() - started, 2)}
    return {"status": out.get("status"), "why": out.get("why"), "proposal": out.get("proposal"),
            "task": out.get("task"), "problems": out.get("problems") or [],
            "confidence": out.get("confidence"),
            "seconds": round(time.monotonic() - started, 2)}


#: N is part of the protocol and not a detail: a rate over three routings of a sentence and a rate over five are two
#: different measurements, and the report states N per sentence, in the summary and beside every sentence.
PROTOCOL = ("each sentence of tools/measure_route.py CASES is routed N times through "
            "POST /api/chats/{id}/tasks/propose, which builds the envelope and runs nothing. A run is CORRECT when "
            "the proposal validates against check_proposal AND its area, the set of its question types and every "
            "governed value the sentence names are the expected ones. The expectations come from the harnesses the "
            "sentences belong to (tools/verify_families.py PROSE and tools/verify_envelopes.py's envelopes) and from "
            "the engines' declared vocabularies -- never from a model's answer.")

#: counted from CASES and never written by hand. The first run of this harness (2026-09-25) published "18 sentences"
#: from a hand-written string while its own summary counted 19; a corpus size that can disagree with the corpus is a
#: fact about the report nobody should have to notice, so it is computed.
CORPUS = (f"tools/verify_families.py PROSE and tools/verify_envelopes.py prompts ({len(CASES)} sentences; "
          f"expectations in tools/measure_route.py CASES)")


def fingerprint(case, base, runs):
    """What a stored sentence result is only valid for: this case, this instance, this many runs.

    A checkpoint written for a different expectation, a different workbench or a different N is not a partial result
    of THIS measurement, and resuming from it would publish a rate over a mixture of protocols."""
    import hashlib
    payload = json.dumps({"case": case, "base": base, "runs": runs, "protocol": PROTOCOL},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def checkpoint_path(directory, case):
    import hashlib
    name = hashlib.sha256(f"{case['provider']}|{case['prompt']}".encode()).hexdigest()[:16]
    return Path(directory) / f"sentence_{name}.json"


def load_checkpoint(directory, case, base, runs):
    """A sentence already measured under exactly this protocol, or None. Nothing partial is ever resumed."""
    if not directory:
        return None
    path = checkpoint_path(directory, case)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if stored.get("fingerprint") != fingerprint(case, base, runs):
        return None
    entry = stored.get("sentence")
    return entry if isinstance(entry, dict) and entry.get("runs") == runs else None


def save_checkpoint(directory, case, base, runs, entry):
    """One sentence's N runs, written the moment they are complete.

    A measurement that only exists in memory until the last sentence is a measurement one kill loses entirely; this
    one is written per sentence, so a restart re-does at most the sentence that was in flight."""
    if not directory:
        return
    path = checkpoint_path(directory, case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": REPORT_SCHEMA + ".checkpoint",
                                "fingerprint": fingerprint(case, base, runs), "sentence": entry},
                               indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def measure(base, runs, timeout, checkpoint=None):
    catalog = call(base, "GET", "/api/catalog")
    report = {"schema": REPORT_SCHEMA, "base": base, "runs_per_sentence": runs,
              "measured_at": datetime.now(timezone.utc).isoformat(),
              "interpreter": catalog.get("interpreter") or {},
              "route": catalog.get("route"),
              "abstention": (catalog.get("abstention") or {}).get("paths", {}).get("route"),
              "protocol": PROTOCOL, "corpus": CORPUS, "sentences": []}
    examples = catalog["examples"]
    for case in CASES:
        chosen = [e for e in examples if e["config"]["provider"] == case["provider"]
                  and case["fragment"].lower() in e["title"].lower()]
        if not chosen:
            report["sentences"].append({**{k: case[k] for k in ("provider", "prompt", "area")},
                                        "kind": "NO_EXAMPLE", "runs": 0, "correct": 0, "verdicts": {},
                                        "why": f"no shipped example of {case['provider']} whose title contains "
                                               f"{case['fragment']!r}"})
            continue
        resumed = load_checkpoint(checkpoint, case, base, runs)
        if resumed is not None:
            report["sentences"].append(dict(resumed, resumed_from_checkpoint=True))
            continue
        cid, file_ids = prepare(base, chosen[0], case["prompt"])
        results, verdicts = [], []
        for _ in range(runs):
            outcome = one_run(base, cid, file_ids, case["prompt"], timeout)
            verdict, detail = verdict_of(case, outcome)
            verdicts.append(verdict)
            results.append({"verdict": verdict, "detail": detail, "status": outcome.get("status"),
                            "seconds": outcome.get("seconds"), "why": outcome.get("why")})
        distinct = sorted({json.dumps({"area": r["detail"].get("area"), "types": r["detail"].get("types"),
                                       "values": r["detail"].get("values")}, sort_keys=True, ensure_ascii=False)
                           for r in results})
        entry = {
            "provider": case["provider"], "prompt": case["prompt"], "example": chosen[0]["title"],
            "kind": "ROUTED", "area": case["area"], "types": sorted(case["types"]),
            "values": case.get("values") or {}, "why_expected": case.get("why"),
            "runs": len(results), "correct": verdicts.count(CORRECT),
            "verdicts": {name: verdicts.count(name) for name in sorted(set(verdicts))},
            "distinct_resolutions": distinct, "stable": len(distinct) == 1,
            "seconds_mean": round(sum(r["seconds"] or 0 for r in results) / len(results), 2),
            "results": results}
        save_checkpoint(checkpoint, case, base, runs, entry)
        report["sentences"].append(entry)

    routed = [s for s in report["sentences"] if s["kind"] == "ROUTED"]
    total = sum(s["runs"] for s in routed)
    tally = {name: sum(s["verdicts"].get(name, 0) for s in routed) for name in VERDICTS}
    kinds = {}
    for sentence in routed:
        for result in sentence["results"]:
            for kind in (result["detail"].get("problem_kinds") or ()):
                kinds[kind] = kinds.get(kind, 0) + 1
    proposed = total - tally[REFUSED]
    report["summary"] = {
        "sentences": len(routed),
        "sentences_without_an_example": sum(1 for s in report["sentences"] if s["kind"] == "NO_EXAMPLE"),
        "runs": total, "correct": tally[CORRECT], "verdicts": tally,
        "invalid_proposal_problems": kinds,
        "reliability": round(tally[CORRECT] / total, 4) if total else None,
        "reliability_when_it_proposed": round(tally[CORRECT] / proposed, 4) if proposed else None,
        "sentences_always_correct": sum(1 for s in routed if s["correct"] == s["runs"] and s["runs"]),
        "sentences_never_correct": sum(1 for s in routed if s["correct"] == 0 and s["runs"]),
        "sentences_stable": sum(1 for s in routed if s["stable"]),
        # a resumed sentence was measured under the same protocol, the same instance and the same N, or it was not
        # resumed at all; the count is published so a reader knows the run was not one uninterrupted sitting
        "sentences_resumed_from_checkpoint": sum(1 for s in routed if s.get("resumed_from_checkpoint")),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8784")
    parser.add_argument("--runs", type=int, default=5, help="how many times each sentence is routed (declare it)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--out")
    parser.add_argument("--token", default=os.environ.get("M5PHET_CHAT_TOKEN"))
    parser.add_argument("--checkpoint", help="directory to write each sentence's runs to as they complete, and to "
                                             "resume from. A stored sentence is reused only when the case, the "
                                             "instance, N and the protocol are identical")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.token:
        login(args.base, args.token)
    try:
        report = measure(args.base, args.runs, args.timeout, checkpoint=args.checkpoint)
    except (urllib.error.URLError, OSError) as error:
        print(f"the workbench is not answering at {args.base}: {error}")
        return 2

    identity = report["interpreter"] or {}
    print(f"router: {identity.get('plugin')} · {identity.get('model')} · confidence="
          f"{(report.get('abstention') or {}).get('confidence')} · {report['measured_at'][:10]} · "
          f"{args.runs} runs per sentence")
    print(f"{'area':<15} {'correct':>8}  {'verdicts':<40} {'s/call':>7} sentence")
    for row in report["sentences"]:
        if row["kind"] == "NO_EXAMPLE":
            print(f"{row['area']:<15} {'NO_EXAMPLE':>8}  {'-':<40} {'-':>7} {row['prompt'][:40]}")
            continue
        verdicts = " ".join(f"{name}:{count}" for name, count in sorted(row["verdicts"].items()))
        print(f"{row['area']:<15} {row['correct']}/{row['runs']:<6}  {verdicts:<40} {row['seconds_mean']:>7} "
              f"{row['prompt'][:40]}")
    resumed = report["summary"].get("sentences_resumed_from_checkpoint") or 0
    if resumed:
        print(f"({resumed} sentence(s) read from the checkpoint directory, measured earlier under the same protocol, "
              f"the same instance and the same N)")
    print(json.dumps(report["summary"], ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
