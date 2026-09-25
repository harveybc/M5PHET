#!/usr/bin/env python3
"""Measure the INTERPRETER, which nothing in this repository had ever measured.

Every quality number M5PHET has written is about an engine: WP09 scored the classification checkpoint on 450
independently labelled rows, the closure tables score forecasts against naive baselines on held-out rows. The
interpreter -- the language model that reads a person's sentence and picks, among the values a provider DECLARED,
which target, which horizon, which study, which policy the engine is to answer about -- was never scored at all. It
was argued about ("it can only choose among declared values, so the worst case is a refusal"), which is a statement
about the blast radius of its mistakes and not about how often it makes one.

This harness measures how often it makes one. The framework already owns the only labelled corpus that exists for the
task: the 14 sentences of `tools/verify_families.py`'s PROSE list, each written for one shipped example, each with a
resolution that is known because the engine behind it has exactly one fitted answer to give.

It measures TWO things, and keeps them apart because they answer different questions.

**The product path** runs each sentence N times through the RUNNING workbench's review window
(`POST /api/chats/{id}/preview`, which resolves the sentence into the typed request and runs nothing) and counts how
often the resolved parameters are the expected ones. That is what a person typing gets -- and on this corpus it is
mostly not the interpreter's doing at all: `m5phet.interpret` runs a deterministic pass first, and when the declared
values and aliases settle every field, no model is consulted.

**The interpreter path** asks the configured interpreter the same sentence with the same declared vocabulary and
scores ITS answer -- `Interpreter.propose`, exactly what `interpret()` would call if the words settled nothing,
validated the same way (a value outside the declared list is not a near miss, it is a refusal). This is the number
that was missing: the interpreter's own reliability, with no deterministic pass in front of it.

A miss there is not one thing, so it is not reported as one. `WRONG_VALUE` is the model choosing a declared value and
choosing the wrong one -- the only outcome that is a misreading of the sentence. `DECLINED` is it returning null,
which is what the instruction asks for when the sentence genuinely does not name a value, and two of the fourteen
sentences are exactly that case (the field is then settled by the engine, not by the model). `OUTSIDE_DECLARED` is a
value nobody declared, which `interpret()` refuses rather than rounding. So the report carries the strict rate
(`reliability` = correct / runs) and, beside it, `reliability_when_it_chose` = correct / (correct + wrong). Quoting
either one alone would be a different claim.

Three things the product path reports separately, because collapsing them would overstate what was measured:

* a sentence whose parameters the DETERMINISTIC pass settles never reaches the interpreter. `m5phet.interpret` says so
  in `sources` (`QUESTION_TEXT` against `INTERPRETER`), and a sentence with no `INTERPRETER` source in any run is
  reported as `DETERMINISTIC` and left out of the interpreter's own rate. It is not evidence about the model;
* a provider that declares no slots (classification) has no parameters to resolve, and its sentences are `NO_SLOTS`;
* a field the engine has exactly one value for cannot distinguish a model that read the sentence from one that picked
  the only option. Those fields are counted and named `SINGLE_VALUED` in the report, so nobody reads the rate as
  harder than it is.

    python3 tools/measure_interpreter.py --base http://127.0.0.1:8780 --runs 5 --out interpreter_reliability.json

Nothing is fitted, nothing is trained and no engine is run: `preview` stops at the resolved request.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_families import PROSE, call, login                          # noqa: E402  the same corpus, read once

REPORT_SCHEMA = "m5phet_interpreter_reliability.v1"

#: What each sentence MUST resolve to, and why that is the answer. The expectation is taken from the engine's own
#: declared vocabulary and from the example the sentence is written against -- never from what the model happened to
#: answer, which would make the measurement circular. Only the fields named here are scored; a field the engine
#: settles on its own (which bundle serves a fitted pair, which fitted reference a regime task belongs to) is recorded
#: in the report and not counted as a question the sentence asked.
#:
#: Keyed by the sentence exactly as `verify_families.PROSE` carries it.
EXPECTED = {
    # classification: `laya_news` declares no slots. The sentence IS the question put to the checkpoint, so there is
    # no parameter for an interpreter to choose and nothing here to measure.
    "Which economy is named in this news?": None,
    "¿De qué economía habla esta noticia?": None,

    # forecasting: the household bundle is fitted for exactly one pair, and "one hour" is 60 steps of a minute series
    "predict household power one hour ahead": {"target": "Global_active_power", "horizon": 60},
    "¿cuánta potencia habrá en la próxima hora?": {"target": "Global_active_power", "horizon": 60},
    # the direction bundle: the sentence names both the target and the horizon in the engine's own words
    "what is the direction_long probability at horizon 1?": {"target": "direction_long", "horizon": 1},
    "¿cuál es la probabilidad de direction_long a horizonte 1?": {"target": "direction_long", "horizon": 1},

    # unsupervised: WP05's point. "cuerpo alto" / "a large body" is the declared metric `highest body_pipettes`, and
    # a sentence that asks for an assignment and no description is `none (assignment only)` -- which the sentence does
    # not spell out either, so a null from the model is a DECLINED and not a misreading
    "describe el grupo de velas con cuerpo alto": {"target_metric": "highest body_pipettes"},
    "describe the cluster with a large body": {"target_metric": "highest body_pipettes"},
    "assign hierarchical regimes to these rows": {"target_metric": "none (assignment only)"},
    "asigna los regímenes jerárquicos a estas filas": {"target_metric": "none (assignment only)"},

    # causal: the sentence names the fitted study in the study's own words
    "Report ATE of treatment on outcome, with its uncertainty.": {"study": "ATE of treatment on outcome"},
    "¿Cuál es el ATE of treatment on outcome y su incertidumbre?": {"study": "ATE of treatment on outcome"},

    # rl: one fitted policy. The English sentence names it. The Spanish one does NOT name it -- the request still has
    # to resolve to it, because the engine has one policy and the slot is optional, but a model that returns null here
    # is declining rather than misreading, which is why DECLINED is tallied apart from WRONG_VALUE.
    "What action does eth_4h_sac_current_stack_anchor_v1 propose?":
        {"policy_id": "eth_4h_sac_current_stack_anchor_v1"},
    "¿Qué acción propone la política para estas barras?":
        {"policy_id": "eth_4h_sac_current_stack_anchor_v1"},
}

UNSUPPORTED = "UNSUPPORTED_VALUE"
NOT_PROPOSED = "NOT_PROPOSED"
INTERPRETER_FAILED = "INTERPRETER_FAILED"

NO_SLOTS = "NO_SLOTS"
DETERMINISTIC = "DETERMINISTIC"
INTERPRETED = "INTERPRETED"
NO_EXAMPLE = "NO_EXAMPLE"


def upload(base, cid, example):
    data = example.get("data")
    if not data:
        return []
    suffix = {"json": "json", "csv": "csv"}.get(example["config"].get("input"), "txt")
    blob = data if isinstance(data, str) else json.dumps(data)
    uploaded = call(base, "POST", f"/api/chats/{cid}/files", raw=blob.encode(), filename=f"example.{suffix}")
    return [uploaded["id"]]


def prepare(base, example, prompt):
    """One chat, configured for this example, with its data attached. Reused for every run of the same sentence."""
    cid = call(base, "POST", "/api/chats", {"title": prompt[:36]})["id"]
    defaults = call(base, "GET", "/api/catalog")["defaults"]
    call(base, "PATCH", f"/api/chats/{cid}", {"title": prompt[:36], "config": dict(defaults, **example["config"])})
    return cid, upload(base, cid, example)


def one_run(base, cid, file_ids, prompt, timeout):
    """One resolution of one sentence. `preview` builds the request the sentence means and runs nothing."""
    started = time.monotonic()
    try:
        out = call(base, "POST", f"/api/chats/{cid}/preview", {"prompt": prompt, "file_ids": file_ids},
                   timeout=timeout)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        return {"status": "ERROR", "why": body[:300], "parameters": {}, "sources": {},
                "seconds": round(time.monotonic() - started, 3)}
    except (urllib.error.URLError, OSError) as error:
        return {"status": "ERROR", "why": f"{type(error).__name__}: {error}", "parameters": {}, "sources": {},
                "seconds": round(time.monotonic() - started, 3)}
    interpretation = out.get("interpretation") or {}
    return {"status": out.get("status") or ("REFUSED" if out.get("request") is None else "OK"),
            "refusal": out.get("refusal"),
            "why": out.get("why"),
            "parameters": interpretation.get("parameters") or {},
            "sources": interpretation.get("sources") or {},
            "interpreted": interpretation.get("status"),
            "seconds": round(time.monotonic() - started, 3)}


def scored(expected, parameters):
    """A run matches when every field the sentence was written to settle came out as the expected value."""
    return all(parameters.get(field) == value for field, value in expected.items())


def single_valued(catalog_areas, provider, fields):
    """The scored fields the engine has exactly one value for -- where any choice is the right one."""
    for area in catalog_areas.values():
        if area.get("provider") == provider:
            allowed = area.get("parameters") or {}
            return sorted(f for f in fields if len(allowed.get(f) or []) == 1)
    return []


def measure(base, runs, timeout):
    catalog = call(base, "GET", "/api/catalog")
    areas = call(base, "GET", "/api/tasks/catalog")["areas"]
    report = {"schema": REPORT_SCHEMA, "base": base, "runs_per_sentence": runs,
              "measured_at": datetime.now(timezone.utc).isoformat(),
              "interpreter": catalog.get("interpreter"),
              "abstention": (catalog.get("abstention") or {}).get("interpreter"),
              "protocol": ("each sentence of tools/verify_families.py PROSE is sent N times to "
                           "POST /api/chats/{id}/preview, which resolves it into the typed request and runs nothing; "
                           "a run matches when every field named in tools/measure_interpreter.py EXPECTED resolves to "
                           "its expected value. Expectations come from the engines' declared vocabularies and the "
                           "shipped example each sentence is written against, never from a model's answer."),
              "sentences": []}

    for provider, fragment, prompt in PROSE:
        expected = EXPECTED.get(prompt)
        chosen = [e for e in catalog["examples"]
                  if e["config"]["provider"] == provider and fragment.lower() in e["title"].lower()]
        if not chosen:
            report["sentences"].append({"provider": provider, "prompt": prompt, "kind": NO_EXAMPLE,
                                        "why": f"no shipped example of {provider} whose title contains {fragment!r}",
                                        "runs": 0, "matched": 0})
            continue
        if expected is None:
            report["sentences"].append({"provider": provider, "prompt": prompt, "kind": NO_SLOTS, "runs": 0,
                                        "matched": 0, "example": chosen[0]["title"],
                                        "why": "this provider declares no slots; the sentence is the question itself "
                                               "and carries no parameter for an interpreter to choose"})
            continue
        cid, file_ids = prepare(base, chosen[0], prompt)
        results = [one_run(base, cid, file_ids, prompt, timeout) for _ in range(runs)]
        matched = sum(1 for r in results if r["status"] == "OK" and scored(expected, r["parameters"]))
        by_interpreter = sorted({field for r in results for field, source in r["sources"].items()
                                 if source == "INTERPRETER" and field in expected})
        distinct = sorted({json.dumps({f: r["parameters"].get(f) for f in expected}, sort_keys=True)
                           for r in results})
        report["sentences"].append({
            "provider": provider, "prompt": prompt, "example": chosen[0]["title"],
            "kind": INTERPRETED if by_interpreter else DETERMINISTIC,
            "expected": expected, "runs": len(results), "matched": matched,
            "scored_fields": sorted(expected),
            "fields_the_model_chose": by_interpreter,
            "single_valued_fields": single_valued(areas, provider, sorted(expected)),
            "distinct_resolutions": distinct,
            "stable": len(distinct) == 1,
            "seconds_mean": round(sum(r["seconds"] for r in results) / len(results), 2),
            "results": results})

    interpreted = [s for s in report["sentences"] if s["kind"] == INTERPRETED]
    deterministic = [s for s in report["sentences"] if s["kind"] == DETERMINISTIC]
    report["summary"] = {
        "sentences_declared": len(PROSE),
        "sentences_with_parameters": len(interpreted) + len(deterministic),
        "sentences_no_slots": sum(1 for s in report["sentences"] if s["kind"] == NO_SLOTS),
        "sentences_without_an_example": sum(1 for s in report["sentences"] if s["kind"] == NO_EXAMPLE),
        # the interpreter's own reliability: only the sentences where a model actually chose a scored field
        "interpreter_sentences": len(interpreted),
        "interpreter_runs": sum(s["runs"] for s in interpreted),
        "interpreter_matched": sum(s["matched"] for s in interpreted),
        "interpreter_reliability": (round(sum(s["matched"] for s in interpreted)
                                          / sum(s["runs"] for s in interpreted), 4)
                                    if sum(s["runs"] for s in interpreted) else None),
        "interpreter_sentences_stable": sum(1 for s in interpreted if s["stable"]),
        # the words alone, measured the same way and reported apart: this is not evidence about the model
        "deterministic_sentences": len(deterministic),
        "deterministic_runs": sum(s["runs"] for s in deterministic),
        "deterministic_matched": sum(s["matched"] for s in deterministic),
        # every sentence that has an expectation, model-chosen or not: what a person typing sees
        "end_to_end_runs": sum(s["runs"] for s in interpreted + deterministic),
        "end_to_end_matched": sum(s["matched"] for s in interpreted + deterministic),
    }
    return report


def slots_for(area, scored):
    """The slot list the interpreter is shown, rebuilt from what the instance PUBLISHES about its own engines.

    `/api/tasks/catalog` carries each area's declared values and the ordinary phrasings for them -- the same two
    things `m5phet.interpret` hands a model. Rebuilding them here keeps this harness on the product's published
    vocabulary instead of importing five providers, and it is the vocabulary the running instance actually has."""
    allowed = area.get("parameters") or {}
    aliases = area.get("aliases") or {}
    out = []
    for name in sorted(allowed):
        if name not in scored:
            continue                                    # only the fields this sentence is scored on are asked about
        values = list(allowed[name])
        slot = {"name": name, "allowed": values, "aliases": aliases.get(name) or {}}
        if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            slot["type"] = "integer"
        out.append(slot)
    return out


def validated(proposed, slots):
    """What `m5phet.interpret` would KEEP of this proposal: a value outside the declared list is not kept at all."""
    out, problems = {}, {}
    for slot in slots:
        value = proposed.get(slot["name"]) if isinstance(proposed, dict) else None
        if value is None:
            problems[slot["name"]] = NOT_PROPOSED
            continue
        if slot.get("type") == "integer":
            try:
                value = int(value)
            except (TypeError, ValueError):
                problems[slot["name"]] = UNSUPPORTED
                continue
        allowed = [int(v) for v in slot["allowed"]] if slot.get("type") == "integer" else list(slot["allowed"])
        if value not in allowed:
            problems[slot["name"]] = UNSUPPORTED
            continue
        out[slot["name"]] = value
    return out, problems


#: the four ways one run can end, kept apart because they are four different facts about the model
CORRECT = "CORRECT"
#: it chose a declared value and chose the wrong one -- the only outcome that is a mistake about the sentence
WRONG_VALUE = "WRONG_VALUE"
#: it returned null: "the request does not name one". Declining is what the instruction asks for when the sentence
#: really does not name a value, so this is NOT counted as a wrong reading -- it is counted as not having read one
DECLINED = "DECLINED"
#: it proposed something the provider never declared. `interpret()` refuses it rather than rounding to a neighbour,
#: so the person sees a refusal; it is recorded here because it is a different failure from choosing the wrong option
OUTSIDE_DECLARED = "OUTSIDE_DECLARED"

VERDICTS = (CORRECT, WRONG_VALUE, DECLINED, OUTSIDE_DECLARED, INTERPRETER_FAILED)


def verdict_of(expected, result):
    """How one run ended. A miss is not one thing: declining, choosing wrongly and inventing are three."""
    if result["status"] != "OK":
        return INTERPRETER_FAILED
    if scored(expected, result["parameters"]):
        return CORRECT
    problems = result.get("problems") or {}
    for field in expected:
        if problems.get(field) == UNSUPPORTED:
            return OUTSIDE_DECLARED
    for field in expected:
        if field in result["parameters"]:
            return WRONG_VALUE                  # it had an opinion among the declared values and it was the wrong one
    return DECLINED


def area_of(areas, provider):
    for entry in areas.values():
        if entry.get("provider") == provider:
            return entry
    return {}


def ask_the_interpreter(base, runs, timeout):
    """The interpreter alone: the same sentences, the same declared vocabulary, no deterministic pass in front."""
    from m5phet.interpret import build

    interpreter = build()
    areas = call(base, "GET", "/api/tasks/catalog")["areas"]
    out = {"interpreter": interpreter.identity(), "runs_per_sentence": runs,
           "protocol": ("Interpreter.propose(sentence, slots) is called N times per sentence with the area's declared "
                        "values and aliases as published by /api/tasks/catalog -- what interpret() shows a model when "
                        "the words settle nothing. A proposal outside the declared values is not kept, exactly as "
                        "interpret() does not keep it. No deterministic pass runs first: this is the model's own "
                        "reading of the sentence and nothing else."),
           "sentences": []}
    if not interpreter.available:
        out["refusal"] = "NO_INTERPRETER_CONFIGURED"
        return out
    for provider, _fragment, prompt in PROSE:
        expected = EXPECTED.get(prompt)
        if expected is None:
            out["sentences"].append({"provider": provider, "prompt": prompt, "kind": NO_SLOTS, "runs": 0,
                                     "matched": 0})
            continue
        slots = slots_for(area_of(areas, provider), set(expected))
        if not slots:
            out["sentences"].append({"provider": provider, "prompt": prompt, "kind": NO_SLOTS, "runs": 0,
                                     "matched": 0, "why": "the running instance declares none of the scored fields"})
            continue
        results = []
        for _ in range(runs):
            started = time.monotonic()
            try:
                proposed = interpreter.propose(prompt, slots)
            except Exception as error:                                          # noqa: BLE001
                results.append({"status": INTERPRETER_FAILED, "why": f"{type(error).__name__}: {error}"[:200],
                                "parameters": {}, "problems": {}, "seconds": round(time.monotonic() - started, 2)})
                continue
            kept, problems = validated(proposed, slots)
            results.append({"status": "OK", "parameters": kept, "problems": problems,
                            "proposed": proposed if isinstance(proposed, dict) else None,
                            "seconds": round(time.monotonic() - started, 2)})
        matched = sum(1 for r in results if r["status"] == "OK" and scored(expected, r["parameters"]))
        fields, verdicts = {}, []
        for field, value in expected.items():
            fields[field] = sum(1 for r in results if r["parameters"].get(field) == value)
        for result in results:
            verdicts.append(verdict_of(expected, result))
        distinct = sorted({json.dumps({f: r["parameters"].get(f) for f in expected}, sort_keys=True)
                           for r in results})
        out["sentences"].append({
            "provider": provider, "prompt": prompt, "kind": INTERPRETED, "expected": expected,
            "runs": len(results), "matched": matched, "per_field_matched": fields,
            "verdicts": {name: verdicts.count(name) for name in sorted(set(verdicts))},
            "distinct_resolutions": distinct, "stable": len(distinct) == 1,
            "seconds_mean": round(sum(r["seconds"] for r in results) / len(results), 2),
            "results": results})
    asked = [s for s in out["sentences"] if s["kind"] == INTERPRETED]
    runs_total = sum(s["runs"] for s in asked)
    tally = {name: sum(s["verdicts"].get(name, 0) for s in asked) for name in VERDICTS}
    out["summary"] = {"sentences": len(asked), "runs": runs_total,
                      "matched": sum(s["matched"] for s in asked),
                      "verdicts": tally,
                      # of the runs where the model DID pick a declared value, how often it picked the right one
                      "reliability_when_it_chose": (round(tally[CORRECT] / (tally[CORRECT] + tally[WRONG_VALUE]), 4)
                                                    if tally[CORRECT] + tally[WRONG_VALUE] else None),
                      "reliability": round(sum(s["matched"] for s in asked) / runs_total, 4) if runs_total else None,
                      "sentences_fully_correct": sum(1 for s in asked if s["matched"] == s["runs"] and s["runs"]),
                      "sentences_never_correct": sum(1 for s in asked if s["matched"] == 0 and s["runs"]),
                      "sentences_stable": sum(1 for s in asked if s["stable"]),
                      "failed_calls": sum(1 for s in asked for r in s["results"]
                                          if r["status"] == INTERPRETER_FAILED)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8780")
    parser.add_argument("--runs", type=int, default=5, help="how many times each sentence is resolved (declare it)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--out")
    parser.add_argument("--token", default=os.environ.get("M5PHET_CHAT_TOKEN"))
    parser.add_argument("--product-only", action="store_true",
                        help="skip the interpreter path (which calls the configured model once per sentence per run)")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.token:
        login(args.base, args.token)
    try:
        report = measure(args.base, args.runs, args.timeout)
        if not args.product_only:
            report["interpreter_path"] = ask_the_interpreter(args.base, args.runs, args.timeout)
    except (urllib.error.URLError, OSError) as error:
        print(f"the workbench is not answering at {args.base}: {error}")
        return 2

    identity = report["interpreter"] or {}
    print(f"interpreter: {identity.get('plugin')} · {identity.get('model')} · "
          f"reports_confidence={identity.get('reports_confidence')} · {report['measured_at'][:10]} · "
          f"{args.runs} runs per sentence")
    print("\nA. product path -- the sentence through the review window, deterministic pass first")
    print(f"{'provider':<34} {'kind':<14} {'match':>7}  {'stable':<7} sentence")
    for row in report["sentences"]:
        matched = f"{row['matched']}/{row['runs']}" if row["runs"] else "-"
        stable = ("yes" if row.get("stable") else "NO") if row["runs"] else "-"
        print(f"{row['provider']:<34} {row['kind']:<14} {matched:>7}  {stable:<7} {row['prompt'][:46]}")
    print(json.dumps(report["summary"]))
    path = report.get("interpreter_path")
    if path:
        print("\nB. interpreter path -- the same sentences asked of the configured model alone")
        if path.get("refusal"):
            print(f"   {path['refusal']}")
        print(f"{'provider':<34} {'match':>7}  {'verdicts':<34} {'s/call':>7} sentence")
        for row in path["sentences"]:
            if row["kind"] == NO_SLOTS:
                print(f"{row['provider']:<34} {NO_SLOTS:>7}  {'-':<34} {'-':>7} {row['prompt'][:38]}")
                continue
            verdicts = " ".join(f"{name}:{count}" for name, count in sorted(row["verdicts"].items()))
            print(f"{row['provider']:<34} {row['matched']}/{row['runs']:<5}  {verdicts:<34} "
                  f"{row['seconds_mean']:>7} {row['prompt'][:38]}")
        print(json.dumps(path.get("summary")))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
