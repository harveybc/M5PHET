#!/usr/bin/env python
"""Run WP18 steps 2, 3b, 4 and 5 on one household artifact set, and write the `m5phet.pipeline.v1` spec.

Every choice goes to the real classification checkpoint through `m5phet.web.engine.Engine` -- the same worker route
the workbench uses -- and every decision is written to the record directory as a content-addressed file. Nothing here
fits anything: WP18 step 7 (fit, score, closure table) is a different job and had not run when this was written.

    set -a; source ~/.config/m5phet/chat.env; set +a
    crispdm-run -m 4G -t 900 -n wp18-laya -- python tools/wp18_pipeline.py \
        --metrics feature_metrics.json --groups groups.json --candidates candidates.json \
        --out pipeline_spec.json

Refusals are printed with their names and the run still writes what it could: a plan with a refused feature is a
plan that says which feature has no decision, not a plan with a guess in its place.
"""

import argparse
import json
import sys
from pathlib import Path

from m5phet import pipeline
from m5phet.web.engine import Engine

DEFAULT_RECORDS = "~/.local/state/m5phet/decisions"


def _load(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _probabilities(entry, width=6):
    if entry.get("status") != "OK":
        return f"REFUSED {entry.get('refusal')}: {str(entry.get('why'))[:160]}"
    if not entry.get("probabilities"):
        return f"{entry['chosen']:<{width}} {entry.get('why', '')}"
    ordered = sorted(entry["probabilities"].items(), key=lambda item: -item[1])
    return f"{entry['chosen']:<{width}} " + ", ".join(f"{key} {value}" for key, value in ordered)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wp18_pipeline", description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", required=True, help="the m5phet.feature_metrics.v1 sheet (WP18 step 1)")
    parser.add_argument("--groups", required=True, help="the m5phet.feature_groups.v1 document (WP18 step 3)")
    parser.add_argument("--candidates", required=True, help="the WP06 representation design document")
    parser.add_argument("--candidate", help="which candidate_id to carry; the first one by default")
    parser.add_argument("--records", default=DEFAULT_RECORDS, help=f"where decision records go (default {DEFAULT_RECORDS})")
    parser.add_argument("--out", required=True, help="where the pipeline spec is written")
    parser.add_argument("--replay", action="store_true",
                        help="rebuild from the decision records already in --records instead of asking anything; a "
                             "state with no record is refused by name, and nothing new is asked")
    parser.add_argument("--reask", action="append", default=[],
                        choices=["preprocessing", "grouping", "extractors", "core"],
                        help="with --replay: ask THIS step again against the model and replay the others. Use it "
                             "when a registry changed under one step's records, so that step's choice is made over "
                             "the option list that exists now; repeat the flag for several steps.")
    args = parser.parse_args(argv)
    if args.reask and not args.replay:
        print("REFUSED REASK_NEEDS_REPLAY: --reask says which steps to ask again INSTEAD of replaying them, so it "
              "only means something with --replay", file=sys.stderr)
        return 2

    sheet, groups, design = _load(args.metrics), _load(args.groups), _load(args.candidates)
    candidates = design["candidates"]
    representation = next((c for c in candidates if c.get("candidate_id") == args.candidate), None) if args.candidate \
        else candidates[0]
    if representation is None:
        print(f"REFUSED UNKNOWN_CANDIDATE: {args.candidate!r} is not one of "
              f"{[c.get('candidate_id') for c in candidates]}", file=sys.stderr)
        return 2

    records = Path(args.records).expanduser()
    index = pipeline.load_records(records) if args.replay else None
    reask = set(args.reask)
    replay = {step: (None if step in reask else index) for step in ("preprocessing", "grouping", "extractors", "core")}
    engine = None if (args.replay and not reask) else Engine()
    if index is not None:
        print(f"[replay] {len(index)} recorded decision(s) in {records}; "
              + (f"asking again: {sorted(reask)}" if reask else "nothing will be asked"))
    preprocessors, extractors = pipeline.catalog_preprocessors(), pipeline.catalog_extractors()
    cores = pipeline.catalog_cores()
    for catalog in (preprocessors, extractors, cores):
        print(f"[catalog] {catalog['role']}: {len(catalog['options'])} declared option(s)"
              + (f"; problems: {catalog['problems']}" if catalog["problems"] else ""))

    print("\n[step 2] one decision per feature -- preprocessing")
    plan = pipeline.choose_preprocessing(engine, sheet, preprocessors, record_dir=records,
                                         replay=replay["preprocessing"])
    for feature, entry in plan.get("features", {}).items():
        print(f"  {feature:<22} {_probabilities(entry, 22)}")

    print("\n[step 3b] one decision among the declared cuts")
    grouping = pipeline.confirm_grouping(engine, groups, record_dir=records, replay=replay["grouping"])
    print(f"  cut {_probabilities(grouping, 4)}")

    cut = grouping.get("cut")
    extraction = {"options": pipeline.options(extractors), "groups": {}, "decisions": {}}
    core = {"status": "REFUSED", "options": pipeline.options(cores), "core": pipeline.NOT_AVAILABLE_MULTI_BRANCH,
            "refusal": "GROUPING_REFUSED", "why": "no cut was chosen, so there are no branches to read"}
    if cut is not None:
        print("\n[step 4] one decision per group -- extractor")
        extraction = pipeline.choose_extractors(engine, cut, extractors, record_dir=records,
                                                replay=replay["extractors"])
        for group_id, entry in extraction.get("groups", {}).items():
            members = next(g["members"] for g in cut["groups"] if g["group_id"] == group_id)
            print(f"  {group_id} {','.join(members)}\n      {_probabilities(entry, 10)}")

        print("\n[step 5] one decision over the fused branches -- core")
        core = pipeline.choose_core(engine, cut, cores, record_dir=records, extractor_plan=extraction,
                                    replay=replay["core"])
        if core.get("status") == "OK":
            print(f"  {core['core']} ({core['chosen_by']}): {core.get('why') or _probabilities(core, 10)}")
        else:
            print(f"  {core['refusal']}: {core['why']}")
        if core.get("plan"):
            print(f"  plan: {core['plan']}")

    spec = pipeline.build_pipeline_spec(representation, plan, grouping, extraction, core,
                                        catalogs={"preprocessing": preprocessors, "extractor": extractors,
                                                  "core": cores})
    pipeline.validate_pipeline(spec, catalogs={"preprocessing": preprocessors, "extractor": extractors,
                                               "core": cores}, record_dir=records)
    Path(args.out).expanduser().write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mapping = spec["core"]["encoder_mapping"]
    print(f"\n[core] {json.dumps(spec['core'], sort_keys=True)}")
    print(f"[encoder_mapping] {mapping['status']}: " +
          ", ".join(f"{group}: extractor {branch['extractor']} -> encoder {branch['encoder']}"
                    for group, branch in mapping.get("branches", {}).items()))
    print(f"\n[step 6] {args.out}: representation {representation.get('candidate_id')} "
          f"({spec['representation_id'][:12]}...), {len(spec['decisions'])} decision(s) in {records}")
    print("These are uncalibrated choices, not measurements. Nothing has been fitted; WP18 step 7 has not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
