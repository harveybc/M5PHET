#!/usr/bin/env python3
"""Render the answers a real run produced with every installed output procedure, and prove none of them says more.

    python3 tools/verify_envelopes.py --base http://127.0.0.1:8766 --out /tmp/e.json
    python3 tools/verify_outputs.py  --report /tmp/e.json [--out /tmp/o.json]

`verify_envelopes.py` stores, per envelope, the answers the engines returned -- the point forecast at the bundle's own
precision, the critic value, the named refusals. This harness takes those answers, renders them with each procedure
registered under `m5phet.outputs` (`default` and `telegram` ship), and checks the text with the same guard that checks
a language model's narration: every figure in it must be one the answers carry, a percent must stand on a declared
probability, and no wording may turn a reading into a profit, a loss or an order.

It is the plugin half of the invariant the product rests on. The engines refuse rather than invent; the narrator may
say less but never more; and now the procedure that writes what the person actually reads is held to the same rule,
whoever wrote it.
"""

import argparse
import json
import sys

from m5phet.orchestrate import narration_problems
from m5phet.outputs import load, names, response_view
from m5phet.outputs.telegram import CLOSING, LIMIT


def responses(report):
    """One response per envelope, rebuilt from what the run stored."""
    out = []
    for envelope in report.get("envelopes") or []:
        answers = envelope.get("answers") or {}
        answered = envelope.get("answered")
        refused = envelope.get("refused")
        if answered is None:
            answered = sum(1 for a in answers.values() if isinstance(a, dict) and a.get("status") == "OK")
        if refused is None:
            refused = len(answers) - answered
        out.append({"area": envelope.get("area"), "answers": answers, "answered": answered, "refused": refused,
                    # WP31: the quality block the run's own answer carried, so what is rendered here is what the
                    # person read and not a shape someone typed into a harness
                    "quality": envelope.get("quality")})
    return out


def check(response, plugin_name):
    plugin = load(plugin_name)()
    rendered = plugin.render(response["area"], response, "es")
    text = rendered.get("text") or ""
    problems = narration_problems(text, response_view(response))
    quality_lines = [line for line in text.splitlines() if line.startswith("quality (")]
    if response.get("quality") is not None and len(quality_lines) != 1:
        problems.append(f"the response carries a quality block and this rendering states it on "
                        f"{len(quality_lines)} lines; exactly one is the rule")
    row = {"area": response["area"], "plugin": plugin_name, "questions": list(response["answers"]),
           "characters": len(text), "problems": problems, "text": text,
           "quality_line": quality_lines[0] if quality_lines else None}
    if plugin_name == "telegram":
        if len(text) > LIMIT:
            problems.append(f"the message is {len(text)} characters and the bound is {LIMIT}")
        if not text.endswith(CLOSING):
            problems.append("the message does not end with the execution_authorized line")
    header = plugin.header(response["area"])
    for name, answer in response["answers"].items():
        if isinstance(answer, dict) and answer.get("status") == "REFUSED":
            if answer.get("refusal") not in header["refusal_codes"]:
                problems.append(f"{name}: refusal {answer.get('refusal')!r} is not one this area's header declares "
                                f"({header['refusal_codes']})")
    row["header"] = header
    row["ok"] = not problems
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, help="the --out JSON of a tools/verify_envelopes.py run")
    parser.add_argument("--plugins", default=",".join(sorted(names())),
                        help="comma-separated output plugins to render with (default: every installed one)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    plugins = [p.strip() for p in args.plugins.split(",") if p.strip()]
    rows = []
    for response in responses(report):
        if not response["answers"]:
            rows.append({"area": response["area"], "plugin": None, "ok": False,
                         "problems": ["the run stored no answers for this envelope"], "text": ""})
            continue
        for plugin_name in plugins:
            rows.append(check(response, plugin_name))
    for row in rows:
        mark = "OK " if row["ok"] else "!! "
        first = (row.get("text") or "").splitlines()[0] if row.get("text") else ""
        print(f"{mark}{str(row['area']):<15} {str(row['plugin']):<10} {row['characters'] if 'characters' in row else 0:>5}c "
              f"{first[:96]}")
        for problem in row["problems"]:
            print(f"    problem: {problem}")
    summary = {"schema": "m5phet_output_verification.v1", "plugins": plugins,
               "renderings": len(rows), "renderings_faithful": sum(1 for r in rows if r["ok"]),
               "figures_not_in_the_answers": sum(len(r["problems"]) for r in rows)}
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({**summary, "renderings_detail": rows}, handle, indent=1, ensure_ascii=False, default=str)
    return 0 if rows and all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
