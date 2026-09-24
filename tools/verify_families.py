#!/usr/bin/env python3
"""Drive the running workbench through its HTTP API, exactly as the browser does, once per family.

This is the product's own acceptance run: it starts nothing, imports no provider and reaches no engine directly. It asks the
server the same questions a person would and reports what came back, so a green result means the path a person uses works --
not that a library imports.

It also asks each family a question in ORDINARY words and a question the fitted model cannot answer, because a system that
answers everything is a system that is answering the wrong thing.

    python3 tools/verify_families.py [--base http://127.0.0.1:8765] [--out report.json]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

#: questions that must be REFUSED, and the fragment of the reason that shows it was refused for the right cause
REFUSALS = {
    "predictor_forecast": [
        ("forecast Global_active_power at 90 steps", "only has"),
        # `Voltage` is a column this bundle reads and does not forecast. The bundle declares it as known-and-unsupported,
        # so naming it is refused before any interpreter is consulted -- which is what makes this refusal deterministic.
        ("forecast Voltage at 60 steps", "answering the nearest one would answer a different question"),
    ],
}


def call(base, method, path, body=None, raw=None, filename=None, timeout=300):
    if raw is not None:
        boundary = "----m5phetverify"
        payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                   f"Content-Type: application/octet-stream\r\n\r\n").encode() + raw + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(base + path, data=payload, method="POST",
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                                  "Origin": base})
    else:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json", "Origin": base})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode()
    return json.loads(text) if text.strip() else None


def ask(base, example, prompt, timeout_seconds=180):
    """One question, through the same endpoints the browser uses."""
    config = example["config"]
    chat = call(base, "POST", "/api/chats", {"title": prompt[:36]})
    cid = chat["id"]
    defaults = call(base, "GET", "/api/catalog")["defaults"]
    call(base, "PATCH", f"/api/chats/{cid}", {"title": prompt[:36], "config": dict(defaults, **config)})
    file_ids = []
    data = example.get("data")
    if data:
        suffix = {"json": "json", "csv": "csv"}.get(config.get("input"), "txt")
        blob = data if isinstance(data, str) else json.dumps(data)
        uploaded = call(base, "POST", f"/api/chats/{cid}/files", raw=blob.encode(), filename=f"example.{suffix}")
        file_ids = [uploaded["id"]]
    sent = call(base, "POST", f"/api/chats/{cid}/messages",
                {"prompt": prompt, "file_ids": file_ids, "client_id": "verify-" + cid[:8]})
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(1.0)
        state = call(base, "GET", f"/api/chats/{cid}")
        found = [m for m in state["messages"] if m["id"] == sent["message_id"]]
        if found and found[0]["status"] not in ("QUEUED", "RUNNING"):
            return found[0]
    return {"status": "TIMEOUT", "content": f"no answer within {timeout_seconds}s", "detail": {}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    try:
        catalog = call(args.base, "GET", "/api/catalog")
    except (urllib.error.URLError, OSError) as error:
        print(f"the workbench is not answering at {args.base}: {error}")
        return 2

    report = {"schema": "m5phet_family_verification.v1", "base": args.base,
              "providers": [p["name"] for p in catalog["providers"]],
              "interpreter": catalog.get("interpreter"), "families": [], "refusals": []}
    examples = {e["config"]["provider"]: e for e in catalog["examples"]}
    for provider in sorted(examples):
        example = examples[provider]
        answer = ask(args.base, example, example["prompt"])
        detail = answer.get("detail") or {}
        result = detail.get("result") or {}
        outputs = result.get("outputs") or {}
        first = next(iter(outputs.values()), {}) if outputs else {}
        report["families"].append({
            "provider": provider, "title": example["title"], "prompt": example["prompt"],
            "status": answer.get("status"), "backend": detail.get("backend"),
            "elapsed_seconds": detail.get("elapsed_seconds"),
            "outputs": sorted(outputs), "payload_keys": sorted((first.get("payload") or {})),
            "interpretation": detail.get("interpretation"),
            "execution_authorized": detail.get("execution_authorized"),
            "why": answer.get("content") if answer.get("status") != "OK" else None})

    for provider, cases in REFUSALS.items():
        if provider not in examples:
            continue
        for prompt, fragment in cases:
            answer = ask(args.base, examples[provider], prompt)
            refused = answer.get("status") != "OK"
            report["refusals"].append({"provider": provider, "prompt": prompt, "status": answer.get("status"),
                                       "refused": refused,
                                       "for_the_right_reason": refused and fragment in str(answer.get("content")),
                                       "content": str(answer.get("content"))[:200]})

    # A provider that ships an example a person clicks, which its own slots then refuse, is broken in the way that
    # matters most: the first thing anyone tries fails. Observed for real once, so it is checked every run.
    for family in report["families"]:
        family["example_resolves"] = family["status"] == "OK"

    answered = [f for f in report["families"] if f["status"] == "OK"]
    report["summary"] = {"families_answering": len(answered), "families": len(report["families"]),
                         "examples_that_resolve": sum(1 for f in report["families"] if f["example_resolves"]),
                         "refusals_correct": sum(1 for r in report["refusals"] if r["for_the_right_reason"]),
                         "refusals": len(report["refusals"]),
                         "any_execution_authorized": any(f["execution_authorized"] for f in report["families"])}
    for family in report["families"]:
        mark = "OK " if family["status"] == "OK" else "!! "
        print(f"{mark}{family['provider']:<34} {str(family['status']):<9} {family['title'][:52]}")
        if family["why"]:
            print(f"      {family['why'][:150]}")
    for refusal in report["refusals"]:
        mark = "OK " if refusal["for_the_right_reason"] else "!! "
        print(f"{mark}refusal {refusal['provider']:<26} {refusal['prompt'][:46]}")
    print(json.dumps(report["summary"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
    ok = (report["summary"]["families_answering"] == report["summary"]["families"]
          and report["summary"]["refusals_correct"] == report["summary"]["refusals"]
          and not report["summary"]["any_execution_authorized"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
