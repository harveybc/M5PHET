#!/usr/bin/env python3
"""CB05: start the product in a FRESH process the way the service starts it, and ask it one real question.

Why this exists as a script and not as an in-process test. The backend contradiction of 2026-09-28 was invisible to
every in-process test in this repository, because a test that injects a registry never resolves the operator's
configuration at all, and a test that monkeypatches the transport never finds out which backend the transport would
have reached. The defect lived exactly in the part no in-process test touched: `create_app()` reading the environment
of a process nobody had prepared for it. So this probe takes no fixtures and patches nothing: it imports the public
entry point, lets it read the environment it was given, and prints ONE JSON line saying what happened.

    python3 tests/backend_startup_probe.py --state DIR [--ask "Which economy is named in this news?"]

    {"startup": "REFUSED", "code": "CLASSIFICATION_BACKEND_CONTRADICTION", "detail": "..."}   -> exit 3
    {"startup": "OK", "classification_backend": {...}, "answer": {...}}                       -> exit 0

It is used by `tests/test_backend_startup_process.py` and, unchanged, by the deployment smoke, so the isolated smoke
and the suite exercise the same door.
"""
import argparse
import json
import pathlib
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="a state directory of this probe's own; never the owner's")
    parser.add_argument("--ask", default=None, help="a classification question to actually put to laya_news")
    parser.add_argument("--context", default="The European Central Bank left its deposit facility rate unchanged.")
    args = parser.parse_args(argv)

    report = {"startup": None}
    try:
        from m5phet.web.app import create_app
        app = create_app(pathlib.Path(args.state))
    except Exception as error:                                  # the startup refusal IS the product's answer here
        code = getattr(error, "code", None) or type(error).__name__
        print(json.dumps({"startup": "REFUSED", "code": code, "detail": str(error)}, ensure_ascii=False))
        return 3
    report["startup"] = "OK"

    from fastapi.testclient import TestClient
    with TestClient(app, base_url="http://127.0.0.1") as client:
        catalog = client.get("/api/catalog").json()
        report["classification_backend"] = catalog.get("classification_backend")
        report["providers"] = {entry["name"]: {"backend": (entry.get("capabilities") or {}).get("backend"),
                                               "refused": (entry.get("capabilities") or {}).get("refused")}
                               for entry in catalog.get("providers", [])}
        if args.ask:
            chat = client.post("/api/chats", json={"title": "cb05 probe"}).json()
            config = dict(chat["config"], provider="laya_news", input="text", family="classification",
                          output_kind="typed_questions", context=args.context,
                          options=[["euro_area", "Euro area"], ["united_states", "United States"],
                                   ["other", "Another economy"]])
            patched = client.patch(f"/api/chats/{chat['id']}", json={"config": config})
            report["configured"] = patched.status_code
            client.post(f"/api/chats/{chat['id']}/messages", json={"prompt": args.ask, "client_id": "cb05"})
            for _ in range(600):
                state = client.get(f"/api/chats/{chat['id']}").json()
                last = state["messages"][-1] if state["messages"] else None
                if last and last["status"] != "RUNNING":
                    break
                time.sleep(0.05)
            else:
                print(json.dumps({"startup": "OK", "answer": "DID_NOT_COMPLETE"}))
                return 4
            detail = last.get("detail") or {}
            report["answer"] = {
                "status": last["status"],
                "content": last["content"],
                # what the receipt records about what ANSWERED -- the whole point of the exercise
                "backend": detail.get("backend"),
                "classification_backend": detail.get("classification_backend"),
                # and whether anything was actually produced, so a refusal cannot be mistaken for a quiet answer
                "outputs": sorted((detail.get("result") or {}).get("outputs") or {}),
            }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
