#!/usr/bin/env python3
"""AP01: an answer a real provider gave is still that answer after the process comes back, and a family that cannot
be served is never offered.

Four properties of ONE conversation, proven against a workbench this script starts and stops itself, on its own port
and its own state directory. It never signals a process it did not start.

    input        an envelope the person sends, with the attachment they attached;
    output       the answer, with the receipt of WHAT answered beside it;
    persistence  the same answer read back out of the store, byte for byte;
    restart      the process stopped and a NEW one started on the SAME state directory -- the conversation, the
                 answer, the receipt and the narration are unchanged, and a request the stop interrupted says so by
                 name instead of waiting for ever or pretending to have finished.

    python3 tools/verify_restart.py --launcher <script> --port 8771 --state <dir> [--case <name>] --out report.json

`--launcher` is an executable that runs the workbench in the FOREGROUND, reading PORT and STATE from its environment.
Nothing in this file knows where anything is installed, which is what lets the same proof run against any build.
"""

import argparse
import hashlib
import http.cookiejar
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

RUNNING = ("QUEUED", "RUNNING")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Client:
    """One cookie jar per process, so a restart is a new session exactly as a person's browser would find it."""

    def __init__(self, base):
        self.base = base
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def call(self, method, path, body=None, raw=None, filename=None, timeout=300):
        if raw is not None:
            boundary = "----m5phetap01"
            payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                       f"Content-Type: application/octet-stream\r\n\r\n").encode() + raw + f"\r\n--{boundary}--\r\n".encode()
            request = urllib.request.Request(self.base + path, data=payload, method="POST",
                                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                                      "Origin": self.base})
        else:
            request = urllib.request.Request(self.base + path,
                                             data=json.dumps(body).encode() if body is not None else None,
                                             method=method,
                                             headers={"Content-Type": "application/json", "Origin": self.base})
        with self.opener.open(request, timeout=timeout) as response:
            text = response.read().decode()
        return json.loads(text) if text.strip() else None

    def login(self, token):
        if token:
            self.call("POST", "/api/login", {"token": token})


class Workbench:
    """A workbench process this script owns: it starts it, it stops it, and it never signals another one."""

    def __init__(self, launcher, port, state, case, log):
        self.launcher, self.port, self.state, self.case, self.log = launcher, port, state, case, log
        self.process = None

    def start(self):
        environment = dict(os.environ, PORT=str(self.port), STATE=str(self.state))
        if self.case:
            environment["AP_CASE"] = self.case
        self.handle = open(self.log, "ab")
        self.process = subprocess.Popen([self.launcher], env=environment, stdout=self.handle, stderr=self.handle,
                                        start_new_session=True)
        base = f"http://127.0.0.1:{self.port}"
        for _ in range(60):
            if self.process.poll() is not None:
                raise RuntimeError(f"the workbench exited with {self.process.returncode} before serving; see {self.log}")
            try:
                urllib.request.urlopen(base + "/api/catalog", timeout=2)
                return
            except urllib.error.HTTPError:
                return                      # 401 is the instance answering: it is up and it wants the owner's cookie
            except OSError:
                time.sleep(1.0)
        raise RuntimeError("the workbench never answered on its port")

    def stop(self, grace=25):
        """SIGTERM to the process group this script created, and nothing else. `pid` is always one we started."""
        if self.process is None or self.process.poll() is not None:
            return self.process.returncode if self.process else None
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=10)
        self.handle.close()
        return self.process.returncode


def offered_areas(task_catalog):
    """What `GET /api/tasks/catalog` puts in front of a person: per area, the provider and the types it offers."""
    return {area: {"provider": entry.get("provider"),
                   "question_types": sorted(entry.get("question_types") or {}),
                   "unavailable": entry.get("unavailable"),
                   "error": entry.get("error")}
            for area, entry in sorted((task_catalog.get("areas") or {}).items())}


def example_named(catalog, fragment):
    for example in catalog.get("examples") or []:
        if fragment.lower() in example["title"].lower():
            return example
    raise LookupError(f"no shipped example whose title contains {fragment!r}")


def point_bundle(catalog, target):
    """Which fitted bundle this envelope will name, read from the catalog the running instance publishes.

    Not a constant: three configured bundles serve `Global_active_power` on this installation (a point head, a
    quantile head and the searched representation), so a `point_forecast` that names none is refused
    `STATE_REQUIRED` -- and that refusal is the rule working, not a fault. The harness reads the declaration and
    names ONE, so the answer it proves is an answer and not a refusal, and it records which one answered."""
    bundles = [b for provider in catalog.get("providers") or []
               if provider.get("name") == "predictor_forecast"
               for b in ((provider.get("capabilities") or {}).get("bundles") or [])
               if target in (b.get("targets") or ())]
    point = sorted((b for b in bundles if "quantile" not in (b.get("heads") or ())),
                   key=lambda b: str(b.get("state_id")))
    if not point:
        raise LookupError(f"no configured point bundle serves {target!r}")
    return point[0]


def message_view(chat, message_id):
    found = [m for m in chat["messages"] if m["id"] == message_id]
    return found[0] if found else None


def answered_by(message):
    """WHICH engine answered, read from the stored receipt rather than from anybody's configuration."""
    detail = message.get("detail") or {}
    response = detail.get("response") or {}
    return {"provider": response.get("provider"), "backend": response.get("backend"),
            "fitted_state_ref": response.get("fitted_state_ref"),
            "classification_backend": (detail.get("classification_backend") or {}).get("mode"),
            "execution_authorized": detail.get("execution_authorized"),
            "quality": (response.get("quality") or {}).get("status") if isinstance(response.get("quality"), dict)
                       else response.get("quality")}


def run_and_wait(client, cid, task, prompt, file_ids, client_id, seconds=300):
    sent = client.call("POST", f"/api/chats/{cid}/tasks/run",
                       {"prompt": prompt, "task": task, "file_ids": file_ids, "client_id": client_id, "language": "es"})
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(1.0)
        message = message_view(client.call("GET", f"/api/chats/{cid}"), sent["message_id"])
        if message and message["status"] not in RUNNING:
            return sent["message_id"], message
    return sent["message_id"], {"status": "TIMEOUT", "content": "", "detail": {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--state", required=True)
    parser.add_argument("--case", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-interrupt", action="store_true",
                        help="do not prove the interrupted-request case (it stops the process mid-answer)")
    arguments = parser.parse_args()
    base = f"http://127.0.0.1:{arguments.port}"
    token = os.environ.get("M5PHET_CHAT_TOKEN", "")
    log = os.path.join(os.path.dirname(arguments.state.rstrip("/")) or ".", f"workbench-{arguments.port}.log")
    report = {"schema": "m5phet.ap01_restart.v1", "case": arguments.case, "port": arguments.port,
              "state_dir": arguments.state, "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    bench = Workbench(arguments.launcher, arguments.port, arguments.state, arguments.case, log)
    bench.start()
    report["process_1_pid_owned_by_this_script"] = True
    try:
        client = Client(base)
        client.login(token)
        catalog = client.call("GET", "/api/catalog")
        tasks = client.call("GET", "/api/tasks/catalog")
        report["before_restart"] = {
            "classification_backend": catalog.get("classification_backend"),
            "classification_selection": catalog.get("classification_selection"),
            "offered": offered_areas(tasks),
        }
        # --- input: the shipped household example, attached exactly as the `+` button attaches it -------------------
        example = example_named(catalog, "household-power")
        chat = client.call("POST", "/api/chats", {"title": "AP01 restart proof"})
        cid = chat["id"]
        client.call("PATCH", f"/api/chats/{cid}", {"title": "AP01 restart proof",
                                                   "config": dict(catalog["defaults"], **example["config"])})
        blob = example["data"] if isinstance(example["data"], str) else json.dumps(example["data"])
        attached = client.call("POST", f"/api/chats/{cid}/files", raw=blob.encode(), filename="window.json")
        bundle = point_bundle(catalog, "Global_active_power")
        report["bundle_named"] = {"state_id": bundle.get("state_id"), "state_ref": bundle.get("state_ref"),
                                  "heads": bundle.get("heads")}
        forecast = {"area": "forecasting", "state": {"target_variable": "Global_active_power"},
                    "questions": {"prediccion": {"type": "point_forecast", "horizon": 60,
                                                 "bundle": bundle.get("state_id") or bundle["state_ref"]}}}
        mid, message = run_and_wait(client, cid, forecast, "pronostica la potencia a 60 pasos", [attached["id"]],
                                    "ap01-forecast")
        report["chat_id"], report["message_id"], report["attachment_sha256"] = cid, mid, attached["sha256"]
        report["before_restart"]["answer"] = {
            "status": message["status"], "content": message["content"],
            "answers": ((message.get("detail") or {}).get("response") or {}).get("answers"),
            "answered_by": answered_by(message),
            "detail_sha256": digest(message.get("detail")),
            "message_sha256": digest({k: message[k] for k in ("status", "content", "detail")}),
        }
        # --- a request the restart interrupts: submitted, then the process stopped while it runs -------------------
        interrupted_id = None
        if not arguments.skip_interrupt:
            sent = client.call("POST", f"/api/chats/{cid}/tasks/run",
                               {"prompt": "pronostica otra vez", "task": forecast, "file_ids": [attached["id"]],
                                "client_id": "ap01-interrupted", "language": "es"})
            interrupted_id = sent["message_id"]
            time.sleep(float(os.environ.get("AP_INTERRUPT_AFTER", "0.8")))
            report["before_restart"]["interrupted_status_at_stop"] = \
                message_view(client.call("GET", f"/api/chats/{cid}"), interrupted_id)["status"]
        report["stop"] = {"signal": "SIGTERM", "returncode": bench.stop()}
    finally:
        bench.stop()

    # --- restart: a NEW process, the SAME state directory, a NEW session ---------------------------------------------
    again = Workbench(arguments.launcher, arguments.port, arguments.state, arguments.case, log)
    again.start()
    try:
        client = Client(base)
        client.login(token)
        catalog = client.call("GET", "/api/catalog")
        tasks = client.call("GET", "/api/tasks/catalog")
        chats = client.call("GET", "/api/chats")
        chat = client.call("GET", f"/api/chats/{report['chat_id']}")
        message = message_view(chat, report["message_id"])
        after = {
            "classification_backend": catalog.get("classification_backend"),
            "offered": offered_areas(tasks),
            "chat_listed": any(entry["id"] == report["chat_id"] for entry in chats),
            "attachment_present": [f["sha256"] for f in chat["files"]] == [report["attachment_sha256"]],
            "answer": {"status": message["status"], "content": message["content"],
                       "answers": ((message.get("detail") or {}).get("response") or {}).get("answers"),
                       "answered_by": answered_by(message),
                       "detail_sha256": digest(message.get("detail")),
                       "message_sha256": digest({k: message[k] for k in ("status", "content", "detail")})},
        }
        if interrupted_id:
            recovered = message_view(chat, interrupted_id)
            after["interrupted"] = {"status": recovered["status"], "content": recovered["content"]}
            # and a person who sends the same request again must not be handed the interrupted one for ever
            try:
                retry = client.call("POST", f"/api/chats/{report['chat_id']}/tasks/run",
                                    {"prompt": "pronostica otra vez", "task": forecast_task(chat, interrupted_id),
                                     "file_ids": [f["id"] for f in chat["files"]],
                                     "client_id": "ap01-interrupted", "language": "es"})
                after["interrupted"]["retry_message_id"] = retry["message_id"]
                after["interrupted"]["retry_is_the_same_message"] = retry["message_id"] == interrupted_id
                # and it must RUN: a 202 for work that never begins is a promise nobody keeps
                deadline = time.monotonic() + 300
                final = None
                while time.monotonic() < deadline:
                    time.sleep(1.0)
                    final = message_view(client.call("GET", f"/api/chats/{report['chat_id']}"), retry["message_id"])
                    if final and final["status"] not in RUNNING:
                        break
                after["interrupted"]["after_resend"] = {
                    "status": (final or {}).get("status"),
                    "answers": (((final or {}).get("detail") or {}).get("response") or {}).get("answers"),
                    "answered_by": answered_by(final or {})}
            except urllib.error.HTTPError as error:
                after["interrupted"]["retry_refused"] = f"{error.code}: {error.read().decode()[:200]}"
        report["after_restart"] = after
        before = report["before_restart"]["answer"]
        report["verdict"] = {
            "input_attached": bool(report["attachment_sha256"]),
            "output_answered": before["status"] in ("OK", "PARTIAL"),
            "persisted_identically": before["message_sha256"] == after["answer"]["message_sha256"],
            "conversation_survived_restart": after["chat_listed"] and after["attachment_present"],
            "receipt_survived_restart": before["answered_by"] == after["answer"]["answered_by"],
            "interrupted_named": (after.get("interrupted") or {}).get("status") == "INTERRUPTED"
                                 if interrupted_id else None,
            "interrupted_request_reran": ((after.get("interrupted") or {}).get("after_resend") or {}).get("status")
                                         in ("OK", "PARTIAL") if interrupted_id else None,
            "offer_matches_servability": offer_matches(after["classification_backend"], after["offered"]),
        }
        report["verdict"]["all"] = all(v is True for v in report["verdict"].values() if v is not None)
    finally:
        again.stop()

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if arguments.out:
        with open(arguments.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    print(json.dumps(report["verdict"]))
    return 0 if report["verdict"]["all"] else 1


def forecast_task(chat, message_id):
    """The envelope the interrupted message carried, taken from the store rather than retyped."""
    for message in chat["messages"]:
        if message["id"] == message_id:
            return (message.get("detail") or {}).get("envelope")
    raise LookupError("the interrupted message is not in the conversation")


def offer_matches(effective, offered):
    """A family the product cannot serve must not be offered as if it could be.

    The rule, in one sentence: when the declared classification backend is NOT validated, `GET /api/tasks/catalog`
    must say so on that area and offer no question types for it."""
    if not effective or effective.get("validated"):
        return offered.get("classification", {}).get("question_types") != []
    entry = offered.get("classification") or {}
    return bool(entry.get("unavailable")) and entry.get("question_types") == []


if __name__ == "__main__":
    sys.exit(main())
