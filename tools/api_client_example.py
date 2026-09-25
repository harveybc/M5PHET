#!/usr/bin/env python3
"""An external program talking to the M5PHET workbench over its local HTTP API, with a bearer token and no browser.

This is the worked example `docs/API.md` refers to: the standard library only, no dependency on this package, so it
can be copied into another system and read as the contract. It walks the whole path a program takes --

    login -> catalog -> create a chat -> upload a CSV from a path -> preview a sentence -> propose an envelope
          -> run it with a fresh client_id -> poll -> read the answers, the refusals and the receipt

-- and with ``--verify`` it drives the envelopes of ``tools/verify_envelopes.py`` (the same specifications, imported
from that file, so the two cannot drift apart) through the token instead of the owner's cookie.

    python3 tools/api_client_example.py --base http://127.0.0.1:8772 --token-file /path/to/token
    python3 tools/api_client_example.py --base http://127.0.0.1:8772 --token-file /path/to/token --verify

Three rules this client keeps, because a client is where they are usually broken:

* the token is read from a FILE and never taken as a command-line argument -- an argument is visible to every other
  process on the machine -- and it is never printed, not in an error and not in a receipt;
* the base URL must be loopback. Sending someone's news, bars or prompts off the machine is a decision a person
  makes explicitly, so a non-loopback URL is refused unless ``--i-know-this-leaves-the-machine`` says otherwise;
* nothing here authorizes anything. Every answer carries ``execution_authorized: false``, and this client prints it.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}
HERE = Path(__file__).resolve().parent


class ApiError(RuntimeError):
    """An HTTP status the API answered with. Carries the status and the server's own `detail`, never the token."""

    def __init__(self, status, path, detail):
        super().__init__(f"{status} on {path}: {detail}")
        self.status, self.path, self.detail = status, path, detail


# --- the transport ------------------------------------------------------------------------------------------------

class Client:
    """Every call carries `Authorization: Bearer <token>`; the login also leaves a cookie the client then reuses."""

    def __init__(self, base, token, timeout=300):
        self.base, self.token, self.timeout = base.rstrip("/"), token, timeout
        import http.cookiejar
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def headers(self, extra=None):
        # Origin is sent because the API refuses a cross-origin request; it is the same origin we are calling.
        head = {"Origin": self.base, "Authorization": f"Bearer {self.token}"}
        head.update(extra or {})
        return head

    def call(self, base, method, path, body=None, raw=None, filename=None, timeout=None):
        """The signature `tools/verify_envelopes.py` uses, so this client can drive that harness unchanged."""
        if raw is not None:
            boundary = "----m5phetapiclient"
            payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                       f"Content-Type: application/octet-stream\r\n\r\n").encode() + raw \
                + f"\r\n--{boundary}--\r\n".encode()
            head = self.headers({"Content-Type": f"multipart/form-data; boundary={boundary}"})
        else:
            payload = json.dumps(body).encode() if body is not None else None
            head = self.headers({"Content-Type": "application/json"})
        request = urllib.request.Request((base or self.base) + path, data=payload, method=method, headers=head)
        try:
            with self.opener.open(request, timeout=timeout or self.timeout) as response:
                text = response.read().decode()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except ValueError:
                pass
            raise ApiError(error.code, path, detail) from None
        return json.loads(text) if text.strip() else None

    # a short form for this file's own calls; `call` keeps the harness's signature
    def get(self, path):
        return self.call(None, "GET", path)

    def post(self, path, body=None, **kw):
        return self.call(None, "POST", path, body, **kw)

    def patch(self, path, body):
        return self.call(None, "PATCH", path, body)


def read_token(path):
    """The token, from the file the operator named. Never echoed; only its length and file are ever reported."""
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(f"{path}: the token file is empty; bearer access is off until it holds a token")
    return token


# --- the walkthrough -----------------------------------------------------------------------------------------------

def loopback(base):
    return (urlsplit(base).hostname or "") in LOOPBACK or (urlsplit(base).hostname or "").startswith("127.")


def choose_spec(harness, catalog):
    """Which of the harness's envelopes the walkthrough uses: one whose example is a CSV, so the upload step is real.

    Today that is the policy example's bars (the regimes example ships JSON rows), and any installation simply walks
    through the first table example its own catalog publishes. The negative-path envelopes are left out: they exist
    to be refused, and a walkthrough should show what an answer looks like."""
    specs = [s for s in harness.envelopes(catalog["examples"], catalog.get("providers") or ())
             if (s["example"].get("config") or {}).get("input") == "csv" and s["example"].get("data")
             and not s.get("data_transform")]
    if not specs:
        raise SystemExit("this instance publishes no table example to walk through; check /api/catalog")
    return next((s for s in specs if s["area"] == "unsupervised"), specs[0])


def answers_of(message):
    detail = message.get("detail") or {}
    return (detail.get("response") or {}).get("answers") or {}


def print_answers(message):
    """What the person (or the program) came for: each answer with its numbers, each refusal with its code."""
    detail = message.get("detail") or {}
    response = detail.get("response") or {}
    for name, answer in answers_of(message).items():
        if answer.get("status") == "REFUSED":
            print(f"    REFUSED {name:<14} {answer.get('type'):<20} {answer.get('refusal')}: "
                  f"{str(answer.get('why'))[:100]}")
        else:
            fields = {k: v for k, v in answer.items() if k not in ("status", "type", "why")}
            print(f"    OK      {name:<14} {answer.get('type'):<20} {json.dumps(fields, default=str)[:160]}")
    print(f"    answered={response.get('answered')} refused={response.get('refused')} "
          f"request_sha256={str(response.get('request_sha256'))[:16]}… "
          f"execution_authorized={detail.get('execution_authorized')}")
    if detail.get("dataset"):
        dataset = detail["dataset"]
        receipt = dataset.get("rows_receipt") or {}
        print(f"    dataset={dataset.get('id')} rows={dataset.get('rows')} chosen_by={dataset.get('source_of_choice')} "
              f"rows_origin={receipt.get('rows_origin')} rows_sha256={str(receipt.get('rows_sha256'))[:16]}…")
    if message.get("content"):
        print(f"    narration: {message['content'][:200]}")


def poll(client, cid, message_id, seconds=240):
    """A run is accepted with 202 and finishes in the background; the chat is where its state is read."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(1.0)
        state = client.get(f"/api/chats/{cid}")
        found = [m for m in state["messages"] if m["id"] == message_id]
        if found and found[0]["status"] not in ("QUEUED", "RUNNING"):
            return found[0]
    return {"status": "TIMEOUT", "content": "", "detail": {}}


def walkthrough(client, spec, catalog, csv_path):
    """Every step of the API in the order a program takes them, printing what each one answered."""
    example = spec["example"]

    print("1. login with the bearer token (the cookie it sets is reused; the token is still sent on every call)")
    client.post("/api/login", {})

    print(f"2. catalog: config_source={catalog.get('config_source')} "
          f"providers={[p['name'] for p in catalog.get('providers', [])]} "
          f"interpreter={(catalog.get('interpreter') or {}).get('model')} "
          f"execution_authorized={catalog.get('execution_authorized')}")
    areas = client.get("/api/tasks/catalog")
    print("   areas:", {name: sorted((spec_.get("question_types") or {})) for name, spec_ in areas["areas"].items()})

    chat = client.post("/api/chats", {"title": "from a program"})
    cid = chat["id"]
    print(f"3. chat created: {cid}")
    client.patch(f"/api/chats/{cid}", {"config": dict(catalog["defaults"], **example["config"])})
    print("   settings applied (the provider, reader and output contract this chat uses)")

    blob = example["data"] if isinstance(example["data"], str) else json.dumps(example["data"])
    if csv_path is None:
        csv_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"m5phet-api-client-{uuid.uuid4().hex[:8]}.csv"
        csv_path.write_text(blob, encoding="utf-8")
    raw = Path(csv_path).read_bytes()
    uploaded = client.post(f"/api/chats/{cid}/files", raw=raw, filename=Path(csv_path).name)
    print(f"4. uploaded {uploaded['name']} ({uploaded['size']} bytes, sha256 {uploaded['sha256'][:16]}…)")
    file_ids = [uploaded["id"]]

    # the example's OWN sentence, because the sentence path resolves against the chat's settings and the provider's
    # declared values: a question that does not name one of them is refused there by name, which is correct and is
    # not what this step is for. The envelope path below takes the freer sentence.
    sentence = example.get("prompt") or spec["prompt"]
    print(f"5. preview {sentence!r}: the sentence resolved into the typed request it WOULD run; nothing ran")
    try:
        preview = client.post(f"/api/chats/{cid}/preview", {"prompt": sentence, "file_ids": file_ids})
        parameters = (preview.get("interpretation") or {}).get("parameters")
        print(f"   ran={preview['ran']} request={json.dumps(preview['request'])[:180]}")
        print(f"   parameters={json.dumps(parameters, default=str)} "
              f"execution_authorized={preview['execution_authorized']}")
    except ApiError as error:
        print(f"   REFUSED {error.status}: {str(error.detail)[:200]}")

    print("6. propose: the same sentence as an envelope, for review before anything runs")
    task = spec["task"]
    try:
        proposal = client.post(f"/api/chats/{cid}/tasks/propose", {"prompt": spec["prompt"], "file_ids": file_ids})
        print(f"   status={proposal['status']} proposed={json.dumps(proposal.get('task'), default=str)[:200]}")
        if proposal.get("problems"):
            print(f"   problems={proposal['problems']}")
        if proposal.get("why"):
            print(f"   why={proposal['why']}")
    except ApiError as error:
        print(f"   REFUSED {error.status}: {str(error.detail)[:200]}")
    print("   this client runs the envelope it declares, which is what a reviewer would have accepted; a program "
          "that trusts the proposal runs proposal['task'] instead")

    client_id = "api-client-" + uuid.uuid4().hex[:12]
    print(f"7. run the envelope with a fresh client_id ({client_id})")
    sent = client.post(f"/api/chats/{cid}/tasks/run",
                       {"prompt": spec["prompt"], "task": task, "file_ids": file_ids,
                        "client_id": client_id, "language": "es"})
    message = poll(client, cid, sent["message_id"])
    print(f"8. finished: status={message['status']}")
    print_answers(message)

    print("9. the client_id rule, both halves:")
    again = client.post(f"/api/chats/{cid}/tasks/run",
                        {"prompt": spec["prompt"], "task": task, "file_ids": file_ids,
                         "client_id": client_id, "language": "es"})
    print(f"   the same id with the same input returns the same message "
          f"({'same' if again['message_id'] == sent['message_id'] else 'DIFFERENT'}), nothing ran twice")
    edited = json.loads(json.dumps(task))
    edited["questions"] = dict(list(edited["questions"].items())[:1])
    # an envelope of one question is unchanged by the line above; then the sentence is what differs, and the rule is
    # the same rule: the request's identity is the prompt, the attachments AND the envelope together
    prompt = spec["prompt"] if edited != task else spec["prompt"] + " (revisado)"
    try:
        client.post(f"/api/chats/{cid}/tasks/run",
                    {"prompt": prompt, "task": edited, "file_ids": file_ids,
                     "client_id": client_id, "language": "es"})
        print("   !! the same id with a DIFFERENT envelope was accepted; it must be 409")
        return 1
    except ApiError as error:
        ok = error.status == 409
        print(f"   {'   ' if ok else '!! '}the same id with a different envelope: {error.status} {error.detail}")
        if not ok:
            return 1
    return 0


# --- the five envelopes, driven with the token ----------------------------------------------------------------------

def load_harness():
    """`tools/verify_envelopes.py` as a module: its envelope specifications and its notion of "as expected"."""
    spec = importlib.util.spec_from_file_location("m5phet_verify_envelopes", HERE / "verify_envelopes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(client, harness, catalog, out=None):
    """The harness's own envelopes, run over the API with the bearer token instead of the owner's cookie."""
    harness.call = client.call                      # same signature; the harness drives the token from here on
    report = {"schema": "m5phet_envelope_verification.v1", "credential": "bearer_token", "envelopes": []}
    for spec in harness.envelopes(catalog["examples"], catalog.get("providers") or ()):
        message = harness.run_envelope(client.base, spec, catalog["defaults"])
        report["envelopes"].append(harness.evaluate(spec, message))
    for entry in report["envelopes"]:
        harness.show(entry)
    status = harness.summarize(report)
    if out:
        Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8765", help="the workbench's base URL (loopback)")
    parser.add_argument("--token-file", default=os.environ.get("M5PHET_API_TOKEN_FILE"),
                        help="file holding the bearer token; defaults to $M5PHET_API_TOKEN_FILE. There is no "
                             "--token: a token in a command line is visible to every process on the machine")
    parser.add_argument("--csv", help="a CSV to upload; without it the catalog example's own table is written to a "
                                      "temporary file and uploaded from there")
    parser.add_argument("--verify", action="store_true",
                        help="also run every envelope of tools/verify_envelopes.py through the token")
    parser.add_argument("--out", help="write the --verify report as JSON")
    parser.add_argument("--i-know-this-leaves-the-machine", dest="offsite", action="store_true",
                        help="allow a non-loopback base URL: the data and prompts then leave this machine")
    args = parser.parse_args(argv)

    if not loopback(args.base) and not args.offsite:
        parser.error(f"{args.base} is not loopback. Everything sent to it -- prompts, tables, news -- leaves this "
                     "machine. Pass --i-know-this-leaves-the-machine if that is what you mean, and use TLS.")
    if not args.token_file:
        parser.error("no token file: pass --token-file or set M5PHET_API_TOKEN_FILE. The instance must have been "
                     "started with the same file (surfaces.api.token_file, or the same variable)")
    client = Client(args.base, read_token(args.token_file))
    print(f"# {args.base} · bearer token from {args.token_file} (never printed)")

    harness = load_harness()
    try:
        catalog = client.get("/api/catalog")
    except ApiError as error:
        if error.status == 401:
            raise SystemExit("401: this instance did not accept the token. It reads its token file once, at "
                             "start-up; check it was started with the file you are pointing at.") from None
        raise
    status = walkthrough(client, choose_spec(harness, catalog), catalog, Path(args.csv) if args.csv else None)
    if args.verify:
        print("\n# the envelopes of tools/verify_envelopes.py, over the API, with the token only")
        status |= verify(client, harness, catalog, args.out)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
