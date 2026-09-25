#!/usr/bin/env python3
"""One envelope per area, in the owner's shape, through the running product: named typed questions in, named typed
answers out. A question the engine cannot answer must come back REFUSED with its reason -- and never with a number.

    python3 tools/verify_envelopes.py [--base http://127.0.0.1:8766] [--out report.json]
"""

import argparse
import json
import os
import sys
import time
import urllib.request


def login(base, token):
    """Open the owner's session when the instance requires the access token (M5PHET_CHAT_TOKEN in chat.env).

    The cookie the login sets is kept for every later call by a process-wide cookie jar; without a token nothing
    changes, and an instance that requires one answers 401 to the first call, which is the right failure."""
    import http.cookiejar
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())))
    call(base, "POST", "/api/login", {"token": token})


def call(base, method, path, body=None, raw=None, filename=None, timeout=300):
    if raw is not None:
        boundary = "----m5phetenv"
        payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                   f"Content-Type: application/octet-stream\r\n\r\n").encode() + raw + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(base + path, data=payload, method="POST",
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                                  "Origin": base})
    else:
        request = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                         method=method, headers={"Content-Type": "application/json", "Origin": base})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode()
    return json.loads(text) if text.strip() else None


#: for each area: which catalog example supplies the data, the envelope in the owner's shape, and what each question
#: must come back as. `expect` is the status; a REFUSED expectation also names the refusal code.
def envelopes(examples):
    def data_of(fragment):
        return next(e for e in examples if fragment.lower() in e["title"].lower())

    return [
        {"area": "forecasting", "example": data_of("household-power"), "prompt": "pronostica la potencia y dame un rango",
         "task": {"area": "forecasting", "state": {"target_variable": "Global_active_power"},
                  "questions": {"prediccion": {"type": "point_forecast", "horizon": 60},
                                "rango": {"type": "interval", "horizon": 60, "confidence_level": 0.95},
                                "riesgo": {"type": "anomaly_risk", "threshold": "< 0.3"}}},
         "expect": {"prediccion": "OK", "rango": "REFUSED:NOT_ESTIMABLE", "riesgo": "REFUSED:NOT_ESTIMABLE"}},
        # the causal provider is inference-only over a study fitted beforehand: attaching rows would mean "fit", which it
        # refuses. The envelope names the graph; it attaches nothing.
        {"area": "causal", "example": {**data_of("ATE"), "data": None},
         "prompt": "cual fue el efecto del tratamiento y en jovenes",
         "task": {"area": "causal",
                  "state": {"causal_graph": {"treatment": "treatment", "outcome": "outcome", "confounders": ["baseline"]}},
                  "questions": {"efecto": {"type": "ate"},
                                "jovenes": {"type": "cate", "condition": "baseline == 1"}}},
         "expect": {"efecto": "OK", "jovenes": "REFUSED:NOT_ESTIMABLE"}},
        {"area": "rl", "example": data_of("market data"), "prompt": "que accion propone y que retorno espera",
         "task": {"area": "rl", "state": {"policy_id": "eth_4h_sac_current_stack_anchor_v1"},
                  "questions": {"accion": {"type": "next_action"},
                                "retorno": {"type": "value_estimation"}}},
         "expect": {"accion": "OK", "retorno": "OK"}},
        # --- RL negative paths (WP08): the same policy, unusable inputs, refused by name and with no number ----------
        {"area": "rl", "example": data_of("market data"), "prompt": "que accion propone con estas pocas barras",
         "data_transform": lambda csv: "\n".join(csv.splitlines()[:6]) + "\n",
         "task": {"area": "rl", "state": {"policy_id": "eth_4h_sac_current_stack_anchor_v1"},
                  "questions": {"accion": {"type": "next_action"}}},
         "expect": {"accion": "REFUSED:STATE_REQUIRED"}, "why_contains": {"accion": "TOO_FEW_ROWS"}},
        {"area": "rl", "example": data_of("market data"), "prompt": "que accion propone sin una de las columnas",
         "data_transform": lambda csv: _drop_last_column(csv),
         "task": {"area": "rl", "state": {"policy_id": "eth_4h_sac_current_stack_anchor_v1"},
                  "questions": {"accion": {"type": "next_action"}}},
         "expect": {"accion": "REFUSED:STATE_REQUIRED"}, "why_contains": {"accion": "MISSING_COLUMNS"}},
        {"area": "rl", "example": data_of("all-zero"), "prompt": "que accion propone para este vector corto",
         "data_transform": lambda vector: json.dumps([0.0] * 10),
         "task": {"area": "rl", "state": {"policy_id": "eth_4h_sac_current_stack_anchor_v1"},
                  "questions": {"accion": {"type": "next_action"}, "valor": {"type": "value_estimation"}}},
         "expect": {"accion": "REFUSED:STATE_REQUIRED", "valor": "REFUSED:STATE_REQUIRED"},
         "why_contains": {"accion": "OBSERVATION_SIZE_MISMATCH", "valor": "OBSERVATION_SIZE_MISMATCH"}},
        {"area": "unsupervised", "example": data_of("OHLC"), "prompt": "segmenta estas filas y describe el cluster alto",
         "task": {"area": "unsupervised", "state": {},
                  "questions": {"segmentacion": {"type": "clustering", "method": "auto"},
                                "perfil": {"type": "cluster_description", "target_metric": "body_pipettes > 0"}}},
         "expect": {"segmentacion": "OK", "perfil": "OK"}},
        {"area": "classification", "example": data_of("Noticia"), "prompt": "de que economia habla y con que tono",
         "task": {"area": "classification", "state": {"asset": "EURUSD", "language": "en"},
                  "questions": {"economia": {"type": "choice", "instructions": "Which economy is named in this news?",
                                             "options": [["euro_area", "Euro area"], ["united_states", "United States"],
                                                         ["other", "Another economy"]]},
                                "tono": {"type": "choice", "instructions": "What tone does the news take?",
                                         "options": [["hawkish", "tighter policy"], ["dovish", "easier policy"],
                                                     ["neutral", "neither"]]}}},
         "expect": {"economia": "OK", "tono": "OK"}},
    ]


def _drop_last_column(csv):
    """The bars without their last fitted column, so the table is market data missing one thing it needs."""
    rows = [line.split(",") for line in csv.splitlines() if line]
    return "\n".join(",".join(row[:-1]) for row in rows) + "\n"


def run_envelope(base, spec, defaults):
    example = spec["example"]
    chat = call(base, "POST", "/api/chats", {"title": "sobre " + spec["area"]})
    cid = chat["id"]
    call(base, "PATCH", f"/api/chats/{cid}", {"title": "sobre " + spec["area"],
                                              "config": dict(defaults, **example["config"])})
    file_ids = []
    if example.get("data"):
        suffix = {"json": "json", "csv": "csv"}.get(example["config"].get("input"), "txt")
        blob = example["data"] if isinstance(example["data"], str) else json.dumps(example["data"])
        if spec.get("data_transform"):
            blob = spec["data_transform"](blob)
        file_ids = [call(base, "POST", f"/api/chats/{cid}/files", raw=blob.encode(), filename=f"d.{suffix}")["id"]]
    sent = call(base, "POST", f"/api/chats/{cid}/tasks/run",
                {"prompt": spec["prompt"], "task": spec["task"], "file_ids": file_ids,
                 "client_id": "env-" + cid[:8], "language": "es"})
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        time.sleep(1.0)
        state = call(base, "GET", f"/api/chats/{cid}")
        found = [m for m in state["messages"] if m["id"] == sent["message_id"]]
        if found and found[0]["status"] not in ("QUEUED", "RUNNING"):
            return found[0]
    return {"status": "TIMEOUT", "content": "", "detail": {}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8766")
    parser.add_argument("--out")
    parser.add_argument("--token", default=os.environ.get("M5PHET_CHAT_TOKEN"),
                        help="owner access token; defaults to M5PHET_CHAT_TOKEN")
    args = parser.parse_args(argv)
    if args.token:
        login(args.base, args.token)
    catalog = call(args.base, "GET", "/api/catalog")
    areas = call(args.base, "GET", "/api/tasks/catalog")["areas"]
    report = {"schema": "m5phet_envelope_verification.v1", "areas": areas, "envelopes": []}
    for spec in envelopes(catalog["examples"]):
        message = run_envelope(args.base, spec, catalog["defaults"])
        detail = message.get("detail") or {}
        response = detail.get("response") or {}
        answers = response.get("answers") or {}
        rows = []
        for name, expected in spec["expect"].items():
            answer = answers.get(name) or {}
            got = answer.get("status", "MISSING")
            if got == "REFUSED":
                got = f"REFUSED:{answer.get('refusal')}"
            invented = got.startswith("REFUSED") and any(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                         for k, v in answer.items() if k not in ("type",))
            wanted = (spec.get("why_contains") or {}).get(name)
            why_ok = wanted is None or wanted in str(answer.get("why", ""))
            rows.append({"question": name, "type": answer.get("type"), "expected": expected, "got": got,
                         "as_expected": got == expected and why_ok, "number_in_refusal": invented,
                         "why_expected": wanted,
                         "why": answer.get("why"), "preview": {k: v for k, v in list(answer.items())[:5]}})
        narration = detail.get("narration") or {}
        report["envelopes"].append({"area": spec["area"], "message_status": message.get("status"),
                                    "questions": rows, "narration_source": narration.get("source"),
                                    "narration_plugin": narration.get("output_plugin"),
                                    "narration": (message.get("content") or "")[:300],
                                    # the answers verbatim, so `tools/verify_outputs.py` can render THESE -- the ones
                                    # the engines really returned -- instead of a shape someone typed into a test
                                    "answers": answers, "answered": response.get("answered"),
                                    "refused": response.get("refused"),
                                    "execution_authorized": detail.get("execution_authorized")})
    ok_total = 0
    for env in report["envelopes"]:
        for row in env["questions"]:
            mark = "OK " if row["as_expected"] and not row["number_in_refusal"] else "!! "
            ok_total += row["as_expected"] and not row["number_in_refusal"]
            print(f"{mark}{env['area']:<15} {row['question']:<13} {str(row['type']):<20} {row['got']:<24} "
                  f"{(row['why'] or '')[:60]}")
        print(f"    narration[{env['narration_source']}]: {env['narration'][:140]}")
    total = sum(len(e["questions"]) for e in report["envelopes"])
    report["summary"] = {"questions_as_expected": ok_total, "questions": total,
                         "any_execution_authorized": any(e["execution_authorized"] for e in report["envelopes"])}
    print(json.dumps(report["summary"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, ensure_ascii=False, default=str)
    return 0 if ok_total == total and not report["summary"]["any_execution_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
