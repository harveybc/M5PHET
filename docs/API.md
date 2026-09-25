# The local API

The workbench's browser client is one caller of a small HTTP API; another program can be a second. This document is
that API: every endpoint, what it takes, what it answers, how it refuses, and one `curl` that works against a running
instance. It is the contract `tools/api_client_example.py` is written against.

What this API is **not**: a prediction service, a multi-tenant account system, or an internet deployment. It binds to
loopback, it serves one owner, it holds no execution authority, and every answer it returns says so
(`execution_authorized: false`). Nothing here trains a model, downloads weights, or reaches a broker.

Every example below uses `http://127.0.0.1:8765` — the owner's own instance. A verification instance runs on another
port with its own `--state-dir`; substitute it.

```bash
BASE=http://127.0.0.1:8765
TOKEN_FILE=~/.config/m5phet/api.token       # outside any repository, mode 600
auth=(-H "Authorization: Bearer $(cat "$TOKEN_FILE")")
```

---

## 1. Getting in

There are two credentials, and they are equivalent. Which one you use depends on whether you have a browser.

| Credential | Who uses it | How |
|---|---|---|
| **Owner cookie** | the browser client | `POST /api/login` with the owner token → `m5phet_owner` HttpOnly SameSite=Strict cookie |
| **Bearer token** | a program | `Authorization: Bearer <token>` on every `/api/*` call |

**The bearer token is the contents of a file the operator names**, never a value written in a configuration file and
never an argument on a command line:

```json
{"schema": "m5phet.config.v1", "surfaces": {"api": {"token_file": "~/.config/m5phet/api.token"}}}
```

```bash
head -c 32 /dev/urandom | base64 > ~/.config/m5phet/api.token && chmod 600 ~/.config/m5phet/api.token
M5PHET_API_TOKEN_FILE=~/.config/m5phet/api.token tools/start_chat.sh      # or bind it in m5phet.json
```

Rules, each of them checked by `tests/test_web_api_token.py`:

* the file is read **once, at start-up**. A file written afterwards does not quietly turn access on, and one deleted
  does not turn it off in the middle of a run;
* **no file, no bearer access.** A missing, empty or unreadable file leaves the header ignored and the owner's cookie
  as the only way in — the safe direction: an API must not open because a permission bit was wrong;
* the token is compared with `hmac.compare_digest`, and **never appears in a response, a header or a log line**;
* `$M5PHET_API_TOKEN_FILE` wins over `surfaces.api.token_file`. It is a *path* override of the same class as
  `M5PHET_CONFIG`, so a verification instance can run with its own token and never read the owner's. (Value bindings
  follow the opposite precedence: there, the JSON wins.)
* the **host allow-list and the cross-origin rules apply to a bearer request exactly as to a browser's**. A token is
  not a way around them.
* a program holding the token may also `POST /api/login` with it and keep the cookie instead of resending the header.

When the instance was started **without** an owner token (`M5PHET_CHAT_TOKEN` unset, the loopback default), no
credential is required at all and the `Authorization` header is simply unused.

## 2. Conventions and limits

| | |
|---|---|
| Content type | `application/json` in and out; uploads are `multipart/form-data` with one `file` part |
| Origin | a cross-origin `Origin`, or `Sec-Fetch-Site: cross-site`, is refused. Send `Origin: $BASE` or none |
| Host | the `Host` header must be in the allow-list (`127.0.0.1`, `localhost`, `::1`, plus any `--allowed-host`) |
| Body | 9 MiB for any request; 8 MiB for one attachment; 250 MiB for all attachments of the instance |
| Prompt | 4000 characters on `/preview`, `/tasks/propose` and `/tasks/run`; 64000 on `/messages` |
| `file_ids` | at most 8 per call, 5 on `/messages`; an id must belong to the chat it is sent to |
| Attachments | UTF-8 `.txt`, `.md`, `.json`, `.csv`. JSON refuses duplicate keys and non-finite numbers; CSV refuses duplicate or empty column names and ragged rows |
| Response headers | `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store`, a strict `Content-Security-Policy` |
| Concurrency | one model call at a time, process-wide. A second run while one is in flight is `409` |

### Status codes

| Code | When |
|---|---|
| `200` / `201` / `202` / `204` | read / created / **accepted and running in the background** / deleted |
| `401` | no credential, a wrong bearer token, or a wrong token at `/api/login` |
| `403` | host not allowed, cross-origin `Origin`, or `Sec-Fetch-Site: cross-site` |
| `404` | no such chat, or an attachment id that belongs to another chat |
| `409` | **the same `client_id` with different input**, another request already running, or deleting a chat mid-run |
| `413` | an attachment over 8 MiB (refused on the declared length, before the body is parsed) or a body over 9 MiB |
| `422` | anything the request says that cannot be read: unknown configuration fields, an empty question, a malformed CSV or JSON, an envelope that is not an object, an unsupported attachment type |

A refusal is a JSON object `{"detail": "<what was wrong, in words>"}`. **A refused *question* is not an HTTP error**:
an envelope whose questions an engine cannot answer completes with `200`-class status and each answer carries its own
`status: "REFUSED"` and a refusal code (`NOT_ESTIMABLE`, `STATE_REQUIRED`, `MISSING_COLUMNS`, …). The API never
substitutes a number for a refusal.

---

## 3. Endpoints

### `POST /api/login` — open a session

Body: `{"token": "<owner token>"}`, or an empty object with a valid bearer header. Answers `{"ok": true}` and sets the
`m5phet_owner` cookie; `401 {"detail": "Invalid token"}` otherwise.

```bash
curl -si $BASE/api/login "${auth[@]}" -H 'Content-Type: application/json' -d '{}' | head -1
```

### `GET /api/catalog` — what is installed

Answers the providers and their declared capabilities, the catalog examples (title, prompt, data, config), the
interpreter's identity, the defaults a chat starts with, which configuration is in force (`config_source`:
`m5phet.json` or `env`) and the provider/output plugin bound per area. Core settings — paths, hosts — are
deliberately **not** published. `execution_authorized` is `false`.

```bash
curl -s $BASE/api/catalog "${auth[@]}" | jq '{config_source, providers: [.providers[].name], examples: [.examples[].title]}'
```

### `GET /api/tasks/catalog` — what can be asked, per area

```json
{"areas": {"forecasting": {"provider": "predictor_forecast",
                           "question_types": {"point_forecast": {"required": ["horizon"], "optional": ["target"]}},
                           "parameters": {"target": ["Global_active_power"], "horizon": [1, 60]},
                           "aliases": {}, "combinations": [], "data_requirement": {...}}},
 "interpreter": {"model": "...", "available": true},
 "outputs": {"forecasting": {"plugin": "default", "output_kind": "point_forecast", "refusals": ["NOT_ESTIMABLE"]}},
 "execution_authorized": false}
```

`parameters` are the values the fitted engine actually has: a column name from your table is **not** a substitute for
one. `outputs` is the output header — read it and you know the shape of an answer before asking anything.

```bash
curl -s $BASE/api/tasks/catalog "${auth[@]}" | jq '.areas | map_values(.question_types | keys)'
```

### `GET /api/chats` — list

```bash
curl -s $BASE/api/chats "${auth[@]}" | jq '.[0]'          # {id, title, created, updated}
```

### `POST /api/chats` — create (`201`)

Body `{"title": "..."}` (1–120 characters; defaults to `Nuevo chat`). Answers the whole chat, including the default
configuration.

```bash
CID=$(curl -s $BASE/api/chats "${auth[@]}" -H 'Content-Type: application/json' \
      -d '{"title":"from a program"}' | jq -r .id)
```

### `GET /api/chats/{cid}` — the chat, its messages and its attachments

```json
{"id": "...", "title": "...", "config": {...}, "created": "...", "updated": "...",
 "messages": [{"id": "...", "client_id": "...", "request_digest": "...", "role": "assistant",
               "content": "<the narration, or the refusal>", "status": "OK", "detail": {...}}],
 "files": [{"id": "...", "name": "d.csv", "sha256": "...", "size": 1234}]}
```

Message `status` is `SENT` (the user's turn), `RUNNING`, `OK`, `PARTIAL` (some questions answered, some refused),
`REFUSED`, or `INTERRUPTED` (the server restarted mid-run; it is never retried automatically). `detail` holds the
exact request or envelope, the answers, the narration, the settings and attachment digests of that turn, and
`execution_authorized: false`.

```bash
curl -s $BASE/api/chats/$CID "${auth[@]}" | jq '.messages[-1] | {status, content}'
```

### `PATCH /api/chats/{cid}` — rename, or change this chat's settings

Body: `{"title": "...", "config": {...}}`, both optional. `config` takes the fields of `/api/catalog`'s `defaults`
and an unknown one is `422`. It is merged **over the defaults, not over the chat's current settings**: a field you
leave out goes back to its default rather than staying as it was, so send a whole configuration — one of
`/api/catalog`'s examples can be sent exactly as it is read from there. A queued turn keeps the settings it was sent
with.

```bash
curl -s -X PATCH $BASE/api/chats/$CID "${auth[@]}" -H 'Content-Type: application/json' \
  -d '{"title":"bars","config":{"input":"csv","provider":"trading_policy","family":"policy","output_kind":"policy_action"}}' \
  | jq '{title, provider: .config.provider, input: .config.input}'
```

`DELETE /api/chats/{cid}` answers `204`, or `409` while a request of that chat is running.
`GET /api/chats/{cid}/export` answers the same document as `GET /api/chats/{cid}` with a
`Content-Disposition: attachment` header, for keeping a conversation outside the store.

```bash
curl -s $BASE/api/chats/$CID/export "${auth[@]}" -o chat.json
```

### `POST /api/chats/{cid}/files` — attach a table, a document or a news event (`201`)

`multipart/form-data`, one part named `file`. Answers `{"id", "name", "sha256", "size"}`. The file is parsed before it
is stored, so a malformed CSV or JSON is refused now (`422`) rather than at run time. The uploaded name is a label:
directories are stripped and non-printable characters removed. **8 MiB** is the limit; a larger declared length is
refused with `413` before the body is read, naming the limit and the size sent.

```bash
FID=$(curl -s $BASE/api/chats/$CID/files "${auth[@]}" -F 'file=@bars.csv' | jq -r .id)
```

`GET /api/chats/{cid}/files/{fid}` returns the bytes back, and refuses (`404`) an id belonging to another chat.

### `POST /api/chats/{cid}/preview` — what the sentence resolves to, without running it

Body `{"prompt": "...", "file_ids": ["..."]}`. Answers the typed request **as it would run**, which words or which
interpreter resolved each field, the settings used, and `ran: false`. Nothing executes and nothing is recorded.

```bash
curl -s $BASE/api/chats/$CID/preview "${auth[@]}" -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"segmenta estas filas y describe el cluster alto\",\"file_ids\":[\"$FID\"]}" \
  | jq '{ran, request, interpretation: .interpretation.parameters}'
```

### `POST /api/chats/{cid}/tasks/propose` — a sentence becomes an envelope, for review

Body `{"prompt": "...", "file_ids": ["..."]}`. Answers:

```json
{"status": "OK", "task": {"area": "...", "state": {...}, "questions": {...}}, "proposal": {...},
 "problems": [], "profile": {"kind": "table", "columns": ["open", "high"], "rows": 500},
 "dataset": null, "dataset_resolution": {...}, "interpreter": {...}, "catalog": {...}}
```

`status` is `OK` (with a validated `task`), `INVALID_PROPOSAL` (the model named something the catalog or the data does
not have — `problems` says what), or `REFUSED` (`why` says why: no interpreter, an empty or over-long question, no
served area). **Nothing runs and nothing is recorded here.** The interpreter is shown the question, the declared
vocabulary and the *shape* of the data — column names, types, a row count — never a row.

```bash
curl -s $BASE/api/chats/$CID/tasks/propose "${auth[@]}" -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"segmenta estas filas y describe el cluster alto\",\"file_ids\":[\"$FID\"]}" \
  | jq '{status, task, problems}'
```

### `POST /api/chats/{cid}/tasks/run` — run the envelope that was reviewed (`202`)

Body:

```json
{"prompt": "segmenta estas filas", "task": {"area": "...", "state": {...}, "questions": {...}},
 "file_ids": ["..."], "client_id": "<yours, unique per request>", "language": "es"}
```

Answers `202 {"message_id": "..."}` immediately; the run happens on the single worker and the result is read from
`GET /api/chats/{cid}`. `language` is `es` or `en`. The envelope may be the one `/tasks/propose` returned, one the
person edited, or one written by hand — it is validated against the area's declared question types either way.

```bash
MID=$(curl -s $BASE/api/chats/$CID/tasks/run "${auth[@]}" -H 'Content-Type: application/json' -d "{
  \"prompt\": \"segmenta estas filas y describe el cluster alto\",
  \"task\": {\"area\":\"unsupervised\",\"state\":{},\"questions\":{
      \"segmentacion\": {\"type\":\"clustering\",\"method\":\"auto\"},
      \"perfil\": {\"type\":\"cluster_description\",\"target_metric\":\"body_pipettes > 0\"}}},
  \"file_ids\": [\"$FID\"], \"client_id\": \"run-$(date +%s)\", \"language\": \"es\"}" | jq -r .message_id)

until curl -s $BASE/api/chats/$CID "${auth[@]}" \
      | jq -e --arg m "$MID" '.messages[] | select(.id==$m) | select(.status!="RUNNING")' >/dev/null
do sleep 1; done
curl -s $BASE/api/chats/$CID "${auth[@]}" | jq --arg m "$MID" '.messages[] | select(.id==$m) | .detail.response.answers'
```

### `POST /api/chats/{cid}/messages` — the sentence path (`202`)

Body `{"prompt": "...", "client_id": "...", "file_ids": [...]}`. Resolves the sentence against the chat's settings and
runs it on the selected provider — the same thing `/preview` shows you without running. Answers
`202 {"message_id": "..."}`; poll the chat as above. An empty prompt is `422`.

```bash
curl -s $BASE/api/chats/$CID/messages "${auth[@]}" -H 'Content-Type: application/json' \
  -d '{"prompt":"Which economy is named in this news?","client_id":"ask-1","file_ids":[]}'
```

---

## 4. The rules a program has to know

### Review before run

The sentence path and the envelope path each have a window, and neither runs anything:

| Path | Window | Then |
|---|---|---|
| sentence | `POST /preview` — the typed request as it *would* run, with the resolved parameters and who resolved them | `POST /messages` with the same sentence |
| envelope | `POST /tasks/propose` — the validated envelope, its problems, the data profile | `POST /tasks/run` with the envelope you accepted, edited or wrote |

A program that does not want a review simply skips the first call. A program that offers one to a person must show
what came back and send what was shown: the envelope is part of the request's identity (below), so an envelope edited
after the review is a *different request*, and the API will say so rather than run the reviewed one.

### `client_id` and the 409 rule

`client_id` is **yours**: a string you choose, unique per request (a UUID does). The server stores, beside it, a
digest of `{prompt, file_ids, task}`.

* **Same `client_id`, same input** → the same `message_id` comes back and **nothing runs twice**. That is what makes a
  retry after a dropped connection safe.
* **Same `client_id`, different input** → `409 {"detail": "Request ID was already used for different input"}` and
  **nothing runs**. An edited envelope, a different sentence or a different attachment is a new request; reusing the
  id would file the new answer under the old question.
* A second request while one is running → `409 {"detail": "Another request is running; try again when it finishes"}`.

### Naming a dataset instead of attaching a file

With **no** attachment, the sentence (or the envelope's `state.dataset`) may name a dataset of the local catalog. The
proposal then carries what will be read, before anything runs:

```json
{"dataset": {"id": "household_power", "source": "...", "rows": 50400, "columns": ["Global_active_power"],
             "governed": false, "scale": "STANDARDIZED_BY_DECLARED_SCALER",
             "source_of_choice": "EXPLICIT_ID | QUESTION_TEXT | LAYA",
             "decision": {...}, "decision_record": "<path>"}}
```

* an **explicit id** wins over every word; then a declared alias; then the words themselves;
* when several datasets fit the words, **Laya chooses among those candidates and among nothing else**, and the
  decision record (what it was shown by digest, what it could choose from, what it chose, with which uncalibrated
  probabilities) travels beside the proposal;
* ambiguity with no decider, a name the catalog does not hold, a **governed** resource, or a panel whose units are
  unknown are each **refused by name** — never served from whatever else was to hand;
* an attachment always wins: no catalog may quietly replace the data a person chose.

The resolved id is written into `task.state.dataset`, so what runs reads what was reviewed.

### The receipt

Every answer carries how its inputs came to be what the engine was handed, so a number can be traced without trusting
the sentence around it:

| Field | Meaning |
|---|---|
| `response.request_sha256` | the digest of the validated envelope; a worker's answer is bound to it or refused |
| `detail.dataset.rows_receipt.rows_origin` | `as_stored`, or `inverse_standardized_from_manifest_scaler` when a standardized panel was inverted to original units |
| `detail.dataset.rows_receipt.scaler_sha256` | the scaler the manifest declared, when rows were inverted |
| `detail.dataset.rows_receipt.target_column` | which column the target is, and where it came from |
| `detail.dataset.rows_receipt.rows_sha256` / `rows_handed_over` | the digest and count of the rows actually handed over, in the order they were handed over |
| `detail.narration.source` / `output_plugin` | whether the sentence was written by the interpreter or rendered deterministically, and by which output plugin |
| `detail.profile` | `LOCAL_UNGOVERNED` — this is the owner's workbench, not governed evidence |
| `detail.elapsed_seconds` | wall time of the run |

A narration that states a number the answers do not carry is discarded in favour of the deterministic rendering, and
the receipt says which plugin produced the discarded text.

### `execution_authorized: false`

It appears on `/api/catalog`, on `/api/tasks/catalog`, on every preview, on every envelope response and on every
stored message. It is not decoration: a policy's action is a reading in the policy's own scale, never an order, and
this API exposes no method that could place one. A client that treats an answer as an instruction is doing so alone.

---

## 5. The worked client

`tools/api_client_example.py` is the whole path above in the standard library alone — login, catalog, chat, upload
from a path, preview, propose, run with a fresh `client_id`, poll, print the answers and the refusals, and both halves
of the `client_id` rule:

```bash
python3 tools/api_client_example.py --base http://127.0.0.1:8765 --token-file ~/.config/m5phet/api.token
python3 tools/api_client_example.py --base http://127.0.0.1:8765 --token-file ~/.config/m5phet/api.token --verify
```

`--verify` additionally drives the envelopes of `tools/verify_envelopes.py` — imported from that file, so the two
cannot drift apart — over the token instead of the owner's cookie, and prints the same last JSON line
(`{"questions_as_expected": …, "questions": …, "any_execution_authorized": false}`).

The client refuses a non-loopback `--base` unless `--i-know-this-leaves-the-machine` is passed, and takes the token
from a file only: a token in a command line is visible to every process on the machine.
