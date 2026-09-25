# Telegram, through the owner's Hermes agent (WP11)

M5PHET builds nothing new for Telegram. `python -m m5phet.mcp_server` already exposes the three tools over stdio;
the owner's Hermes gateway already speaks Telegram. This document is the wiring and the acceptance test.

**No token, no chat id, no host and no path of a private worker appears in this repository.** Everything secret
stays in Hermes' own files (`~/.hermes/.env`, mode `600`) and in the operator's environment file
(`~/.config/m5phet/chat.env`). If a step below asks for a secret, it is typed by the owner into one of those two
files, never into a commit, never into a chat, never into this document.

---

## 1. What is already registered (done 2026-09-24, no owner action needed)

| Thing | Where | State |
|---|---|---|
| MCP server `m5phet` | `~/.hermes/config.yaml`, key `mcp_servers.m5phet.command` → `tools/mcp_hermes.sh` of this repository | registered, 3/3 tools enabled |
| Launcher | `tools/mcp_hermes.sh` (committed) — loads `~/.config/m5phet/chat.env`, forces `CUDA_VISIBLE_DEVICES=""`, runs the chat venv's `python -m m5phet.mcp_server` | committed here |
| Hermes skill | `~/.hermes/skills/m5phet/SKILL.md`; the committed copy is `tools/hermes_skill_m5phet/SKILL.md` | installed, listed by `hermes skills list` as `local / enabled` |

Proof to re-run at any time:

```bash
hermes mcp list                 # m5phet · stdio · all tools · enabled
hermes mcp test m5phet          # ✓ Connected · ✓ Tools discovered: 3
```

The three tools are `m5phet_catalog`, `m5phet_propose_task`, `m5phet_execute_ml_task`.

---

## 2. What the OWNER must do

### 2.1 The bot token — already present on this machine

Hermes reads the Telegram credential from `~/.hermes/.env`. On this machine the key `TELEGRAM_BOT_TOKEN` is
**already set** (verified by key name only; the value was never read, printed or copied). Nothing to create.

Only if a *new* bot is ever wanted: `hermes gateway setup` → Telegram → paste the token @BotFather gives. Hermes
writes it to `~/.hermes/.env` itself. Never paste a token into a terminal that is being recorded, into a repository,
or into a chat with an agent.

### 2.2 Who may talk to the bot — already configured

Also in `~/.hermes/.env`: `TELEGRAM_ALLOWED_USERS`, `TELEGRAM_GROUP_ALLOWED_CHATS`, `TELEGRAM_HOME_CHANNEL`
(all already set). In `~/.hermes/config.yaml`, section `telegram`: `require_mention: true` and
`free_response_chats` (a chat id, already set — not reproduced here). The owner changes these only if he wants to
open or close access; the chat id itself is an identifier and stays out of this repository.

### 2.3 The one required action: restart the gateway

Hermes has **no hot reload for MCP servers** (`native-mcp` skill, "Adding or removing servers requires restarting
the agent"). The gateway service on this machine has been running since before `m5phet` was registered, so it does
not know the three tools yet. This agent did **not** touch it: it is the owner's live messaging service.

```bash
hermes gateway status     # read-only; shows the running unit
hermes gateway restart    # the owner runs this when a restart is acceptable
hermes gateway status
```

After the restart, MCP tools are injected into every platform toolset, Telegram included (`platform_toolsets.telegram`
needs no edit).

### 2.4 Making the agent use the skill on Telegram

Hermes shows the agent the skill index and lets it load `m5phet` by name. To make it deterministic, either
- start the first message with "use the m5phet skill", or
- put that instruction in `telegram.channel_prompts` for the chat in `~/.hermes/config.yaml`.

The skill is what enforces the contract: catalog first, envelope shown for review, `ok` required before running,
one line per answer, refusals named, **no number that the engine did not return**, and the closing line
`execution_authorized: false — this is not an instruction to act`.

Since WP31 the `telegram` output procedure adds **one** line above that closing: what was measured about how well the
area that answered actually answers — a macro-F1 and its calibration, a held-out error and its skill against a named
naive reference (and the interval's coverage where an interval was returned), a refusal by name where the quantity
does not exist, or `NOT_MEASURED`. It carries the corpus, the protocol digest and the seal, and it is held to the
same guard as every other figure in the message.

---

## 3. Acceptance test (the owner runs it from Telegram)

Send message 1, mentioning the bot, then reply `ok` to the envelope it shows; then message 2, same pattern.

**Message 1 — classification (the news example):**

```
use the m5phet skill. Classify this news: "The European Central Bank left its deposit facility rate unchanged.
Incoming euro-area inflation data will guide its next decision." Question 1 "economia": Which economy is named in
this news? options euro_area / united_states / other. Question 2 "tono": What tone does the news take?
options hawkish / dovish / neutral.
```

**Message 2 — causal (the ATE sentence):**

```
use the m5phet skill. Report the ATE of treatment on outcome, with its uncertainty.
```

### What must come back

Message 2 is the strict comparison. Measured on 2026-09-24 through Hermes' own MCP client (`hermes -s m5phet -z`,
same server the Telegram agent will use) and identical to the workbench's stored answer:

```
efecto: ATE = 2.0293820021534623 synthetic outcome units, 95% interval
        [1.931308938896731, 2.1274550654101936], development: true. Assumptions: consistency, constant_effect,
        iid_sampling, no_interference, nuisance_models_correct, positivity, pre_treatment_adjustment,
        sufficient_adjustment.
jovenes: REFUSED NOT_ESTIMABLE — … not fitted with an effect modifier …      (only if a cate question is asked)

execution_authorized: false — this is not an instruction to act
```

Accept only if: the digits match character for character, no p-value appears, and the last line is present.

Message 1 has a **known gap, and it is not a Telegram problem.** The MCP surface answers classification through the
backend this host declares in `~/.config/m5phet/chat.env` (`NEWS_SIGNAL_BACKEND`). The real Laya checkpoint lives on
the private worker and is reached by the **web workbench's** own route, which the MCP server does not have. So on
this machine Telegram will return the local backend's label and probabilities (verified 2026-09-24: `other` at
`1.0`), not the workbench's `euro_area 0.9666`. Worse, that answer carries no `NON_MODEL_FIXTURE` marker: it looks
like a checkpoint answer. Until the MCP path routes classification to the worker the way `web/engine.py` does,
**do not compare classification numbers between Telegram and the workbench**, and do not quote a Telegram
classification probability as a model result. The other four areas answer from the same fitted states as the
workbench.

### Where the evidence goes

Screenshots or copied text of the two replies go in `docs/audits/evidence/TELEGRAM_<YYYYMMDD>/`, next to the
coordinator-side evidence already committed in `docs/audits/evidence/WP11_HERMES_MCP_20260924/`.

---

## 4. If something refuses

| Symptom | Reading |
|---|---|
| the agent says it has no `m5phet_*` tool | the gateway was not restarted after registration (§2.3) |
| `hermes mcp test m5phet` fails | `~/.config/m5phet/chat.env` is missing or the chat venv moved; the launcher names both through `M5PHET_CHAT_ENV` / `M5PHET_MCP_PYTHON` overrides |
| a question comes back `REFUSED STATE_REQUIRED` | the fitted state that area needs is not configured in `chat.env` on this machine |
| a forecast comes back `REFUSED PROVIDER_ERROR` | the forecasting bundle needs its windowed input attached (`columns`, `values`, `scale`, `scaler_digest`); a sentence alone cannot produce it, and Telegram is a poor place to attach it |
| `interval`, `anomaly_risk`, `cate` come back `NOT_ESTIMABLE` | correct behaviour: no quantile head, no effect modifier. WP07 and WP10 are what change that |
