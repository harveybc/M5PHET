#!/usr/bin/env bash
# AP01: the work plan's two standing acceptance harnesses against MY staged build, on MY port and state directory.
set -uo pipefail
S="$HOME/.local/share/m5phet/staging-ap-20260929"
W="$HOME/Documents/GitHub/.worktrees/m5phet-next-acceptance"
PORT="${PORT:-8773}"
rm -rf "$S/state-harness"; mkdir -p "$S/state-harness"
AP_CASE=real_forecast_fixture_classification PORT="$PORT" STATE="$S/state-harness" \
  timeout --kill-after=10s "${LIFETIME:-900}" "$S/run-workbench.sh" >>"$S/harness-server.log" 2>&1 &
server=$!
cleanup() { kill -TERM "$server" 2>/dev/null; wait "$server" 2>/dev/null; }
trap cleanup EXIT INT TERM HUP
for _ in $(seq 1 40); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/api/catalog")" != "000" ] && break
  "$S/venv/bin/python" -c "import time; time.sleep(1)"
done
"$S/venv/bin/python" "$W/tools/verify_families.py"  --base "http://127.0.0.1:$PORT" --out "$S/families.json"  | tail -3
"$S/venv/bin/python" "$W/tools/verify_envelopes.py" --base "http://127.0.0.1:$PORT" --out "$S/envelopes.json" | tail -3
