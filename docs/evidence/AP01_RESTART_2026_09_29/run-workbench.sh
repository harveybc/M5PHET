#!/usr/bin/env bash
# AP01 verification workbench: MY staged build, MY port, MY state directory.
# It never touches port 8765, the owner's conversations, or the m5phet-chat service, and never restarts it.
#
#   AP_CASE=real_forecast_fixture_classification | pinned_checkpoint_mismatch
set -uo pipefail
S="$HOME/.local/share/m5phet/staging-ap-20260929"
set -a; . "$HOME/.config/m5phet/chat.env"; set +a
export CUDA_VISIBLE_DEVICES=""
unset M5PHET_CHAT_ALLOWED_HOST
# The operator's chat.env is the contradiction CB05 refuses by name. This instance repairs it IN ITS OWN
# ENVIRONMENT ONLY -- his file is never edited -- and declares one backend, per case.
unset NEWS_SIGNAL_BACKEND M5PHET_CHAT_LAYA_WORKER M5PHET_CHAT_LAYA_COMMAND \
      M5PHET_CLASSIFICATION_MODE M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT
case "${AP_CASE:-real_forecast_fixture_classification}" in
  real_forecast_fixture_classification)
      # classification = the DECLARED non-model fixture, in this process. Forecasting, causal, rl and unsupervised
      # keep the operator's real fitted states and answer through their own interpreters.
      export NEWS_SIGNAL_BACKEND=fixture NEWS_SIGNAL_DEVICE=cpu ;;
  pinned_checkpoint_mismatch)
      # a bound worker that describes a checkpoint OTHER than the pinned one. The transport is the LOCAL FAKE on
      # PATH; no packet leaves this machine and no real worker is contacted.
      export PATH="$S/fake-ssh:$PATH" AP_FAKE_WORKER=other_checkpoint
      export M5PHET_CLASSIFICATION_MODE=remote_worker
      export M5PHET_CHAT_LAYA_WORKER=worker.invalid
      export M5PHET_CHAT_LAYA_COMMAND=ap-worker
      export M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT="laya-checkpoint:$(printf 'a%.0s' $(seq 64))" ;;
  *) echo "unknown AP_CASE" >&2; exit 2 ;;
esac
exec "$S/venv/bin/m5phet-chat" --host 127.0.0.1 --port "${PORT:-8771}" --state-dir "${STATE:-$S/state-8771}"
