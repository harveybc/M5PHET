#!/usr/bin/env bash
# Start the M5PHET workbench with all five families configured.
#
# Every provider's engine lives where its dependencies already are: TensorFlow, Stable-Baselines3 and EconML never enter the
# interface's environment. This file is the operator's declaration of where those engines and their fitted states are. A
# prompt cannot change any of it.
set -uo pipefail
CHAT_VENV="${CHAT_VENV:-$HOME/.local/share/m5phet/chat-venv}"
STATE="${STATE:-$HOME/.local/state/m5phet}"

# The interface itself never touches a GPU.
export CUDA_VISIBLE_DEVICES=""

# 1. Classification -- news-signal + Laya.
#    `fixture` is the declared NON_MODEL_FIXTURE and runs anywhere; every receipt it produces says so.
#    For the real checkpoint set NEWS_SIGNAL_BACKEND=laya with CHECKPOINT/MANIFEST/DEVICE/GPU_UUID, or point the workbench
#    at the private worker with M5PHET_CHAT_LAYA_WORKER and M5PHET_CHAT_LAYA_COMMAND.
export NEWS_SIGNAL_BACKEND="${NEWS_SIGNAL_BACKEND:-fixture}"
#    The private worker, when the operator has one. Set M5PHET_CHAT_LAYA_WORKER to the ssh host in your own environment;
#    it is never written into this repository. The command is fixed here and a prompt cannot change it.
if [ -n "${M5PHET_CHAT_LAYA_WORKER:-}" ]; then
    export M5PHET_CHAT_LAYA_COMMAND="${M5PHET_CHAT_LAYA_COMMAND:-bash \$HOME/work/m5phet-chat-worker/chat_laya_worker.sh}"
    echo "classification: private worker (real weights)"
else
    echo "classification: ${NEWS_SIGNAL_BACKEND} backend"
fi

# 2. Forecasting -- predictor/prediction_provider, TensorFlow in its own interpreter.
export M5PHET_FORECAST_PYTHON="${M5PHET_FORECAST_PYTHON:-$HOME/.local/share/m5phet/forecast-native-venv/bin/python}"
export M5PHET_FORECAST_BUNDLE="${M5PHET_FORECAST_BUNDLE:-$STATE/forecast-household-dev-20260924}"

# 3. Hierarchical regimes -- feature-eng, fitted reference and its assignments.
export FEATURE_ENG_REGIMES_DEMO_DIR="${FEATURE_ENG_REGIMES_DEMO_DIR:-$STATE/examples/regimes}"

# 4. Causal inference -- causal-inference, inference over studies fitted explicitly beforehand.
export CAUSAL_INFERENCE_STATE_DIR="${CAUSAL_INFERENCE_STATE_DIR:-$HOME/.local/share/causal-inference-m5phet/studies}"

# 5. Policy -- agent-multi, Stable-Baselines3 in its own interpreter.
export M5PHET_POLICY_PYTHON="${M5PHET_POLICY_PYTHON:-$HOME/anaconda3/envs/trading-stack/bin/python}"
export M5PHET_POLICY_BUNDLE="${M5PHET_POLICY_BUNDLE:-$STATE/policy-eth4h-dev-20260924}"

# JSON configuration (schema m5phet.config.v1). When ~/.config/m5phet/m5phet.json exists -- or M5PHET_CONFIG names
# another file -- the app reads its bindings through m5phet.config and they WIN over the variables above; everything the
# file does not bind stays exactly as this script left it. Nothing is required: without a file the variables above are
# the configuration. The path is only passed through here; the app validates it and refuses an unknown key, an unset
# $VARIABLE or a host literal. `tools/m5phet.json.example` is the template.
if [ -n "${M5PHET_CONFIG:-}" ]; then
    export M5PHET_CONFIG
    echo "configuration: ${M5PHET_CONFIG} (JSON bindings win)"
elif [ -f "$HOME/.config/m5phet/m5phet.json" ]; then
    echo "configuration: $HOME/.config/m5phet/m5phet.json (JSON bindings win)"
else
    echo "configuration: environment (no m5phet.json)"
fi

PORT="${PORT:-8765}"
echo "M5PHET workbench on http://127.0.0.1:${PORT}"
exec "$CHAT_VENV/bin/m5phet-chat" --port "$PORT" "$@"
