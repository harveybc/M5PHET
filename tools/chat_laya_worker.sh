#!/usr/bin/env bash
set -euo pipefail
# Deploy this wrapper beside worker.py on the private worker. No user-controlled commands.
: "${M5PHET_LAYA_ROOT:=$HOME/.local/state/crispdm-data-foundation/laya-pilot-20260924}"
export NEWS_SIGNAL_CHECKPOINT="$M5PHET_LAYA_ROOT/checkpoint"
export NEWS_SIGNAL_MANIFEST="$M5PHET_LAYA_ROOT/manifest.json"
export NEWS_SIGNAL_BACKEND=laya
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
GPU_UUID=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '$1 == "NVIDIA GeForce RTX 5090" {print $2}')
if [[ -z "$GPU_UUID" || "$GPU_UUID" == *$'\n'* ]]; then
    printf '%s\n' '{"transport_error":"Exactly one RTX 5090 is required; no fallback"}'
    exit 2
fi
export NEWS_SIGNAL_GPU_UUID="$GPU_UUID" CUDA_VISIBLE_DEVICES="$GPU_UUID" NEWS_SIGNAL_DEVICE=cuda:0
mkdir -p "$HOME/.local/state/m5phet"
exec 9>"$HOME/.local/state/m5phet/chat-gpu.lock"
# One call at a time on the GPU. A second caller waits its turn for up to 60 s instead of being refused on arrival:
# two workbench instances verifying at once used to make one of them fail its whole classification lane.
if ! flock -w 60 9; then
    printf '%s\n' '{"transport_error":"Chat worker is occupied"}'
    exit 2
fi
exec timeout 175 "$M5PHET_LAYA_ROOT/venv/bin/python" "$(dirname "$0")/worker.py"
