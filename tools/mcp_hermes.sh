#!/usr/bin/env bash
# M5PHET MCP server for Hermes (stdio JSON-RPC 2.0, protocol 2025-06-18).
#
# Hermes launches this script as a stdio MCP server; it exposes the three tools
# m5phet_catalog, m5phet_propose_task and m5phet_execute_ml_task.
#
# The operator's declaration of where every engine and fitted state lives is
# ~/.config/m5phet/chat.env. It is read here and never copied into this
# repository: no host, no path of a private worker appears in this file.
#
# The coordinator never uses a GPU (standing rule 4): CUDA_VISIBLE_DEVICES is
# forced empty. The admitted GPU belongs to the worker and is reached through
# the classification engine's own command.
set -euo pipefail

env_file="${M5PHET_CHAT_ENV:-$HOME/.config/m5phet/chat.env}"
if [ -f "$env_file" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi
export CUDA_VISIBLE_DEVICES=""

python_bin="${M5PHET_MCP_PYTHON:-$HOME/.local/share/m5phet/chat-venv/bin/python}"
exec "$python_bin" -m m5phet.mcp_server "$@"
