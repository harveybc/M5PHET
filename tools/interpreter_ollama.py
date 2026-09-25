#!/usr/bin/env python3
"""A second interpreter behind M5PHET_INTERPRETER_COMMAND, through a local ollama server instead of hermes.

    M5PHET_INTERPRETER_COMMAND="/path/to/tools/interpreter_ollama.py qwen3:4b"

The workbench calls its interpreter as `COMMAND -z PROMPT` and reads stdout; this forwards the prompt to ollama's chat
API and prints the reply. It asks for CPU only (`num_gpu: 0`): on this fleet the only GPU admitted for work is the
external one on the worker, never the coordinator's. Thinking is off and the reply is bounded. Which values the model
may choose is not decided here: m5phet.interpret shows it the declared vocabulary and refuses anything outside it.
"""
import json
import os
import sys
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def main(argv):
    if len(argv) < 3 or argv[1] == "-z":
        sys.exit("usage: interpreter_ollama.py MODEL -z PROMPT")
    model, prompt = argv[1], argv[argv.index("-z") + 1]
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_gpu": 0, "num_predict": 512, "temperature": 0}}
    request = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=float(os.environ.get("M5PHET_INTERPRETER_TIMEOUT", 170))) as reply:
        answer = json.load(reply)
    text = (answer.get("message") or {}).get("content", "")
    if "</think>" in text:                         # a model that thought anyway: only what follows is the reply
        text = text.split("</think>", 1)[1]
    sys.stdout.write(text.strip() + "\n")


if __name__ == "__main__":
    main(sys.argv)
