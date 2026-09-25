#!/usr/bin/env python3
"""A thin shim over the `ollama` interpreter plugin, for an operator who configures by environment only.

The implementation moved into the package at WP03 (`m5phet.interpreters.ollama`), where it is selected by

    {"interpreter": {"plugin": "ollama", "model": "llama3.2:3b"}}

in `~/.config/m5phet/m5phet.json`. This script stays because `M5PHET_INTERPRETER_COMMAND` is still a supported way to
run, and because the shell contract it implements -- `COMMAND -z PROMPT`, reply on stdout -- is what the `command`
plugin calls. It adds nothing of its own: the request body, the CPU-only option and the `</think>` strip all live in
the plugin now, so the two routes cannot drift apart.

    M5PHET_INTERPRETER_COMMAND="/path/to/tools/interpreter_ollama.py llama3.2:3b"
"""
import sys

from m5phet.interpreters.ollama import OllamaInterpreter


def main(argv):
    if len(argv) < 3 or argv[1] == "-z" or "-z" not in argv:
        sys.exit("usage: interpreter_ollama.py MODEL -z PROMPT")
    model, prompt = argv[1], argv[argv.index("-z") + 1]
    sys.stdout.write(OllamaInterpreter({"model": model})._ask(prompt) + "\n")


if __name__ == "__main__":
    main(sys.argv)
