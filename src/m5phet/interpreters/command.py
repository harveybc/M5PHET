"""The `command` plugin: the interpreter the operator already has, called as `COMMAND -z PROMPT`.

This is the contract the workbench has used since the interpreter existed, unchanged and now named. The command is a
program on this machine -- `hermes --ignore-user-config -m deepseek-v4-flash --provider opencode-go` on the owner's
host -- and the prompt is passed as one argument. Its standard output is the reply.

It is given no tool, no path and no shell: the argument vector is built here, never a shell string, so nothing in a
person's sentence can become a command. `CUDA_VISIBLE_DEVICES=""` is forced for the same reason the rest of this
package forces it: the only GPU admitted for work on this fleet is the worker's external one.
"""

from ..interpret import Interpreter


class CommandInterpreter(Interpreter):
    """Today's interpreter, selected by `"plugin": "command"` (the default when nothing is configured)."""

    plugin = "command"

    def __init__(self, settings=None, environ=None):
        settings = dict(settings or {})
        super().__init__(command=settings.get("command"), model=settings.get("model"), environ=environ)
        if settings.get("timeout_seconds"):
            self.timeout = int(settings["timeout_seconds"])
