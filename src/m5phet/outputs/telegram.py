"""The `telegram` procedure: everything a person needs in one short plain-text message.

A chat application is not a workbench. There is no table, no JSON pane and no place to expand a field, and a message
is cut at a few thousand characters by the transport itself. So this procedure states, per answer, exactly the fields
the area's header declares as the ones carrying its numbers and units, one answer per line, with the numbers exactly
as the engine returned them -- no rounding, because a rounded number is a different number and the person may quote it
back. A refusal is named with its code and its reason, and carries no number.

Every message ends with the same sentence. On a surface where the previous message might have been a person telling
someone to buy something, the line that says this is not an instruction to act is not decoration; it is the one part
of the message that must never be cut, which is why the body is bounded to leave room for it.

The header does real work here: `unit_fields` is what a short rendering shows. That is also why the unsupervised
header names the cluster counts and not the per-row assignments -- a list of ten thousand row ids is not a message.
"""

from . import clip, header as area_header, narratable, one_line, value_text

#: Telegram cuts a message at 4096 characters; the bound is stated at 4000 so the closing line is never the part that
#: does not fit, and so a forwarded message keeps its ending.
LIMIT = 4000

#: never omitted, never rewritten, always last
CLOSING = "execution_authorized: false — this is not an instruction to act"

VALUE_LIMIT = 120
WHY_LIMIT = 300

#: shown when the answer carries none of the header's declared fields, so a line is never empty
FALLBACK_FIELDS = 4

NOT_REPORTED = ("type", "status", "execution_authorized", "sdk_answer", "refusal", "why")


def answer_line(name, answer, header):
    """One answer, one line: the fields the area's header declares, with the numbers exactly as returned."""
    if not isinstance(answer, dict):
        return f"{name}: {value_text(answer, VALUE_LIMIT)}"
    if answer.get("status") == "REFUSED":
        reason = one_line(answer.get("why") or "no reason was recorded")
        return (f"{name} ({answer.get('type')}): REFUSED {answer.get('refusal')} — "
                f"{clip(reason, WHY_LIMIT)}")
    declared = [field for field in header.get("unit_fields") or () if field in answer]
    if not declared:
        declared = [k for k in answer if k not in NOT_REPORTED][:FALLBACK_FIELDS]
    shown = ", ".join(f"{field}={value_text(answer[field], VALUE_LIMIT)}" for field in declared)
    return f"{name} ({answer.get('type')}): {shown}" if shown else f"{name} ({answer.get('type')}): answered"


class TelegramOutput:
    """A plain-text message, selected by `"plugin": "telegram"` for an area served over a chat application."""

    name = "telegram"

    #: no model is asked: a chat surface gets the numbers, not a paragraph about them
    narrates = False

    def __init__(self, settings=None):
        self.settings = dict(settings or {})
        self.language = self.settings.get("language") or "es"
        self.limit = int(self.settings.get("limit") or LIMIT)

    def header(self, area):
        return area_header(area)

    def render(self, area, response, language="es"):
        header = self.header(area)
        answers = narratable(response.get("answers") or {})
        body = "\n".join(answer_line(name, answer, header) for name, answer in answers.items())
        tail = (f"{response.get('answered', 0)} answered, {response.get('refused', 0)} refused.\n{CLOSING}")
        room = self.limit - len(tail) - 1
        text = (clip(body, room, mark="…") if len(body) > room else body)
        return {"text": (text + "\n" + tail) if text else tail,
                "table": [{"question": name, "line": answer_line(name, answer, header)}
                          for name, answer in answers.items()],
                "json": {"area": area, "header": header, "answers": answers,
                         "answered": response.get("answered", 0), "refused": response.get("refused", 0),
                         "execution_authorized": False},
                "language": language or self.language, "output_plugin": self.name}


__all__ = ["CLOSING", "LIMIT", "TelegramOutput", "answer_line"]
