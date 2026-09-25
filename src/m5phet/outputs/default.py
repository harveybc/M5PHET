"""The `default` procedure: the workbench's own rendering, unchanged.

This is what the product has done since the envelope existed, now named and selectable. Its text is one deterministic
line per answer -- the answer's name, its type, and the first fields it carries with their numbers as the engine
returned them -- and a closing line saying how many were answered, how many refused, and that nothing here is an
instruction to act. A refused question is named with its reason and carries no number, because a refusal that carries
a number is indistinguishable from an answer.

The plugin declares `narrates = True`: after this deterministic text is produced, `m5phet.orchestrate.narrate` may ask
the configured interpreter for a sentence about the same answers, and keeps it ONLY if every number in it is one the
answers carry. That is why this rendering matters even when a model is configured -- it is what the guard falls back
to, and it is faithful by construction.
"""

from . import clip, header as area_header, narratable, value_text

#: fields that describe the answer rather than report it; the type and the status are already in the line's prefix
NOT_REPORTED = ("type", "status", "execution_authorized", "sdk_answer")

#: how many of an answer's fields one line shows, and how long each value may be
FIELDS_PER_ANSWER = 6
VALUE_LIMIT = 80


def answer_line(name, answer):
    """One answer as one line, or its refusal with the reason and no number."""
    if not isinstance(answer, dict):
        return f"{name}: {value_text(answer, VALUE_LIMIT)}"
    if answer.get("status") == "REFUSED":
        return f"{name}: not answered -- {answer.get('why')}"
    fields = {k: v for k, v in answer.items() if k not in NOT_REPORTED}
    summary = ", ".join(f"{k}={value_text(v, VALUE_LIMIT)}" for k, v in list(fields.items())[:FIELDS_PER_ANSWER])
    return f"{name} ({answer.get('type')}): {summary}"


def text(response):
    """A deterministic sentence per answer. Plain, and always faithful by construction.

    It renders the same view every procedure and the guard read: the backend's verbatim object and the digests are
    not part of what a person is shown, and a digest printed in a line is a string of figures the answers do not
    carry -- the guard says so, and it is right."""
    lines = [answer_line(name, answer) for name, answer in narratable(response.get("answers") or {}).items()]
    lines.append(f"{response.get('answered', 0)} answered, {response.get('refused', 0)} refused; nothing here is an "
                 f"instruction to act.")
    return "\n".join(lines)


def table(response):
    """The same answers as rows, for a surface that draws a table instead of reading a paragraph."""
    rows = []
    for name, answer in narratable(response.get("answers") or {}).items():
        answer = answer if isinstance(answer, dict) else {"value": answer}
        row = {"question": name, "type": answer.get("type"), "status": answer.get("status", "OK"),
               "refusal": answer.get("refusal"), "why": answer.get("why")}
        row["fields"] = {k: v for k, v in answer.items() if k not in NOT_REPORTED and k not in ("refusal", "why")}
        rows.append(row)
    return rows


class DefaultOutput:
    """The workbench's procedure, selected by `"plugin": "default"` (the default when nothing is configured)."""

    name = "default"

    #: this procedure may additionally ask the configured interpreter for a sentence, which the guard then checks
    narrates = True

    def __init__(self, settings=None):
        self.settings = dict(settings or {})
        self.language = self.settings.get("language") or "es"

    def header(self, area):
        return area_header(area)

    def render(self, area, response, language="es"):
        return {"text": text(response), "table": table(response),
                "json": {"area": area, "header": self.header(area),
                         "answers": narratable(response.get("answers") or {}),
                         "answered": response.get("answered", 0), "refused": response.get("refused", 0),
                         "execution_authorized": False},
                "language": language or self.language, "output_plugin": self.name}


__all__ = ["DefaultOutput", "answer_line", "clip", "table", "text"]
