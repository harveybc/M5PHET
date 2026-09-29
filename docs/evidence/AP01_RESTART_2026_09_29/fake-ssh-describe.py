#!/home/harveybc/.local/share/m5phet/staging-ap-20260929/venv/bin/python
"""A local stand-in for `ssh`, for AP01's backend cases. Nothing leaves this machine: no packet, no host name, no key.

It is a TRANSPORT FAKE and never a classification provider: what it returns is labelled a fake in every report that
uses it. AP_FAKE_WORKER picks what the imagined worker declares -- ok | other_checkpoint | fail.
"""
import json, os, sys

mode = os.environ.get("AP_FAKE_WORKER", "ok")
if mode == "fail":
    sys.stderr.write("ap fake worker: refusing to connect\n")
    sys.exit(255)
command = json.load(sys.stdin)
checkpoint = "laya-checkpoint:" + ("a" * 64 if mode != "other_checkpoint" else "b" * 64)
caps = {"provider": "laya_news", "operations": ["infer"], "families": ["classification"],
        "output_kinds": ["typed_questions"], "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
        "supported": [{"operation": "infer", "family": "classification", "output_kind": "typed_questions"}],
        "known_states": [checkpoint], "question_types": {"choice": {"required": ["options"]}},
        "tasks": ["news_relevance"], "device": "cuda:0", "backend": "laya", "weights_present": True,
        "quality": "NOT_MEASURED"}
if command.get("action") == "describe":
    print(json.dumps({"capabilities": caps, "discovery": {"registered": ["laya_news"], "refused": {}}}))
    sys.exit(0)
print(json.dumps({"transport_error": "ap fake worker: this fake answers 'describe' only, by design"}))
