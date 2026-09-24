"""First implemented contract; other task families remain separate designs."""

import hashlib
import json
import math
import re


class ContractError(ValueError):
    pass


def make_classification_result(outputs, *, expected_labels, input_sha256, model_sha256,
                               task_sha256, probability_decimals=None):
    """Validate and snapshot a complete set of uncalibrated class distributions.

    Decimal precision describes the producer's rounding, never scientific
    uncertainty. The result is evidence data, not permission to execute actions.
    """
    identities = {"input_sha256": input_sha256, "model_sha256": model_sha256, "task_sha256": task_sha256}
    for name, value in identities.items():
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ContractError(f"{name}: lowercase SHA256 required")
    if probability_decimals is not None and (type(probability_decimals) is not int or not 1 <= probability_decimals <= 16):
        raise ContractError("probability_decimals: integer 1..16 or None required")
    if not isinstance(expected_labels, dict) or not expected_labels:
        raise ContractError("nonempty declared task population required")
    for task, labels in expected_labels.items():
        if not isinstance(task, str) or not task.strip() or not isinstance(labels, list) or not labels:
            raise ContractError("task and label list required")
        if any(not isinstance(label, str) or not label.strip() for label in labels) or len(set(labels)) != len(labels):
            raise ContractError("unique nonempty label names required")
    if not isinstance(outputs, dict) or set(outputs) != set(expected_labels):
        raise ContractError("output population differs from declared task")
    for task, labels in expected_labels.items():
        result = outputs[task]
        if not isinstance(result, dict) or set(result) != {"label", "uncalibrated_probabilities"}:
            raise ContractError("classification fields mismatch")
        probs = result["uncalibrated_probabilities"]
        if not isinstance(probs, dict) or set(probs) != set(labels):
            raise ContractError("probability labels mismatch")
        if any(type(v) not in (int, float) or not 0 <= v <= 1 or not math.isfinite(v) for v in probs.values()):
            raise ContractError("finite nonboolean probabilities in [0,1] required")
        tolerance = 1e-12 if probability_decimals is None else len(labels) * 0.5 * 10**(-probability_decimals) + 1e-12
        if abs(math.fsum(probs.values()) - 1.0) > tolerance:
            raise ContractError("probability sum outside declared precision")
        label = result["label"]
        if not isinstance(label, str) or label not in probs or probs[label] != max(probs.values()):
            raise ContractError("label must be a maximal-probability declared class")
    # JSON snapshot prevents later mutation of caller-owned nested dictionaries.
    payload = json.loads(json.dumps({
        "schema": "m5phet.classification.v1", "family": "classification",
        "uncertainty": "UNCALIBRATED_CLASS_PROBABILITIES",
        "execution_authorized": False, "provenance": identities,
        "expected_labels": expected_labels, "outputs": outputs,
        "probability_decimals": probability_decimals,
    }, allow_nan=False))
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload
