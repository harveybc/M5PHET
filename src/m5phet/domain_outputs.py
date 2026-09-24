"""Minimal structured output contracts; validation is not scientific acceptance."""
import math


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_domain_output(kind, payload, schema):
    if not isinstance(payload, dict):
        return "Domain payload must be an object"
    if kind == "point_forecast":
        targets, horizons, values = payload.get("targets"), payload.get("horizons"), payload.get("values")
        if not targets or targets != schema.get("targets") or horizons != schema.get("horizons"):
            return "Forecast target/horizon order differs from the declared contract"
        if not isinstance(values, list) or len(values) != len(targets) or any(
            not isinstance(row, list) or len(row) != len(horizons) or not all(number(v) for v in row) for row in values
        ):
            return "Forecast values must be finite [target][horizon] values"
        if not text(payload.get("unit")) or not text(payload.get("scale")):
            return "Forecast unit and scale are required"
    elif kind == "hierarchical_regimes":
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            return "Regime assignments require rows"
        ids = []
        for row in rows:
            if not isinstance(row, dict) or type(row.get("row_id")) not in (str, int):
                return "Each assignment needs a typed row_id"
            ids.append(row["row_id"])
            path = row.get("cluster_path")
            if not isinstance(path, list) or not path or any(type(v) not in (str, int) for v in path):
                return "Each assignment needs a cluster path"
            if not number(row.get("novelty_score")) or row["novelty_score"] < 0:
                return "Novelty must be finite and nonnegative"
        if len(set(ids)) != len(ids):
            return "Duplicate assignment rows"
        if not text(payload.get("model_version")) or payload["model_version"] != schema.get("model_version"):
            return "Regime model version differs from the declared version"
    elif kind == "causal_effect":
        interval = payload.get("interval")
        if not text(payload.get("estimand")) or not number(payload.get("estimate")) or not text(payload.get("unit")):
            return "Effect requires an estimand, finite estimate and unit"
        if not isinstance(interval, list) or len(interval) != 2 or not all(number(v) for v in interval) or interval[0] > interval[1]:
            return "Effect interval must have ordered finite bounds"
        assumptions = payload.get("assumptions")
        if not isinstance(assumptions, list) or not assumptions or not all(text(v) for v in assumptions):
            return "Causal assumptions must be declared"
        if not isinstance(payload.get("diagnostics"), dict):
            return "Causal diagnostics must be declared"
    elif kind == "policy_action":
        action = payload.get("action")
        if not isinstance(action, list) or not action or not all(number(v) for v in action):
            return "Policy action must be a finite numeric vector"
        if any(not text(payload.get(k)) for k in ("unit", "action_space", "policy_id")):
            return "Policy identity, action space and unit are required"
        if payload.get("execution_authorized") is not False:
            return "Policy proposals cannot authorize execution"
    else:
        return "Unknown domain output kind"
    return None
