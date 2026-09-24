import copy

import pytest

from m5phet.domain_outputs import validate_domain_output


GOOD = {
    "point_forecast": ({"targets": ["price"], "horizons": [1, 2], "values": [[2.0, 3.0]], "unit": "USD", "scale": "original"}, {"targets": ["price"], "horizons": [1, 2]}),
    "hierarchical_regimes": ({"rows": [{"row_id": "r1", "cluster_path": [0, 1], "novelty_score": .4}], "model_version": "a" * 64}, {"targets": ["regimes"], "model_version": "a" * 64}),
    "causal_effect": ({"estimand": "ATE", "estimate": 2., "interval": [1., 3.], "unit": "return", "assumptions": ["exchangeability"], "diagnostics": {}}, {"targets": ["effect"]}),
    "policy_action": ({"action": [0.], "action_space": "continuous", "unit": "target_fraction", "policy_id": "p", "execution_authorized": False}, {"targets": ["policy"]}),
}


@pytest.mark.parametrize("kind", GOOD)
def test_domain_positive(kind):
    payload, schema = GOOD[kind]
    assert validate_domain_output(kind, payload, schema) is None


@pytest.mark.parametrize("kind", GOOD)
def test_missing_fields(kind):
    payload, schema = GOOD[kind]
    for key in payload:
        altered = copy.deepcopy(payload)
        del altered[key]
        assert validate_domain_output(kind, altered, schema), (kind, key)


def test_shape_order_and_semantics():
    assert validate_domain_output("point_forecast", GOOD["point_forecast"][0] | {"values": [[1]]}, GOOD["point_forecast"][1])
    assert validate_domain_output("point_forecast", GOOD["point_forecast"][0] | {"horizons": [2, 1]}, GOOD["point_forecast"][1])
    assert validate_domain_output("hierarchical_regimes", GOOD["hierarchical_regimes"][0] | {"model_version": "b" * 64}, GOOD["hierarchical_regimes"][1])
    assert validate_domain_output("causal_effect", GOOD["causal_effect"][0] | {"interval": [3, 1]}, GOOD["causal_effect"][1])
    assert validate_domain_output("policy_action", GOOD["policy_action"][0] | {"execution_authorized": True}, GOOD["policy_action"][1])
