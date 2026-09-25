"""WP31: every answer carries what was measured about the area that answered it -- and nothing else.

What these tests pin. A number is published only from a record somebody measured, with its corpus, its protocol and
its seal; a report of another family, or (for forecasting) one no configured bundle references, is refused by name
instead of borrowed. An area with no record says `NOT_MEASURED` out loud rather than saying nothing. Causal and rl
carry the refusal `m5phet_evaluation` owns, never a softer restatement of it.

And the rule that made this harder than it looks: a quality number is a FIGURE, so the narration guard applies to it.
It passes not because quality is exempt but because the answer carries it -- which is why the last tests here fabricate
one and watch it be discarded exactly as an invented forecast would be.
"""

import json

import pytest

from m5phet import quality
from m5phet.orchestrate import narration_problems, render
from m5phet.outputs import quality_line, response_view
from m5phet.outputs.default import DefaultOutput, text
from m5phet.outputs.telegram import CLOSING, TelegramOutput
from m5phet.questions import catalog as question_catalog, run_task
from m5phet.runtime import Registry


# --- fixtures: the shapes the real records have ----------------------------------------------------------------------

CLASSIFICATION_RECORD = {
    "schema": "news_signal.quality.v1", "corpus_id": "corpus-7e4789e5", "n": 450,
    "macro_f1": 0.37775954555995367,
    "calibration": {"status": "UNCALIBRATED", "expected_calibration_error": 0.13151177777777778,
                    "brier": 0.6472929861999999, "bins": 10},
    "corpus_seal": "31257d47b87aa3cbc9bc915d0fda418ad40e19dbbccd7b447e351af4f0f9af61",
    "protocol_digest": "b9aefb3cf32daff28a4c8b698a02fbc5b52021c6b82cce704e813f50d6102176",
    "label_provenance": "INDEPENDENT_LABELS", "naive": {"metric": "macro_f1", "majority_class": 0.16666666666666666},
    "skill": {"skill": 0.2533114546719444, "status": "OK"},
    "reading": "macro-F1 on 450 sealed, independently labelled rows of one task"}

FORECAST_REPORT = {
    "version": quality.REPORT_VERSION, "family": "forecast", "label_provenance": "REALISED_OUTCOME",
    "corpus_seal": "33820b552ddffff02b1d4aafe72266be281c94ecab85025ae55b9e2c230024fc",
    "protocol_digest": "d0ebd9a4bc7535e679705f1e0d92745cdff5cfa0e4a0dd9d3d322e14fea15df4",
    "sealed_row_count": 9824, "sealed_at": "2026-09-25T00:00:00Z", "minimum_rows": 1000, "flags": [],
    "scale": "kW", "target": "Global_active_power", "horizon": "h+60",
    "metric_sets": [
        {"name": "forecast", "family": "forecast", "rows_declared": 9824,
         "counts": {"scored_rows": 9824, "declared_rows": 9824},
         "values": {"mae": 0.526293524060822, "rmse": 0.8151394116711527, "skill_mae": 0.12185848183739156},
         "baseline": {"name": "last_value", "mae": 0.5993265472312703, "same_rows_as_model": True}},
        {"name": "interval_coverage", "family": "forecast", "rows_declared": 9824,
         "counts": {"scored_rows": 9824, "inside_rows": 9097},
         "values": {"empirical_coverage": 0.9259975570032574, "nominal_level": 0.95}, "baseline": None}]}

REGIMES_REPORT = {
    "version": quality.REPORT_VERSION, "family": "regimes", "label_provenance": "AUTHOR_WRITTEN_SMOKE",
    "corpus_seal": "94bf06df7b411684d778740dfbcf46491b2b5f8e393016a5749892c00941b121",
    "protocol_digest": "196e1cd9834512a0b98efdd7cdca7612062bd1a662de8fcfa38a18949058294c",
    "sealed_row_count": 10080, "sealed_at": "2026-09-25T00:00:00Z", "minimum_rows": 32,
    "flags": ["AUTHOR_WRITTEN_SMOKE"],
    "indices_omitted": {"silhouette": "INDEX_NOT_DEFINED: 1 cluster"},
    "metric_sets": [{"name": "regimes_internal_indices", "family": "regimes", "rows_declared": 10080,
                     "counts": {"scored_rows": 10080, "clusters_assigned": 1},
                     "values": {"stability_index": 0.9998015873015873}, "baseline": None}]}


def write(tmp_path, document, name="report.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def bundle(tmp_path, digest, name="quantile-household-95"):
    """A forecast bundle directory whose manifest names the report that measured it, as the real ones do."""
    directory = tmp_path / "bundles" / name
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "schema": "prediction_provider.forecast_bundle.v2", "state_id": name + "-20260925",
        "provenance": {"quality": "UNMEASURED",
                       "evaluation": {"protocol_digest": FORECAST_REPORT["protocol_digest"],
                                      "corpus_seal": FORECAST_REPORT["corpus_seal"], "sealed_rows": 9824,
                                      "report_sha256": digest}}}), encoding="utf-8")
    return directory.parent


def digest_of(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- a measurement travels with its protocol, or it does not travel ----------------------------------------------------

def test_the_classification_record_is_published_with_its_corpus_protocol_and_seal():
    block = quality.for_area("classification", {"quality": CLASSIFICATION_RECORD})
    assert block["status"] == quality.MEASURED
    assert block["values"]["macro_f1"] == CLASSIFICATION_RECORD["macro_f1"]
    assert block["values"]["expected_calibration_error"] == 0.13151177777777778
    assert block["corpus_id"] == "corpus-7e4789e5"
    assert block["protocol_digest"] == CLASSIFICATION_RECORD["protocol_digest"]
    assert block["corpus_seal"] == CLASSIFICATION_RECORD["corpus_seal"]
    assert block["skill"] == 0.2533114546719444


def test_a_provider_that_publishes_no_quality_is_not_measured_and_says_so():
    block = quality.for_area("classification", {"quality": "NOT_MEASURED"})
    assert block["status"] == quality.NOT_MEASURED and block["why"]
    assert quality.line(block).startswith("quality (classification): NOT_MEASURED")


def test_a_provider_whose_own_record_refused_itself_keeps_that_refusal_in_its_words():
    block = quality.for_area("classification", {"quality": "QUALITY_RECORD_REFUSED: INCOMPLETE_QUALITY_RECORD: ..."})
    assert block["status"] == quality.REFUSED and block["refusal"] == quality.QUALITY_RECORD_REFUSED
    assert "INCOMPLETE_QUALITY_RECORD" in block["why"]


def test_a_forecast_report_a_configured_bundle_names_is_published_with_its_error_skill_and_coverage(tmp_path):
    path = write(tmp_path, FORECAST_REPORT)
    bundles = bundle(tmp_path, digest_of(path))
    block = quality.for_area("forecasting", None, None,
                             {"M5PHET_FORECASTING_QUALITY_REPORT": str(path),
                              "M5PHET_FORECAST_BUNDLE": str(bundles)})
    assert block["status"] == quality.MEASURED and block["kind"] == "forecast"
    assert block["values"]["mae"] == 0.526293524060822
    assert block["skill"] == 0.12185848183739156 and block["baseline"]["name"] == "last_value"
    assert block["interval_coverage"]["empirical_coverage"] == 0.9259975570032574
    assert block["measured_on"] == "quantile-household-95-20260925"
    assert block["report_sha256"] == digest_of(path)
    assert str(path) not in json.dumps(block), \
        "the digest travels, never the local path: where the operator keeps his files proves nothing"


def test_a_report_no_configured_bundle_references_is_refused_rather_than_borrowed(tmp_path):
    path = write(tmp_path, FORECAST_REPORT)
    bundles = bundle(tmp_path, "0" * 64)
    block = quality.for_area("forecasting", None, None,
                             {"M5PHET_FORECASTING_QUALITY_REPORT": str(path),
                              "M5PHET_FORECAST_BUNDLE": str(bundles)})
    assert block["status"] == quality.REFUSED
    assert block["refusal"] == quality.QUALITY_MEASURED_ON_ANOTHER_STATE


def test_a_report_of_another_family_is_refused_for_this_area(tmp_path):
    path = write(tmp_path, FORECAST_REPORT)
    block = quality.for_area("unsupervised", None, None, {"M5PHET_UNSUPERVISED_QUALITY_REPORT": str(path)})
    assert block["refusal"] == quality.QUALITY_MEASURED_ON_ANOTHER_FAMILY


def test_a_document_that_is_not_an_evaluation_report_is_refused_by_name(tmp_path):
    path = tmp_path / "not_a_report.json"
    path.write_text(json.dumps({"mae": 0.1}), encoding="utf-8")
    block = quality.for_area("unsupervised", None, None, {"M5PHET_UNSUPERVISED_QUALITY_REPORT": str(path)})
    assert block["refusal"] == quality.QUALITY_REPORT_UNREADABLE


def test_a_report_missing_its_seal_or_its_counts_is_not_a_measurement(tmp_path):
    incomplete = {k: v for k, v in REGIMES_REPORT.items() if k != "corpus_seal"}
    path = write(tmp_path, incomplete)
    block = quality.for_area("unsupervised", None, None, {"M5PHET_UNSUPERVISED_QUALITY_REPORT": str(path)})
    assert block["refusal"] == quality.QUALITY_REPORT_UNREADABLE and "corpus_seal" in block["why"]


def test_the_representation_report_publishes_its_internal_indices_and_what_was_not_defined(tmp_path):
    path = write(tmp_path, REGIMES_REPORT)
    block = quality.for_area("unsupervised", None, None, {"M5PHET_UNSUPERVISED_QUALITY_REPORT": str(path)})
    assert block["status"] == quality.MEASURED and block["kind"] == "regimes"
    assert block["values"]["stability_index"] == 0.9998015873015873
    assert "silhouette" in block["indices_omitted"]
    assert block["flags"] == ["AUTHOR_WRITTEN_SMOKE"]


def test_an_area_with_no_declared_report_is_not_measured_and_names_no_number(tmp_path):
    block = quality.for_area("unsupervised", None, None, {})
    assert block["status"] == quality.NOT_MEASURED
    assert not any(key in block for key in ("values", "skill", "baseline"))


def test_a_bundle_that_names_where_its_numbers_live_still_publishes_no_number(tmp_path):
    path = write(tmp_path, FORECAST_REPORT)
    bundles = bundle(tmp_path, digest_of(path))
    block = quality.for_area("forecasting", None, None, {"M5PHET_FORECAST_BUNDLE": str(bundles)})
    assert block["status"] == quality.NOT_MEASURED
    assert block["evaluation_references"][0]["evaluation"]["report_sha256"] == digest_of(path)
    assert "0.52" not in json.dumps(block), "a pointer to a report is not the report's numbers"


# --- the two areas nothing can measure ---------------------------------------------------------------------------------

@pytest.mark.parametrize("area,name", [("causal", "causal_accuracy"), ("rl", "policy_profitability")])
def test_causal_and_policy_refuse_the_quantity_with_the_reason_the_evaluation_package_owns(area, name):
    block = quality.for_area(area)
    assert block["status"] == quality.REFUSED and block["refusal"] == name
    assert name in block["reason_source"]
    owned = quality.refused_metric_reason(name)
    if owned is not None:
        assert block["why"] == owned, "the reason is imported, never restated more softly here"


def test_a_refusal_whose_reason_the_guard_admits_is_rendered_in_full():
    block = quality.for_area("causal")
    rendered = quality.line(block)
    if quality.refused_metric_reason("causal_accuracy"):
        assert block["why"][:40] in rendered, "the reason travels in the line when the guard admits it"
    assert not narration_problems(rendered, {"quality": block})


def test_a_refusal_the_guard_cannot_admit_points_at_its_reason_instead_of_rewording_it():
    """`policy_profitability`'s reason ends with a clause the narration guard reads as a claim of profit.

    It is a statement that profit does NOT exist here, and the guard is right by its own rule: the word appears in a
    sentence that does not negate it. The rule is not relaxed and the reason is not softened -- the line names the
    refusal and says where the reason is, and the block still carries it verbatim."""
    block = quality.for_area("rl")
    rendered = quality.line(block)
    assert not narration_problems(rendered, {"quality": block}), "the line a person reads passes the guard"
    if quality.refused_metric_reason("policy_profitability"):
        assert narration_problems(f"policy_profitability — {block['why']}", {"quality": block}), \
            "this test only means something while the full reason is the thing the guard refuses"
        assert "quality.why" in rendered and block["why"] not in rendered
        assert block["why"] == quality.refused_metric_reason("policy_profitability")


# --- one line, in the answer, held to the guard --------------------------------------------------------------------------

def response(area, answers, block):
    answered = sum(1 for a in answers.values() if a.get("status", "OK") == "OK")
    return {"area": area, "answers": answers, "answered": answered,
            "refused": len(answers) - answered, "quality": block}


def test_the_rendered_quality_line_passes_the_narration_guard(tmp_path):
    path = write(tmp_path, FORECAST_REPORT)
    bundles = bundle(tmp_path, digest_of(path))
    block = quality.for_area("forecasting", None, None,
                             {"M5PHET_FORECASTING_QUALITY_REPORT": str(path),
                              "M5PHET_FORECAST_BUNDLE": str(bundles)})
    out = response("forecasting", {"rango": {"type": "interval", "status": "OK", "lower": 1.2, "upper": 3.4}}, block)
    rendered = text(out)
    lines = [line for line in rendered.splitlines() if line.startswith("quality (")]
    assert len(lines) == 1, "exactly one line, never two"
    assert "0.526293524060822" in lines[0] and "last_value" in lines[0]
    assert "0.9259975570032574" in lines[0], "an interval was returned, so its coverage is stated"
    assert not narration_problems(rendered, response_view(out))


def test_the_same_figures_are_refused_when_the_answer_does_not_carry_the_quality(tmp_path):
    path = write(tmp_path, FORECAST_REPORT)
    bundles = bundle(tmp_path, digest_of(path))
    block = quality.for_area("forecasting", None, None,
                             {"M5PHET_FORECASTING_QUALITY_REPORT": str(path),
                              "M5PHET_FORECAST_BUNDLE": str(bundles)})
    carried = response("forecasting", {"p": {"type": "point_forecast", "status": "OK", "values": [0.5]}}, block)
    line = quality.line(block)
    bare = {k: v for k, v in carried.items() if k != "quality"}
    assert not narration_problems(line, response_view(carried))
    assert narration_problems(line, response_view(bare)), \
        "the line passes because the ANSWER carries the quality, not because a quality is exempt"


def test_a_fabricated_quality_figure_is_discarded_like_any_other_invention(tmp_path):
    path = write(tmp_path, FORECAST_REPORT)
    bundles = bundle(tmp_path, digest_of(path))
    block = quality.for_area("forecasting", None, None,
                             {"M5PHET_FORECASTING_QUALITY_REPORT": str(path),
                              "M5PHET_FORECAST_BUNDLE": str(bundles)})
    out = response("forecasting", {"p": {"type": "point_forecast", "status": "OK", "values": [0.5]}}, block)
    invented = "quality (forecasting): MAE 0.031 — 9824 scored rows"
    problems = narration_problems(invented, response_view(out))
    assert problems and "0.031" in problems[0]


def test_an_interval_coverage_is_stated_only_where_an_interval_was_returned(tmp_path):
    path = write(tmp_path, FORECAST_REPORT)
    bundles = bundle(tmp_path, digest_of(path))
    block = quality.for_area("forecasting", None, None,
                             {"M5PHET_FORECASTING_QUALITY_REPORT": str(path),
                              "M5PHET_FORECAST_BUNDLE": str(bundles)})
    point = response("forecasting", {"p": {"type": "point_forecast", "status": "OK", "values": [0.5]}}, block)
    assert "coverage" not in quality.line(block, point)
    interval = response("forecasting", {"r": {"type": "interval", "status": "OK", "lower": 0.1}}, block)
    assert "coverage" in quality.line(block, interval)


def test_the_classification_line_names_its_protocol_and_its_seal_and_does_not_call_a_metric_a_baseline():
    block = quality.for_area("classification", {"quality": CLASSIFICATION_RECORD})
    rendered = quality.line(block)
    assert "protocol b9aefb3cf32d" in rendered and "seal 31257d47b87a" in rendered
    assert "vs macro_f1" not in rendered, "macro_f1 is the metric; the naive references are named in the block"
    assert "0.37775954555995367" in rendered
    out = response("classification", {"e": {"type": "choice", "status": "OK", "label": "euro_area"}}, block)
    assert not narration_problems(rendered, response_view(out))


def test_the_telegram_message_carries_the_line_and_still_ends_with_its_closing():
    block = quality.for_area("classification", {"quality": CLASSIFICATION_RECORD})
    out = response("classification", {"economia": {"type": "choice", "status": "OK", "label": "euro_area"}}, block)
    rendered = TelegramOutput().render("classification", out, "es")["text"]
    assert rendered.endswith(CLOSING)
    assert sum(1 for line in rendered.splitlines() if line.startswith("quality (")) == 1
    assert not narration_problems(rendered, response_view(out))


def test_an_answer_that_carries_no_quality_renders_exactly_as_it_always_did():
    answers = {"p": {"type": "point_forecast", "status": "OK", "values": [0.5412]}}
    before = {"area": "forecasting", "answers": answers, "answered": 1, "refused": 0}
    assert "quality" not in text(before)
    assert text(before) == render(before)
    assert quality_line(before) is None
    telegram = TelegramOutput().render("forecasting", before, "es")["text"]
    assert "quality (" not in telegram


def test_the_json_a_procedure_returns_carries_the_block_itself():
    block = quality.for_area("classification", {"quality": CLASSIFICATION_RECORD})
    out = response("classification", {"e": {"type": "choice", "status": "OK", "label": "euro_area"}}, block)
    for plugin in (DefaultOutput(), TelegramOutput()):
        assert plugin.render("classification", out, "es")["json"]["quality"] == block


# --- the answer and the catalog both carry it ----------------------------------------------------------------------------

class Forecaster:
    name, area = "fake_forecaster", "forecasting"

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["regression_forecasting"],
                "output_kinds": ["point_forecast"], "uncertainty_methods": ["none"],
                "supported": [{"operation": "infer", "family": "regression_forecasting",
                               "output_kind": "point_forecast"}],
                "known_states": ["s1"]}

    def question_types(self):
        return {"point_forecast": {"required": ["horizon"], "optional": ["target"]}}

    def answer_questions(self, state, questions, data, as_of):
        return {n: {"type": "point_forecast", "values": [0.5412], "unit": "kW"} for n in questions}


def registry():
    r = Registry()
    r.register(Forecaster())
    return r


def test_every_answer_of_an_envelope_carries_its_area_s_quality():
    out = run_task({"area": "forecasting", "state": {}, "questions": {"p": {"type": "point_forecast", "horizon": 1}}},
                   registry(), environ={})
    assert out["quality"]["status"] == quality.NOT_MEASURED
    assert out["quality"]["area"] == "forecasting"


def test_an_envelope_no_provider_serves_still_states_what_is_known_about_that_area():
    out = run_task({"area": "causal", "state": {}, "questions": {"e": {"type": "ate"}}}, Registry(), environ={})
    assert out["quality"]["refusal"] == "causal_accuracy"


def test_every_catalog_entry_carries_a_quality_block():
    entries = question_catalog(registry())
    assert set(entries) >= {"classification", "forecasting", "causal", "rl", "unsupervised"}
    for area, entry in entries.items():
        assert entry["quality"]["status"] in (quality.MEASURED, quality.NOT_MEASURED, quality.REFUSED)
        assert entry["quality"]["area"] == area
