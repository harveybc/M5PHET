"""WP23 step 1: a decision record linked to one closure-table row, and every link that must be refused instead.

The rule these tests exist to enforce is one sentence long: **a person's opinion is never a label; only a table row
is.** So the only way a decision acquires an outcome is through a row that `evaluation/compare_stages.py` measured,
found `COMPARABLE`, and ranked. A row that was not comparable, a row that carries no rank, and a record that does not
load are each refused by their own name, and nothing is written.

Every row below is produced by the real generator over reports written by the evaluation package's own writer, so a
change to the table's shape breaks these tests rather than leaving `outcome()` reading a field that no longer exists.
"""

import hashlib
import json

import pytest

import closure_fixtures
from m5phet import decide

OPTIONS = [["a", "keep the level"], ["b", "normalise per feature"], ["c", "log return"]]


def decision(*, kind="feature_preprocessing", question="preprocessing", chosen="b", salt="0",
             probabilities=None, options=None):
    return {"schema": decide.DECISION_SCHEMA,
            "kind": kind,
            "state_sha256": hashlib.sha256(salt.encode()).hexdigest(),
            "question": question,
            "options": [list(pair) for pair in (options or OPTIONS)],
            "chosen": chosen,
            "probabilities": probabilities or {"a": 0.34, "b": 0.35, "c": 0.31},
            "probability_decimals": 4,
            "checkpoint": "laya-checkpoint:bd12df887789",
            "backend": "laya",
            "as_of": "2026-09-25T00:00:00+00:00",
            "execution_authorized": False}


def written(tmp_path, **fields):
    return decide.record(decision(**fields), tmp_path / "decisions")


def comparable_rows(tmp_path):
    """Two stages on one holdout: both COMPARABLE, ranks 1 and 2, the better error ranked first."""
    return closure_fixtures.table(tmp_path, {"designed": 0.25, "baseline": 0.5})


# --------------------------------------------------------------------------------------------------------------------
# the link that is allowed
# --------------------------------------------------------------------------------------------------------------------

def test_a_comparable_ranked_row_produces_a_content_addressed_outcome_record(tmp_path):
    rows = comparable_rows(tmp_path)
    path = written(tmp_path, chosen="b")
    out = decide.outcome(path, rows["designed"], out_dir=tmp_path / "outcomes")

    assert out["status"] == "OK"
    written_record = json.loads(out["record_path"] and open(out["record_path"]).read())
    assert written_record == out["outcome"]
    assert out["outcome"]["schema"] == decide.OUTCOME_SCHEMA == "m5phet.decision_outcome.v1"
    assert out["outcome"]["decision_sha256"] == path.stem
    assert out["outcome"]["kind"] == "feature_preprocessing"
    assert out["outcome"]["question"] == "preprocessing"
    assert out["outcome"]["chosen"] == "b"
    assert out["outcome"]["options"] == OPTIONS
    assert out["outcome"]["stage"] == "designed"
    assert out["outcome"]["rank"] == 1
    assert out["outcome"]["comparability"] == "COMPARABLE"
    # the file is named after its own content, exactly as a decision record is
    assert out["record_path"].endswith(decide.outcome_sha256(out["outcome"]) + ".json")


def test_the_row_digest_binds_the_numbers_the_outcome_does_not_copy(tmp_path):
    rows = comparable_rows(tmp_path)
    path = written(tmp_path)
    out = decide.outcome(path, rows["designed"], out_dir=tmp_path / "outcomes")

    expected = hashlib.sha256(json.dumps(rows["designed"], sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode("utf-8")).hexdigest()
    assert out["outcome"]["table_row_sha256"] == expected
    # the model error, the naive error and the skill are in the row, not in the outcome: the digest is what binds them
    assert rows["designed"]["model_error"] == pytest.approx(0.25)
    assert rows["designed"]["naive"]["error"] == pytest.approx(1.0)
    assert rows["designed"]["skill"] == pytest.approx(0.75)
    assert not {"model_error", "skill", "naive"} & set(out["outcome"])


def test_the_first_ranked_stage_settles_the_best_ranked_option_and_no_other_stage_does(tmp_path):
    rows = comparable_rows(tmp_path)
    first = decide.outcome(written(tmp_path, chosen="b", salt="first"), rows["designed"],
                           out_dir=tmp_path / "outcomes")
    second = decide.outcome(written(tmp_path, chosen="a", salt="second"), rows["baseline"],
                            out_dir=tmp_path / "outcomes")

    assert first["outcome"]["rank"] == 1 and first["outcome"]["best_ranked_option"] == "b"
    assert second["outcome"]["rank"] == 2 and "best_ranked_option" not in second["outcome"]


def test_the_same_decision_and_the_same_row_are_the_same_file_twice(tmp_path):
    rows = comparable_rows(tmp_path)
    path = written(tmp_path)
    first = decide.outcome(path, rows["designed"], out_dir=tmp_path / "outcomes")
    second = decide.outcome(path, rows["designed"], out_dir=tmp_path / "outcomes")

    assert first["record_path"] == second["record_path"]
    assert len(list((tmp_path / "outcomes").glob("*.json"))) == 1


# --------------------------------------------------------------------------------------------------------------------
# the links that are refused, each by its own name
# --------------------------------------------------------------------------------------------------------------------

def test_a_row_that_is_not_comparable_is_refused_as_NOT_COMPARABLE_and_nothing_is_written(tmp_path):
    rows = closure_fixtures.table(tmp_path, {"designed": 0.25,
                                             "other_holdout": (0.1, closure_fixtures.OTHER_TRUTH)})
    row = rows["other_holdout"]
    assert row["comparability"].startswith("NOT_COMPARABLE")

    out = decide.outcome(written(tmp_path), row, out_dir=tmp_path / "outcomes")
    assert out["status"] == "REFUSED"
    assert out["refusal"] == decide.NOT_COMPARABLE == "NOT_COMPARABLE"
    assert "holdout differs" in out["why"]
    assert not (tmp_path / "outcomes").exists()


def test_a_row_with_no_measurement_is_refused_as_NOT_COMPARABLE(tmp_path):
    rows = closure_fixtures.table(tmp_path, {"designed": 0.25})
    row = dict(rows["designed"], status="NO_NEW_MEASUREMENT", comparability="NO_NEW_MEASUREMENT",
               model_error=None, rank=None)

    out = decide.outcome(written(tmp_path), row, out_dir=tmp_path / "outcomes")
    assert out["status"] == "REFUSED" and out["refusal"] == decide.NOT_COMPARABLE


def test_a_comparable_row_that_carries_no_rank_is_refused_as_NOT_RANKED(tmp_path):
    rows = closure_fixtures.table(tmp_path, {"designed": 0.25})
    row = dict(rows["designed"], rank=None)

    out = decide.outcome(written(tmp_path), row, out_dir=tmp_path / "outcomes")
    assert out["status"] == "REFUSED"
    assert out["refusal"] == decide.NOT_RANKED == "NOT_RANKED"
    assert not (tmp_path / "outcomes").exists()


def test_a_record_that_is_not_there_is_refused_as_DECISION_NOT_FOUND(tmp_path):
    rows = comparable_rows(tmp_path)
    out = decide.outcome(tmp_path / "decisions" / "nothing.json", rows["designed"], out_dir=tmp_path / "outcomes")
    assert out["status"] == "REFUSED"
    assert out["refusal"] == decide.DECISION_NOT_FOUND == "DECISION_NOT_FOUND"


def test_a_record_that_is_not_a_laya_decision_is_refused_as_DECISION_NOT_FOUND(tmp_path):
    folder = tmp_path / "decisions"
    folder.mkdir()
    path = folder / "not_a_decision.json"
    path.write_text(json.dumps({"schema": "something.else.v1"}))
    out = decide.outcome(path, comparable_rows(tmp_path)["designed"], out_dir=tmp_path / "outcomes")
    assert out["status"] == "REFUSED" and out["refusal"] == decide.DECISION_NOT_FOUND


def test_a_tampered_record_is_refused_as_DIGEST_MISMATCH(tmp_path):
    rows = comparable_rows(tmp_path)
    path = written(tmp_path, chosen="b")
    altered = json.loads(path.read_text())
    altered["chosen"] = "a"                                  # the file name still says what the record used to be
    path.write_text(json.dumps(altered, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    out = decide.outcome(path, rows["designed"], out_dir=tmp_path / "outcomes")
    assert out["status"] == "REFUSED"
    assert out["refusal"] == decide.DIGEST_MISMATCH == "DIGEST_MISMATCH"
    assert not (tmp_path / "outcomes").exists()


def test_a_row_that_is_not_a_closure_table_row_is_refused_by_name(tmp_path):
    out = decide.outcome(written(tmp_path), {"stage": "designed"}, out_dir=tmp_path / "outcomes")
    assert out["status"] == "REFUSED" and out["refusal"] == decide.MALFORMED_TABLE_ROW


def test_an_outcome_record_reloads_and_a_tampered_one_does_not(tmp_path):
    rows = comparable_rows(tmp_path)
    out = decide.outcome(written(tmp_path), rows["designed"], out_dir=tmp_path / "outcomes")
    assert decide.load_outcome(out["record_path"]) == out["outcome"]

    altered = json.loads(open(out["record_path"]).read())
    altered["rank"] = 99
    open(out["record_path"], "w").write(json.dumps(altered, sort_keys=True, separators=(",", ":")))
    with pytest.raises(decide.DecisionError):
        decide.load_outcome(out["record_path"])
