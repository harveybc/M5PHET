"""CB05 correction 3: the router corpus scoped honestly, and a held-out set nobody may tune against.

Two things were overstated about the 2026-09-28 completion-pass comparison, and both are about the denominator rather
than the arithmetic:

* it is **19 prompts x 5 repeats**, not 95 independent examples. Five runs of one sentence are five observations of
  one item, so a difference of six runs is not six examples' worth of evidence, and the report has to say which of
  the two it is holding;
* it measures the **router** -- a language model writing an envelope -- and not the classifier. A router number is
  not a classification score, and this corpus is not a classification benchmark under any reading.

And one thing had to be protected rather than described: the whole corpus has now been used to tune a switch, so the
same 19 sentences can never again serve as untouched validation. A frozen, held-apart set of unseen paraphrases is
pinned here by digest, so a later generalization test has something the code has never been fitted to -- and so that
anyone who edits it has to change a digest in this file and explain why.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import measure_route                                                                    # noqa: E402
from measure_route import CASES, corpus_scope                                           # noqa: E402

REPO = Path(__file__).resolve().parent.parent
HELDOUT = REPO / "docs/evidence/ROUTE_GENERALIZATION_HELDOUT_2026_09_28/heldout_paraphrases.json"

#: the freeze. A held-out set whose contents can drift is not held out; changing it means changing this line.
HELDOUT_SHA256 = "f0e0b0cad84e1b6f25dcf7e3f2f3c3e7d9d9e0a2d7ed96d0dcb0b4d80f4b8b6a"


def test_the_scope_block_declares_prompts_and_repeats_and_not_an_example_count():
    scope = corpus_scope(CASES, None, runs=5)
    assert scope["scoped"] is False
    assert scope["prompts"] == len(CASES)
    assert scope["repeats"] == 5
    assert scope["runs"] == len(CASES) * 5
    # the number that must never be read as a sample size
    assert scope["independent_examples"] is False
    assert "repeats" in scope["reading"] and "not" in scope["reading"]


def test_the_scope_block_says_what_is_measured_and_what_this_corpus_is_not():
    scope = corpus_scope(CASES, None, runs=5)
    assert scope["measures"] == "router"
    assert scope["role"] == measure_route.CORPUS_ROLE
    assert scope["not_a_classification_benchmark"] is True
    assert "classification benchmark" in scope["reading"].lower()


def test_a_scoped_run_still_says_it_is_a_subset_and_counts_its_own_prompts():
    scope = corpus_scope(CASES, ["cuerpo alto"], runs=3)
    assert scope["scoped"] is True
    assert scope["prompts"] == 1 and scope["repeats"] == 3 and scope["runs"] == 3
    assert scope["of"] == len(CASES)
    assert scope["independent_examples"] is False
    assert "SUBSET" in scope["reading"]


def test_the_role_is_stated_once_and_used_everywhere():
    assert "ROUTER" in measure_route.CORPUS_ROLE
    assert "CLASSIFICATION" in measure_route.CORPUS_ROLE
    assert measure_route.CORPUS_ROLE in measure_route.PROTOCOL


# --- the frozen held-out paraphrases ------------------------------------------------------------------------------------

def heldout():
    return json.loads(HELDOUT.read_text(encoding="utf-8"))


def test_the_heldout_file_is_the_one_this_test_pins():
    assert HELDOUT.is_file(), "the held-out paraphrases must ship with the branch that promised them"
    assert hashlib.sha256(HELDOUT.read_bytes()).hexdigest() == HELDOUT_SHA256


def test_the_heldout_file_declares_what_it_is_for_and_that_it_has_not_been_measured():
    document = heldout()
    assert document["schema"] == "m5phet_route_heldout_paraphrases.v1"
    assert document["status"] == "FROZEN_NOT_MEASURED"
    assert document["measures"] == "router"
    assert document["not_a_classification_benchmark"] is True
    assert "tun" in document["rule"].lower()               # the rule forbids tuning against it, in its own words


def test_every_heldout_prompt_is_unseen_by_the_development_corpus():
    """Held apart means held apart: not one of these sentences may already be in CASES."""
    seen = {case["prompt"].strip().lower() for case in CASES}
    prompts = [item["prompt"] for item in heldout()["paraphrases"]]
    assert len(prompts) == len(set(prompts)), "a duplicate inside the held-out set is one fewer unseen sentence"
    assert not (seen & {prompt.strip().lower() for prompt in prompts})


def test_every_heldout_prompt_carries_the_expectation_a_later_run_would_score_it_against():
    areas = {"classification", "forecasting", "unsupervised", "causal", "rl"}
    for item in heldout()["paraphrases"]:
        assert item["area"] in areas
        assert item["types"] and isinstance(item["types"], list)
        assert isinstance(item.get("values", {}), dict)
        assert item["paraphrase_of"] in {case["prompt"] for case in CASES}


def test_the_heldout_set_covers_every_area_the_development_corpus_covers():
    covered = {item["area"] for item in heldout()["paraphrases"]}
    assert covered == {case["area"] for case in CASES}
