"""CB05: a contradictory backend selection must REFUSE, and what answered must be recorded, not inferred.

The defect these tests exist for was live in the operator's own configuration on 2026-09-28: `chat.env` set
`NEWS_SIGNAL_BACKEND="fixture"` AND bound `M5PHET_CHAT_LAYA_WORKER`, and the worker binding silently won. Real
weights answered while the configuration said fixture. It resolved that day in the useful direction; nothing in the
code made that the direction it had to resolve in, and nobody reading the configuration could have told which.

So the rules asserted here, each of them a refusal and never a repair:

* an explicit fixture selection together with a real-worker binding is a CONTRADICTION and is refused BY NAME, with
  both variables named. A declared mode does not rescue it: a mode that overrides an explicit selection is the same
  silent override wearing a label;
* two real-weights selections (a local `laya` and a bound worker) are ambiguous rather than contradictory, and
  require ONE unambiguous explicit mode -- that is the only thing `M5PHET_CLASSIFICATION_MODE` may settle;
* the effective backend is VALIDATED against what the declared mode requires, at start-up and again on the answer:
  a worker that describes itself as a fixture, a provider with no weights, or a checkpoint other than the pinned
  one is refused, never served;
* the receipt carries the backend and the checkpoint that ANSWERED, read from the capabilities of the path that was
  taken, so what answered is recorded rather than inferred from the configuration;
* nothing here ever carries the worker's host: the resolution publishes a boolean, never a name.
"""
import pytest

from m5phet.classification_backend import (
    BACKEND_MISMATCH, BACKEND_UNKNOWN, BACKEND_VARIABLE, CHECKPOINT_MISMATCH, CHECKPOINT_PIN_VARIABLE,
    COMMAND_VARIABLE, CONTRADICTION, FIXTURE, LOCAL_WEIGHTS, MODES, MODE_CONTRADICTS, MODE_REQUIRED, MODE_UNKNOWN,
    MODE_VARIABLE, PATH_MISMATCH, REMOTE_WORKER, SCHEMA, WEIGHTS_ABSENT, WORKER_COMMAND_NOT_BOUND,
    WORKER_NOT_BOUND, WORKER_VARIABLE, BackendRefusal, resolve, validate,
)

#: never a real host; the operator's worker is only ever the value of $M5PHET_CHAT_LAYA_WORKER in his own environment
WORKER_PLACEHOLDER = "worker.invalid"
WORKER_COMMAND = "m5phet-worker"
CHECKPOINT = "laya-checkpoint:" + "b" * 64

LAYA_CAPS = {"provider": "laya_news", "backend": "laya", "weights_present": True, "device": "cuda:0",
             "known_states": [CHECKPOINT]}
FIXTURE_CAPS = {"provider": "laya_news", "backend": "fixture", "weights_present": False, "device": "cpu",
                "known_states": ["laya-checkpoint:" + "c" * 64]}


def env(**values):
    return {key: value for key, value in values.items() if value is not None}


# --- the unambiguous configurations, which must keep working --------------------------------------------------------

def test_an_explicit_fixture_alone_is_the_fixture_mode():
    resolution = resolve(env(NEWS_SIGNAL_BACKEND="fixture"))
    assert resolution["schema"] == SCHEMA
    assert resolution["mode"] == FIXTURE
    assert resolution["expects_backend"] == "fixture"
    assert resolution["expects_weights"] is False
    assert resolution["worker_bound"] is False


def test_an_explicit_laya_alone_is_the_local_weights_mode():
    resolution = resolve(env(NEWS_SIGNAL_BACKEND="laya"))
    assert resolution["mode"] == LOCAL_WEIGHTS
    assert resolution["expects_backend"] == "laya"
    assert resolution["expects_weights"] is True


def test_a_worker_alone_is_the_remote_worker_mode():
    resolution = resolve(env(M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER, M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND))
    assert resolution["mode"] == REMOTE_WORKER
    assert resolution["worker_bound"] is True
    assert resolution["expects_backend"] == "laya"


def test_nothing_declared_is_the_providers_own_default_and_says_so():
    resolution = resolve({})
    assert resolution["mode"] == LOCAL_WEIGHTS
    assert resolution["mode_source"] == "PROVIDER_DEFAULT"
    assert resolution["mode_declared"] is False


# --- the contradiction this whole file exists for ---------------------------------------------------------------------

def test_an_explicit_fixture_with_a_bound_worker_refuses_by_name():
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(NEWS_SIGNAL_BACKEND="fixture", M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER,
                    M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND))
    assert caught.value.code == CONTRADICTION
    # the refusal must name BOTH selections; an operator who cannot see which two settings disagree cannot repair it
    assert BACKEND_VARIABLE in str(caught.value) and WORKER_VARIABLE in str(caught.value)
    # and it must never print the host it was given
    assert WORKER_PLACEHOLDER not in str(caught.value)


@pytest.mark.parametrize("mode", MODES)
def test_no_declared_mode_rescues_that_contradiction(mode):
    """A mode that silently settles an explicit disagreement is the same override with a label on it."""
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(NEWS_SIGNAL_BACKEND="fixture", M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER,
                    M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND, M5PHET_CLASSIFICATION_MODE=mode))
    assert caught.value.code == CONTRADICTION


def test_the_contradiction_refusal_states_both_repairs():
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(NEWS_SIGNAL_BACKEND="fixture", M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER))
    detail = caught.value.detail
    assert "unset" in detail.lower() and MODE_VARIABLE in detail


# --- ambiguity, which IS what one explicit mode may settle -------------------------------------------------------------

def test_two_real_weights_selections_require_one_explicit_mode():
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(NEWS_SIGNAL_BACKEND="laya", M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER,
                    M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND))
    assert caught.value.code == MODE_REQUIRED
    assert MODE_VARIABLE in str(caught.value)


def test_the_explicit_mode_settles_that_ambiguity_in_either_direction():
    remote = resolve(env(NEWS_SIGNAL_BACKEND="laya", M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER,
                         M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND, M5PHET_CLASSIFICATION_MODE=REMOTE_WORKER))
    assert remote["mode"] == REMOTE_WORKER and remote["mode_source"] == "DECLARED"
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(NEWS_SIGNAL_BACKEND="laya", M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER,
                    M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND, M5PHET_CLASSIFICATION_MODE=LOCAL_WEIGHTS))
    # local weights WITH a worker still bound is not an unambiguous local installation: unbind the worker
    assert caught.value.code == MODE_CONTRADICTS


def test_a_declared_mode_must_match_the_selection_it_names():
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(NEWS_SIGNAL_BACKEND="laya", M5PHET_CLASSIFICATION_MODE=FIXTURE))
    assert caught.value.code == MODE_CONTRADICTS


def test_an_unknown_mode_is_refused_naming_what_exists():
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(M5PHET_CLASSIFICATION_MODE="real"))
    assert caught.value.code == MODE_UNKNOWN
    assert all(mode in str(caught.value) for mode in MODES)


def test_an_unknown_backend_is_refused_at_startup_and_not_at_the_first_question():
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(NEWS_SIGNAL_BACKEND="Laya"))
    assert caught.value.code == BACKEND_UNKNOWN


# --- the missing and the half-bound worker -----------------------------------------------------------------------------

def test_the_remote_mode_without_a_worker_refuses_by_name():
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(M5PHET_CLASSIFICATION_MODE=REMOTE_WORKER))
    assert caught.value.code == WORKER_NOT_BOUND
    assert WORKER_VARIABLE in str(caught.value)


def test_a_bound_worker_with_no_command_refuses_at_startup():
    """It used to be discovered at the first question, after the catalog had already shown the local provider."""
    with pytest.raises(BackendRefusal) as caught:
        resolve(env(M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER))
    assert caught.value.code == WORKER_COMMAND_NOT_BOUND
    assert COMMAND_VARIABLE in str(caught.value)


def test_the_resolution_never_carries_the_host():
    resolution = resolve(env(M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER, M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND))
    assert WORKER_PLACEHOLDER not in repr(resolution)
    assert resolution["worker_bound"] is True


def test_an_injected_registry_reports_a_worker_binding_as_ignored_rather_than_dropping_it():
    resolution = resolve(env(NEWS_SIGNAL_BACKEND="fixture", M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER),
                         remote_permitted=False)
    assert resolution["mode"] == FIXTURE
    assert resolution["worker_binding_ignored"] is True


# --- validation of what actually answered --------------------------------------------------------------------------

def test_the_effective_block_is_read_from_the_capabilities_of_the_path_taken():
    resolution = resolve(env(M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER, M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND))
    effective = validate(resolution, LAYA_CAPS, answered_by="remote_worker")
    assert effective["mode"] == REMOTE_WORKER
    assert effective["answered_by"] == "remote_worker"
    assert effective["backend"] == "laya"
    assert effective["checkpoint"] == CHECKPOINT
    assert effective["weights_present"] is True
    assert effective["validated"] is True


def test_a_worker_that_describes_itself_as_a_fixture_is_refused_under_a_real_mode():
    resolution = resolve(env(M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER, M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND))
    with pytest.raises(BackendRefusal) as caught:
        validate(resolution, FIXTURE_CAPS, answered_by="remote_worker")
    assert caught.value.code == BACKEND_MISMATCH


def test_real_weights_answering_a_declared_fixture_are_refused_too():
    """Both directions. A declared non-model that answers from a checkpoint is as wrong as the reverse."""
    resolution = resolve(env(NEWS_SIGNAL_BACKEND="fixture"))
    with pytest.raises(BackendRefusal) as caught:
        validate(resolution, LAYA_CAPS, answered_by="in_process")
    assert caught.value.code == BACKEND_MISMATCH


def test_absent_weights_under_a_real_mode_are_refused():
    resolution = resolve(env(NEWS_SIGNAL_BACKEND="laya"))
    with pytest.raises(BackendRefusal) as caught:
        validate(resolution, dict(LAYA_CAPS, weights_present=False), answered_by="in_process")
    assert caught.value.code == WEIGHTS_ABSENT


def test_a_pinned_checkpoint_that_is_not_the_one_serving_is_refused():
    resolution = resolve(env(M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER, M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND,
                             M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT="laya-checkpoint:" + "d" * 64))
    with pytest.raises(BackendRefusal) as caught:
        validate(resolution, LAYA_CAPS, answered_by="remote_worker")
    assert caught.value.code == CHECKPOINT_MISMATCH
    assert CHECKPOINT_PIN_VARIABLE in str(caught.value)


def test_a_pinned_checkpoint_that_matches_validates():
    resolution = resolve(env(M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER, M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND,
                             M5PHET_CLASSIFICATION_EXPECT_CHECKPOINT=CHECKPOINT))
    assert validate(resolution, LAYA_CAPS, answered_by="remote_worker")["checkpoint"] == CHECKPOINT


def test_the_local_path_answering_under_a_remote_mode_is_refused():
    """The hidden fallback, asserted directly: in-process capabilities may not satisfy a remote declaration."""
    resolution = resolve(env(M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER, M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND))
    with pytest.raises(BackendRefusal) as caught:
        validate(resolution, LAYA_CAPS, answered_by="in_process")
    assert caught.value.code == PATH_MISMATCH


def test_capabilities_that_declare_no_backend_are_refused_rather_than_assumed():
    resolution = resolve(env(NEWS_SIGNAL_BACKEND="laya"))
    with pytest.raises(BackendRefusal) as caught:
        validate(resolution, {"provider": "laya_news", "known_states": []}, answered_by="in_process")
    assert caught.value.code == BACKEND_MISMATCH


# --- the two contracts this module has with code outside it ------------------------------------------------------------

def test_the_backends_named_here_are_the_ones_the_installed_provider_accepts():
    """A start-up refusal for a misspelled backend is only useful if it refuses the same set the provider does."""
    provider = pytest.importorskip("news_signal.provider")
    from m5phet.classification_backend import BACKENDS
    for backend in BACKENDS:
        assert provider.configuration({"NEWS_SIGNAL_BACKEND": backend})["backend"] == backend
    with pytest.raises(Exception):
        provider.configuration({"NEWS_SIGNAL_BACKEND": "not-a-backend"})


def test_the_json_configuration_can_declare_the_mode_and_the_pin():
    """The operator's file binds them, so "which backend answers" lives where a person reads the configuration."""
    from m5phet import config as config_module
    mapping = config_module.CORE_ENVIRONMENT["classification"]
    assert mapping["mode"] == MODE_VARIABLE
    assert mapping["expect_checkpoint"] == CHECKPOINT_PIN_VARIABLE


def test_a_supplied_provider_is_reported_as_supplied_and_never_as_validated():
    from m5phet.classification_backend import declared
    block = declared(FIXTURE_CAPS)
    assert block["mode"] == "PROVIDER_SUPPLIED" and block["status"] == "PROVIDER_SUPPLIED"
    assert block["backend"] == "fixture"
    assert "unchecked" in block["reading"]


def test_an_unavailable_backend_block_invents_no_backend():
    from m5phet.classification_backend import WORKER_UNREACHABLE, unavailable
    resolution = resolve(env(M5PHET_CHAT_LAYA_WORKER=WORKER_PLACEHOLDER, M5PHET_CHAT_LAYA_COMMAND=WORKER_COMMAND))
    block = unavailable(resolution, WORKER_UNREACHABLE, "the worker was occupied")
    assert block["status"] == WORKER_UNREACHABLE and block["validated"] is False
    assert block["backend"] is None and block["checkpoint"] is None and block["answered_by"] is None
    assert block["mode"] == REMOTE_WORKER                      # the declaration survives the refusal


def test_the_operator_script_no_longer_declares_a_fixture_beside_a_bound_worker():
    """The shipped configuration itself was the contradiction; a start-up refusal is not a licence to keep shipping it."""
    from pathlib import Path
    script = (Path(__file__).resolve().parent.parent / "tools/start_chat.sh").read_text(encoding="utf-8")
    worker_branch = script.index('if [ -n "${M5PHET_CHAT_LAYA_WORKER')
    default = script.index("NEWS_SIGNAL_BACKEND:-fixture")
    assert default > worker_branch, "the fixture default must live in the branch where no worker is bound"
    assert f"{MODE_VARIABLE}:-{REMOTE_WORKER}" in script, "the worker branch must declare one unambiguous mode"
    assert CHECKPOINT_PIN_VARIABLE in script
