"""Shared guards for the whole suite.

The one guard that must exist: a run of these tests never reaches the owner's data-gov. WP15 reads a governed
resource through data-gov when `DATA_GOV_*` is bound, and this process inherits the operator environment whenever the
suite is run from a sourced `chat.env` or by the service. Tests that exercise the governed path bind those variables
themselves, at their own fake server; every other test must see an unconfigured installation, which is also the state
a fresh checkout is in.
"""

import pytest

from m5phet import classification_backend, datasets


@pytest.fixture(autouse=True)
def _no_inherited_governed_identity(monkeypatch):
    for variable in (*datasets.GOVERNED_VARIABLES, datasets.CHECKOUT_VARIABLE, datasets.CACHE_VARIABLE):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture(autouse=True)
def _no_inherited_classification_backend(monkeypatch):
    """CB05: and no test may inherit the operator's backend selection either.

    A suite run from a sourced `chat.env` -- which is how it is run on the operator's own machine -- would otherwise
    let `Engine()` resolve HIS classification configuration, so a contradictory environment would fail tests that have
    nothing to do with it, and, worse, a repaired environment would make the contradiction tests pass for the wrong
    reason. Every test that exercises this contract binds these variables itself, in its own child process.
    """
    for variable in (classification_backend.BACKEND_VARIABLE, classification_backend.WORKER_VARIABLE,
                     classification_backend.COMMAND_VARIABLE, classification_backend.MODE_VARIABLE,
                     classification_backend.CHECKPOINT_PIN_VARIABLE):
        monkeypatch.delenv(variable, raising=False)
