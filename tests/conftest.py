"""Shared guards for the whole suite.

The one guard that must exist: a run of these tests never reaches the owner's data-gov. WP15 reads a governed
resource through data-gov when `DATA_GOV_*` is bound, and this process inherits the operator environment whenever the
suite is run from a sourced `chat.env` or by the service. Tests that exercise the governed path bind those variables
themselves, at their own fake server; every other test must see an unconfigured installation, which is also the state
a fresh checkout is in.
"""

import pytest

from m5phet import datasets


@pytest.fixture(autouse=True)
def _no_inherited_governed_identity(monkeypatch):
    for variable in (*datasets.GOVERNED_VARIABLES, datasets.CHECKOUT_VARIABLE, datasets.CACHE_VARIABLE):
        monkeypatch.delenv(variable, raising=False)
