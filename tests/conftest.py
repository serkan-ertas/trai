"""Pytest configuration for the project test suite.

Registers the ``slow`` marker (used for long-running smoke tests, e.g. the
AMS resume test) and skips slow tests by default. Run with ``-m slow`` to
include them.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"', "
        "or include exclusively with '-m slow').",
    )


def pytest_collection_modifyitems(config, items):
    # If the user explicitly asked for slow tests, leave selection to pytest.
    markexpr = config.getoption("-m") or ""
    if "slow" in markexpr:
        return
    skip_slow = pytest.mark.skip(reason="slow test; pass '-m slow' to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
