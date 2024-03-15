"""Truth isolation guard.

Evaluation outcomes live under a `_ground_truth` directory inside a run. The
feature and prediction code paths must never open anything under that
directory. ``isolation_context`` installs a guard that raises on any file
access to a `_ground_truth` path; the replay engine wraps every feature- and
prediction-building step in it. Tests assert both the guard itself and that
deleting `_ground_truth` leaves prediction outputs byte-identical.
"""

from __future__ import annotations

import builtins
import contextlib
import os

_GROUND_TRUTH_DIRNAME = "_ground_truth"


class IsolationViolation(RuntimeError):
    """Raised when a guarded code path touches the ground-truth directory."""


class _GuardedOpen:
    def __init__(self, original_open):
        self._original = original_open

    def __call__(self, file, *args, **kwargs):
        path = os.fspath(file)
        if _GROUND_TRUTH_DIRNAME in path.split(os.sep):
            raise IsolationViolation(
                f"ground-truth access blocked for feature path: {path!r}"
            )
        return self._original(file, *args, **kwargs)


@contextlib.contextmanager
def isolation_context():
    """Block reads/writes of `_ground_truth` paths inside the context."""
    original = builtins.open
    builtins.open = _GuardedOpen(original)
    try:
        yield
    finally:
        builtins.open = original


def assert_no_ground_truth_access():
    """Return True if the guard is currently installed (used by tests)."""
    return isinstance(builtins.open, _GuardedOpen)
