"""Workaround for error reporting in tt-lang-sim 1.1.6.

When a kernel has a problem (an unsupported copy pattern, a deadlock), the
simulator formats the message with ttl/diagnostics.py, which it looks for at
site-packages/ttl/ttl/diagnostics.py. The sim-only wheel doesn't ship that
file, so the real message is replaced by a FileNotFoundError. This swaps in a
plain printer so the kernel error is visible. Importing it is a no-op outside
the simulator.
"""

from __future__ import annotations

import sys

_MODULES = (
    "ttl.sim.diagnostics",
    "ttl.sim.program",
    "ttl.sim.greenlet_scheduler",
    "ttl.sim.analysis",
    "ttl.sim.ttlang_sim",
)


def _plain_diagnostic(name, message, source_file, source_line, source_col=1):
    print(f"{source_file}:{source_line}:{source_col}: [{name}] {message}", file=sys.stderr)


def install() -> None:
    for mod_name in _MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "print_diagnostic_error"):
            mod.print_diagnostic_error = _plain_diagnostic


install()
