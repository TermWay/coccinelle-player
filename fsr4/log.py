"""One log sink for the whole package: the CLI prints, the UI routes lines into its log pane."""
from __future__ import annotations

from typing import Callable

_sink: Callable[[str], None] = print


def emit(message: str) -> None:
    """Send one line to the current sink."""
    _sink(message)


def set_sink(sink: Callable[[str], None] | None) -> None:
    """Replace the sink (`None` restores printing to stdout)."""
    global _sink
    _sink = sink if sink is not None else print
