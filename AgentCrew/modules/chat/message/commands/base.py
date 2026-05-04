from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommandResult:
    """Result of command processing."""

    handled: bool = False
    exit_flag: bool = False
    clear_flag: bool = False
    message: str = ""
