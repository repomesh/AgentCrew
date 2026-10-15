from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class StreamSession:
    session_id: int
    status: str = "idle"
    loop: asyncio.AbstractEventLoop | None = None
    task: asyncio.Task | None = None
    cancel_requested: bool = False
    first_chunk_received: bool = False
    first_chunk_timeout: float = 300.0
    finished: asyncio.Event = field(default_factory=asyncio.Event)

    def bind(self, loop: asyncio.AbstractEventLoop, task: asyncio.Task) -> None:
        self.loop = loop
        self.task = task
        self.status = "opening"

    def begin_segment(self) -> None:
        self.first_chunk_received = False
        self.status = "opening"

    def mark_streaming(self) -> None:
        self.first_chunk_received = True
        self.status = "streaming"

    def mark_cancel_requested(self) -> bool:
        if self.finished.is_set() or self.cancel_requested:
            return False
        self.cancel_requested = True
        if self.status not in {"completed", "failed", "canceled", "timed_out"}:
            self.status = "cancel_requested"
        return True

    def finalize(self, status: str) -> None:
        self.status = status
        self.finished.set()
