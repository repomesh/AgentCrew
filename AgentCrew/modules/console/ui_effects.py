"""
UI effects and animations for console interface.
Handles loading animations, live displays, and other visual effects.
"""

from __future__ import annotations

import time
import threading
import itertools
import random
from rich.live import Live
from rich.box import HORIZONTALS
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .constants import CODE_THEME, RICH_STYLE_GRAY, RICH_STYLE_GREEN

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .console_ui import ConsoleUI


class UIEffects:
    """Handles UI effects like loading animations and live displays."""

    def __init__(self, console_ui: ConsoleUI):
        """Initialize UI effects with a console instance."""
        self.console = console_ui.console
        self.live = None
        self._loading_stop_event = None
        self._loading_thread = None
        self._visible_buffer = -1
        self._tracking_buffer = 0
        self.message_handler = console_ui.message_handler
        self.spinner = itertools.cycle(["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
        self.updated_text = ""
        # Delegate animation state
        # key: tool_use_id, value: (agent_name, start_time)
        self._delegate_agents: dict[str, tuple[str, float]] = {}
        self._delegate_lock = threading.Lock()
        self._delegate_stop_event: threading.Event | None = None
        self._delegate_thread: threading.Thread | None = None

    def _loading_animation(self, stop_event):
        """Display a loading animation in the terminal."""
        fun_words = [
            "Pondering",
            "Cogitating",
            "Ruminating",
            "Contemplating",
            "Brainstorming",
            "Calculating",
            "Processing",
            "Analyzing",
            "Deciphering",
            "Meditating",
            "Daydreaming",
            "Scheming",
            "Brewing",
            "Conjuring",
            "Inventing",
            "Imagining",
        ]
        fun_word = random.choice(fun_words)

        with Live(
            "", console=self.console, auto_refresh=True, refresh_per_second=10
        ) as live:
            while not stop_event.is_set():
                live.update(f" {fun_word} {next(self.spinner)}")
                time.sleep(0.1)  # Control animation speed
            live.update("")  # Clear the live display when done
            live.stop()  # Stop the live display
            import sys

            sys.stdout.write("\x1b[1A")  # cursor up one line
            sys.stdout.write("\x1b[2K")

    def _delegate_animation(self, stop_event):
        """Animate active delegate agents with per-agent spinners and elapsed time."""
        spinner = itertools.cycle(["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
        with Live(
            "", console=self.console, auto_refresh=True, refresh_per_second=10
        ) as live:
            while not stop_event.is_set():
                spin = next(spinner)
                with self._delegate_lock:
                    agents = dict(self._delegate_agents)
                if agents:
                    now = time.time()
                    parts = []
                    for agent_name, start in agents.values():
                        elapsed = int(now - start)
                        parts.append(
                            f"📋 [bold yellow]{agent_name}[/] [dim]{spin} {elapsed}s[/]"
                        )
                    live.update("   " + "   [dim]|[/]   ".join(parts))
                else:
                    live.update("")
                time.sleep(0.1)
            live.update("")
            live.stop()
            import sys

            sys.stdout.write("\x1b[1A")
            sys.stdout.write("\x1b[2K")

    def start_delegate_animation(self, tool_use_id: str, agent_name: str):
        """Register a delegation by tool_use_id and start/keep the animation running."""
        with self._delegate_lock:
            self._delegate_agents[tool_use_id] = (agent_name, time.time())
        if self._delegate_thread and self._delegate_thread.is_alive():
            return  # Already running
        self._delegate_stop_event = threading.Event()
        self._delegate_thread = threading.Thread(
            target=self._delegate_animation, args=(self._delegate_stop_event,)
        )
        self._delegate_thread.daemon = True
        self._delegate_thread.start()

    def stop_delegate_animation(self, tool_use_id: str):
        """Unregister a delegation by tool_use_id. Stop animation only when all are done."""
        with self._delegate_lock:
            self._delegate_agents.pop(tool_use_id, None)
            any_remaining = bool(self._delegate_agents)
        if not any_remaining:
            if self._delegate_stop_event:
                self._delegate_stop_event.set()
                self._delegate_stop_event = None
            if self._delegate_thread and self._delegate_thread.is_alive():
                self._delegate_thread.join(timeout=0.5)
                self._delegate_thread = None

    def start_loading_animation(self):
        """Start the loading animation."""
        if self._loading_thread and self._loading_thread.is_alive():
            return  # Already running

        self._loading_stop_event = threading.Event()
        self._loading_thread = threading.Thread(
            target=self._loading_animation, args=(self._loading_stop_event,)
        )
        self._loading_thread.daemon = True
        self._loading_thread.start()

    def stop_loading_animation(self):
        """Stop the loading animation."""
        if self._loading_stop_event:
            self._loading_stop_event.set()
            self._loading_stop_event = None
        if self._loading_thread and self._loading_thread.is_alive():
            self._loading_thread.join(timeout=0.5)
            self._loading_thread = None

    def start_streaming_response(self, agent_name: str, is_thinking=False):
        """Start streaming the assistant's response."""
        from .constants import RICH_STYLE_GREEN_BOLD

        header = Text(
            f"💭 {agent_name.upper()}'s thinking:"
            if is_thinking
            else f"🤖 {agent_name.upper()}:",
            style=RICH_STYLE_GRAY if is_thinking else RICH_STYLE_GREEN_BOLD,
        )

        live_panel = Panel(
            "",
            title=header,
            box=HORIZONTALS,
            border_style=RICH_STYLE_GRAY if is_thinking else RICH_STYLE_GREEN,
        )

        self.live = Live(
            live_panel,
            auto_refresh=False,
            console=self.console,
            vertical_overflow="crop",
        )
        self.live.start()

    def scroll_live_display(self, direction: str):
        speed = 10
        if self._visible_buffer == -1:
            self._visible_buffer = self._tracking_buffer
        if direction == "up":
            self._visible_buffer = max(0, self._visible_buffer - speed)
        elif direction == "down":
            self._visible_buffer += speed
        if self.live:
            self.update_live_display(self.updated_text)

    def update_live_display(self, chunk: str, is_thinking: bool = False):
        """Update the live display with a new chunk of the response."""
        if not self.live:
            self.start_streaming_response(self.message_handler.agent.name, is_thinking)

        if chunk != self.updated_text:
            self.updated_text = self.updated_text + chunk if is_thinking else chunk

        # Only show the last part that fits in the console
        lines = self.updated_text.split("\n")
        height_limit = int(self.console.size.height * 0.9)
        if len(lines) > height_limit:
            if (
                self._visible_buffer == -1
                or self._visible_buffer > len(lines) - height_limit
            ):
                self._tracking_buffer = len(lines) - height_limit
                self._visible_buffer = -1
                lines = lines[-height_limit:]
            else:
                lines = lines[
                    self._visible_buffer : self._visible_buffer + height_limit
                ]

        if self.live:
            from .constants import RICH_STYLE_GREEN_BOLD
            from rich.text import Text

            agent_name = self.message_handler.agent.name

            header = Text(
                f"💭 {agent_name.upper()}'s thinking:"
                if is_thinking
                else f"🤖 {agent_name.upper()}:",
                style=RICH_STYLE_GRAY if is_thinking else RICH_STYLE_GREEN_BOLD,
            )
            subtitle = Text(
                "(Use Ctrl+U/Ctrl+D to scroll)",
                style=RICH_STYLE_GRAY,
            )
            live_panel = Panel(
                Markdown("\n".join(lines), code_theme=CODE_THEME),
                title=header,
                box=HORIZONTALS,
                subtitle=subtitle if not is_thinking else None,
                title_align="left",
                expand=False,
                height=min(height_limit, len(lines)),
                border_style=RICH_STYLE_GRAY if is_thinking else RICH_STYLE_GREEN,
            )
            self.live.update(live_panel, refresh=True)

    def finish_live_update(self):
        """Stop the live update display."""
        self._visible_buffer = -1
        self._tracking_buffer = 0
        self.updated_text = ""
        if self.live:
            self.console.print(self.live.get_renderable())
            self.live.update("")
            self.live.stop()
            self.live = None

    def finish_response(self, response: str, is_thinking: bool = False):
        """Finalize and display the complete response."""
        from .constants import RICH_STYLE_GREEN_BOLD

        self._visible_buffer = -1
        self._tracking_buffer = 0
        self.updated_text = ""

        if self.live:
            self.live.update(Text("", end=""))
            self.live.stop()
            self.live = None

        # Replace \n with two spaces followed by \n for proper Markdown line breaks
        markdown_formatted_response = response.replace("\n", "  \n")

        if not markdown_formatted_response.strip():
            return

        agent_name = self.message_handler.agent.name

        header = Text(
            f"💭 {agent_name.upper()}'s thinking:"
            if is_thinking
            else f"🤖 {agent_name.upper()}:",
            style=RICH_STYLE_GRAY if is_thinking else RICH_STYLE_GREEN_BOLD,
        )
        assistant_panel = Panel(
            Markdown(markdown_formatted_response, code_theme=CODE_THEME),
            title=header,
            box=HORIZONTALS,
            title_align="left",
            border_style=RICH_STYLE_GRAY if is_thinking else RICH_STYLE_GREEN,
        )
        self.console.print(assistant_panel)

    def cleanup(self):
        """Clean up all running effects."""
        self.stop_loading_animation()
        self.finish_live_update()
        with self._delegate_lock:
            self._delegate_agents.clear()
        if self._delegate_stop_event:
            self._delegate_stop_event.set()
            self._delegate_stop_event = None
        if self._delegate_thread and self._delegate_thread.is_alive():
            self._delegate_thread.join(timeout=0.5)
            self._delegate_thread = None
