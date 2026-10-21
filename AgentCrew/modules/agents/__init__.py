from .agent_runner import run_agent_loop
from .local_agent import LocalAgent
from .manager import AgentManager, AgentMode

__all__ = [
    "AgentManager",
    "AgentMode",
    "CursorAgent",
    "LocalAgent",
    "run_agent_loop",
]


# Reduce the initial import cost by lazy load class
def __getattr__(name):
    if name == "RemoteAgent":
        from .remote_agent import RemoteAgent

        return RemoteAgent
    if name == "CursorAgent":
        from .cursor_agent import CursorAgent

        return CursorAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
