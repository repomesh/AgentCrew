import asyncio
import atexit
import os
import shlex
import sys
import time
from pathlib import Path

import pytest

from AgentCrew.modules.command_execution.service import CommandExecutionService
from AgentCrew.modules.command_execution.tool import get_run_command_tool_handler


@pytest.fixture
def service():
    instance = CommandExecutionService()
    atexit.unregister(instance.shutdown)
    yield instance
    instance.shutdown()


def sleep_command(seconds):
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(f'import time; time.sleep({seconds})')}"


@pytest.mark.asyncio
async def test_run_command_keeps_event_loop_responsive(service):
    handler = get_run_command_tool_handler(service)
    task = asyncio.create_task(handler(command=sleep_command(3)))
    try:
        started = time.monotonic()
        await asyncio.wait_for(asyncio.sleep(0.05), timeout=0.5)
        assert time.monotonic() - started < 0.5
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert service.list_running_commands()["count"] == 0


@pytest.mark.asyncio
async def test_cancel_terminates_owned_process_and_cleans_registry(service):
    handler = get_run_command_tool_handler(service)
    task = asyncio.create_task(handler(command=sleep_command(10)))
    while not service._instances:
        await asyncio.sleep(0.01)
    command_id, owned = next(iter(service._instances.items()))
    pid = owned.process.pid
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert time.monotonic() - started < 3
    assert owned.process.poll() is not None
    assert service.get_command_status(command_id)["status"] == "error"
    assert service.list_running_commands()["count"] == 0
    assert all(not thread.is_alive() for thread in owned.reader_threads)
    if os.name != "nt":
        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)


@pytest.mark.asyncio
async def test_cancel_kills_unix_child_when_shell_exits(service, tmp_path):
    if os.name == "nt":
        pytest.skip("Unix process groups only")
    marker = tmp_path / "child_pid"
    child = f"echo $BASHPID > {shlex.quote(str(marker))}; exec sleep 10"
    command = f"({child}) & wait"
    task = asyncio.create_task(service.execute_command_async(command, timeout=5))
    while not marker.exists():
        await asyncio.sleep(0.01)
    child_pid = int(marker.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert service.list_running_commands()["count"] == 0
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        pass
    else:
        if sys.platform == "linux":
            try:
                state = await asyncio.to_thread(
                    lambda: Path(f"/proc/{child_pid}/stat").read_text().split()[2]
                )
            except FileNotFoundError:
                state = "Z"
            assert state == "Z"
        else:
            pytest.fail("Child process survived cancellation")


@pytest.mark.asyncio
async def test_cancel_escalates_for_term_ignoring_group(service):
    if os.name == "nt":
        pytest.skip("Unix process groups only")
    task = asyncio.create_task(
        service.execute_command_async("trap '' TERM; exec sleep 10", timeout=5)
    )
    while not service._instances:
        await asyncio.sleep(0.01)
    process = next(iter(service._instances.values())).process
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert process.poll() is not None
    assert service.list_running_commands()["count"] == 0


@pytest.mark.asyncio
async def test_completion_timeout_and_sync_api(service):
    handler = get_run_command_tool_handler(service)
    output = await handler(command="echo command-ready")
    assert output.startswith("ok exit=0 dur=")
    assert "command-ready" in output
    result = await service.execute_command_async(sleep_command(2), timeout=0.05)
    assert result["status"] == "running"
    assert result["timeout_seconds"] == 0.05
    assert service.terminate_command(result["command_id"])["status"] == "success"
    sync_result = service.execute_command("echo sync-ready", timeout=2)
    assert sync_result["status"] == "completed"
    assert "sync-ready" in sync_result["output"]
    assert service.list_running_commands()["count"] == 0
