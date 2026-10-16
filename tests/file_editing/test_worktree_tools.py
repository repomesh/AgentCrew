import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from AgentCrew.modules.agents.tool_registrar import AgentToolRegistrar
from AgentCrew.modules.file_editing.tool import (
    get_close_worktree_tool_definition,
    get_create_worktree_tool_definition,
    get_list_worktree_tool_definition,
)
from AgentCrew.modules.file_editing.worktree_service import WorktreeService
from AgentCrew.modules.tools.parallel_executor import is_sequential_tool


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "readme.txt").write_text("first\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "first")
    return root


def test_registration_schemas_and_prompt():
    agent = SimpleNamespace(
        services={"file_editing": object()},
        tools=["file_editing"],
        tool_prompts=[],
        tool_definitions={},
        voice_enabled="disabled",
        llm=None,
    )

    def register_tool(definition, handler_factory, service):
        agent.tool_definitions[definition()["function"]["name"]] = (
            definition,
            handler_factory,
            service,
        )

    agent.register_tool = register_tool
    AgentToolRegistrar(agent).register_tools()
    assert set(agent.tool_definitions) == {
        "write_file",
        "create_worktree",
        "close_worktree",
        "list_worktree",
    }
    create = get_create_worktree_tool_definition()["function"]["parameters"]
    close = get_close_worktree_tool_definition()["function"]["parameters"]
    listing = get_list_worktree_tool_definition()["function"]["parameters"]
    assert set(create["required"]) == {"worktree_path", "branch_name"}
    assert create["properties"]["repository_path"]["default"] == "."
    assert create["properties"]["start_point"]["default"] == "HEAD"
    assert close["required"] == ["worktree_path"]
    assert listing["required"] == []
    assert set(listing["properties"]) == {"repository_path"}
    assert listing["properties"]["repository_path"]["default"] == "."
    assert all(
        is_sequential_tool(tool)
        for tool in ("create_worktree", "close_worktree", "list_worktree")
    )
    assert len(agent.tool_prompts) == 1
    prompt = agent.tool_prompts[0]
    assert "Prefer Git worktrees" in prompt
    assert "current checkout is also allowed" in prompt
    assert "reads, writes, and commands within the returned worktree path" in prompt
    assert "list_worktree" in prompt
    assert "committing or discarding changes" in prompt


def test_registered_handlers_create_and_close(repo: Path):
    agent = SimpleNamespace(tool_definitions={})
    agent.register_tool = lambda definition, handler, service: (
        agent.tool_definitions.update(
            {definition()["function"]["name"]: (handler, service)}
        )
    )
    from AgentCrew.modules.file_editing.tool import register

    register(agent=agent)
    path = repo.parent / "linked"
    factory, service = agent.tool_definitions["create_worktree"]
    assert "Created" in asyncio.run(
        factory(service)(
            repository_path=str(repo), worktree_path=str(path), branch_name="new"
        )
    )
    list_factory, list_service = agent.tool_definitions["list_worktree"]
    assert list_service is service
    assert str(path) in asyncio.run(
        list_factory(list_service)(repository_path=str(repo))
    )
    factory, service = agent.tool_definitions["close_worktree"]
    assert list_service is service
    assert "preserved" in asyncio.run(factory(service)(worktree_path=str(path)))
    assert git(repo, "branch", "--list", "new") == "new"


def test_list_default_repository_path(repo: Path, monkeypatch):
    monkeypatch.chdir(repo)
    assert str(repo) in WorktreeService().list_worktree()


def test_list_main_linked_and_spaces(repo: Path):
    service = WorktreeService()
    linked = repo.parent / "linked with spaces"
    service.create_worktree(str(linked), "feature/new", str(repo))
    head = git(repo, "rev-parse", "HEAD")
    main_branch = git(repo, "branch", "--show-current")
    subdir = repo / "nested"
    subdir.mkdir()
    lines = service.list_worktree(str(subdir)).splitlines()
    assert lines == [
        f"{repo} | {main_branch} | HEAD {head}",
        f"{linked} | feature/new | HEAD {head}",
    ]
    assert service.list_worktree(str(linked)) == "\n".join(lines)


def test_list_detached_and_locked(repo: Path):
    service = WorktreeService()
    linked = repo.parent / "detached"
    git(repo, "worktree", "add", "--detach", str(linked), "HEAD")
    git(repo, "worktree", "lock", "--reason", "hold for review", str(linked))
    head = git(repo, "rev-parse", "HEAD")
    lines = service.list_worktree(str(repo)).splitlines()
    assert lines[1] == f"{linked} | detached | HEAD {head} | locked: hold for review"
    assert (
        git(repo, "worktree", "list", "--porcelain").count("locked hold for review")
        == 1
    )


def test_list_prunable_metadata(repo: Path, monkeypatch):
    service = WorktreeService()
    original_git = service._git

    def listing_with_prunable(cwd: Path, *args: str):
        result = original_git(cwd, *args)
        if args == ("worktree", "list", "--porcelain", "-z"):
            return subprocess.CompletedProcess(
                result.args,
                result.returncode,
                result.stdout.replace(
                    "\x00\x00", "\x00prunable missing directory\x00\x00", 1
                ),
                result.stderr,
            )
        return result

    monkeypatch.setattr(service, "_git", listing_with_prunable)
    assert "prunable: missing directory" in service.list_worktree(str(repo))


def test_list_invalid_repository_and_git_failure(repo: Path, monkeypatch):
    service = WorktreeService()
    bare = repo.parent / "bare.git"
    git(repo.parent, "init", "-q", "--bare", str(bare))
    for path in (str(repo.parent), str(repo.parent / "missing"), str(bare), ""):
        with pytest.raises(ValueError, match="repository_path"):
            service.list_worktree(path)

    original_git = service._git

    def failed_listing(cwd: Path, *args: str):
        result = original_git(cwd, *args)
        if args == ("worktree", "list", "--porcelain", "-z"):
            return subprocess.CompletedProcess(result.args, 1, "", "failed")
        return result

    monkeypatch.setattr(service, "_git", failed_listing)
    with pytest.raises(ValueError, match="Could not list worktrees"):
        service.list_worktree(str(repo))


def test_create_from_start_point_and_rejections(repo: Path):
    service = WorktreeService()
    first = git(repo, "rev-parse", "HEAD")
    (repo / "readme.txt").write_text("second\n")
    git(repo, "commit", "-qam", "second")
    linked = repo.parent / "linked"
    service.create_worktree(str(linked), "new", str(repo), first)
    assert git(linked, "rev-parse", "HEAD") == first
    assert git(linked, "branch", "--show-current") == "new"
    with pytest.raises(ValueError, match="already exists"):
        service.create_worktree(str(repo.parent / "other"), "new", str(repo))
    with pytest.raises(ValueError, match="already exists"):
        service.create_worktree(str(linked), "another", str(repo))
    with pytest.raises(ValueError, match="repository_path"):
        service.create_worktree(str(repo.parent / "other"), "another", str(repo.parent))
    with pytest.raises(ValueError, match="start_point"):
        service.create_worktree(
            str(repo.parent / "other"), "another", str(repo), "missing-ref"
        )
    with pytest.raises(ValueError, match="branch_name"):
        service.create_worktree(str(repo.parent / "other"), "-bad", str(repo))
    dangling = repo.parent / "dangling"
    dangling.symlink_to(repo.parent / "missing")
    with pytest.raises(ValueError, match="already exists"):
        service.create_worktree(str(dangling), "another", str(repo))
    service.close_worktree(str(linked))


@pytest.mark.parametrize("kind", ["unstaged", "staged", "untracked"])
def test_dirty_worktree_refused(repo: Path, kind: str):
    service = WorktreeService()
    linked = repo.parent / "linked"
    service.create_worktree(str(linked), "new", str(repo))
    if kind == "untracked":
        (linked / "new.txt").write_text("new\n")
    else:
        (linked / "readme.txt").write_text("edited\n")
        if kind == "staged":
            git(linked, "add", "readme.txt")
    with pytest.raises(ValueError, match="changes"):
        service.close_worktree(str(linked))
    assert linked.exists()
    if kind == "untracked":
        (linked / "new.txt").unlink()
    else:
        git(linked, "restore", "--staged", "--worktree", "readme.txt")
    service.close_worktree(str(linked))
    assert not linked.exists()
    assert git(repo, "branch", "--list", "new") == "new"


def test_close_uses_git_metadata_instead_of_worktree_list_order(
    repo: Path, monkeypatch
):
    service = WorktreeService()
    linked = repo.parent / "linked"
    service.create_worktree(str(linked), "new", str(repo))
    original_git = service._git

    def reversed_listing(cwd: Path, *args: str):
        result = original_git(cwd, *args)
        if args == ("worktree", "list", "--porcelain", "-z"):
            entries = [entry for entry in result.stdout.split("\x00\x00") if entry]
            return subprocess.CompletedProcess(
                result.args,
                result.returncode,
                "\x00\x00".join(reversed(entries)) + "\x00\x00",
                result.stderr,
            )
        return result

    monkeypatch.setattr(service, "_git", reversed_listing)
    with pytest.raises(ValueError, match="main worktree"):
        service.close_worktree(str(repo))
    assert repo.exists()
    assert "preserved" in service.close_worktree(str(linked))
    assert not linked.exists()
    assert git(repo, "branch", "--list", "new") == "new"


def test_main_and_non_worktree_refused(repo: Path):
    service = WorktreeService()
    with pytest.raises(ValueError, match="main worktree"):
        service.close_worktree(str(repo))
    with pytest.raises(ValueError, match="registered linked worktree"):
        service.close_worktree(str(repo.parent))
