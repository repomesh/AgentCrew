"""Safe Git linked-worktree lifecycle operations."""

import os
import subprocess
from pathlib import Path


class WorktreeService:
    """List registered worktrees, create branches, and remove clean linked worktrees."""

    _TIMEOUT = 30

    @classmethod
    def _git(cls, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=cls._TIMEOUT,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                "Git is not installed or is not available on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                "Git command timed out; check the repository and try again."
            ) from exc
        except OSError as exc:
            raise ValueError("Unable to run Git in the requested repository.") from exc

    @staticmethod
    def _path(value: str, name: str) -> Path:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{name} must be a non-empty path.")
        return Path(value).expanduser().resolve()

    def list_worktree(self, repository_path: str = ".") -> str:
        """List registered worktrees without changing their state."""
        repo = self._path(repository_path, "repository_path")
        if (
            not repo.is_dir()
            or self._git(repo, "rev-parse", "--is-inside-work-tree").stdout.strip()
            != "true"
        ):
            raise ValueError("repository_path must point inside a Git working tree.")
        listing = self._git(repo, "worktree", "list", "--porcelain", "-z")
        if listing.returncode:
            raise ValueError(
                "Could not list worktrees; check the repository and Git permissions."
            )
        lines = []
        for entry in listing.stdout.split("\x00\x00"):
            if not entry:
                continue
            fields = entry.split("\x00")
            if not fields[0].startswith("worktree "):
                raise ValueError(
                    "Could not parse Git worktree list; check the repository."
                )
            path = fields[0][len("worktree ") :]
            head = next(
                (field[5:] for field in fields if field.startswith("HEAD ")), None
            )
            if not path or not head:
                raise ValueError(
                    "Could not parse Git worktree list; check the repository."
                )
            branch = next(
                (
                    field[len("branch refs/heads/") :]
                    for field in fields
                    if field.startswith("branch refs/heads/")
                ),
                "detached",
            )
            details = [path, branch, f"HEAD {head}"]
            for field in fields:
                for marker in ("locked", "prunable"):
                    if field == marker or field.startswith(f"{marker} "):
                        details.append(
                            field.replace(" ", ": ", 1) if field != marker else marker
                        )
            lines.append(" | ".join(details))
        return "\n".join(lines)

    def create_worktree(
        self,
        worktree_path: str,
        branch_name: str,
        repository_path: str = ".",
        start_point: str = "HEAD",
    ) -> str:
        repo = self._path(repository_path, "repository_path")
        if (
            not isinstance(worktree_path, str)
            or not worktree_path.strip()
            or "\x00" in worktree_path
        ):
            raise ValueError("worktree_path must be a non-empty path.")
        raw_destination = Path(os.path.abspath(Path(worktree_path).expanduser()))
        if (
            not repo.is_dir()
            or self._git(repo, "rev-parse", "--is-inside-work-tree").stdout.strip()
            != "true"
        ):
            raise ValueError("repository_path must point inside a Git working tree.")
        if os.path.lexists(raw_destination):
            raise ValueError("worktree_path already exists; choose a new destination.")
        destination = raw_destination.resolve()
        if os.path.lexists(destination):
            raise ValueError("worktree_path already exists; choose a new destination.")
        if (
            not isinstance(branch_name, str)
            or not branch_name
            or branch_name.startswith("-")
        ):
            raise ValueError("branch_name must be a valid new local branch name.")
        branch_check = self._git(repo, "check-ref-format", "--branch", branch_name)
        if branch_check.returncode or branch_check.stdout.strip() != branch_name:
            raise ValueError("branch_name must be a valid new local branch name.")
        if (
            not isinstance(start_point, str)
            or not start_point.strip()
            or start_point.startswith("-")
        ):
            raise ValueError("start_point must identify an existing commit.")
        if self._git(
            repo,
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"{start_point}^{{commit}}",
        ).returncode:
            raise ValueError("start_point must identify an existing commit.")
        if (
            self._git(
                repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"
            ).returncode
            == 0
        ):
            raise ValueError("branch_name already exists; choose a new branch.")
        result = self._git(
            repo,
            "worktree",
            "add",
            "-b",
            branch_name,
            "--",
            str(destination),
            start_point,
        )
        if result.returncode:
            raise ValueError(
                "Could not create worktree; check that the branch and destination are unused "
                "and that the repository is writable."
            )
        return f"Created worktree at {destination} on new branch {branch_name}."

    def close_worktree(self, worktree_path: str) -> str:
        target = self._path(worktree_path, "worktree_path")
        if not target.is_dir():
            raise ValueError(
                "worktree_path must be an existing registered linked worktree."
            )
        listing = self._git(target, "worktree", "list", "--porcelain", "-z")
        if listing.returncode:
            raise ValueError(
                "worktree_path must be an existing registered linked worktree."
            )
        entries = [entry for entry in listing.stdout.split("\x00\x00") if entry]
        paths = [
            Path(entry.split("\x00", 1)[0][len("worktree ") :]).resolve()
            for entry in entries
            if entry.startswith("worktree ")
        ]
        if target not in paths:
            raise ValueError(
                "worktree_path must be an existing registered linked worktree."
            )
        git_dir = self._git(
            target, "rev-parse", "--path-format=absolute", "--absolute-git-dir"
        )
        common_dir = self._git(
            target, "rev-parse", "--path-format=absolute", "--git-common-dir"
        )
        if (
            git_dir.returncode
            or common_dir.returncode
            or not git_dir.stdout.strip()
            or not common_dir.stdout.strip()
        ):
            raise ValueError(
                "Could not verify worktree Git metadata; no files were removed."
            )
        git_directory = Path(git_dir.stdout.strip()).resolve()
        common_directory = Path(common_dir.stdout.strip()).resolve()
        if git_directory == common_directory:
            raise ValueError(
                "Cannot close the main worktree; only linked worktrees can be closed."
            )
        status = self._git(
            target,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        if status.returncode:
            raise ValueError("Could not check worktree status; no files were removed.")
        if status.stdout:
            raise ValueError(
                "Worktree has staged, unstaged, untracked, conflicted, or submodule changes; "
                "commit or discard them before closing."
            )
        result = self._git(common_directory, "worktree", "remove", "--", str(target))
        if result.returncode:
            raise ValueError(
                "Could not remove worktree; check for new changes or a worktree lock."
            )
        return f"Closed worktree at {target}; its branch was preserved."
