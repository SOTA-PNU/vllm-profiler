"""Path validation for immutable profiler evaluations.

The runner never resolves a user-supplied symlink into an accepted path.  New
outputs must have a real, existing parent and must not overlap immutable
inputs.  Existing experiment paths used for resume are checked with the same
rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
from typing import Iterable


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")


class ExperimentPathError(ValueError):
    """An experiment input or output path is unsafe."""


def validate_safe_name(value: str, *, field: str = "name") -> str:
    """Return a safe single path component."""

    if not isinstance(value, str) or _SAFE_NAME_RE.fullmatch(value) is None:
        raise ExperimentPathError(f"{field} must be a safe single path component")
    return value


def validate_absolute_path(
    value: str | os.PathLike[str],
    *,
    field: str = "path",
    allow_missing: bool = True,
) -> Path:
    """Validate an absolute, normalized path with no symlink components.

    ``allow_missing`` permits the first absent component and everything below
    it.  Existing components are always inspected with ``lstat`` so a symlink
    is never silently followed.
    """

    if not isinstance(value, (str, os.PathLike)):
        raise ExperimentPathError(f"{field} must be a filesystem path")
    path = Path(value)
    if not path.is_absolute():
        raise ExperimentPathError(f"{field} must be absolute")
    if ".." in path.parts:
        raise ExperimentPathError(f"{field} must not contain '..'")

    normalized = Path(os.path.normpath(os.fspath(path)))
    if path != normalized:
        raise ExperimentPathError(f"{field} must be normalized")

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                return path
            raise ExperimentPathError(f"{field} does not exist: {path}") from None
        except OSError as error:
            raise ExperimentPathError(
                f"{field} component cannot be inspected: {current}"
            ) from error
        if stat.S_ISLNK(current_stat.st_mode):
            raise ExperimentPathError(
                f"{field} must not contain a symlink component: {current}"
            )
    return path


def validate_existing_real_path(
    value: str | os.PathLike[str],
    *,
    field: str = "path",
    kind: str | None = None,
) -> Path:
    """Validate an existing real file or directory.

    ``kind`` may be ``"file"`` or ``"directory"``.
    """

    path = validate_absolute_path(value, field=field, allow_missing=False)
    try:
        path_stat = path.lstat()
    except OSError as error:
        raise ExperimentPathError(f"{field} cannot be inspected: {path}") from error
    if kind == "file" and not stat.S_ISREG(path_stat.st_mode):
        raise ExperimentPathError(f"{field} must be a real regular file")
    if kind == "directory" and not stat.S_ISDIR(path_stat.st_mode):
        raise ExperimentPathError(f"{field} must be a real directory")
    if kind not in {None, "file", "directory"}:
        raise ValueError("kind must be 'file', 'directory', or None")
    if kind is None and not (
        stat.S_ISREG(path_stat.st_mode) or stat.S_ISDIR(path_stat.st_mode)
    ):
        raise ExperimentPathError(f"{field} must be a real file or directory")
    return path


def paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either absolute path contains the other."""

    return (
        left == right
        or left in right.parents
        or right in left.parents
    )


def validate_new_output_directory(
    value: str | os.PathLike[str],
    *,
    immutable_roots: Iterable[Path] = (),
    field: str = "output",
) -> Path:
    """Validate a not-yet-existing output without creating it."""

    output = validate_absolute_path(value, field=field)
    validate_safe_name(output.name, field=f"{field} directory name")
    parent = validate_existing_real_path(
        output.parent,
        field=f"{field} parent",
        kind="directory",
    )
    output = parent / output.name
    if os.path.lexists(output):
        raise FileExistsError(f"{field} already exists: {output}")

    for index, raw_root in enumerate(immutable_roots):
        root = validate_existing_real_path(
            raw_root,
            field=f"immutable_roots[{index}]",
        )
        if paths_overlap(output, root):
            raise ExperimentPathError(
                f"{field} must not overlap an immutable input: {root}"
            )
    return output


def validate_resume_directory(
    value: str | os.PathLike[str],
    *,
    immutable_roots: Iterable[Path] = (),
    field: str = "experiment",
) -> Path:
    """Validate an existing experiment directory used for resume."""

    root = validate_existing_real_path(value, field=field, kind="directory")
    validate_safe_name(root.name, field=f"{field} directory name")
    for index, raw_source in enumerate(immutable_roots):
        source = validate_existing_real_path(
            raw_source,
            field=f"immutable_roots[{index}]",
        )
        if paths_overlap(root, source):
            raise ExperimentPathError(
                f"{field} must not overlap an immutable input: {source}"
            )
    return root


@dataclass(frozen=True)
class ExperimentPaths:
    """Stable paths below one experiment root."""

    root: Path

    def __post_init__(self) -> None:
        root = validate_absolute_path(self.root, field="experiment root")
        validate_safe_name(root.name, field="experiment directory name")
        object.__setattr__(self, "root", root)

    @classmethod
    def plan_new(
        cls,
        output: str | os.PathLike[str],
        *,
        immutable_roots: Iterable[Path] = (),
    ) -> "ExperimentPaths":
        return cls(
            validate_new_output_directory(
                output,
                immutable_roots=immutable_roots,
            )
        )

    @classmethod
    def for_resume(
        cls,
        experiment: str | os.PathLike[str],
        *,
        immutable_roots: Iterable[Path] = (),
    ) -> "ExperimentPaths":
        return cls(
            validate_resume_directory(
                experiment,
                immutable_roots=immutable_roots,
            )
        )

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def manifest(self) -> Path:
        return self.root / "experiment_manifest.json"

    @property
    def trials(self) -> Path:
        return self.root / "trials"

    @property
    def staging(self) -> Path:
        return self.root / ".staging"

    def trial(self, attempt_id: str) -> Path:
        return self.trials / validate_safe_name(
            attempt_id,
            field="attempt_id",
        )
