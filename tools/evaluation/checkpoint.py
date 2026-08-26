"""Atomic evaluation checkpoints and conservative resume decisions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from .failure import FailureClass
from .paths import (
    ExperimentPathError,
    validate_absolute_path,
    validate_existing_real_path,
    validate_safe_name,
)
from .schedule import (
    MAX_HARDWARE_ATTEMPTS,
    make_attempt_id,
    validate_attempt_id,
    validate_logical_trial_id,
)


CHECKPOINT_VERSION = "1.0.0"
PROCESS_EVIDENCE_OWNERSHIP = "evidence_only"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class CheckpointError(RuntimeError):
    """A checkpoint cannot be trusted or safely updated."""


class CheckpointIdentityError(CheckpointError):
    """Resume was requested with a different config or schedule."""


class CheckpointIntegrityError(CheckpointError):
    """Stored state or a successful trial failed fresh validation."""


class AttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass(frozen=True)
class ProcessEvidence:
    """Diagnostic PID/PGID evidence that never conveys signal ownership."""

    role: str
    pid: int
    pgid: int
    observed_state: str
    ownership: str = field(
        default=PROCESS_EVIDENCE_OWNERSHIP,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or _ROLE_RE.fullmatch(self.role) is None:
            raise CheckpointIntegrityError("process evidence role is unsafe")
        for name in ("pid", "pgid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CheckpointIntegrityError(
                    f"process evidence {name} must be a positive integer"
                )
        if not isinstance(self.observed_state, str) or not self.observed_state:
            raise CheckpointIntegrityError(
                "process evidence observed_state must be non-empty"
            )

    @property
    def grants_ownership(self) -> bool:
        """Persisted process identity is evidence only, never authority."""

        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "pid": self.pid,
            "pgid": self.pgid,
            "observed_state": self.observed_state,
            "ownership": self.ownership,
        }


@dataclass(frozen=True)
class AttemptRecord:
    """One immutable-on-terminal hardware attempt."""

    attempt_id: str
    logical_trial_id: str
    attempt_number: int
    status: AttemptStatus
    relative_directory: str
    failure_class: FailureClass | None = None
    failure_summary: str | None = None
    process_evidence: tuple[ProcessEvidence, ...] = ()
    artifact_validation_valid: bool | None = None
    environment_fingerprint: str | None = None

    def __post_init__(self) -> None:
        validate_logical_trial_id(self.logical_trial_id)
        validate_attempt_id(self.attempt_id)
        if (
            isinstance(self.attempt_number, bool)
            or not isinstance(self.attempt_number, int)
            or self.attempt_number not in {1, 2}
        ):
            raise CheckpointIntegrityError("attempt_number must be 1 or 2")
        if self.attempt_id != make_attempt_id(
            self.logical_trial_id,
            self.attempt_number,
        ):
            raise CheckpointIntegrityError(
                "attempt_id does not match logical_trial_id and attempt_number"
            )
        try:
            validate_safe_name(
                self.relative_directory,
                field="relative_directory",
            )
        except ExperimentPathError as error:
            raise CheckpointIntegrityError(str(error)) from error
        if self.relative_directory != self.attempt_id:
            raise CheckpointIntegrityError(
                "relative_directory must equal attempt_id"
            )
        if not isinstance(self.status, AttemptStatus):
            object.__setattr__(self, "status", AttemptStatus(self.status))
        if self.failure_class is not None and not isinstance(
            self.failure_class,
            FailureClass,
        ):
            object.__setattr__(
                self,
                "failure_class",
                FailureClass(self.failure_class),
            )
        object.__setattr__(
            self,
            "process_evidence",
            tuple(self.process_evidence),
        )
        for evidence in self.process_evidence:
            if not isinstance(evidence, ProcessEvidence):
                raise CheckpointIntegrityError(
                    "process_evidence entries must be ProcessEvidence"
                )

        failed = self.status in {AttemptStatus.FAILED, AttemptStatus.PARTIAL}
        if failed:
            if self.failure_class is None:
                raise CheckpointIntegrityError(
                    "failed or partial attempt requires failure_class"
                )
            if (
                not isinstance(self.failure_summary, str)
                or not self.failure_summary
                or len(self.failure_summary) > 512
            ):
                raise CheckpointIntegrityError(
                    "failed or partial attempt requires a bounded failure_summary"
                )
        elif self.failure_class is not None or self.failure_summary is not None:
            raise CheckpointIntegrityError(
                "running or succeeded attempt must not contain failure data"
            )
        if self.artifact_validation_valid is not None and not isinstance(
            self.artifact_validation_valid, bool
        ):
            raise CheckpointIntegrityError(
                "artifact_validation_valid must be boolean or null"
            )
        if self.environment_fingerprint is not None and (
            not isinstance(self.environment_fingerprint, str)
            or _SHA256_RE.fullmatch(self.environment_fingerprint) is None
        ):
            raise CheckpointIntegrityError(
                "environment_fingerprint must be a lowercase SHA-256 or null"
            )
        if self.status is AttemptStatus.SUCCEEDED and (
            self.artifact_validation_valid is not True
            or self.environment_fingerprint is None
        ):
            raise CheckpointIntegrityError(
                "succeeded attempt requires valid artifacts and environment fingerprint"
            )

    @property
    def terminal(self) -> bool:
        return self.status is not AttemptStatus.RUNNING

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "logical_trial_id": self.logical_trial_id,
            "attempt_number": self.attempt_number,
            "status": self.status.value,
            "relative_directory": self.relative_directory,
            "failure_class": (
                self.failure_class.value
                if self.failure_class is not None
                else None
            ),
            "failure_summary": self.failure_summary,
            "process_evidence": [
                evidence.to_dict() for evidence in self.process_evidence
            ],
            "artifact_validation_valid": self.artifact_validation_valid,
            "environment_fingerprint": self.environment_fingerprint,
        }


@dataclass(frozen=True)
class ExperimentCheckpoint:
    """Identity-bound experiment state."""

    config_sha256: str
    schedule_sha256: str
    max_hardware_attempts: int
    generation: int = 0
    attempts: tuple[AttemptRecord, ...] = ()
    checkpoint_version: str = CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        if self.checkpoint_version != CHECKPOINT_VERSION:
            raise CheckpointIntegrityError(
                f"unsupported checkpoint_version: {self.checkpoint_version!r}"
            )
        for name in ("config_sha256", "schedule_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise CheckpointIntegrityError(
                    f"{name} must be a lowercase SHA-256 digest"
                )
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise CheckpointIntegrityError(
                "checkpoint generation must be a non-negative integer"
            )
        if (
            isinstance(self.max_hardware_attempts, bool)
            or not isinstance(self.max_hardware_attempts, int)
            or not 36
            <= self.max_hardware_attempts
            <= MAX_HARDWARE_ATTEMPTS
        ):
            raise CheckpointIntegrityError(
                "max_hardware_attempts must be between 36 and 42"
            )
        object.__setattr__(self, "attempts", tuple(self.attempts))
        if len(self.attempts) > self.max_hardware_attempts:
            raise CheckpointIntegrityError("hardware attempt limit exceeded")
        self._validate_attempt_history()

    @classmethod
    def new(
        cls,
        *,
        config_sha256: str,
        schedule_sha256: str,
        max_hardware_attempts: int = MAX_HARDWARE_ATTEMPTS,
    ) -> "ExperimentCheckpoint":
        return cls(
            config_sha256=config_sha256,
            schedule_sha256=schedule_sha256,
            max_hardware_attempts=max_hardware_attempts,
        )

    def _validate_attempt_history(self) -> None:
        attempt_ids: set[str] = set()
        directories: set[str] = set()
        per_logical: dict[str, list[AttemptRecord]] = {}
        for attempt in self.attempts:
            if not isinstance(attempt, AttemptRecord):
                raise CheckpointIntegrityError(
                    "attempts entries must be AttemptRecord"
                )
            if attempt.attempt_id in attempt_ids:
                raise CheckpointIntegrityError("attempt_id must be unique")
            if attempt.relative_directory in directories:
                raise CheckpointIntegrityError(
                    "attempt directories must be unique"
                )
            attempt_ids.add(attempt.attempt_id)
            directories.add(attempt.relative_directory)
            per_logical.setdefault(attempt.logical_trial_id, []).append(attempt)

        for logical_trial_id, attempts in per_logical.items():
            numbers = [attempt.attempt_number for attempt in attempts]
            if numbers != list(range(1, len(attempts) + 1)):
                raise CheckpointIntegrityError(
                    f"attempt numbers for {logical_trial_id} must be contiguous"
                )
            if len(attempts) > 2:
                raise CheckpointIntegrityError(
                    f"{logical_trial_id} exceeds the one-retry policy"
                )
            for index, attempt in enumerate(attempts):
                if (
                    attempt.status is AttemptStatus.SUCCEEDED
                    and index != len(attempts) - 1
                ):
                    raise CheckpointIntegrityError(
                        "no retry may follow a succeeded attempt"
                    )

    def attempts_for(self, logical_trial_id: str) -> tuple[AttemptRecord, ...]:
        validate_logical_trial_id(logical_trial_id)
        return tuple(
            attempt
            for attempt in self.attempts
            if attempt.logical_trial_id == logical_trial_id
        )

    def with_attempt(self, record: AttemptRecord) -> "ExperimentCheckpoint":
        """Append an attempt or finalize its nonterminal record."""

        if not isinstance(record, AttemptRecord):
            raise TypeError("record must be AttemptRecord")
        existing_index = next(
            (
                index
                for index, attempt in enumerate(self.attempts)
                if attempt.attempt_id == record.attempt_id
            ),
            None,
        )
        attempts = list(self.attempts)
        if existing_index is not None:
            previous = attempts[existing_index]
            _validate_attempt_record_transition(previous, record)
            attempts[existing_index] = record
        else:
            if len(attempts) >= self.max_hardware_attempts:
                raise CheckpointIntegrityError("hardware attempt limit reached")
            logical_attempts = self.attempts_for(record.logical_trial_id)
            if logical_attempts and logical_attempts[-1].status is AttemptStatus.RUNNING:
                raise CheckpointIntegrityError(
                    "running attempt must be finalized before retry"
                )
            if any(
                attempt.status is AttemptStatus.SUCCEEDED
                for attempt in logical_attempts
            ):
                raise CheckpointIntegrityError(
                    "cannot append a retry after success"
                )
            expected_number = len(logical_attempts) + 1
            if record.attempt_number != expected_number:
                raise CheckpointIntegrityError(
                    f"next attempt_number must be {expected_number}"
                )
            attempts.append(record)
        return replace(
            self,
            generation=self.generation + 1,
            attempts=tuple(attempts),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "config_sha256": self.config_sha256,
            "schedule_sha256": self.schedule_sha256,
            "max_hardware_attempts": self.max_hardware_attempts,
            "generation": self.generation,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class ResumeDecision:
    checkpoint: ExperimentCheckpoint
    skip_logical_trial_ids: tuple[str, ...]
    failed_attempt_ids: tuple[str, ...]
    incomplete_attempt_ids: tuple[str, ...]

    @property
    def hardware_attempt_count(self) -> int:
        return len(self.checkpoint.attempts)

    @property
    def attempt_budget_remaining(self) -> int:
        return (
            self.checkpoint.max_hardware_attempts
            - self.hardware_attempt_count
        )


def _validate_attempt_record_transition(
    previous: AttemptRecord,
    updated: AttemptRecord,
) -> None:
    identity_fields = (
        "attempt_id",
        "logical_trial_id",
        "attempt_number",
        "relative_directory",
    )
    if any(
        getattr(previous, name) != getattr(updated, name)
        for name in identity_fields
    ):
        raise CheckpointIntegrityError("attempt identity is immutable")
    if previous.terminal:
        if previous != updated:
            raise CheckpointIntegrityError(
                "terminal attempt records are immutable"
            )
        return
    if updated.status is AttemptStatus.RUNNING and previous != updated:
        raise CheckpointIntegrityError(
            "running attempt may only transition to a terminal state"
        )
    if updated.status is AttemptStatus.RUNNING:
        return
    if updated.process_evidence[: len(previous.process_evidence)] != (
        previous.process_evidence
    ):
        raise CheckpointIntegrityError(
            "finalization must preserve prior process evidence"
        )


def validate_checkpoint_transition(
    previous: ExperimentCheckpoint,
    updated: ExperimentCheckpoint,
) -> None:
    """Reject identity changes, history deletion, and terminal mutation."""

    if updated.generation != previous.generation + 1:
        raise CheckpointIntegrityError(
            "checkpoint generation must increase by exactly one"
        )
    identity_fields = (
        "checkpoint_version",
        "config_sha256",
        "schedule_sha256",
        "max_hardware_attempts",
    )
    if any(
        getattr(previous, name) != getattr(updated, name)
        for name in identity_fields
    ):
        raise CheckpointIntegrityError("checkpoint identity is immutable")
    if len(updated.attempts) < len(previous.attempts):
        raise CheckpointIntegrityError("attempt history must not shrink")
    for index, old_attempt in enumerate(previous.attempts):
        new_attempt = updated.attempts[index]
        if old_attempt.attempt_id != new_attempt.attempt_id:
            raise CheckpointIntegrityError(
                "attempt history must not be reordered"
            )
        _validate_attempt_record_transition(old_attempt, new_attempt)
    if len(updated.attempts) > len(previous.attempts) + 1:
        raise CheckpointIntegrityError(
            "only one hardware attempt may be appended per checkpoint update"
        )


def canonical_checkpoint_bytes(checkpoint: ExperimentCheckpoint) -> bytes:
    return (
        json.dumps(
            checkpoint.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _strict_keys(
    value: Any,
    *,
    path: str,
    fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError(f"{path} must be an object")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise CheckpointIntegrityError(f"{path}.{unknown[0]} is unknown")
    if missing:
        raise CheckpointIntegrityError(f"{path}.{missing[0]} is required")
    return value


def _process_evidence_from_dict(
    value: Any,
    *,
    path: str,
) -> ProcessEvidence:
    data = _strict_keys(
        value,
        path=path,
        fields={
            "role",
            "pid",
            "pgid",
            "observed_state",
            "ownership",
        },
    )
    if data["ownership"] != PROCESS_EVIDENCE_OWNERSHIP:
        raise CheckpointIntegrityError(
            f"{path}.ownership must be {PROCESS_EVIDENCE_OWNERSHIP!r}"
        )
    return ProcessEvidence(
        role=data["role"],
        pid=data["pid"],
        pgid=data["pgid"],
        observed_state=data["observed_state"],
    )


def _attempt_from_dict(value: Any, *, path: str) -> AttemptRecord:
    data = _strict_keys(
        value,
        path=path,
        fields={
            "attempt_id",
            "logical_trial_id",
            "attempt_number",
            "status",
            "relative_directory",
            "failure_class",
            "failure_summary",
            "process_evidence",
            "artifact_validation_valid",
            "environment_fingerprint",
        },
    )
    evidence_value = data["process_evidence"]
    if not isinstance(evidence_value, list):
        raise CheckpointIntegrityError(f"{path}.process_evidence must be an array")
    try:
        status = AttemptStatus(data["status"])
        failure_class = (
            FailureClass(data["failure_class"])
            if data["failure_class"] is not None
            else None
        )
    except (TypeError, ValueError) as error:
        raise CheckpointIntegrityError(
            f"{path} contains an invalid enum value"
        ) from error
    return AttemptRecord(
        attempt_id=data["attempt_id"],
        logical_trial_id=data["logical_trial_id"],
        attempt_number=data["attempt_number"],
        status=status,
        relative_directory=data["relative_directory"],
        failure_class=failure_class,
        failure_summary=data["failure_summary"],
        process_evidence=tuple(
            _process_evidence_from_dict(item, path=f"{path}.process_evidence[{index}]")
            for index, item in enumerate(evidence_value)
        ),
        artifact_validation_valid=data["artifact_validation_valid"],
        environment_fingerprint=data["environment_fingerprint"],
    )


def checkpoint_from_dict(value: Any) -> ExperimentCheckpoint:
    data = _strict_keys(
        value,
        path="$",
        fields={
            "checkpoint_version",
            "config_sha256",
            "schedule_sha256",
            "max_hardware_attempts",
            "generation",
            "attempts",
        },
    )
    attempts_value = data["attempts"]
    if not isinstance(attempts_value, list):
        raise CheckpointIntegrityError("$.attempts must be an array")
    return ExperimentCheckpoint(
        checkpoint_version=data["checkpoint_version"],
        config_sha256=data["config_sha256"],
        schedule_sha256=data["schedule_sha256"],
        max_hardware_attempts=data["max_hardware_attempts"],
        generation=data["generation"],
        attempts=tuple(
            _attempt_from_dict(item, path=f"$.attempts[{index}]")
            for index, item in enumerate(attempts_value)
        ),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CheckpointIntegrityError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temp(parent: Path, name: str, data: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{name}.checkpoint-",
        dir=parent,
    )
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise CheckpointIntegrityError(
            "checkpoint must be a real regular file"
        )
    return (
        path_stat.st_dev,
        path_stat.st_ino,
        path_stat.st_size,
        path_stat.st_mtime_ns,
    )


class CheckpointStore:
    """Atomic single-writer checkpoint storage."""

    def __init__(self, path: str | os.PathLike[str]):
        try:
            checkpoint_path = validate_absolute_path(
                path,
                field="checkpoint",
            )
            validate_safe_name(
                checkpoint_path.name,
                field="checkpoint filename",
            )
            validate_existing_real_path(
                checkpoint_path.parent,
                field="checkpoint parent",
                kind="directory",
            )
        except ExperimentPathError as error:
            raise CheckpointError(str(error)) from error
        self.path = checkpoint_path

    def initialize(self, checkpoint: ExperimentCheckpoint) -> None:
        """Atomically create a generation-zero checkpoint without overwrite."""

        if checkpoint.generation != 0:
            raise CheckpointIntegrityError(
                "initial checkpoint generation must be zero"
            )
        data = canonical_checkpoint_bytes(checkpoint)
        temporary = _write_temp(self.path.parent, self.path.name, data)
        try:
            os.link(temporary, self.path)
            _fsync_directory(self.path.parent)
        except FileExistsError:
            raise FileExistsError(
                f"checkpoint already exists: {self.path}"
            ) from None
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def load(self) -> ExperimentCheckpoint:
        """Read and validate a real checkpoint file."""

        try:
            validate_existing_real_path(
                self.path,
                field="checkpoint",
                kind="file",
            )
            before = _file_identity(self.path)
            raw = self.path.read_text(encoding="utf-8")
            after = _file_identity(self.path)
        except (OSError, UnicodeDecodeError, ExperimentPathError) as error:
            raise CheckpointIntegrityError(
                f"checkpoint cannot be read safely: {self.path}"
            ) from error
        if before != after:
            raise CheckpointIntegrityError("checkpoint changed while being read")
        try:
            value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise CheckpointIntegrityError(
                f"checkpoint contains invalid JSON at line {error.lineno}"
            ) from error
        return checkpoint_from_dict(value)

    def update(self, checkpoint: ExperimentCheckpoint) -> None:
        """Atomically replace a checkpoint after validating its transition."""

        previous_identity = _file_identity(self.path)
        previous = self.load()
        validate_checkpoint_transition(previous, checkpoint)
        data = canonical_checkpoint_bytes(checkpoint)
        temporary = _write_temp(self.path.parent, self.path.name, data)
        try:
            if _file_identity(self.path) != previous_identity:
                raise CheckpointIntegrityError(
                    "checkpoint changed before update publication"
                )
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def load_for_resume(
        self,
        *,
        expected_config_sha256: str,
        expected_schedule_sha256: str,
        validate_success: Callable[[AttemptRecord], bool],
    ) -> ResumeDecision:
        """Load resume state and freshly validate every skipped success."""

        if not callable(validate_success):
            raise TypeError("validate_success must be callable")
        checkpoint = self.load()
        if checkpoint.config_sha256 != expected_config_sha256:
            raise CheckpointIdentityError(
                "resume config hash does not match the experiment"
            )
        if checkpoint.schedule_sha256 != expected_schedule_sha256:
            raise CheckpointIdentityError(
                "resume schedule hash does not match the experiment"
            )

        skipped: list[str] = []
        failed: list[str] = []
        incomplete: list[str] = []
        for attempt in checkpoint.attempts:
            if attempt.status is AttemptStatus.SUCCEEDED:
                try:
                    valid = validate_success(attempt)
                except Exception as error:
                    raise CheckpointIntegrityError(
                        f"fresh validation failed for {attempt.attempt_id}"
                    ) from error
                if valid is not True:
                    raise CheckpointIntegrityError(
                        f"successful trial failed fresh validation: "
                        f"{attempt.attempt_id}"
                    )
                skipped.append(attempt.logical_trial_id)
            elif attempt.status is AttemptStatus.FAILED:
                failed.append(attempt.attempt_id)
            else:
                incomplete.append(attempt.attempt_id)
        return ResumeDecision(
            checkpoint=checkpoint,
            skip_logical_trial_ids=tuple(skipped),
            failed_attempt_ids=tuple(failed),
            incomplete_attempt_ids=tuple(incomplete),
        )


def checkpoint_digest(checkpoint: ExperimentCheckpoint) -> str:
    return hashlib.sha256(canonical_checkpoint_bytes(checkpoint)).hexdigest()
