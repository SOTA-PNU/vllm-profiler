"""Stable evaluation failure classes based on structured evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import errno as errno_module
from typing import Mapping


class FailureClass(str, Enum):
    CONFIG_INVALID = "config_invalid"
    UNSAFE_PATH = "unsafe_path"
    PORT_IN_USE = "port_in_use"
    SERVER_START_FAILED = "server_start_failed"
    READINESS_TIMEOUT = "readiness_timeout"
    REQUEST_FAILED = "request_failed"
    ACCURACY_FAILED = "accuracy_failed"
    ENVIRONMENT_NOT_IDLE = "environment_not_idle"
    TRACE_VALIDATION_FAILED = "trace_validation_failed"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    PREFLIGHT_FAILED = "preflight_failed"
    PORT_COLLISION = "port_collision"
    SERVER_NOT_STARTED = "server_not_started"
    SERVER_STARTUP_FAILED = "server_startup_failed"
    HEALTH_NOT_READY = "health_not_ready"
    HEALTH_TIMEOUT = "health_timeout"
    WRONG_HOST_OR_PORT = "wrong_host_or_port"
    PARTIAL_TOPOLOGY = "partial_topology"
    SERVER_EARLY_EXIT = "server_early_exit"
    CLIENT_CONNECTION_REFUSED = "client_connection_refused"
    CLIENT_TIMEOUT = "client_timeout"
    CLIENT_CLEANUP_FAILED = "client_cleanup_failed"
    PROFILER_START_FAILED = "profiler_start_failed"
    PROFILER_STOP_FAILED = "profiler_stop_failed"
    PROFILER_FINALIZE_FAILED = "profiler_finalize_failed"
    TELEMETRY_MALFORMED = "telemetry_malformed"
    TELEMETRY_COMMAND_FAILED = "telemetry_command_failed"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    ARTIFACT_UNSAFE = "artifact_unsafe"
    PUBLICATION_CONFLICT = "publication_conflict"
    PUBLICATION_FAILED = "publication_failed"
    CLEANUP_FAILED = "cleanup_failed"
    INTERRUPTED = "interrupted"
    INTERNAL_ERROR = "internal_error"


class FailurePhase(str, Enum):
    PLAN = "plan"
    PREFLIGHT = "preflight"
    STARTUP = "startup"
    HEALTH = "health"
    WARMUP = "warmup"
    PROFILER_START = "profiler_start"
    MEASURED_REQUEST = "measured_request"
    PROFILER_STOP = "profiler_stop"
    PROFILER_FINALIZE = "profiler_finalize"
    TELEMETRY = "telemetry"
    ARTIFACT_VALIDATION = "artifact_validation"
    PUBLICATION = "publication"
    CLEANUP = "cleanup"


@dataclass(frozen=True)
class ConnectionEvidence:
    """Facts captured at a failed server or client connection boundary."""

    phase: FailurePhase
    process_role: str
    expected_host: str
    expected_port: int
    process_start_called: bool
    process_ready: bool
    process_returncode: int | None
    expected_listener_present: bool | None
    ready_roles: tuple[str, ...] = ()
    required_roles: tuple[str, ...] = ()
    error_number: int | None = None
    timed_out: bool = False
    preflight_port_collision: bool = False
    contacted_expected_endpoint: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.phase, FailurePhase):
            object.__setattr__(self, "phase", FailurePhase(self.phase))
        if not self.process_role:
            raise ValueError("process_role must be non-empty")
        if self.expected_host != "127.0.0.1":
            raise ValueError("expected_host must be 127.0.0.1")
        if (
            isinstance(self.expected_port, bool)
            or not isinstance(self.expected_port, int)
            or not 1 <= self.expected_port <= 65535
        ):
            raise ValueError("expected_port must be between 1 and 65535")
        if len(set(self.ready_roles)) != len(self.ready_roles):
            raise ValueError("ready_roles must be unique")
        if len(set(self.required_roles)) != len(self.required_roles):
            raise ValueError("required_roles must be unique")
        if not set(self.ready_roles).issubset(self.required_roles):
            raise ValueError("ready_roles must be a subset of required_roles")


def classify_connection_failure(evidence: ConnectionEvidence) -> FailureClass:
    """Classify a connection failure without parsing an exception message."""

    if evidence.preflight_port_collision:
        return FailureClass.PORT_COLLISION
    if not evidence.contacted_expected_endpoint:
        return FailureClass.WRONG_HOST_OR_PORT
    if not evidence.process_start_called:
        return FailureClass.SERVER_NOT_STARTED

    ready = set(evidence.ready_roles)
    required = set(evidence.required_roles)
    if required and ready and ready != required:
        return FailureClass.PARTIAL_TOPOLOGY
    if evidence.process_ready and evidence.process_returncode is not None:
        return FailureClass.SERVER_EARLY_EXIT
    if not evidence.process_ready and evidence.process_returncode is not None:
        return FailureClass.SERVER_STARTUP_FAILED

    if evidence.phase is FailurePhase.HEALTH:
        if evidence.timed_out:
            return FailureClass.HEALTH_TIMEOUT
        return FailureClass.HEALTH_NOT_READY
    if evidence.phase is FailurePhase.MEASURED_REQUEST:
        if evidence.timed_out:
            return FailureClass.CLIENT_TIMEOUT
        if (
            evidence.error_number == errno_module.ECONNREFUSED
            and evidence.process_ready
            and evidence.expected_listener_present is False
        ):
            return FailureClass.CLIENT_CONNECTION_REFUSED
    return FailureClass.INTERNAL_ERROR


@dataclass(frozen=True)
class FailureRecord:
    """Serializable failure classification with bounded diagnostic evidence."""

    failure_class: FailureClass
    phase: FailurePhase
    summary: str
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.failure_class, FailureClass):
            object.__setattr__(
                self,
                "failure_class",
                FailureClass(self.failure_class),
            )
        if not isinstance(self.phase, FailurePhase):
            object.__setattr__(self, "phase", FailurePhase(self.phase))
        if not isinstance(self.summary, str) or not self.summary:
            raise ValueError("failure summary must be non-empty")
        if len(self.summary) > 512:
            raise ValueError("failure summary must be at most 512 characters")
        object.__setattr__(self, "evidence", dict(self.evidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_class": self.failure_class.value,
            "phase": self.phase.value,
            "summary": self.summary,
            "evidence": dict(self.evidence),
        }
