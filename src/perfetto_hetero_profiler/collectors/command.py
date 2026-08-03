"""Safe child-command specification and redaction helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping


DEFAULT_ENV_ALLOWLIST = (
    "PATH",
    "PYTHONPATH",
    "LANG",
    "LC_ALL",
    "CUDA_VISIBLE_DEVICES",
)
_SENSITIVE_MARKERS = (
    "TOKEN",
    "PASSWORD",
    "SECRET",
    "API_KEY",
    "API-KEY",
    "AUTH",
    "CREDENTIAL",
)


def _sensitive(name: str) -> bool:
    normalized = name.upper()
    return any(marker in normalized for marker in _SENSITIVE_MARKERS)


def mask_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return a copy with values of credential-like keys redacted."""
    return {
        key: "***" if _sensitive(key) else value
        for key, value in sorted(environment.items())
    }


def mask_command(argv: tuple[str, ...] | list[str]) -> list[str]:
    """Redact common secret-bearing command options without joining a shell string."""
    masked: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            masked.append("***")
            redact_next = False
            continue
        if "=" in argument:
            key, _value = argument.split("=", 1)
            if _sensitive(key):
                masked.append(f"{key}=***")
                continue
        masked.append(argument)
        if argument.startswith("-") and _sensitive(argument):
            redact_next = True
    return masked


@dataclass(frozen=True)
class CommandSpec:
    """An argv-only child command with an explicit environment boundary."""

    argv: tuple[str, ...]
    cwd: Path | None = None
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    env_allowlist: tuple[str, ...] = DEFAULT_ENV_ALLOWLIST
    timeout_sec: float | None = None
    terminate_grace_sec: float = 2.0

    def __post_init__(self) -> None:
        if not self.argv or any(not isinstance(item, str) or not item for item in self.argv):
            raise ValueError("argv must contain non-empty strings")
        if self.timeout_sec is not None and self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        if self.terminate_grace_sec < 0:
            raise ValueError("terminate_grace_sec must be >= 0")
        disallowed = sorted(set(self.env_overrides) - set(self.env_allowlist))
        if disallowed:
            raise ValueError(f"environment override is not allowlisted: {disallowed[0]}")

    def safe_plan(self) -> dict[str, object]:
        return {
            "argv": mask_command(self.argv),
            "cwd": str(self.cwd) if self.cwd is not None else None,
            "environment": mask_environment(self.env_overrides),
            "timeout_sec": self.timeout_sec,
        }


def build_environment(
    spec: CommandSpec, source: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build the child environment from allowlisted inherited values and overrides."""
    inherited = os.environ if source is None else source
    environment = {
        key: inherited[key] for key in spec.env_allowlist if key in inherited
    }
    environment.update(spec.env_overrides)
    return environment
