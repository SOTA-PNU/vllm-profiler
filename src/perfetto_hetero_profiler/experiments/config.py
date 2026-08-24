"""Strict configuration for fixed hybrid profiler experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from ..hybrid.runner_config import HybridRunnerConfig, load_hybrid_runner_config
from .paths import validate_existing_real_path, validate_safe_name
from .compatibility import LEGACY_SCHEDULE_SEED_DOMAIN
from .schedule import (
    MAX_HARDWARE_ATTEMPTS,
    SCHEDULE_SEED_DOMAIN,
    ExperimentSchedule,
    build_schedule,
    canonical_schedule_bytes,
)


CONFIG_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExperimentConfigError(ValueError):
    pass


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{name} must be an object")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise ExperimentConfigError(f"unknown {name} field: {unknown[0]}")
    if missing:
        raise ExperimentConfigError(f"missing {name} field: {missing[0]}")
    return dict(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    config_path: Path
    experiment_id: str
    hybrid_config_path: Path
    hybrid_config_sha256: str
    seed: int
    max_hardware_attempts: int
    schedule_seed_domain: str = SCHEDULE_SEED_DOMAIN

    @property
    def schedule(self) -> ExperimentSchedule:
        return build_schedule(
            seed=self.seed,
            max_hardware_attempts=self.max_hardware_attempts,
            seed_domain=self.schedule_seed_domain,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONFIG_VERSION,
            "experiment_id": self.experiment_id,
            "hybrid_config": {
                "path": str(self.hybrid_config_path),
                "sha256": self.hybrid_config_sha256,
            },
            "schedule": {
                "seed": self.seed,
                "max_hardware_attempts": self.max_hardware_attempts,
            },
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_config_bytes(self)).hexdigest()

    def load_hybrid(self) -> HybridRunnerConfig:
        if sha256_file(self.hybrid_config_path) != self.hybrid_config_sha256:
            raise ExperimentConfigError("hybrid config SHA-256 mismatch")
        config = load_hybrid_runner_config(self.hybrid_config_path)
        workload = config.workload
        fixed = {
            "prompt": workload.prompt_text(),
            "warmup_requests": workload.warmup_requests,
            "measured_requests": workload.measured_requests,
            "max_output_tokens": workload.max_output_tokens,
            "temperature": workload.temperature,
            "streaming": workload.streaming,
            "max_num_seqs": config.max_num_seqs,
            "offline": config.offline,
        }
        expected = {
            "prompt": "Capital of South Korea is",
            "warmup_requests": 2,
            "measured_requests": 10,
            "max_output_tokens": 8,
            "temperature": 0.0,
            "streaming": True,
            "max_num_seqs": 1,
            "offline": True,
        }
        if fixed != expected:
            raise ExperimentConfigError(
                f"hybrid workload does not match the fixed experiment policy: {fixed}"
            )
        if config.served_model_name != "Qwen3-0.6B":
            raise ExperimentConfigError("served model must be Qwen3-0.6B")
        return config


def canonical_config_bytes(config: ExperimentConfig) -> bytes:
    return (
        json.dumps(
            config.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def load_experiment_config(path: Path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_absolute():
        raise ExperimentConfigError("--config must be absolute")
    try:
        path = validate_existing_real_path(path, field="config", kind="file")
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ExperimentConfigError):
            raise
        raise ExperimentConfigError(f"cannot load config: {error}") from error
    root = _object(
        value,
        "config",
        {"schema_version", "experiment_id", "hybrid_config", "schedule"},
    )
    if root["schema_version"] != CONFIG_VERSION:
        raise ExperimentConfigError(f"schema_version must be {CONFIG_VERSION!r}")
    experiment_id = root["experiment_id"]
    if not isinstance(experiment_id, str):
        raise ExperimentConfigError("experiment_id must be a string")
    validate_safe_name(experiment_id, field="experiment_id")
    hybrid = _object(root["hybrid_config"], "hybrid_config", {"path", "sha256"})
    if not isinstance(hybrid["path"], str) or not Path(hybrid["path"]).is_absolute():
        raise ExperimentConfigError("hybrid_config.path must be absolute")
    hybrid_path = validate_existing_real_path(
        hybrid["path"], field="hybrid_config.path", kind="file"
    )
    if not isinstance(hybrid["sha256"], str) or _SHA256.fullmatch(hybrid["sha256"]) is None:
        raise ExperimentConfigError("hybrid_config.sha256 must be lowercase SHA-256")
    schedule = _object(root["schedule"], "schedule", {"seed", "max_hardware_attempts"})
    seed = schedule["seed"]
    maximum = schedule["max_hardware_attempts"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ExperimentConfigError("schedule.seed must be in [0, 2^32-1]")
    if maximum != MAX_HARDWARE_ATTEMPTS:
        raise ExperimentConfigError("schedule.max_hardware_attempts must equal 42")
    seed_domain = SCHEDULE_SEED_DOMAIN
    stored_schedule = path.parent / "schedule.json"
    if path.name == "config.json" and stored_schedule.is_file():
        legacy = build_schedule(
            seed=seed,
            max_hardware_attempts=maximum,
            seed_domain=LEGACY_SCHEDULE_SEED_DOMAIN,
        )
        try:
            if stored_schedule.read_bytes() == canonical_schedule_bytes(legacy):
                seed_domain = LEGACY_SCHEDULE_SEED_DOMAIN
        except OSError as error:
            raise ExperimentConfigError(f"cannot inspect stored schedule: {error}") from error
    result = ExperimentConfig(
        config_path=path,
        experiment_id=experiment_id,
        hybrid_config_path=hybrid_path,
        hybrid_config_sha256=hybrid["sha256"],
        seed=seed,
        max_hardware_attempts=maximum,
        schedule_seed_domain=seed_domain,
    )
    result.load_hybrid()
    return result


__all__ = [
    "CONFIG_VERSION",
    "ExperimentConfig",
    "ExperimentConfigError",
    "canonical_config_bytes",
    "load_experiment_config",
    "sha256_file",
]
