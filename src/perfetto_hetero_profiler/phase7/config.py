"""Strict configuration for the fixed Phase 7B experiment."""

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
from .schedule import MAX_HARDWARE_ATTEMPTS, Phase7Schedule, build_schedule


CONFIG_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Phase7ConfigError(ValueError):
    pass


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase7ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Phase7ConfigError(f"{name} must be an object")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise Phase7ConfigError(f"unknown {name} field: {unknown[0]}")
    if missing:
        raise Phase7ConfigError(f"missing {name} field: {missing[0]}")
    return dict(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Phase7Config:
    config_path: Path
    experiment_id: str
    hybrid_config_path: Path
    hybrid_config_sha256: str
    seed: int
    max_hardware_attempts: int

    @property
    def schedule(self) -> Phase7Schedule:
        return build_schedule(
            seed=self.seed,
            max_hardware_attempts=self.max_hardware_attempts,
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
            raise Phase7ConfigError("hybrid config SHA-256 mismatch")
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
            raise Phase7ConfigError(
                f"hybrid workload does not match fixed Phase 7B policy: {fixed}"
            )
        if config.served_model_name != "Qwen3-0.6B":
            raise Phase7ConfigError("served model must be Qwen3-0.6B")
        return config


def canonical_config_bytes(config: Phase7Config) -> bytes:
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


def load_phase7_config(path: Path) -> Phase7Config:
    path = Path(path)
    if not path.is_absolute():
        raise Phase7ConfigError("--config must be absolute")
    try:
        path = validate_existing_real_path(path, field="config", kind="file")
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, Phase7ConfigError):
            raise
        raise Phase7ConfigError(f"cannot load config: {error}") from error
    root = _object(
        value,
        "config",
        {"schema_version", "experiment_id", "hybrid_config", "schedule"},
    )
    if root["schema_version"] != CONFIG_VERSION:
        raise Phase7ConfigError(f"schema_version must be {CONFIG_VERSION!r}")
    experiment_id = root["experiment_id"]
    if not isinstance(experiment_id, str):
        raise Phase7ConfigError("experiment_id must be a string")
    validate_safe_name(experiment_id, field="experiment_id")
    hybrid = _object(root["hybrid_config"], "hybrid_config", {"path", "sha256"})
    if not isinstance(hybrid["path"], str) or not Path(hybrid["path"]).is_absolute():
        raise Phase7ConfigError("hybrid_config.path must be absolute")
    hybrid_path = validate_existing_real_path(
        hybrid["path"], field="hybrid_config.path", kind="file"
    )
    if not isinstance(hybrid["sha256"], str) or _SHA256.fullmatch(hybrid["sha256"]) is None:
        raise Phase7ConfigError("hybrid_config.sha256 must be lowercase SHA-256")
    schedule = _object(root["schedule"], "schedule", {"seed", "max_hardware_attempts"})
    seed = schedule["seed"]
    maximum = schedule["max_hardware_attempts"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise Phase7ConfigError("schedule.seed must be in [0, 2^32-1]")
    if maximum != MAX_HARDWARE_ATTEMPTS:
        raise Phase7ConfigError("schedule.max_hardware_attempts must equal 42")
    result = Phase7Config(
        config_path=path,
        experiment_id=experiment_id,
        hybrid_config_path=hybrid_path,
        hybrid_config_sha256=hybrid["sha256"],
        seed=seed,
        max_hardware_attempts=maximum,
    )
    result.load_hybrid()
    return result


__all__ = [
    "CONFIG_VERSION",
    "Phase7Config",
    "Phase7ConfigError",
    "canonical_config_bytes",
    "load_phase7_config",
    "sha256_file",
]
