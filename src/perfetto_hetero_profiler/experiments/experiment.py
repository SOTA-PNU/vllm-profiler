"""Repeatability experiment orchestration on top of the HybridRunner."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from ..hybrid.runner import HybridRunner
from ..hybrid.runner_config import validate_hybrid_invocation
from ..schema.records import RunStatus
from .checkpoint import (
    AttemptRecord,
    AttemptStatus,
    CheckpointIntegrityError,
    CheckpointStore,
    ExperimentCheckpoint,
)
from .config import ExperimentConfig, canonical_config_bytes, load_experiment_config, sha256_file
from .environment import (
    EnvironmentNotIdleError,
    canonical_bytes,
    capture_environment,
    idle_reasons,
    wait_for_idle,
)
from .failure import FailureClass
from .paths import ExperimentPaths, validate_new_output_directory
from .report import build_report, canonical_json, render_report_html
from .schedule import Condition, TrialKind, canonical_schedule_bytes, schedule_by_logical_id
from .validation import TrialValidationError, validate_trial


CONDITION_MODE = {
    Condition.REFERENCE: ("monitor", False),
    Condition.MONITOR: ("monitor", True),
    Condition.GPU_TORCH: ("gpu-torch", True),
    Condition.GPU_NSYS: ("gpu-nsys", True),
    Condition.NPU_TORCH: ("npu-torch", True),
    Condition.NPU_RBLN: ("npu-rbln", True),
}

RETRYABLE = frozenset(
    {
        FailureClass.SERVER_START_FAILED,
        FailureClass.READINESS_TIMEOUT,
        FailureClass.CLIENT_CONNECTION_REFUSED,
        FailureClass.CLIENT_TIMEOUT,
        FailureClass.REQUEST_FAILED,
        FailureClass.PROFILER_START_FAILED,
        FailureClass.PROFILER_STOP_FAILED,
        FailureClass.CLEANUP_FAILED,
        FailureClass.ARTIFACT_MISSING,
        FailureClass.ARTIFACT_MISMATCH,
        FailureClass.TRACE_VALIDATION_FAILED,
        FailureClass.PUBLICATION_FAILED,
        FailureClass.INTERRUPTED,
    }
)


class ExperimentError(RuntimeError):
    pass


def _atomic_write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    _atomic_write(path, canonical_bytes(value), exclusive=exclusive)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentError(f"expected JSON object: {path}")
    return value


def _policy() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "logical_trials": 36,
        "maximum_hardware_attempts": 42,
        "maximum_retries_per_logical_trial": 1,
        "pilot_required_before_formal": True,
        "pilot_in_formal_statistics": False,
        "outlier_exclusion": False,
        "conditions": {
            condition.value: {
                "hybrid_profile_mode": mode,
                "resource_telemetry": telemetry,
                "detailed_profiler": None if mode == "monitor" else mode,
            }
            for condition, (mode, telemetry) in CONDITION_MODE.items()
        },
        "accuracy": {
            "e2e_ttft": "absolute_error <= max(2000000 ns, reference*0.02)",
            "tpot": "absolute_error <= max(1000000 ns, reference*0.05)",
            "boundary_equality_passes": True,
            "counts_markers_ids": "exact equality",
        },
        "latency_definitions": {
            "e2e": "request_start_to_stream_done",
            "ttft": "request_start_to_first_valid_output_token",
            "tpot": "(last_valid_token-first_valid_token)/(output_tokens-1)",
        },
        "retryable_failure_classes": sorted(item.value for item in RETRYABLE),
        "reference_limitation": "runtime marker emission remains enabled",
    }


def build_plan(config: ExperimentConfig, experiment_root: Path) -> dict[str, object]:
    schedule = config.schedule
    return {
        "executes": False,
        "creates_output": False,
        "experiment_root": str(experiment_root),
        "config_sha256": config.sha256,
        "schedule_sha256": schedule.sha256,
        "logical_trial_count": len(schedule.trials),
        "pilot_count": len(schedule.pilot_trials),
        "formal_count": len(schedule.formal_trials),
        "maximum_hardware_attempts": schedule.max_hardware_attempts,
        "trials": [item.to_dict() for item in schedule.trials],
    }


def _classify_message(message: str) -> FailureClass:
    lower = message.lower()
    if "accuracy" in lower or "token" in lower or "marker" in lower or "correlation" in lower:
        return FailureClass.ACCURACY_FAILED
    if "trace" in lower or "perfetto" in lower:
        return FailureClass.TRACE_VALIDATION_FAILED
    if "artifact" in lower or "fingerprint" in lower or "deterministic" in lower:
        return FailureClass.ARTIFACT_MISMATCH
    if "start_profile" in lower:
        return FailureClass.PROFILER_START_FAILED
    if "stop_profile" in lower or "profiler cleanup" in lower:
        return FailureClass.PROFILER_STOP_FAILED
    if "readiness" in lower or "before readiness" in lower:
        return FailureClass.READINESS_TIMEOUT
    if "connection refused" in lower:
        return FailureClass.CLIENT_CONNECTION_REFUSED
    if "timeout" in lower:
        return FailureClass.CLIENT_TIMEOUT
    if "cleanup" in lower or "sigkill" in lower:
        return FailureClass.CLEANUP_FAILED
    if "request" in lower or "completion" in lower:
        return FailureClass.REQUEST_FAILED
    if "publication" in lower or "overview" in lower or "postprocess" in lower:
        return FailureClass.PUBLICATION_FAILED
    if "port" in lower:
        return FailureClass.PORT_IN_USE
    return FailureClass.SERVER_START_FAILED


def _environment_digest(before: dict[str, object] | None, after: dict[str, object] | None) -> str:
    return hashlib.sha256(canonical_bytes({"before": before, "after": after})).hexdigest()


def _fresh_attempt_validation(root: Path, attempt: AttemptRecord, trials: dict[str, Any]) -> bool:
    trial = trials[attempt.logical_trial_id]
    result = validate_trial(
        root / "trials" / attempt.relative_directory,
        attempt_id=attempt.attempt_id,
        condition=trial.condition.value,
    )
    stored = _load_json(root / "trials" / attempt.relative_directory / "validation.json")
    return result == stored and result.get("valid") is True


def _initialize(config: ExperimentConfig, paths: ExperimentPaths) -> ExperimentCheckpoint:
    hybrid = config.load_hybrid()
    paths.root.mkdir(mode=0o755)
    paths.trials.mkdir()
    _atomic_write(paths.root / "config.json", canonical_config_bytes(config), exclusive=True)
    _write_json(paths.root / "policy.json", _policy(), exclusive=True)
    _atomic_write(paths.root / "schedule.json", canonical_schedule_bytes(config.schedule), exclusive=True)
    _write_json(
        paths.root / "environment.json",
        capture_environment(hybrid, stage="experiment_start"),
        exclusive=True,
    )
    _atomic_write(
        paths.root / "hybrid_config.json",
        config.hybrid_config_path.read_bytes(),
        exclusive=True,
    )
    checkpoint = ExperimentCheckpoint.new(
        config_sha256=config.sha256,
        schedule_sha256=config.schedule.sha256,
        max_hardware_attempts=config.max_hardware_attempts,
    )
    CheckpointStore(paths.checkpoint).initialize(checkpoint)
    return checkpoint


def _resume(config: ExperimentConfig, paths: ExperimentPaths) -> ExperimentCheckpoint:
    if (paths.root / "config.json").read_bytes() != canonical_config_bytes(config):
        raise CheckpointIntegrityError("resume config snapshot mismatch")
    if (paths.root / "schedule.json").read_bytes() != canonical_schedule_bytes(config.schedule):
        raise CheckpointIntegrityError("resume schedule snapshot mismatch")
    if sha256_file(paths.root / "hybrid_config.json") != config.hybrid_config_sha256:
        raise CheckpointIntegrityError("resume hybrid config snapshot mismatch")
    store = CheckpointStore(paths.checkpoint)
    checkpoint = store.load()
    if checkpoint.config_sha256 != config.sha256 or checkpoint.schedule_sha256 != config.schedule.sha256:
        raise CheckpointIntegrityError("resume checkpoint identity mismatch")
    trial_map = schedule_by_logical_id(config.schedule)
    for attempt in checkpoint.attempts:
        if attempt.status is AttemptStatus.SUCCEEDED and not _fresh_attempt_validation(paths.root, attempt, trial_map):
            raise CheckpointIntegrityError(f"successful attempt failed fresh validation: {attempt.attempt_id}")
    running = [item for item in checkpoint.attempts if item.status is AttemptStatus.RUNNING]
    if running:
        if len(running) != 1 or running[0] is not checkpoint.attempts[-1]:
            raise CheckpointIntegrityError("checkpoint contains unsafe running attempt history")
        previous = running[0]
        finalized = replace(
            previous,
            status=AttemptStatus.FAILED,
            failure_class=FailureClass.INTERRUPTED,
            failure_summary="previous invocation ended before attempt finalization",
            artifact_validation_valid=False,
            environment_fingerprint=_environment_digest(None, None),
        )
        checkpoint = checkpoint.with_attempt(finalized)
        store.update(checkpoint)
    return checkpoint


def _execute_attempt(
    *,
    config: ExperimentConfig,
    paths: ExperimentPaths,
    checkpoint: ExperimentCheckpoint,
    logical_trial: Any,
) -> tuple[ExperimentCheckpoint, FailureClass | None]:
    store = CheckpointStore(paths.checkpoint)
    prior = checkpoint.attempts_for(logical_trial.logical_trial_id)
    attempt_number = len(prior) + 1
    attempt_id = logical_trial.attempt_id(attempt_number)
    attempt_root = paths.trial(attempt_id)
    attempt_root.mkdir(parents=False)
    (attempt_root / "runs").mkdir()
    mode, telemetry = CONDITION_MODE[logical_trial.condition]
    _write_json(
        attempt_root / "trial.json",
        {
            "attempt_id": attempt_id,
            "logical_trial": logical_trial.to_dict(),
            "profile_mode": mode,
            "resource_telemetry": telemetry,
            "status": "planned",
        },
        exclusive=True,
    )
    hybrid = config.load_hybrid()
    before: dict[str, object] | None = None
    after: dict[str, object] | None = None
    running = AttemptRecord(
        attempt_id=attempt_id,
        logical_trial_id=logical_trial.logical_trial_id,
        attempt_number=attempt_number,
        status=AttemptStatus.RUNNING,
        relative_directory=attempt_id,
    )
    checkpoint = checkpoint.with_attempt(running)
    store.update(checkpoint)
    failure: FailureClass | None = None
    summary: str | None = None
    valid = False
    try:
        before = wait_for_idle(hybrid)
        _write_json(attempt_root / "environment_before.json", before, exclusive=True)
        validate_hybrid_invocation(
            hybrid,
            run_root=attempt_root / "runs",
            run_id=attempt_id,
            profile_mode=mode,
        )
        result = HybridRunner(
            hybrid,
            run_root=attempt_root / "runs",
            run_id=attempt_id,
            profile_mode=mode,
            enable_telemetry=telemetry,
        ).run()
        if result.status is not RunStatus.SUCCEEDED:
            message = "; ".join(result.errors) or "hybrid runner failed"
            raise ExperimentError(message)
        validation = validate_trial(
            attempt_root,
            attempt_id=attempt_id,
            condition=logical_trial.condition.value,
        )
        _write_json(attempt_root / "validation.json", validation, exclusive=True)
        raw = attempt_root / "runs" / f"{attempt_id}-gpu" / "raw/client/measured_requests.jsonl"
        _write_json(
            attempt_root / "independent_client.json",
            {
                "clock": "CLOCK_MONOTONIC_NS",
                "method_id": "independent_streaming_client_v1",
                "source_relative_path": raw.relative_to(attempt_root).as_posix(),
                "sha256": sha256_file(raw),
                "stores_response_content": False,
            },
            exclusive=True,
        )
        valid = True
    except KeyboardInterrupt:
        failure = FailureClass.INTERRUPTED
        summary = "execution interrupted by user"
    except EnvironmentNotIdleError as error:
        failure = FailureClass.ENVIRONMENT_NOT_IDLE
        summary = str(error)
    except TrialValidationError as error:
        failure = _classify_message(str(error))
        summary = str(error)
        _write_json(
            attempt_root / "validation.json",
            {"schema_version": "1.0", "attempt_id": attempt_id, "valid": False, "failure_class": failure.value, "reason": summary},
            exclusive=True,
        )
    except Exception as error:
        failure = _classify_message(str(error))
        summary = f"{type(error).__name__}: {error}"
    finally:
        try:
            after = capture_environment(hybrid, stage="post_trial")
            _write_json(attempt_root / "environment_after.json", after, exclusive=True)
            remaining = idle_reasons(after)
            if valid and remaining:
                failure = FailureClass.CLEANUP_FAILED
                summary = "; ".join(remaining)
                valid = False
        except Exception as error:
            if failure is None:
                failure = FailureClass.CLEANUP_FAILED
                summary = f"post-trial environment capture failed: {error}"
                valid = False

    fingerprint = _environment_digest(before, after)
    final = replace(
        running,
        status=AttemptStatus.SUCCEEDED if valid else AttemptStatus.FAILED,
        failure_class=None if valid else failure or FailureClass.INTERNAL_ERROR,
        failure_summary=None if valid else (summary or "attempt failed")[:512],
        artifact_validation_valid=valid,
        environment_fingerprint=fingerprint,
    )
    checkpoint = checkpoint.with_attempt(final)
    store.update(checkpoint)
    trial_payload = _load_json(attempt_root / "trial.json")
    trial_payload.update(
        {
            "status": final.status.value,
            "failure_class": final.failure_class.value if final.failure_class else None,
            "failure_summary": final.failure_summary,
            "environment_fingerprint": fingerprint,
        }
    )
    _atomic_write(attempt_root / "trial.json", canonical_bytes(trial_payload))
    if failure is FailureClass.INTERRUPTED:
        raise KeyboardInterrupt
    return checkpoint, failure


def run_experiment(
    *,
    config_path: Path,
    experiment_root: Path,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    hybrid = config.load_hybrid()
    if dry_run:
        return build_plan(config, experiment_root)
    paths = (
        ExperimentPaths.for_resume(experiment_root, immutable_roots=(hybrid.model_path, hybrid.rbln_cache_path))
        if resume
        else ExperimentPaths.plan_new(experiment_root, immutable_roots=(hybrid.model_path, hybrid.rbln_cache_path))
    )
    checkpoint = _resume(config, paths) if resume else _initialize(config, paths)
    trial_map = schedule_by_logical_id(config.schedule)
    successful = {
        item.logical_trial_id
        for item in checkpoint.attempts
        if item.status is AttemptStatus.SUCCEEDED
    }
    formal_gate_validated = False
    for trial in config.schedule.trials:
        if trial.logical_trial_id in successful:
            continue
        if trial.phase is TrialKind.FORMAL and not config.schedule.formal_trials_unlocked(successful):
            raise ExperimentError("formal trials are locked until all pilots succeed")
        if trial.phase is TrialKind.FORMAL and not formal_gate_validated:
            successful_attempts = {
                item.logical_trial_id: item
                for item in checkpoint.attempts
                if item.status is AttemptStatus.SUCCEEDED
            }
            for pilot in config.schedule.pilot_trials:
                attempt = successful_attempts[pilot.logical_trial_id]
                if not _fresh_attempt_validation(paths.root, attempt, trial_map):
                    raise CheckpointIntegrityError(
                        f"pilot failed pre-formal fresh validation: {attempt.attempt_id}"
                    )
            formal_gate_validated = True
            print("experiment pilot fresh validation: 6/6; formal trials unlocked", flush=True)
        prior = checkpoint.attempts_for(trial.logical_trial_id)
        if prior and prior[-1].status is AttemptStatus.FAILED:
            if prior[-1].failure_class not in RETRYABLE or len(prior) >= 2:
                raise ExperimentError(
                    f"logical trial cannot be retried: {trial.logical_trial_id} ({prior[-1].failure_class.value})"
                )
        if len(checkpoint.attempts) >= checkpoint.max_hardware_attempts:
            raise ExperimentError("maximum hardware attempt count reached")
        checkpoint, failure = _execute_attempt(
            config=config,
            paths=paths,
            checkpoint=checkpoint,
            logical_trial=trial,
        )
        latest = checkpoint.attempts_for(trial.logical_trial_id)[-1]
        print(
            f"experiment trial {latest.attempt_id}: {latest.status.value}"
            + (f" ({latest.failure_class.value})" if latest.failure_class else ""),
            flush=True,
        )
        if latest.status is AttemptStatus.SUCCEEDED:
            successful.add(trial.logical_trial_id)
        elif failure in RETRYABLE and latest.attempt_number == 1:
            print(f"experiment retry scheduled: {trial.logical_trial_id}", flush=True)
            checkpoint, failure = _execute_attempt(
                config=config,
                paths=paths,
                checkpoint=checkpoint,
                logical_trial=trial,
            )
            latest = checkpoint.attempts_for(trial.logical_trial_id)[-1]
            print(
                f"experiment retry {latest.attempt_id}: {latest.status.value}"
                + (f" ({latest.failure_class.value})" if latest.failure_class else ""),
                flush=True,
            )
            if latest.status is AttemptStatus.SUCCEEDED:
                successful.add(trial.logical_trial_id)
            else:
                raise ExperimentError(f"retry failed: {latest.attempt_id}")
        else:
            raise ExperimentError(f"trial failed without retry: {latest.attempt_id}")
        if trial.phase is TrialKind.PILOT:
            completed_pilots = sum(item.logical_trial_id in successful for item in config.schedule.pilot_trials)
            print(f"experiment pilot progress: {completed_pilots}/6", flush=True)
        elif all(item.logical_trial_id in successful for item in config.schedule.formal_round(trial.round_index)):
            print(f"experiment formal round {trial.round_index}/5 complete", flush=True)

    validation = validate_experiment(
        paths.root,
        output_path=paths.root / "fresh_validation.json",
    )
    report = generate_report(paths.root)
    return {
        "status": "succeeded",
        "experiment_root": str(paths.root),
        "config_sha256": config.sha256,
        "schedule_sha256": config.schedule.sha256,
        "hardware_attempts": len(checkpoint.attempts),
        "validation": validation,
        "report_sha256": hashlib.sha256(canonical_json(report)).hexdigest(),
    }


def experiment_status(experiment_root: Path) -> dict[str, object]:
    paths = ExperimentPaths.for_resume(experiment_root)
    checkpoint = CheckpointStore(paths.checkpoint).load()
    return {
        "experiment_root": str(paths.root),
        "generation": checkpoint.generation,
        "hardware_attempts": len(checkpoint.attempts),
        "successful": sum(item.status is AttemptStatus.SUCCEEDED for item in checkpoint.attempts),
        "failed": sum(item.status is AttemptStatus.FAILED for item in checkpoint.attempts),
        "running": sum(item.status is AttemptStatus.RUNNING for item in checkpoint.attempts),
        "attempts": [item.to_dict() for item in checkpoint.attempts],
    }


def validate_experiment(
    experiment_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Validate successful trials without modifying the experiment by default.

    A newly completed hardware run may explicitly persist its validation
    evidence with ``output_path``.  Read-only validation and the public
    ``experiment validate`` command leave the existing experiment untouched.
    """

    paths = ExperimentPaths.for_resume(experiment_root)
    config = load_experiment_config(paths.root / "config.json")
    checkpoint = CheckpointStore(paths.checkpoint).load()
    if checkpoint.config_sha256 != config.sha256 or checkpoint.schedule_sha256 != config.schedule.sha256:
        raise CheckpointIntegrityError("experiment identity mismatch")
    trial_map = schedule_by_logical_id(config.schedule)
    results: list[dict[str, object]] = []
    for attempt in checkpoint.attempts:
        if attempt.status is AttemptStatus.SUCCEEDED:
            valid = _fresh_attempt_validation(paths.root, attempt, trial_map)
            results.append({"attempt_id": attempt.attempt_id, "valid": valid})
    valid = len(results) == 36 and all(item["valid"] for item in results)
    result = {
        "schema_version": "1.0",
        "valid": valid,
        "successful_trials_checked": len(results),
        "config_sha256": config.sha256,
        "schedule_sha256": config.schedule.sha256,
        "trials": results,
    }
    if not valid:
        raise CheckpointIntegrityError("experiment fresh validation is incomplete or invalid")
    if output_path is not None:
        output_path = Path(output_path)
        if output_path != paths.root / "fresh_validation.json":
            raise ExperimentError(
                "validation output must be the experiment fresh_validation.json"
            )
        _write_json(output_path, result)
    return result


def _artifact_manifest(root: Path) -> dict[str, object]:
    excluded = {"artifact_manifest.json", "artifact_manifest_validation.json"}
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name not in excluded:
            files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"schema_version": "1.0", "files": files}


def generate_report(
    experiment_root: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, object]:
    paths = ExperimentPaths.for_resume(experiment_root)
    config = load_experiment_config(paths.root / "config.json")
    checkpoint = CheckpointStore(paths.checkpoint).load()
    report = build_report(
        root=paths.root,
        config={
            "experiment_config": config.to_dict(),
            "hybrid": _load_json(paths.root / "hybrid_config.json"),
        },
        schedule=config.schedule,
        checkpoint=checkpoint,
    )
    report_bytes = canonical_json(report)
    html_bytes = render_report_html(report)
    if output_root is not None:
        output = validate_new_output_directory(
            output_root,
            immutable_roots=(paths.root,),
            field="report output",
        )
        output.mkdir(mode=0o755)
        source = {
            "schema_version": "1.0",
            "experiment_root": str(paths.root),
            "config_sha256": config.sha256,
            "schedule_sha256": config.schedule.sha256,
            "checkpoint_sha256": sha256_file(paths.checkpoint),
        }
        for path, data in (
            (output / "report.json", report_bytes),
            (output / "report.html", html_bytes),
            (output / "limitations.json", canonical_json({"limitations": report["limitations"]})),
            (output / "source.json", canonical_json(source)),
        ):
            _atomic_write(path, data, exclusive=True)
        manifest = _artifact_manifest(output)
        manifest_bytes = canonical_json(manifest)
        _atomic_write(output / "artifact_manifest.json", manifest_bytes, exclusive=True)
        current = _artifact_manifest(output)
        validation = {
            "schema_version": "1.0",
            "valid": current == manifest,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "mismatches": [] if current == manifest else ["fresh artifact manifest differs"],
            "report_json_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "report_html_sha256": hashlib.sha256(html_bytes).hexdigest(),
            "deterministic_regeneration": True,
        }
        if not validation["valid"]:
            raise ExperimentError("report output fresh validation failed")
        _write_json(output / "artifact_manifest_validation.json", validation)
        return {
            "status": "succeeded",
            "experiment_root": str(paths.root),
            "output_root": str(output),
            **validation,
        }
    for path, data in (
        (paths.root / "report.json", report_bytes),
        (paths.root / "report.html", html_bytes),
        (paths.root / "limitations.json", canonical_json({"limitations": report["limitations"]})),
    ):
        if path.exists() and path.read_bytes() != data:
            raise ExperimentError(f"deterministic report mismatch: {path.name}")
        if not path.exists():
            _atomic_write(path, data, exclusive=True)
    manifest = _artifact_manifest(paths.root)
    manifest_bytes = canonical_json(manifest)
    manifest_path = paths.root / "artifact_manifest.json"
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise ExperimentError("artifact manifest changed on regeneration")
    if not manifest_path.exists():
        _atomic_write(manifest_path, manifest_bytes, exclusive=True)
    current = _artifact_manifest(paths.root)
    validation = {
        "schema_version": "1.0",
        "valid": current == manifest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "mismatches": [] if current == manifest else ["fresh artifact manifest differs"],
        "report_json_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "report_html_sha256": hashlib.sha256(html_bytes).hexdigest(),
        "deterministic_regeneration": True,
    }
    if not validation["valid"]:
        raise ExperimentError("artifact fresh validation failed")
    _write_json(paths.root / "artifact_manifest_validation.json", validation)
    return report


__all__ = [
    "CONDITION_MODE",
    "RETRYABLE",
    "ExperimentError",
    "build_plan",
    "experiment_status",
    "generate_report",
    "run_experiment",
    "validate_experiment",
]
