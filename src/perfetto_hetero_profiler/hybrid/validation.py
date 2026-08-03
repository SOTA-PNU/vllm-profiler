"""Source integrity and cross-run validation for hybrid bundle creation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from ..schema import (
    ArtifactReference,
    ClockDomain,
    DeviceType,
    EventRecord,
    MetricSample,
    RunManifest,
    RunStatus,
    read_json,
    read_jsonl,
    validate_record,
)
from .join import JoinResult


class SourceBundleError(RuntimeError):
    """A source bundle is missing, corrupt, or internally inconsistent."""


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    manifest: RunManifest
    clock_domains: tuple[ClockDomain, ...]
    events: tuple[EventRecord, ...]
    metrics: tuple[MetricSample, ...]
    artifacts: tuple[ArtifactReference, ...]
    manifest_sha256: str


def _typed(rows: list[object], expected: type, path: Path) -> tuple:
    invalid = next((row for row in rows if not isinstance(row, expected)), None)
    if invalid is not None:
        raise SourceBundleError(f"{path} contains an unexpected record type")
    return tuple(rows)


def load_source_bundle(root: Path, expected_device_type: DeviceType) -> SourceBundle:
    root = Path(root)
    required = {
        "manifest": root / "manifest.json",
        "clock": root / "clocks/clock_domains.jsonl",
        "events": root / "events/events.jsonl",
        "metrics": root / "metrics/metrics.jsonl",
        "artifacts": root / "artifacts/artifacts.jsonl",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise SourceBundleError(
            f"source bundle is missing required file(s): {', '.join(missing)}"
        )
    try:
        manifest = read_json(required["manifest"])
        if not isinstance(manifest, RunManifest):
            raise SourceBundleError("source manifest has the wrong record type")
        clocks = _typed(read_jsonl(required["clock"]), ClockDomain, required["clock"])
        events = _typed(read_jsonl(required["events"]), EventRecord, required["events"])
        metrics = _typed(
            read_jsonl(required["metrics"]), MetricSample, required["metrics"]
        )
        artifacts = _typed(
            read_jsonl(required["artifacts"]),
            ArtifactReference,
            required["artifacts"],
        )
    except (OSError, ValueError) as error:
        raise SourceBundleError(f"source schema validation failed: {error}") from error

    if not clocks:
        raise SourceBundleError("source bundle has no clock domain")
    if expected_device_type not in {
        device.device_type for device in manifest.devices
    }:
        raise SourceBundleError(
            f"source manifest does not contain a {expected_device_type.value} device"
        )
    host_ids = {host.host_id for host in manifest.hosts}
    device_keys = {
        (device.host_id, device.device_type, device.device_id)
        for device in manifest.devices
    }
    for collection_name, records in (
        ("clock", clocks),
        ("events", events),
        ("metrics", metrics),
        ("artifacts", artifacts),
    ):
        for record in records:
            if record.run_id != manifest.run_id:
                raise SourceBundleError(
                    f"{collection_name} record run_id does not match source manifest"
                )
    for clock in clocks:
        if clock.host_id not in host_ids:
            raise SourceBundleError(
                f"clock domain {clock.clock_domain_id!r} references an unknown host"
            )
    domains = {clock.clock_domain_id for clock in clocks}
    for event in events:
        if event.host_id not in host_ids:
            raise SourceBundleError(
                f"event {event.event_id!r} references an unknown host"
            )
        if event.clock_domain_id not in domains:
            raise SourceBundleError(
                f"event {event.event_id!r} references an unknown clock domain"
            )
        if event.device_id is not None and (
            event.host_id,
            event.device_type,
            event.device_id,
        ) not in device_keys:
            raise SourceBundleError(
                f"event {event.event_id!r} references an unknown device"
            )
    for metric in metrics:
        if metric.host_id not in host_ids:
            raise SourceBundleError(
                f"metric {metric.metric_name!r} references an unknown host"
            )
        if metric.clock_domain_id not in domains:
            raise SourceBundleError(
                f"metric {metric.metric_name!r} references an unknown clock domain"
            )
        if metric.device_id is not None and (
            metric.host_id,
            metric.device_type,
            metric.device_id,
        ) not in device_keys:
            raise SourceBundleError(
                f"metric {metric.metric_name!r} references an unknown device"
            )
    for artifact in artifacts:
        if artifact.host_id is not None and artifact.host_id not in host_ids:
            raise SourceBundleError(
                f"artifact {artifact.artifact_id!r} references an unknown host"
            )
        path = (root / artifact.relative_path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise SourceBundleError("source artifact escapes its run root") from error
        if not path.is_file():
            raise SourceBundleError(
                f"source artifact does not exist: {artifact.relative_path}"
            )
        size = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if artifact.size_bytes is not None and artifact.size_bytes != size:
            raise SourceBundleError(
                f"source artifact size mismatch: {artifact.relative_path}"
            )
        if artifact.sha256 is not None and artifact.sha256 != digest:
            raise SourceBundleError(
                f"source artifact hash mismatch: {artifact.relative_path}"
            )
    return SourceBundle(
        root=root,
        manifest=manifest,
        clock_domains=clocks,
        events=events,
        metrics=metrics,
        artifacts=artifacts,
        manifest_sha256=hashlib.sha256(required["manifest"].read_bytes()).hexdigest(),
    )


def validate_hybrid_records(
    run_id: str,
    canonical_clock_domain_id: str,
    records: Iterable[object],
) -> None:
    event_ids: set[str] = set()
    for record in records:
        validate_record(record)
        if getattr(record, "run_id", run_id) != run_id:
            raise SourceBundleError("hybrid record run_id mismatch")
        if isinstance(record, EventRecord):
            if record.event_id in event_ids:
                raise SourceBundleError("hybrid event ids must be unique")
            event_ids.add(record.event_id)
            if record.clock_domain_id != canonical_clock_domain_id:
                raise SourceBundleError("hybrid event is not on the canonical clock")
        if isinstance(record, MetricSample):
            if record.clock_domain_id != canonical_clock_domain_id:
                raise SourceBundleError("hybrid metric is not on the canonical clock")


def classify_hybrid_status(
    *,
    gpu_status: RunStatus,
    npu_status: RunStatus,
    joins: tuple[JoinResult, ...],
    alignment_available: bool,
    uncertainty_ns: int,
    maximum_uncertainty_ns: int,
    validation_errors: tuple[str, ...] = (),
) -> tuple[RunStatus, tuple[str, ...]]:
    reasons = list(validation_errors)
    if gpu_status is RunStatus.FAILED or npu_status is RunStatus.FAILED:
        reasons.append("a source run failed")
    if not alignment_available:
        reasons.append("clock alignment is unavailable")
    if validation_errors or not alignment_available or (
        gpu_status is RunStatus.FAILED or npu_status is RunStatus.FAILED
    ):
        return RunStatus.FAILED, tuple(reasons)

    joined = [result for result in joins if result.status in {"joined", "partial"}]
    if not joined:
        reasons.append("no request was joined")
    if gpu_status is RunStatus.PARTIAL or npu_status is RunStatus.PARTIAL:
        reasons.append("a source run is partial")
    if uncertainty_ns > maximum_uncertainty_ns:
        reasons.append("clock uncertainty exceeds the configured maximum")
    if any(result.status != "joined" for result in joins):
        reasons.append("some requests or markers are incomplete")
    if any(
        issue.status == "definite_violation"
        for result in joins
        for issue in result.ordering_violations
    ):
        reasons.append("a definite marker ordering violation was found")
    return (
        (RunStatus.SUCCEEDED if not reasons else RunStatus.PARTIAL),
        tuple(dict.fromkeys(reasons)),
    )
