"""Build one normalized hybrid bundle from immutable GPU and NPU sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

from ..schema import (
    ArtifactKind,
    ArtifactReference,
    Availability,
    ClockDomain,
    ClockTransform,
    ClockType,
    DeviceType,
    MetricKind,
    MetricSample,
    MetricScope,
    ModelDescriptor,
    Phase,
    ProfileMode,
    RunManifest,
    RunMode,
    RunStatus,
    SoftwareDescriptor,
    SyncMethod,
    SyncPoint,
    ValueOrigin,
    WorkloadDescriptor,
    read_json,
    record_to_dict,
    validate_record,
    write_json,
    write_jsonl,
)
from .alignment import (
    AlignmentError,
    TimestampTransform,
    align_event_stream,
    align_metric_stream,
)
from .clock_sync import (
    ClockEstimate,
    ClockProbeTransport,
    FakeClockProbeTransport,
    LocalClockProbeTransport,
    probe_clock,
    same_clock_estimate,
)
from .config import AlignmentMethod, HybridMergeConfig
from .join import JoinResult, join_requests
from .validation import (
    SourceBundle,
    SourceBundleError,
    classify_hybrid_status,
    load_source_bundle,
    validate_hybrid_records,
)


_PHASE_METRICS = {
    "latency.e2e": ("request_received", "response_done", Phase.REQUEST),
    "latency.prefill": ("prefill_start", "prefill_end", Phase.PREFILL),
    "latency.kv_export": ("kv_export_start", "kv_export_end", Phase.KV_EXPORT),
    "latency.kv_transform": (
        "kv_transform_start",
        "kv_transform_end",
        Phase.KV_TRANSFORM,
    ),
    "latency.kv_transfer": (
        "kv_transfer_start",
        "kv_transfer_end",
        Phase.KV_TRANSFER,
    ),
    "latency.decode": ("decode_loop_start", "decode_loop_end", Phase.DECODE),
    "latency.sampling": ("sampling_start", "sampling_end", Phase.SAMPLING),
}


@dataclass(frozen=True)
class HybridMergeResult:
    run_directory: Path
    status: RunStatus
    event_count: int
    metric_count: int
    artifact_count: int
    joined_request_count: int
    unjoined_request_count: int
    uncertainty_ns: int
    reasons: tuple[str, ...]


class HybridBundleMerger:
    def __init__(
        self,
        config: HybridMergeConfig,
        *,
        clock_transport: ClockProbeTransport | None = None,
        unix_time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.config = config
        self.clock_transport = clock_transport
        self.unix_time_ns = unix_time_ns

    def merge(self) -> HybridMergeResult:
        config = self.config
        paths = config.paths
        try:
            gpu = load_source_bundle(config.gpu_run, DeviceType.GPU)
            npu = load_source_bundle(config.npu_run, DeviceType.NPU)
            device_keys = [
                (device.host_id, device.device_id)
                for source in (gpu, npu)
                for device in source.manifest.devices
            ]
            if len(device_keys) != len(set(device_keys)):
                raise SourceBundleError(
                    "source device identity collision: "
                    "(host_id, device_id) must be unique"
                )
        except SourceBundleError as error:
            paths.create()
            return self._write_failed_source_bundle(str(error))

        if not config.allow_non_fake_sources and not all(
            source.manifest.attributes.get("hybrid.fake_source") is True
            for source in (gpu, npu)
        ):
            reason = "Phase 4A executable merge accepts fake source bundles only"
            return HybridMergeResult(
                run_directory=paths.root,
                status=RunStatus.FAILED,
                event_count=0,
                metric_count=0,
                artifact_count=0,
                joined_request_count=0,
                unjoined_request_count=0,
                uncertainty_ns=0,
                reasons=(reason,),
            )

        paths.create()
        estimate, alignment_available, alignment_reason = self._estimate(gpu, npu)
        validation_errors: list[str] = []
        if not alignment_available:
            validation_errors.append(alignment_reason or "clock alignment failed")

        gpu_transforms = self._timestamp_transforms(
            gpu, offset_ns=0, uncertainty_ns=0, method="same_clock_domain"
        )
        npu_transforms = self._timestamp_transforms(
            npu,
            offset_ns=-estimate.offset_ns,
            uncertainty_ns=estimate.uncertainty_ns,
            method=estimate.method,
            available=alignment_available,
            reason=alignment_reason,
        )
        aligned_events = []
        aligned_metrics = []
        joins: tuple[JoinResult, ...] = ()
        try:
            gpu_events = align_event_stream(
                gpu.events,
                hybrid_run_id=config.run_id,
                source_role="gpu",
                transforms=gpu_transforms,
            )
            npu_events = align_event_stream(
                npu.events,
                hybrid_run_id=config.run_id,
                source_role="npu",
                transforms=npu_transforms,
            )
            aligned_events = sorted(
                [*gpu_events, *npu_events],
                key=lambda event: (event.timestamp_ns, event.event_id),
            )
            aligned_metrics = sorted(
                [
                    *align_metric_stream(
                        gpu.metrics,
                        hybrid_run_id=config.run_id,
                        source_role="gpu",
                        transforms=gpu_transforms,
                    ),
                    *align_metric_stream(
                        npu.metrics,
                        hybrid_run_id=config.run_id,
                        source_role="npu",
                        transforms=npu_transforms,
                    ),
                ],
                key=lambda metric: (metric.timestamp_ns, metric.metric_name),
            )
            joins = join_requests(gpu_events, npu_events)
        except AlignmentError as error:
            validation_errors.append(str(error))
            alignment_available = False

        joined_count = sum(
            result.status in {"joined", "partial"} for result in joins
        )
        unjoined_count = len(joins) - joined_count
        derived_metrics = self._hybrid_metrics(
            joins,
            estimate,
            alignment_accepted=(
                alignment_available
                and estimate.uncertainty_ns <= config.max_uncertainty_ns
            ),
            timestamp_ns=max(
                (event.timestamp_ns for event in aligned_events), default=0
            ),
        )
        all_metrics = [*aligned_metrics, *derived_metrics]
        status, reasons = classify_hybrid_status(
            gpu_status=gpu.manifest.status,
            npu_status=npu.manifest.status,
            joins=joins,
            alignment_available=alignment_available,
            uncertainty_ns=estimate.uncertainty_ns,
            maximum_uncertainty_ns=config.max_uncertainty_ns,
            validation_errors=tuple(validation_errors),
        )

        source_descriptors = self._write_source_descriptors(gpu, npu)
        clock_domains, sync_points, transforms = self._clock_records(
            gpu, npu, estimate
        )
        alignment_path = paths.root / "clocks/clock_alignment.jsonl"
        self._write_alignment_metadata(
            alignment_path,
            gpu,
            npu,
            estimate,
            alignment_available,
            alignment_reason,
        )
        summary_path = paths.root / "summary/hybrid_summary.json"
        self._write_summary(
            summary_path,
            status,
            reasons,
            joins,
            estimate,
            source_descriptors,
        )
        artifacts = [
            self._artifact_reference(
                "gpu-source-descriptor",
                source_descriptors["gpu"],
                ArtifactKind.MANIFEST,
                "hybrid-merger",
                "json",
            ),
            self._artifact_reference(
                "npu-source-descriptor",
                source_descriptors["npu"],
                ArtifactKind.MANIFEST,
                "hybrid-merger",
                "json",
            ),
            self._artifact_reference(
                "clock-alignment",
                alignment_path,
                ArtifactKind.OTHER,
                "hybrid-clock-alignment",
                "jsonl",
            ),
            self._artifact_reference(
                "hybrid-summary",
                summary_path,
                ArtifactKind.OTHER,
                "hybrid-merger",
                "json",
            ),
        ]
        manifest = self._manifest(gpu, npu, status, reasons, joins, estimate)
        validate_hybrid_records(
            config.run_id,
            config.canonical_clock_domain_id,
            [*aligned_events, *all_metrics],
        )
        for record in [manifest, *clock_domains, *sync_points, *transforms, *artifacts]:
            validate_record(record)
        write_jsonl(paths.clock_domains, clock_domains)
        write_jsonl(paths.sync_points, sync_points)
        write_jsonl(paths.transforms, transforms)
        write_jsonl(paths.events, aligned_events)
        write_jsonl(paths.metrics, all_metrics)
        write_jsonl(paths.artifacts, artifacts)
        write_json(paths.manifest, manifest)
        return HybridMergeResult(
            run_directory=paths.root,
            status=status,
            event_count=len(aligned_events),
            metric_count=len(all_metrics),
            artifact_count=len(artifacts),
            joined_request_count=joined_count,
            unjoined_request_count=unjoined_count,
            uncertainty_ns=estimate.uncertainty_ns,
            reasons=reasons,
        )

    def _estimate(
        self, gpu: SourceBundle, npu: SourceBundle
    ) -> tuple[ClockEstimate, bool, str | None]:
        method = self.config.alignment_method
        if method is AlignmentMethod.SAME_CLOCK_DOMAIN:
            gpu_domains = {
                (clock.host_id, clock.clock_domain_id)
                for clock in gpu.clock_domains
            }
            npu_domains = {
                (clock.host_id, clock.clock_domain_id)
                for clock in npu.clock_domains
            }
            if gpu_domains != npu_domains:
                return (
                    same_clock_estimate(),
                    False,
                    "same-clock-domain requires identical host_id/clock_domain_id sets",
                )
            return same_clock_estimate(), True, None
        transport = self.clock_transport
        if transport is None and method is AlignmentMethod.LOCAL:
            transport = LocalClockProbeTransport()
        if transport is None:
            transport = FakeClockProbeTransport(
                offset_ns=self.config.fake_offset_ns,
                delay_ns=self.config.fake_delay_ns,
                jitter_ns=self.config.fake_jitter_ns,
                asymmetry_ns=self.config.fake_asymmetry_ns,
            )
        try:
            estimate = probe_clock(
                transport,
                count=self.config.probe_count,
                minimum_samples=self.config.minimum_probe_samples,
            )
        except (RuntimeError, ValueError) as error:
            return same_clock_estimate(), False, str(error)
        return estimate, True, None

    def _timestamp_transforms(
        self,
        source: SourceBundle,
        *,
        offset_ns: int,
        uncertainty_ns: int,
        method: str,
        available: bool = True,
        reason: str | None = None,
    ) -> dict[str, TimestampTransform]:
        return {
            clock.clock_domain_id: TimestampTransform(
                source_clock_domain_id=clock.clock_domain_id,
                target_clock_domain_id=self.config.canonical_clock_domain_id,
                offset_ns=offset_ns,
                uncertainty_ns=uncertainty_ns,
                method=method,
                available=available,
                reason=reason,
            )
            for clock in source.clock_domains
        }

    def _clock_records(
        self, gpu: SourceBundle, npu: SourceBundle, estimate: ClockEstimate
    ) -> tuple[list[ClockDomain], list[SyncPoint], list[ClockTransform]]:
        run_id = self.config.run_id
        canonical = self.config.canonical_clock_domain_id
        domains = [
            ClockDomain(
                run_id=run_id,
                clock_domain_id=canonical,
                host_id=self.config.coordinator_host_id,
                clock_type=ClockType.MONOTONIC,
                unit="ns",
                monotonic=True,
                adjustable=False,
                attributes={"hybrid.role": "canonical"},
            )
        ]
        transforms: list[ClockTransform] = []
        for role, source, offset, uncertainty, method in (
            ("gpu", gpu, 0, 0, "same_clock_domain"),
            (
                "npu",
                npu,
                -estimate.offset_ns,
                estimate.uncertainty_ns,
                estimate.method,
            ),
        ):
            sync_method = (
                SyncMethod.SHARED_EVENT
                if method == "same_clock_domain"
                else SyncMethod.RPC_MIDPOINT
            )
            for clock in source.clock_domains:
                source_domain = f"{role}:{clock.clock_domain_id}"
                domains.append(
                    replace(
                        clock,
                        run_id=run_id,
                        clock_domain_id=source_domain,
                        attributes={
                            **clock.attributes,
                            "hybrid.original_clock_domain_id": clock.clock_domain_id,
                            "hybrid.source_role": role,
                        },
                    )
                )
                transforms.append(
                    ClockTransform(
                        run_id=run_id,
                        transform_id=f"{role}:{clock.clock_domain_id}:to-canonical",
                        source_clock_domain_id=source_domain,
                        target_clock_domain_id=canonical,
                        scale=1.0,
                        offset_ns=offset,
                        uncertainty_ns=uncertainty,
                        method=sync_method,
                        valid_from_source_ns=0,
                        valid_to_source_ns=None,
                        attributes={
                            "hybrid.method": method,
                            "hybrid.source_role": role,
                        },
                    )
                )
        sync_points: list[SyncPoint] = []
        if estimate.samples:
            selected = estimate.samples[estimate.selected_index]
            sync_points.append(
                SyncPoint(
                    run_id=run_id,
                    sync_point_id="npu-selected-probe",
                    source_clock_domain_id=canonical,
                    target_clock_domain_id=(
                        f"npu:{npu.clock_domains[0].clock_domain_id}"
                    ),
                    source_timestamp_ns=(selected.t0_ns + selected.t3_ns) // 2,
                    target_timestamp_ns=(selected.t1_ns + selected.t2_ns) // 2,
                    method=SyncMethod.RPC_MIDPOINT,
                    uncertainty_ns=estimate.uncertainty_ns,
                    attributes={
                        "hybrid.round_trip_ns": estimate.round_trip_ns,
                        "hybrid.sample_count": estimate.sample_count,
                        "hybrid.selected_sample": estimate.selected_index,
                    },
                )
            )
        return domains, sync_points, transforms

    def _hybrid_metrics(
        self,
        joins: tuple[JoinResult, ...],
        estimate: ClockEstimate,
        *,
        alignment_accepted: bool,
        timestamp_ns: int,
    ) -> list[MetricSample]:
        run_id = self.config.run_id
        clock = self.config.canonical_clock_domain_id
        metrics: list[MetricSample] = []
        for result in joins:
            if result.status not in {"joined", "partial"} or result.request_id is None:
                continue
            by_name = {}
            for event in result.events:
                by_name.setdefault(event.event_name, event)
            for metric_name, (start_name, end_name, phase) in _PHASE_METRICS.items():
                start = by_name.get(start_name)
                end = by_name.get(end_name)
                reason = None
                value = None
                availability = Availability.AVAILABLE
                if not alignment_accepted:
                    availability = Availability.NOT_AVAILABLE
                    reason = "alignment uncertainty exceeds the accepted threshold"
                elif start is None or end is None:
                    availability = Availability.NOT_AVAILABLE
                    reason = f"required marker missing: {start_name if start is None else end_name}"
                elif end.timestamp_ns < start.timestamp_ns:
                    availability = Availability.NOT_AVAILABLE
                    reason = "marker ordering does not permit a duration"
                else:
                    value = end.timestamp_ns - start.timestamp_ns
                metric_timestamp = (
                    end.timestamp_ns
                    if end is not None
                    else max((event.timestamp_ns for event in result.events), default=0)
                )
                uncertainty = max(
                    (
                        int(
                            event.attributes.get(
                                "hybrid.alignment_uncertainty_ns", 0
                            )
                        )
                        for event in (start, end)
                        if event is not None
                    ),
                    default=0,
                )
                metrics.append(
                    MetricSample(
                        run_id=run_id,
                        metric_name=metric_name,
                        metric_kind=MetricKind.DURATION,
                        scope=MetricScope.REQUEST,
                        host_id=self.config.coordinator_host_id,
                        clock_domain_id=clock,
                        timestamp_ns=metric_timestamp,
                        availability=availability,
                        origin=ValueOrigin.DERIVED,
                        unit="ns",
                        value=value,
                        request_id=result.request_id,
                        phase=phase,
                        interval_ns=value,
                        reason=reason,
                        source_event_ids=(
                            [start.event_id, end.event_id]
                            if start is not None and end is not None
                            else None
                        ),
                        dimensions={"hybrid.join_method": result.join_method},
                        attributes={
                            "hybrid.confidence": result.confidence,
                            "hybrid.alignment_uncertainty_ns": uncertainty,
                            "hybrid.duration_confidence": (
                                "low"
                                if value is not None and uncertainty > value
                                else "normal"
                            ),
                        },
                    )
                )
        joined_count = sum(
            result.status in {"joined", "partial"} for result in joins
        )
        unjoined_count = len(joins) - joined_count
        for name, value in (
            ("hybrid.joined_requests", joined_count),
            ("hybrid.unjoined_requests", unjoined_count),
        ):
            metrics.append(
                MetricSample(
                    run_id=run_id,
                    metric_name=name,
                    metric_kind=MetricKind.COUNT,
                    scope=MetricScope.RUN,
                    host_id=self.config.coordinator_host_id,
                    clock_domain_id=clock,
                    timestamp_ns=timestamp_ns,
                    availability=Availability.AVAILABLE,
                    origin=ValueOrigin.DERIVED,
                    unit="requests",
                    value=value,
                    dimensions={},
                    attributes={},
                )
            )
        for name, value in (
            ("hybrid.alignment_offset", estimate.offset_ns),
            ("hybrid.alignment_uncertainty", estimate.uncertainty_ns),
        ):
            metrics.append(
                MetricSample(
                    run_id=run_id,
                    metric_name=name,
                    metric_kind=MetricKind.GAUGE,
                    scope=MetricScope.RUN,
                    host_id=self.config.coordinator_host_id,
                    clock_domain_id=clock,
                    timestamp_ns=timestamp_ns,
                    availability=Availability.AVAILABLE,
                    origin=ValueOrigin.ESTIMATED,
                    unit="ns",
                    value=value,
                    dimensions={"hybrid.source_role": "npu"},
                    attributes={"hybrid.method": estimate.method},
                )
            )
        return metrics

    def _manifest(
        self,
        gpu: SourceBundle,
        npu: SourceBundle,
        status: RunStatus,
        reasons: tuple[str, ...],
        joins: tuple[JoinResult, ...],
        estimate: ClockEstimate,
    ) -> RunManifest:
        hosts = []
        seen_hosts = set()
        for host in [*gpu.manifest.hosts, *npu.manifest.hosts]:
            if host.host_id not in seen_hosts:
                hosts.append(host)
                seen_hosts.add(host.host_id)
        software: list[SoftwareDescriptor] = []
        seen_software = set()
        for item in [*gpu.manifest.software, *npu.manifest.software]:
            key = (item.name, item.version, item.role, item.path)
            if key not in seen_software:
                software.append(item)
                seen_software.add(key)
        gpu_model = replace(gpu.manifest.models[0], role="prefill")
        npu_model = replace(npu.manifest.models[0], role="decode")
        joined_count = sum(
            result.status in {"joined", "partial"} for result in joins
        )
        return RunManifest(
            run_id=self.config.run_id,
            mode=RunMode.HYBRID,
            profile_mode=(
                ProfileMode.DETAILED_PROFILE
                if ProfileMode.DETAILED_PROFILE
                in {gpu.manifest.profile_mode, npu.manifest.profile_mode}
                else ProfileMode.MONITOR
            ),
            status=status,
            created_at_unix_ns=self.unix_time_ns(),
            models=[gpu_model, npu_model],
            workload=WorkloadDescriptor(
                request_count=joined_count,
                concurrency=None,
                request_rate_per_s=None,
                input_tokens=None,
                output_tokens=None,
                max_model_len=None,
                warmup_requests=None,
            ),
            hosts=hosts,
            software=software,
            devices=[*gpu.manifest.devices, *npu.manifest.devices],
            configuration={
                "gpu_source_run": gpu.manifest.run_id,
                "npu_source_run": npu.manifest.run_id,
                "alignment_method": self.config.alignment_method.value,
                "maximum_uncertainty_ns": self.config.max_uncertainty_ns,
                "canonical_clock_domain_id": self.config.canonical_clock_domain_id,
            },
            attributes={
                "hybrid.source_gpu_status": gpu.manifest.status.value,
                "hybrid.source_npu_status": npu.manifest.status.value,
                "hybrid.alignment_offset_ns": estimate.offset_ns,
                "hybrid.alignment_uncertainty_ns": estimate.uncertainty_ns,
                "hybrid.status_reasons": list(reasons),
            },
        )

    def _write_source_descriptors(
        self, gpu: SourceBundle, npu: SourceBundle
    ) -> dict[str, Path]:
        source_dir = self.config.paths.root / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        result = {}
        for role, source, device_type in (
            ("gpu", gpu, "gpu"),
            ("npu", npu, "npu"),
        ):
            path = source_dir / f"{role}-source.json"
            payload = {
                "source_role": role,
                "source_run_id": source.manifest.run_id,
                "source_path": str(source.root),
                "source_manifest_sha256": source.manifest_sha256,
                "host_ids": [host.host_id for host in source.manifest.hosts],
                "device_type": device_type,
                "profile_mode": source.manifest.profile_mode.value,
                "source_status": source.manifest.status.value,
                "source_clock_domains": [
                    clock.clock_domain_id for clock in source.clock_domains
                ],
                "ingested_at_unix_ns": self.unix_time_ns(),
                "source_artifacts": [
                    record_to_dict(artifact) for artifact in source.artifacts
                ],
            }
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result[role] = path
        return result

    def _write_alignment_metadata(
        self,
        path: Path,
        gpu: SourceBundle,
        npu: SourceBundle,
        estimate: ClockEstimate,
        available: bool,
        reason: str | None,
    ) -> None:
        rows = [
            {
                "source_clock_domain_id": gpu.clock_domains[0].clock_domain_id,
                "target_clock_domain_id": self.config.canonical_clock_domain_id,
                "offset_ns": 0,
                "uncertainty_ns": 0,
                "round_trip_ns": 0,
                "sample_count": 0,
                "selected_sample": None,
                "method": "same_clock_domain",
                "measured_at_unix_ns": self.unix_time_ns(),
                "availability": "available",
                "reason": None,
            },
            {
                "source_clock_domain_id": npu.clock_domains[0].clock_domain_id,
                "target_clock_domain_id": self.config.canonical_clock_domain_id,
                "offset_ns": -estimate.offset_ns,
                "uncertainty_ns": estimate.uncertainty_ns,
                "round_trip_ns": estimate.round_trip_ns,
                "sample_count": estimate.sample_count,
                "selected_sample": (
                    estimate.selected_index if estimate.samples else None
                ),
                "method": estimate.method,
                "measured_at_unix_ns": self.unix_time_ns(),
                "availability": "available" if available else "not_available",
                "reason": reason,
            },
        ]
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_summary(
        self,
        path: Path,
        status: RunStatus,
        reasons: tuple[str, ...],
        joins: tuple[JoinResult, ...],
        estimate: ClockEstimate,
        descriptors: dict[str, Path],
    ) -> None:
        payload = {
            "run_id": self.config.run_id,
            "status": status.value,
            "reasons": list(reasons),
            "alignment": {
                "method": estimate.method,
                "offset_ns": estimate.offset_ns,
                "uncertainty_ns": estimate.uncertainty_ns,
                "round_trip_ns": estimate.round_trip_ns,
                "sample_count": estimate.sample_count,
                "selected_sample": estimate.selected_index,
                "maximum_accepted_uncertainty_ns": (
                    self.config.max_uncertainty_ns
                ),
            },
            "sources": {
                role: path.relative_to(self.config.paths.root).as_posix()
                for role, path in descriptors.items()
            },
            "joins": [
                {
                    **{
                        key: value
                        for key, value in asdict(result).items()
                        if key != "events"
                    },
                    "ordering_violations": [
                        asdict(issue) for issue in result.ordering_violations
                    ],
                }
                for result in joins
            ],
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _artifact_reference(
        self,
        artifact_id: str,
        path: Path,
        kind: ArtifactKind,
        producer: str,
        format_name: str,
    ) -> ArtifactReference:
        data = path.read_bytes()
        return ArtifactReference(
            run_id=self.config.run_id,
            artifact_id=artifact_id,
            artifact_kind=kind,
            relative_path=path.relative_to(self.config.paths.root).as_posix(),
            format=format_name,
            producer=producer,
            created_at_unix_ns=self.unix_time_ns(),
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            attributes={},
        )

    def _write_failed_source_bundle(self, reason: str) -> HybridMergeResult:
        """Keep a machine-readable failure when source streams are corrupt."""
        path = self.config.paths.root / "summary/hybrid_summary.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": self.config.run_id,
                    "status": "failed",
                    "reasons": [reason],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return HybridMergeResult(
            run_directory=self.config.paths.root,
            status=RunStatus.FAILED,
            event_count=0,
            metric_count=0,
            artifact_count=0,
            joined_request_count=0,
            unjoined_request_count=0,
            uncertainty_ns=0,
            reasons=(reason,),
        )
