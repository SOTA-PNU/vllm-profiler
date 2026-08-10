"""Synthetic schema-valid GPU/NPU source bundles for hybrid tests."""

from __future__ import annotations

from pathlib import Path
import hashlib
import time

from perfetto_hetero_profiler.schema import (
    ArtifactKind,
    ArtifactReference,
    Availability,
    ClockDomain,
    ClockType,
    DeviceDescriptor,
    DeviceType,
    EventRecord,
    EventType,
    HostDescriptor,
    MetricKind,
    MetricSample,
    MetricScope,
    ModelDescriptor,
    Phase,
    ProfileMode,
    RunManifest,
    RunMode,
    RunPaths,
    RunStatus,
    SoftwareDescriptor,
    ValueOrigin,
    WorkloadDescriptor,
    write_json,
    write_jsonl,
)


PHASES = {
    "request_received": Phase.REQUEST,
    "prefill_start": Phase.PREFILL,
    "prefill_end": Phase.PREFILL,
    "kv_export_start": Phase.KV_EXPORT,
    "kv_export_end": Phase.KV_EXPORT,
    "kv_transfer_start": Phase.KV_TRANSFER,
    "kv_transfer_end": Phase.KV_TRANSFER,
    "kv_transform_start": Phase.KV_TRANSFORM,
    "kv_transform_end": Phase.KV_TRANSFORM,
    "decode_loop_start": Phase.DECODE,
    "decode_step_start": Phase.DECODE,
    "decode_step_end": Phase.DECODE,
    "sampling_start": Phase.SAMPLING,
    "sampling_end": Phase.SAMPLING,
    "decode_loop_end": Phase.DECODE,
    "first_token_emitted": Phase.RESPONSE,
    "token_emitted": Phase.RESPONSE,
    "response_done": Phase.RESPONSE,
}

GPU_MARKERS = (
    "request_received",
    "prefill_start",
    "prefill_end",
    "kv_export_start",
    "kv_export_end",
    "kv_transfer_start",
    "kv_transfer_end",
    "kv_transform_start",
    "kv_transform_end",
)
NPU_MARKERS = (
    "decode_loop_start",
    "decode_step_start",
    "decode_step_end",
    "sampling_start",
    "sampling_end",
    "decode_loop_end",
    "response_done",
)


def event(
    *,
    run_id: str,
    event_name: str,
    timestamp_ns: int,
    host_id: str,
    clock_domain_id: str,
    request_id: str | None = "request-1",
    event_id: str | None = None,
    device_type: DeviceType | None = None,
    attributes: dict[str, object] | None = None,
) -> EventRecord:
    return EventRecord(
        run_id=run_id,
        event_id=event_id or f"{event_name}-{timestamp_ns}",
        event_name=event_name,
        event_type=EventType.INSTANT,
        phase=PHASES[event_name],
        host_id=host_id,
        clock_domain_id=clock_domain_id,
        timestamp_ns=timestamp_ns,
        request_id=request_id,
        device_type=device_type,
        device_id=(f"{device_type.value}-0" if device_type else None),
        attributes=attributes or {},
    )


def build_source_bundle(
    root: Path,
    *,
    device_type: DeviceType,
    host_id: str,
    clock_domain_id: str,
    markers: tuple[str, ...],
    timestamps: tuple[int, ...] | None = None,
    request_id: str = "request-1",
    status: RunStatus = RunStatus.SUCCEEDED,
    fake: bool = True,
    marker_attributes: dict[str, dict[str, object]] | None = None,
    include_resource_metric: bool = True,
    include_artifact: bool = False,
) -> Path:
    root = Path(root)
    run_id = root.name
    paths = RunPaths(root.parent, run_id)
    paths.create()
    times = timestamps or tuple(
        1_000_000 + index * 100_000 for index in range(len(markers))
    )
    def attributes_for(name: str) -> dict[str, object]:
        attributes = (
            {"decode.step_index": 0}
            if name
            in {
                "decode_step_start",
                "decode_step_end",
                "sampling_start",
                "sampling_end",
            }
            else {}
        )
        attributes.update((marker_attributes or {}).get(name, {}))
        return attributes

    rows = [
        event(
            run_id=run_id,
            event_name=name,
            timestamp_ns=timestamp,
            host_id=host_id,
            clock_domain_id=clock_domain_id,
            request_id=request_id,
            device_type=device_type,
            attributes=attributes_for(name),
        )
        for name, timestamp in zip(markers, times)
    ]
    manifest = RunManifest(
        run_id=run_id,
        mode=(
            RunMode.GPU_ONLY
            if device_type is DeviceType.GPU
            else RunMode.NPU_ONLY
        ),
        profile_mode=ProfileMode.MONITOR,
        status=status,
        created_at_unix_ns=time.time_ns(),
        models=[
            ModelDescriptor(
                role="served",
                model_id="fake-model",
                revision=None,
                tokenizer_id=None,
                dtype=None,
            )
        ],
        workload=WorkloadDescriptor(
            request_count=1,
            concurrency=1,
            request_rate_per_s=None,
            input_tokens=None,
            output_tokens=None,
            max_model_len=None,
            warmup_requests=0,
        ),
        hosts=[
            HostDescriptor(
                host_id=host_id,
                role=device_type.value,
                hostname=host_id,
                operating_system="test",
                architecture="test",
            )
        ],
        software=[
            SoftwareDescriptor(
                name="fake-runtime",
                version="1",
                role=device_type.value,
                path=None,
            )
        ],
        devices=[
            DeviceDescriptor(
                host_id=host_id,
                device_type=device_type,
                device_id=f"{device_type.value}-0",
                vendor="fake",
                model="fake",
                status="available",
                memory_total_bytes=1,
                attributes={},
            )
        ],
        configuration={},
        attributes={"hybrid.fake_source": fake},
    )
    clock = ClockDomain(
        run_id=run_id,
        clock_domain_id=clock_domain_id,
        host_id=host_id,
        clock_type=ClockType.MONOTONIC,
        unit="ns",
        monotonic=True,
        adjustable=False,
        attributes={},
    )
    metrics = []
    if include_resource_metric:
        metric_name = (
            "resource.gpu.utilization"
            if device_type is DeviceType.GPU
            else "resource.npu.utilization"
        )
        metrics.append(
            MetricSample(
                run_id=run_id,
                metric_name=metric_name,
                metric_kind=MetricKind.GAUGE,
                scope=MetricScope.DEVICE,
                host_id=host_id,
                clock_domain_id=clock_domain_id,
                timestamp_ns=times[-1] if times else 0,
                availability=Availability.AVAILABLE,
                origin=ValueOrigin.MEASURED,
                unit="percent",
                value=10,
                device_type=device_type,
                device_id=f"{device_type.value}-0",
                dimensions={},
                attributes={},
            )
        )
    artifacts = []
    if include_artifact:
        artifact_path = paths.root / "raw/client/source.log"
        artifact_path.write_text("source\n", encoding="utf-8")
        data = artifact_path.read_bytes()
        artifacts.append(
            ArtifactReference(
                run_id=run_id,
                artifact_id="source-log",
                artifact_kind=ArtifactKind.RAW_LOG,
                relative_path="raw/client/source.log",
                format="text",
                producer="fake",
                created_at_unix_ns=time.time_ns(),
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                attributes={},
            )
        )
    write_json(paths.manifest, manifest)
    write_jsonl(paths.clock_domains, [clock])
    write_jsonl(paths.events, rows)
    write_jsonl(paths.metrics, metrics)
    write_jsonl(paths.artifacts, artifacts)
    return paths.root
