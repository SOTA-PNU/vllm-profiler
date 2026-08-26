"""Kineto/PyTorch native trace conversion."""

from __future__ import annotations

from collections import Counter, defaultdict

from .loader import LoadedHybridRun, SourceRunMetadata
from .model import InstantSpec, SliceSpec, TrackSpec
from .native_details import (
    NativeDetailError,
    NativeDetailResult,
    NativeDetailSummary,
    _ChromeEvent,
    _NativeSlice,
    _artifact_path,
    _attach_explicit_flows,
    _chrome_annotations,
    _chrome_category,
    _chrome_category_order,
    _chrome_leaf_identity,
    _chrome_leaf_name,
    _clock_bridge,
    _decimal_or_none,
    _identity,
    _microseconds_to_ns,
    _non_bool_int_or_none,
    _read_alignment,
    _stable_gzip_json,
    _stable_token,
    _stable_uint64,
    _validate_mapped_interval,
)


def chrome_detail_result(
    loaded: LoadedHybridRun,
    source: SourceRunMetadata,
    *,
    profiler_type: str,
    native_clock_domain: str,
    native_timestamp_unit: str,
    host_boundary_uncertainty_ns: int,
) -> NativeDetailResult:
    artifacts = sorted(
        (
            artifact
            for artifact in source.artifacts
            if artifact.clock_domain_id == native_clock_domain
            and artifact.format == "chrome_trace_json_gzip"
        ),
        key=lambda item: item.relative_path,
    )
    if not artifacts:
        raise NativeDetailError(f"{profiler_type} has no Chrome trace artifacts")
    alignment = _read_alignment(source)
    bridge = _clock_bridge(
        loaded,
        source,
        alignment,
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        host_boundary_uncertainty_ns=host_boundary_uncertainty_ns,
    )

    events: list[_ChromeEvent] = []
    base_times: set[int] = set()
    process_names: dict[str, str] = {}
    thread_names: dict[tuple[str, str], str] = {}
    metadata_count = 0
    skipped = 0
    for artifact_index, artifact in enumerate(artifacts):
        path = _artifact_path(source, artifact)
        document = _stable_gzip_json(path, artifact)
        base = document.get("baseTimeNanoseconds")
        if isinstance(base, bool) or not isinstance(base, int) or base < 0:
            raise NativeDetailError("Chrome trace lacks integer baseTimeNanoseconds")
        base_times.add(base)
        raw_events = document.get("traceEvents")
        if not isinstance(raw_events, list):
            raise NativeDetailError("Chrome trace traceEvents must be an array")
        for raw in raw_events:
            if not isinstance(raw, dict):
                raise NativeDetailError("Chrome trace event must be an object")
            phase = str(raw.get("ph", ""))
            category = str(raw.get("cat", ""))
            name = str(raw.get("name", ""))
            pid = _identity(raw.get("pid"))
            tid = _identity(raw.get("tid"))
            args = raw.get("args")
            if not isinstance(args, dict):
                args = {}
            if phase == "M":
                metadata_count += 1
                label = args.get("name")
                if isinstance(label, str) and label:
                    if name == "process_name":
                        process_names[pid] = label
                    elif name == "thread_name":
                        thread_names[(pid, tid)] = label
                continue
            timestamp = _decimal_or_none(raw.get("ts"))
            duration = _decimal_or_none(raw.get("dur"))
            event_id = (
                str(raw["id"]) if raw.get("id") is not None else None
            )
            if phase not in {"X", "i", "I", "s", "f"}:
                skipped += 1
                continue
            events.append(
                _ChromeEvent(
                    artifact_index=artifact_index,
                    phase=phase,
                    category=category,
                    name=name,
                    pid=pid,
                    tid=tid,
                    timestamp=timestamp,
                    duration=duration,
                    event_id=event_id,
                    args=args,
                )
            )
    if len(base_times) != 1:
        raise NativeDetailError("Chrome traces have inconsistent time bases")
    base_time_ns = next(iter(base_times))

    root_key = f"native.{profiler_type}"
    root_name = (
        "GPU Torch native details (partial alignment)"
        if profiler_type == "gpu_torch"
        else "NPU vLLM native details (partial alignment)"
    )
    tracks: dict[str, TrackSpec] = {
        root_key: TrackSpec(
            key=root_key,
            uuid=_stable_uint64(loaded.manifest.run_id, "track", root_key),
            name=root_name,
            kind="group",
            description=(
                "Native profiler events derived from documented Kineto Unix "
                "timestamps and recorded Unix/monotonic samples; alignment is "
                "partial, never exact."
            ),
            parent_key="summary.root",
            child_ordering="explicit",
            sibling_order_rank=5,
        )
    }
    category_keys: dict[str, str] = {}
    native_slices: list[_NativeSlice] = []
    instants: list[InstantSpec] = []
    counts: Counter[str] = Counter()
    flow_host_pids: dict[int, set[str]] = defaultdict(set)
    for event in events:
        _, endpoint_kind = _chrome_category(
            profiler_type,
            event.category,
            event.name,
            event.phase,
        )
        if (
            endpoint_kind == "host_api"
            and _non_bool_int_or_none(
                event.args.get("correlation")
            )
            is not None
        ):
            flow_host_pids[event.artifact_index].add(event.pid)

    category_order = _chrome_category_order(profiler_type)
    chrome_flow_marker_count = 0
    for event in events:
        if event.phase in {"s", "f"}:
            chrome_flow_marker_count += 1
            continue
        if event.timestamp is None:
            raise NativeDetailError("Chrome activity event lacks timestamp")
        unix_ns = base_time_ns + _microseconds_to_ns(event.timestamp)
        timestamp_ns = bridge.unix_to_canonical(unix_ns)
        category_name, endpoint_kind = _chrome_category(
            profiler_type,
            event.category,
            event.name,
            event.phase,
        )
        category_key = category_keys.get(category_name)
        if category_key is None:
            category_key = (
                f"{root_key}.category."
                f"{_stable_token(category_name)}"
            )
            category_keys[category_name] = category_key
            tracks[category_key] = TrackSpec(
                key=category_key,
                uuid=_stable_uint64(
                    loaded.manifest.run_id, "track", category_key
                ),
                name=category_name,
                kind="group",
                description=f"{profiler_type} {category_name} events.",
                parent_key=root_key,
                child_ordering="lexicographic",
                sibling_order_rank=category_order[category_name],
            )
        leaf_identity = _chrome_leaf_identity(event, endpoint_kind)
        leaf_key = (
            f"{category_key}.lane.{_stable_token(leaf_identity)}"
        )
        if leaf_key not in tracks:
            tracks[leaf_key] = TrackSpec(
                key=leaf_key,
                uuid=_stable_uint64(
                    loaded.manifest.run_id, "track", leaf_key
                ),
                name=_chrome_leaf_name(
                    event,
                    endpoint_kind,
                    process_names=process_names,
                    thread_names=thread_names,
                ),
                kind="slice",
                description=(
                    "Original native process/thread/stream identity; timestamp "
                    "point is conditionally mapped and annotated with uncertainty."
                ),
                parent_key=category_key,
            )
        annotations = _chrome_annotations(
            event,
            profiler_type=profiler_type,
            bridge=bridge,
            original_timestamp=event.timestamp,
            original_duration=event.duration,
            process_name=process_names.get(event.pid),
            thread_name=thread_names.get((event.pid, event.tid)),
        )
        correlation = _non_bool_int_or_none(event.args.get("correlation"))
        if event.phase == "X":
            if event.duration is None:
                raise NativeDetailError("Chrome X event lacks duration")
            duration_ns = _microseconds_to_ns(event.duration)
            if duration_ns == 0:
                instant_annotations = dict(annotations)
                instant_annotations[
                    "hetero.native_zero_duration_complete_event"
                ] = True
                instants.append(
                    InstantSpec(
                        track_key=leaf_key,
                        name=event.name,
                        timestamp_ns=timestamp_ns,
                        annotations=tuple(
                            sorted(instant_annotations.items())
                        ),
                    )
                )
                counts[
                    f"{category_name} (zero-duration complete instant)"
                ] += 1
                continue
            host_pids = flow_host_pids[event.artifact_index]
            if len(host_pids) == 1:
                correlation_scope = (
                    f"artifact:{event.artifact_index}:host-pid:"
                    f"{next(iter(host_pids))}"
                )
            elif endpoint_kind == "host_api":
                correlation_scope = (
                    f"artifact:{event.artifact_index}:host-pid:{event.pid}"
                )
            else:
                correlation_scope = (
                    f"artifact:{event.artifact_index}:"
                    "device-without-unique-host-process"
                )
            native_slices.append(
                _NativeSlice(
                    spec=SliceSpec(
                        track_key=leaf_key,
                        name=event.name,
                        timestamp_ns=timestamp_ns,
                        duration_ns=duration_ns,
                        annotations=annotations,
                    ),
                    category=category_name,
                    correlation_id=correlation,
                    endpoint_kind=endpoint_kind,
                    correlation_scope=correlation_scope,
                )
            )
            counts[category_name] += 1
        else:
            instants.append(
                InstantSpec(
                    track_key=leaf_key,
                    name=event.name,
                    timestamp_ns=timestamp_ns,
                    annotations=annotations,
                )
            )
            counts[f"{category_name} (instant)"] += 1

    skipped += chrome_flow_marker_count
    if chrome_flow_marker_count:
        counts["Chrome flow markers (not emitted)"] = (
            chrome_flow_marker_count
        )
    native_slices, flows = _attach_explicit_flows(
        loaded.manifest.run_id,
        profiler_type,
        native_slices,
    )
    valid_interval, mapped_interval = _validate_mapped_interval(
        alignment,
        bridge,
        tuple(item.spec for item in native_slices),
        tuple(instants),
    )
    summary = NativeDetailSummary(
        profiler_type=profiler_type,
        source_role=source.source_role,
        support_status="converted",
        alignment_status="partial_derived",
        alignment_method=(
            "documented_kineto_unix_time_plus_recorded_host_clock_samples"
        ),
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        emitted_event_count=len(native_slices) + len(instants),
        emitted_slice_count=len(native_slices),
        emitted_instant_count=len(instants),
        emitted_flow_count=len(flows),
        metadata_only_event_count=metadata_count,
        skipped_event_count=skipped,
        timestamp_fallback_count=0,
        fabricated_event_count=0,
        alignment_uncertainty_ns=bridge.uncertainty_ns,
        clock_offset_ns=bridge.offset_ns,
        observed_offset_half_range_ns=bridge.observed_half_range_ns,
        native_epoch_base_ns=base_time_ns,
        clock_sample_offsets_ns=bridge.sample_offsets_ns,
        canonical_transform_offset_ns=bridge.canonical_offset_ns,
        clock_formula=(
            "canonical_ns = baseTimeNanoseconds + Decimal(ts_us)*1000 "
            "- clock_offset_ns + canonical_transform_offset_ns"
        ),
        alignment_valid_interval_ns=valid_interval,
        mapped_event_interval_ns=mapped_interval,
        event_counts=tuple(sorted(counts.items())),
        artifact_count=len(artifacts),
        artifact_sha256=tuple(item.sha256 for item in artifacts),
        notes=(
            "baseTimeNanoseconds + exact Decimal(ts_us)*1000 reconstructs Unix ns",
            "Unix/monotonic samples are non-atomic; alignment remains partial",
            "reported uncertainty is not a proven clock-error bound",
            (
                "Chrome s/f markers are counted but not emitted as "
                "API-to-device flows"
            ),
            (
                "no event is classified as NPU device execution; "
                "unrecognized source events remain execution-domain unverified"
                if profiler_type == "npu_vllm"
                else "CUDA flows require one unique explicit correlation ID"
            ),
        ),
    )
    return NativeDetailResult(
        tracks=tuple(tracks.values()),
        slices=tuple(item.spec for item in native_slices),
        instants=tuple(instants),
        flows=flows,
        summaries=(summary,),
    )


__all__ = ["chrome_detail_result"]
