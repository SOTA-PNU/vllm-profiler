"""RBLN native Perfetto trace validation and publication."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
import re
from typing import Any

from google.protobuf.message import DecodeError
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

from .loader import SourceRunMetadata
from .native_details import (
    NativeDetailError,
    NativeDetailResult,
    NativeDetailSummary,
    NativeTraceView,
    _artifact_path,
    _stable_file_identity,
)


def rbln_flow_edge_count(
    endpoints: Mapping[int, Sequence[tuple[int, int, bool]]],
) -> int:
    """Count timestamp-directed Perfetto flow edges without inventing links."""

    edge_count = 0
    for flow_id, rows in endpoints.items():
        active = False
        for _timestamp_ns, _packet_index, terminating in sorted(rows):
            if terminating:
                if not active:
                    raise NativeDetailError(
                        "RBLN Perfetto flow terminates before it starts "
                        f"(flow_id={flow_id})"
                    )
                edge_count += 1
                active = False
            elif active:
                edge_count += 1
            else:
                active = True
    return edge_count


def rbln_native_only_result(
    source: SourceRunMetadata,
    *,
    native_clock_domain: str,
    native_timestamp_unit: str,
) -> NativeDetailResult:
    artifacts = sorted(
        (
            artifact
            for artifact in source.artifacts
            if artifact.clock_domain_id == native_clock_domain
            and artifact.relative_path.endswith(".pb")
        ),
        key=lambda item: item.relative_path,
    )
    if not artifacts:
        raise NativeDetailError("npu_rbln has no PB artifacts")
    parsed: list[tuple[Any, int, int, int, int, int]] = []
    payloads: dict[str, bytes] = {}
    for artifact in artifacts:
        path = _artifact_path(source, artifact)
        _stable_file_identity(path, artifact)
        before = path.lstat()
        payload = path.read_bytes()
        payloads[artifact.relative_path] = payload
        after = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise NativeDetailError("RBLN PB changed while parsing")
        trace = Trace()
        try:
            trace.ParseFromString(payload)
        except DecodeError as error:
            raise NativeDetailError(
                "RBLN PB is not a standard Perfetto Trace protobuf"
            ) from error
        descriptor_uuids = {
            packet.track_descriptor.uuid
            for packet in trace.packet
            if packet.HasField("track_descriptor")
            and packet.track_descriptor.uuid
        }
        depths: Counter[int] = Counter()
        flow_endpoints: defaultdict[int, list[tuple[int, int, bool]]] = (
            defaultdict(list)
        )
        begin_count = 0
        end_count = 0
        instant_count = 0
        used_track_uuids: set[int] = set()
        for packet_index, packet in enumerate(trace.packet):
            if not packet.HasField("track_event"):
                continue
            event = packet.track_event
            if event.type not in {
                TrackEvent.TYPE_SLICE_BEGIN,
                TrackEvent.TYPE_SLICE_END,
                TrackEvent.TYPE_INSTANT,
            }:
                if event.flow_ids or event.terminating_flow_ids:
                    raise NativeDetailError(
                        "RBLN Perfetto flow uses an unsupported TrackEvent type"
                    )
                continue
            if event.track_uuid == 0:
                raise NativeDetailError(
                    "RBLN Perfetto slice lacks an explicit track UUID"
                )
            used_track_uuids.add(event.track_uuid)
            continuing = tuple(int(value) for value in event.flow_ids)
            terminating = tuple(
                int(value) for value in event.terminating_flow_ids
            )
            if (
                any(value <= 0 for value in (*continuing, *terminating))
                or len(set(continuing)) != len(continuing)
                or len(set(terminating)) != len(terminating)
                or set(continuing).intersection(terminating)
            ):
                raise NativeDetailError(
                    "RBLN Perfetto flow identifiers are invalid"
                )
            if continuing or terminating:
                if not packet.HasField("timestamp"):
                    raise NativeDetailError(
                        "RBLN Perfetto flow endpoint lacks an absolute timestamp"
                    )
                for flow_id in continuing:
                    flow_endpoints[flow_id].append(
                        (int(packet.timestamp), packet_index, False)
                    )
                for flow_id in terminating:
                    flow_endpoints[flow_id].append(
                        (int(packet.timestamp), packet_index, True)
                    )
            if event.type == TrackEvent.TYPE_INSTANT:
                instant_count += 1
                continue
            if event.type == TrackEvent.TYPE_SLICE_BEGIN:
                begin_count += 1
                depths[event.track_uuid] += 1
            else:
                end_count += 1
                depths[event.track_uuid] -= 1
                if depths[event.track_uuid] < 0:
                    raise NativeDetailError(
                        "RBLN Perfetto slice stream closes before it opens"
                    )
        expected_flow_count = rbln_flow_edge_count(flow_endpoints)
        descriptor_count = sum(
            packet.HasField("track_descriptor") for packet in trace.packet
        )
        clock_snapshot_count = sum(
            packet.HasField("clock_snapshot") for packet in trace.packet
        )
        if clock_snapshot_count:
            raise NativeDetailError(
                "RBLN clock snapshots require an explicit clock-mapping policy"
            )
        if (
            begin_count <= 0
            or begin_count != end_count
            or descriptor_count <= 0
            or any(depths.values())
            or not used_track_uuids.issubset(descriptor_uuids)
        ):
            raise NativeDetailError(
                "RBLN PB lacks a balanced standard Perfetto slice stream"
            )
        parsed.append(
            (
                artifact,
                begin_count + instant_count,
                descriptor_count,
                len(used_track_uuids),
                expected_flow_count,
                clock_snapshot_count,
            )
        )
    aggregate_rows = []
    for candidate in parsed:
        candidate_path = PurePosixPath(candidate[0].relative_path)
        shard_pattern = re.compile(
            rf"{re.escape(candidate_path.stem)}_\d+\.pb$"
        )
        direct_shards = [
            item
            for item in parsed
            if PurePosixPath(item[0].relative_path).parent
            == candidate_path.parent
            and shard_pattern.fullmatch(
                PurePosixPath(item[0].relative_path).name
            )
        ]
        if len(parsed) == 1 or len(direct_shards) == len(parsed) - 1:
            aggregate_rows.append(candidate)
    if len(aggregate_rows) != 1:
        raise NativeDetailError(
            "RBLN capture must have exactly one unnumbered aggregate PB"
        )
    aggregate = aggregate_rows[0]
    aggregate_slices = aggregate[1]
    aggregate_flows = aggregate[4]
    shard_slices = sum(
        item[1]
        for item in parsed
        if item[0].relative_path != aggregate[0].relative_path
    )
    if len(parsed) > 1 and aggregate_slices != shard_slices:
        raise NativeDetailError(
            "RBLN aggregate/shard Perfetto slice counts do not reconcile"
        )
    shard_flows = sum(
        item[4]
        for item in parsed
        if item[0].relative_path != aggregate[0].relative_path
    )
    if len(parsed) > 1 and aggregate_flows != shard_flows:
        raise NativeDetailError(
            "RBLN aggregate/shard Perfetto flow counts do not reconcile"
        )
    clock_snapshots = sum(item[5] for item in parsed)
    summary = NativeDetailSummary(
        profiler_type="npu_rbln",
        source_role=source.source_role,
        support_status="separate_native_perfetto_trace_unaligned",
        alignment_status="partial_unaligned",
        alignment_method="none_no_clock_snapshot_or_shared_anchor",
        native_clock_domain=native_clock_domain,
        native_timestamp_unit=native_timestamp_unit,
        emitted_event_count=0,
        emitted_slice_count=0,
        emitted_instant_count=0,
        emitted_flow_count=0,
        metadata_only_event_count=0,
        skipped_event_count=0,
        timestamp_fallback_count=0,
        fabricated_event_count=0,
        alignment_uncertainty_ns=None,
        clock_offset_ns=None,
        observed_offset_half_range_ns=None,
        native_epoch_base_ns=None,
        clock_sample_offsets_ns=(),
        canonical_transform_offset_ns=None,
        clock_formula=None,
        alignment_valid_interval_ns=None,
        mapped_event_interval_ns=None,
        event_counts=(
            ("aggregate_perfetto_flow_count", aggregate_flows),
            ("aggregate_perfetto_slice_count", aggregate_slices),
            ("aggregate_track_descriptor_packet_count", aggregate[2]),
            ("aggregate_used_track_count", aggregate[3]),
            ("clock_snapshot_count", clock_snapshots),
            ("shard_perfetto_flow_count", shard_flows),
            ("shard_perfetto_slice_count", shard_slices),
        ),
        artifact_count=len(artifacts),
        artifact_sha256=tuple(item.sha256 for item in artifacts),
        notes=(
            "official Perfetto protobuf schema parses every PB artifact",
            "unnumbered PB is the aggregate; numbered PB files are shards",
            "no clock_snapshot or canonical anchor; canonical merge is forbidden",
            "aggregate PB is published byte-identically as a separate native timeline",
        ),
    )
    aggregate_artifact = aggregate[0]
    view = NativeTraceView(
        profiler_type="npu_rbln",
        source_role=source.source_role,
        source_relative_path=aggregate_artifact.relative_path,
        output_name="trace.rbln-native.pftrace",
        validation_name="trace.rbln-native.validation.json",
        payload=payloads[aggregate_artifact.relative_path],
        size_bytes=aggregate_artifact.size_bytes,
        sha256=aggregate_artifact.sha256,
        expected_slice_count=aggregate_slices,
        expected_track_count=aggregate[3],
        expected_flow_count=aggregate_flows,
    )
    return NativeDetailResult(
        summaries=(summary,),
        separate_traces=(view,),
    )


__all__ = ["rbln_flow_edge_count", "rbln_native_only_result"]
