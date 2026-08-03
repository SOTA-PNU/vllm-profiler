"""Strict runtime-marker ingestion and real-order hybrid merge tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.hybrid import (
    AlignmentMethod,
    CANONICAL_MARKER_PHASES,
    HybridBundleMerger,
    HybridMergeConfig,
    RuntimeMarkerIngestError,
    ingest_runtime_marker_files,
)
from perfetto_hetero_profiler.schema import (
    DeviceType,
    RunPaths,
    RunStatus,
    validate_record,
    write_jsonl,
)

from tests.hybrid_fixtures import (
    GPU_MARKERS,
    NPU_MARKERS,
    build_source_bundle,
)


HOST_ID = "host-0"
CLOCK_DOMAIN_ID = "host-monotonic"
REQUEST_ID = "request-1"


def marker(
    event_name: str,
    timestamp_ns: int,
    *,
    process_role: str = "proxy",
    pid: int = 100,
    thread_id: int = 101,
    source: str | None = None,
    request_id: str = REQUEST_ID,
    attributes: dict[str, object] | None = None,
    correlation_id: str | None = "correlation-1",
    transfer_id: str | None = "transfer-1",
    sequence: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "1.0.0",
        "event_name": event_name,
        "timestamp_ns": timestamp_ns,
        "host_id": HOST_ID,
        "clock_domain_id": CLOCK_DOMAIN_ID,
        "process_role": process_role,
        "pid": pid,
        "thread_id": thread_id,
        "request_id": request_id,
        "phase": CANONICAL_MARKER_PHASES[event_name].value,
        "source": source or f"test.{process_role}",
        "attributes": attributes or {},
    }
    if correlation_id is not None:
        row["correlation_id"] = correlation_id
    if transfer_id is not None:
        row["transfer_id"] = transfer_id
    if sequence is not None:
        row["sequence"] = sequence
    return row


def write_markers(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


class RuntimeMarkerIngestTests(unittest.TestCase):
    def test_multiple_process_files_are_normalized_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            later = write_markers(
                root / "z-proxy.jsonl",
                [
                    {
                        **marker(
                            "response_done",
                            30,
                            process_role="proxy",
                            sequence=2,
                        ),
                        "remote_request_id_suffix": "remote-ab12",
                    }
                ],
            )
            earlier = write_markers(
                root / "a-gpu.jsonl",
                [
                    marker(
                        "prefill_start",
                        10,
                        process_role="gpu_worker",
                        pid=200,
                        thread_id=201,
                        sequence=1,
                    )
                ],
            )
            events = ingest_runtime_marker_files(
                [later, earlier],
                run_id="gpu-source",
                expected_host_id=HOST_ID,
                expected_clock_domain_id=CLOCK_DOMAIN_ID,
                process_devices={
                    "proxy": None,
                    "gpu_worker": (DeviceType.GPU, "gpu-0"),
                },
            )

        self.assertEqual(
            [event.event_name for event in events],
            ["prefill_start", "response_done"],
        )
        gpu, proxy = events
        self.assertEqual(gpu.process_id, 200)
        self.assertEqual(gpu.thread_id, 201)
        self.assertIs(gpu.device_type, DeviceType.GPU)
        self.assertEqual(gpu.device_id, "gpu-0")
        self.assertIsNone(proxy.device_type)
        self.assertEqual(proxy.request_id, REQUEST_ID)
        self.assertEqual(
            proxy.attributes["hybrid.remote_request_id_suffix"],
            "remote-ab12",
        )
        self.assertEqual(proxy.attributes["hybrid.correlation_id"], "correlation-1")
        self.assertEqual(proxy.attributes["hybrid.transfer_id"], "transfer-1")
        self.assertEqual(proxy.attributes["hybrid.marker_sequence"], 2)
        self.assertEqual(
            {event.host_id for event in events},
            {HOST_ID},
        )
        self.assertEqual(
            {event.clock_domain_id for event in events},
            {CLOCK_DOMAIN_ID},
        )
        for event in events:
            validate_record(event)

    def test_event_ids_are_deterministic_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_markers(
                Path(directory) / "markers.jsonl",
                [
                    marker("prefill_start", 10),
                    marker("prefill_end", 20),
                ],
            )
            kwargs = {
                "run_id": "gpu-source",
                "expected_host_id": HOST_ID,
                "expected_clock_domain_id": CLOCK_DOMAIN_ID,
            }
            first = ingest_runtime_marker_files([path], **kwargs)
            second = ingest_runtime_marker_files([path], **kwargs)
        self.assertEqual(
            [event.event_id for event in first],
            [event.event_id for event in second],
        )
        self.assertEqual(len({event.event_id for event in first}), 2)

    def test_no_marker_files_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeMarkerIngestError,
            "at least one marker file",
        ):
            ingest_runtime_marker_files(
                [],
                run_id="gpu-source",
                expected_host_id=HOST_ID,
                expected_clock_domain_id=CLOCK_DOMAIN_ID,
            )

    def test_wrong_phase_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            row = marker("prefill_start", 10)
            row["phase"] = "decode"
            path = write_markers(Path(directory) / "markers.jsonl", [row])
            with self.assertRaisesRegex(RuntimeMarkerIngestError, "does not match"):
                ingest_runtime_marker_files(
                    [path],
                    run_id="gpu-source",
                    expected_host_id=HOST_ID,
                    expected_clock_domain_id=CLOCK_DOMAIN_ID,
                )

    def test_unknown_top_level_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            row = marker("prefill_start", 10)
            row["raw_pointer"] = "0x1234"
            path = write_markers(Path(directory) / "markers.jsonl", [row])
            with self.assertRaisesRegex(RuntimeMarkerIngestError, "unknown field"):
                ingest_runtime_marker_files(
                    [path],
                    run_id="gpu-source",
                    expected_host_id=HOST_ID,
                    expected_clock_domain_id=CLOCK_DOMAIN_ID,
                )

    def test_sensitive_and_unnamespaced_attributes_are_rejected(self):
        for attributes, pattern in (
            ({"nixl.raw_pointer": "0x1234"}, "forbidden payload"),
            (
                {"nixl.metadata": {"raw_pointer": "0x1234"}},
                "nixl.metadata.raw_pointer",
            ),
            ({"unnamespaced": 1}, "attribute key must use a namespace"),
        ):
            with self.subTest(attributes=attributes):
                with tempfile.TemporaryDirectory() as directory:
                    path = write_markers(
                        Path(directory) / "markers.jsonl",
                        [marker("prefill_start", 10, attributes=attributes)],
                    )
                    with self.assertRaisesRegex(
                        RuntimeMarkerIngestError,
                        pattern,
                    ):
                        ingest_runtime_marker_files(
                            [path],
                            run_id="gpu-source",
                            expected_host_id=HOST_ID,
                            expected_clock_domain_id=CLOCK_DOMAIN_ID,
                        )

    def test_host_and_clock_mismatch_are_rejected(self):
        for field, value, pattern in (
            ("host_id", "other-host", "host_id"),
            ("clock_domain_id", "other-clock", "clock_domain_id"),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    row = marker("prefill_start", 10)
                    row[field] = value
                    path = write_markers(
                        Path(directory) / "markers.jsonl",
                        [row],
                    )
                    with self.assertRaisesRegex(RuntimeMarkerIngestError, pattern):
                        ingest_runtime_marker_files(
                            [path],
                            run_id="gpu-source",
                            expected_host_id=HOST_ID,
                            expected_clock_domain_id=CLOCK_DOMAIN_ID,
                        )


class RuntimeMarkerBundleTests(unittest.TestCase):
    def _runtime_sources(self, root: Path):
        raw = root / "raw"
        raw.mkdir()
        gpu_proxy = write_markers(
            raw / "gpu-proxy.jsonl",
            [
                marker("request_received", 0, process_role="proxy", sequence=0),
                marker("response_done", 190, process_role="proxy", sequence=1),
            ],
        )
        gpu_worker = write_markers(
            raw / "gpu-worker.jsonl",
            [
                marker("prefill_start", 10, process_role="gpu_worker"),
                marker("prefill_end", 20, process_role="gpu_worker"),
                marker("kv_export_start", 30, process_role="gpu_connector"),
                marker("kv_export_end", 40, process_role="gpu_connector"),
            ],
        )
        npu_connector = write_markers(
            raw / "npu-connector.jsonl",
            [
                marker("kv_transfer_start", 50, process_role="npu_connector"),
                marker("kv_transfer_end", 60, process_role="npu_connector"),
                marker("kv_transform_start", 70, process_role="npu_connector"),
                marker("kv_transform_end", 80, process_role="npu_connector"),
            ],
        )
        npu_worker = write_markers(
            raw / "npu-worker.jsonl",
            [
                marker("decode_loop_start", 90, process_role="npu_worker"),
                marker(
                    "decode_step_start",
                    100,
                    process_role="npu_worker",
                    attributes={"decode.step_index": 0},
                ),
                marker(
                    "decode_step_end",
                    110,
                    process_role="npu_worker",
                    attributes={"decode.step_index": 0},
                ),
                marker(
                    "sampling_start",
                    120,
                    process_role="npu_sampler",
                    attributes={"decode.step_index": 0},
                ),
                marker(
                    "sampling_end",
                    130,
                    process_role="npu_sampler",
                    attributes={"decode.step_index": 0},
                ),
                marker(
                    "decode_step_start",
                    140,
                    process_role="npu_worker",
                    attributes={"decode.step_index": 1},
                ),
                marker(
                    "decode_step_end",
                    150,
                    process_role="npu_worker",
                    attributes={"decode.step_index": 1},
                ),
                marker(
                    "sampling_start",
                    160,
                    process_role="npu_sampler",
                    attributes={"decode.step_index": 1},
                ),
                marker(
                    "sampling_end",
                    170,
                    process_role="npu_sampler",
                    attributes={"decode.step_index": 1},
                ),
                marker("decode_loop_end", 180, process_role="npu_worker"),
            ],
        )

        gpu_source = build_source_bundle(
            root / "gpu-source",
            device_type=DeviceType.GPU,
            host_id=HOST_ID,
            clock_domain_id=CLOCK_DOMAIN_ID,
            markers=GPU_MARKERS,
        )
        npu_source = build_source_bundle(
            root / "npu-source",
            device_type=DeviceType.NPU,
            host_id=HOST_ID,
            clock_domain_id=CLOCK_DOMAIN_ID,
            markers=NPU_MARKERS,
        )
        gpu_events = ingest_runtime_marker_files(
            [gpu_proxy, gpu_worker],
            run_id=gpu_source.name,
            expected_host_id=HOST_ID,
            expected_clock_domain_id=CLOCK_DOMAIN_ID,
            process_devices={
                "proxy": None,
                "gpu_worker": (DeviceType.GPU, "gpu-0"),
                "gpu_connector": (DeviceType.GPU, "gpu-0"),
            },
        )
        npu_events = ingest_runtime_marker_files(
            [npu_connector, npu_worker],
            run_id=npu_source.name,
            expected_host_id=HOST_ID,
            expected_clock_domain_id=CLOCK_DOMAIN_ID,
            process_devices={
                "npu_connector": (DeviceType.NPU, "npu-0"),
                "npu_worker": (DeviceType.NPU, "npu-0"),
                "npu_sampler": (DeviceType.NPU, "npu-0"),
            },
        )
        write_jsonl(
            RunPaths(gpu_source.parent, gpu_source.name).events,
            gpu_events,
            overwrite=True,
        )
        write_jsonl(
            RunPaths(npu_source.parent, npu_source.name).events,
            npu_events,
            overwrite=True,
        )
        return gpu_source, npu_source, gpu_events, npu_events

    def test_ingested_multi_process_markers_make_hybrid_succeeded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_source, npu_source, gpu_events, npu_events = self._runtime_sources(
                root
            )
            result = HybridBundleMerger(
                HybridMergeConfig(
                    run_root=root / "runs",
                    run_id="hybrid",
                    gpu_run=gpu_source,
                    npu_run=npu_source,
                    alignment_method=AlignmentMethod.SAME_CLOCK_DOMAIN,
                )
            ).merge()
            summary = json.loads(
                (result.run_directory / "summary/hybrid_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertIs(result.status, RunStatus.SUCCEEDED)
        self.assertEqual(result.joined_request_count, 1)
        self.assertEqual(result.unjoined_request_count, 0)
        self.assertEqual(
            {event.request_id for event in (*gpu_events, *npu_events)},
            {REQUEST_ID},
        )
        join = summary["joins"][0]
        self.assertEqual(join["join_method"], "request_id")
        self.assertEqual(join["status"], "joined")
        self.assertFalse(join["missing_markers"])
        self.assertFalse(join["duplicate_markers"])
        self.assertFalse(join["pairing_issues"])
        self.assertFalse(join["ordering_violations"])

    def test_missing_runtime_marker_keeps_hybrid_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_source, npu_source, _, npu_events = self._runtime_sources(root)
            incomplete = tuple(
                event
                for event in npu_events
                if event.event_name != "kv_transform_end"
            )
            write_jsonl(
                RunPaths(npu_source.parent, npu_source.name).events,
                incomplete,
                overwrite=True,
            )
            result = HybridBundleMerger(
                HybridMergeConfig(
                    run_root=root / "runs",
                    run_id="hybrid-partial",
                    gpu_run=gpu_source,
                    npu_run=npu_source,
                    alignment_method=AlignmentMethod.SAME_CLOCK_DOMAIN,
                )
            ).merge()
            summary = json.loads(
                (result.run_directory / "summary/hybrid_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertIs(result.status, RunStatus.PARTIAL)
        self.assertIn(
            "kv_transform_end",
            summary["joins"][0]["missing_markers"],
        )


if __name__ == "__main__":
    unittest.main()
