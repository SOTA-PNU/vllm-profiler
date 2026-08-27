"""CPU-only tests for RBLN runtime collection and normalization."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.npu.runtime_collection import (
    NpuRuntimeCollectionConfig,
    _format_name,
    _profile_artifact_kind,
    _profile_files,
    _relocate_vendor_sidecars,
    _replace_jsonl,
    build_runtime_collection_plan,
)
from perfetto_hetero_profiler.schema import read_jsonl
from perfetto_hetero_profiler.npu.workload import (
    NON_TOKEN_REASON,
    measured_window_metrics,
    observation_events,
    observation_metrics,
    parse_observations,
)
from perfetto_hetero_profiler.schema import (
    ArtifactKind,
    Availability,
    ProfileMode,
    validate_record,
)


def summary(*rows):
    return {
        "measured": [
            {
                "request_id": request_id,
                "started_ns": started_ns,
                "ended_ns": ended_ns,
            }
            for request_id, started_ns, ended_ns in rows
        ]
    }


class NpuWorkloadTests(unittest.TestCase):
    def test_parse_observations_preserves_individual_boundaries(self):
        rows = parse_observations(summary(("a", 10, 20), ("b", 30, 55)))
        self.assertEqual([row.latency_ns for row in rows], [10, 25])

    def test_parse_rejects_missing_measured(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            parse_observations({})

    def test_parse_rejects_negative_duration(self):
        with self.assertRaisesRegex(ValueError, "negative duration"):
            parse_observations(summary(("a", 20, 10)))

    def test_events_use_actual_runtime_boundaries(self):
        observation = parse_observations(summary(("a", 10, 20)))[0]
        events = observation_events("run", observation)
        self.assertEqual(
            [(event.event_name, event.timestamp_ns) for event in events],
            [("request_received", 10), ("response_done", 20)],
        )
        for event in events:
            validate_record(event)

    def test_latency_is_per_measured_inference(self):
        observation = parse_observations(summary(("a", 10, 25)))[0]
        metrics = observation_metrics("run", observation)
        latency = next(item for item in metrics if item.metric_name == "latency.e2e")
        self.assertEqual(latency.value, 15)
        self.assertIs(latency.availability, Availability.AVAILABLE)
        for metric in metrics:
            validate_record(metric)

    def test_non_token_ttft_and_tpot_are_unavailable(self):
        observation = parse_observations(summary(("a", 10, 25)))[0]
        metrics = observation_metrics("run", observation)
        unavailable = [
            item for item in metrics if item.metric_name in {"latency.ttft", "latency.tpot"}
        ]
        self.assertEqual(len(unavailable), 2)
        self.assertTrue(
            all(item.availability is Availability.NOT_AVAILABLE for item in unavailable)
        )
        self.assertTrue(all(item.value is None for item in unavailable))
        self.assertTrue(all(item.reason == NON_TOKEN_REASON for item in unavailable))

    def test_throughput_uses_only_measured_window(self):
        rows = parse_observations(summary(("a", 100, 200), ("b", 300, 600)))
        metrics = measured_window_metrics("run", rows)
        throughput = next(
            item for item in metrics if item.metric_name == "throughput.requests"
        )
        self.assertEqual(throughput.interval_ns, 500)
        self.assertEqual(throughput.value, 2 / 0.0000005)
        self.assertTrue(throughput.attributes["rbln.warmup_excluded"])

    def test_empty_window_has_no_derived_metrics(self):
        self.assertEqual(measured_window_metrics("run", ()), [])


class NpuRuntimePlanningTests(unittest.TestCase):
    def config(self, root, mode=ProfileMode.MONITOR):
        return NpuRuntimeCollectionConfig(
            run_root=Path(root),
            run_id="runtime-dry",
            artifact=Path("/tmp/existing.rbln"),
            runtime_python=Path("/opt/runtime/bin/python"),
            profile_mode=mode,
            min_measured_seconds=0,
        )

    def test_plan_is_side_effect_free(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory) / "runs")
            plan = build_runtime_collection_plan(config)
            self.assertFalse(config.run_directory.exists())
            self.assertIs(plan["executes"], False)

    def test_detailed_plan_uses_public_profiler_api(self):
        plan = build_runtime_collection_plan(
            self.config("/tmp/runs", ProfileMode.DETAILED_PROFILE)
        )
        self.assertIs(plan["profiler"]["enabled"], True)
        self.assertIn("activate_profiler=True", plan["profiler"]["public_api"])
        self.assertIn("rebel.profiler.profile", plan["profiler"]["public_api"])
        self.assertNotIn("core_ori", repr(plan))

    def test_invalid_device_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            NpuRuntimeCollectionConfig(
                run_root=Path("/tmp/runs"),
                run_id="bad",
                artifact=Path("/tmp/a.rbln"),
                runtime_python=Path("/bin/python3"),
                device_id=-1,
            )

    def test_profile_discovery_rejects_zero_byte_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty.pb").touch()
            (root / "actual.pb").write_bytes(b"report")
            self.assertEqual(_profile_files(root), (root / "actual.pb",))

    def test_unknown_extension_is_preserved(self):
        self.assertEqual(_format_name(Path("vendor.traceblob")), "traceblob")
        self.assertIs(
            _profile_artifact_kind(Path("vendor.traceblob")),
            ArtifactKind.RBLN_REPORT,
        )

    def test_profiler_diagnostic_log_is_not_a_report(self):
        self.assertIs(
            _profile_artifact_kind(Path("profiler_error.log")),
            ArtifactKind.RAW_LOG,
        )

    def test_new_vendor_sidecar_is_relocated_to_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cwd = root / "cwd"
            output = root / "output"
            cwd.mkdir()
            output.mkdir()
            old = cwd / "profiler_error_old.log"
            old.write_text("old", encoding="utf-8")
            before = {old}
            new = cwd / "profiler_error_new.log"
            new.write_text("new", encoding="utf-8")
            relocated = _relocate_vendor_sidecars(cwd, output, before)
            self.assertEqual(relocated, (output / new.name,))
            self.assertTrue(old.exists())
            self.assertFalse(new.exists())

    def test_postprocess_jsonl_uses_distinct_run_local_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            rows = measured_window_metrics(
                "run", parse_observations(summary(("a", 10, 20)))
            )
            with mock.patch(
                "perfetto_hetero_profiler.npu.runtime_collection.os.replace",
                wraps=os.replace,
            ) as replace_call:
                _replace_jsonl(path, rows)
            source, destination = replace_call.call_args.args
            self.assertNotEqual(source, destination)
            self.assertEqual(destination, path)
            self.assertEqual(read_jsonl(path), rows)
            self.assertFalse(path.with_name(f".{path.name}.postprocess.tmp").exists())

    def test_cli_help_exposes_runtime_boundary(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                main(["collect", "npu-runtime", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--runtime-python", stdout.getvalue())

    def test_cli_dry_run_creates_no_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "collect",
                        "npu-runtime",
                        "--run-root",
                        str(root),
                        "--run-id",
                        "dry",
                        "--artifact",
                        "/tmp/a.rbln",
                        "--runtime-python",
                        "/bin/python3",
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertFalse(root.exists())
            self.assertIs(json.loads(stdout.getvalue())["executes"], False)


if __name__ == "__main__":
    unittest.main()
