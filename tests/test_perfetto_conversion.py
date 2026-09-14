"""End-to-end Perfetto conversion tests with immutable synthetic inputs."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.perfetto.converter import (
    CONVERSION_MANIFEST_NAME,
    RBLN_NATIVE_TRACE_NAME,
    RBLN_NATIVE_VALIDATION_NAME,
    REQUEST_FOCUSED_TRACE_NAME,
    REQUEST_FOCUSED_VALIDATION_NAME,
    TRACE_ATTRIBUTE_VALIDATION_NAME,
    TRACE_NAME,
    TRACE_VALIDATION_NAME,
    PerfettoConversionConfig,
    convert_perfetto,
    plan_perfetto_conversion,
)
from perfetto_hetero_profiler.perfetto.loader import (
    PerfettoInputError,
    load_hybrid_run,
)
from perfetto_hetero_profiler.perfetto.native_details import NativeDetailError
from perfetto_hetero_profiler.perfetto.native_nsys import (
    _validate_nsys_sqlite_preamble,
)
from perfetto_hetero_profiler.schema import (
    DETACHED_MANIFEST_NAME,
    build_detached_artifact_manifest,
)
from tests.support.perfetto_family import (
    _CLOSEOUT_REQUIRED,
    _OUTPUT_NAMES,
    RUN_ID,
    _build_monitor_family,
    _sha256,
    _tree_state,
)
from tests.support.toolchain import (
    trace_processor_path as _trace_processor_path,
)
from tests.support.toolchain import (
    trace_processor_test_class,
)


@trace_processor_test_class()
class PerfettoConversionIntegrationTests(unittest.TestCase):
    def test_nsys_schema_error_never_publishes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            output = family["runs"] / "invalid-nsys-schema-output"
            with sqlite3.connect(":memory:") as connection:
                connection.execute(
                    "CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)"
                )
                connection.execute(
                    "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
                    ("EXPORT_SCHEMA_VERSION", "3.17.0"),
                )

                def reject_schema(*_):
                    _validate_nsys_sqlite_preamble(connection)

                with mock.patch(
                    "perfetto_hetero_profiler.perfetto.converter."
                    "build_native_detail_plan",
                    side_effect=reject_schema,
                ), self.assertRaisesRegex(NativeDetailError, "unsupported"):
                    convert_perfetto(
                        PerfettoConversionConfig(
                            run_directory=family["hybrid"],
                            output_directory=output,
                            trace_processor_path=_trace_processor_path(),
                            include_native_details=True,
                        )
                    )
            self.assertFalse(output.exists())
            self.assertFalse(
                tuple(
                    family["runs"].glob(
                        f".{output.name}.perfetto-staging-*"
                    )
                )
            )

    def test_default_output_is_a_new_source_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            expected = family["hybrid"].with_name(
                f"{family['hybrid'].name}-perfetto"
            )
            plan = plan_perfetto_conversion(
                PerfettoConversionConfig(
                    run_directory=family["hybrid"],
                    trace_processor_path=_trace_processor_path(),
                )
            )
            self.assertEqual(plan["output_directory"], str(expected))
            self.assertFalse(expected.exists())

    def test_dry_run_conversion_is_deterministic_and_sql_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            input_roots = (
                family["hybrid"],
                family["gpu"],
                family["npu"],
                family["coordinator"],
                family["recovery"],
            )
            before = _tree_state(input_roots)
            first_output = family["runs"] / "perfetto-a"
            second_output = family["runs"] / "perfetto-b"
            dry_output = family["runs"] / "perfetto-dry"
            common = {
                "run_directory": family["hybrid"],
                "trace_processor_path": _trace_processor_path(),
            }

            plan = plan_perfetto_conversion(
                PerfettoConversionConfig(
                    **common,
                    output_directory=dry_output,
                )
            )
            self.assertEqual(plan["status"], "planned")
            self.assertTrue(plan["dry_run"])
            self.assertFalse(plan["hardware_execution"])
            self.assertFalse(dry_output.exists())
            self.assertEqual(_tree_state(input_roots), before)

            first = convert_perfetto(
                PerfettoConversionConfig(
                    **common,
                    output_directory=first_output,
                )
            )
            second = convert_perfetto(
                PerfettoConversionConfig(
                    **common,
                    output_directory=second_output,
                )
            )
            self.assertEqual(first["status"], "succeeded")
            self.assertEqual(second["status"], "succeeded")
            self.assertEqual(
                {path.name for path in first_output.iterdir()},
                _OUTPUT_NAMES,
            )
            self.assertEqual(
                {path.name for path in second_output.iterdir()},
                _OUTPUT_NAMES,
            )
            self.assertEqual(_tree_state(input_roots), before)

            for name in (
                TRACE_NAME,
                CONVERSION_MANIFEST_NAME,
                TRACE_VALIDATION_NAME,
                TRACE_ATTRIBUTE_VALIDATION_NAME,
            ):
                self.assertEqual(
                    (first_output / name).read_bytes(),
                    (second_output / name).read_bytes(),
                    name,
                )

            validation = json.loads(
                (first_output / TRACE_VALIDATION_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["mismatches"], [])
            self.assertTrue(
                all(query["matched"] for query in validation["queries"])
            )
            self.assertTrue(
                all("rows" not in query for query in validation["queries"])
            )
            self.assertLess(
                (first_output / TRACE_VALIDATION_NAME).stat().st_size,
                100_000,
            )
            self.assertGreater(validation["counts"]["slices"], 0)
            self.assertGreater(validation["counts"]["step_annotations"], 0)
            self.assertGreater(validation["counts"]["counters"], 0)
            self.assertGreater(validation["counts"]["flows"], 0)
            self.assertGreater(
                validation["counts"]["timeline_summary_hierarchy"],
                0,
            )
            self.assertEqual(
                validation["counts"]["timeline_summary_slices"],
                10,
            )
            self.assertEqual(
                validation["counts"]["timeline_summary_kpis"],
                0,
            )
            self.assertEqual(
                validation["counts"]["timeline_summary_data_quality"],
                0,
            )
            self.assertEqual(validation["counts"]["dangling_flows"], 0)
            self.assertEqual(validation["counts"]["import_errors"], 0)
            validation_text = json.dumps(validation, sort_keys=True)
            self.assertNotIn(str(_trace_processor_path()), validation_text)
            self.assertNotIn(str(first_output), validation_text)

            manifest_text = (
                first_output / CONVERSION_MANIFEST_NAME
            ).read_text(encoding="utf-8")
            self.assertNotIn(str(_trace_processor_path()), manifest_text)
            self.assertNotIn(str(first_output), manifest_text)
            self.assertNotIn(str(family["hybrid"]), manifest_text)
            manifest = json.loads(
                manifest_text
            )
            self.assertFalse(
                manifest["flow_policy"]["timestamp_proximity_fallback"]
            )
            self.assertEqual(
                manifest["trace_mapping"]["mapping_version"],
                "processing-timeline-info-stats-v1",
            )
            self.assertEqual(
                manifest["trace_mapping"]["root_track"]["name"],
                "Heterogeneous LLM Processing",
            )
            self.assertEqual(
                manifest["trace_mapping"]["kpi_presentation"],
                "info_and_stats_trace_attributes_only",
            )
            self.assertEqual(
                manifest["trace_mapping"]["kpi_counter_mapping"],
                [],
            )
            self.assertEqual(
                manifest["trace_mapping"]["flow_policy"][
                    "representative_location"
                ],
                "detailed_tracks_only",
            )
            self.assertEqual(
                manifest["trace_mapping"]["unavailable_handling"][
                    "timeline_counter_policy"
                ],
                "not_emitted",
            )
            self.assertFalse(
                manifest["trace_mapping"]["resource_grouping"][
                    "counter_samples_copied"
                ]
            )
            self.assertEqual(
                manifest["trace"]["sha256"],
                _sha256(first_output / TRACE_NAME),
            )

    def test_one_conversion_publishes_full_and_request_focused_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                overview_metrics=True,
                measured_token_timestamps=(2_100_000, 2_200_000),
            )
            output = family["runs"] / "combined-perfetto"
            result = convert_perfetto(
                PerfettoConversionConfig(
                    run_directory=family["hybrid"],
                    output_directory=output,
                    trace_processor_path=_trace_processor_path(),
                    request_focused=True,
                )
            )
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    *_OUTPUT_NAMES,
                    REQUEST_FOCUSED_TRACE_NAME,
                    REQUEST_FOCUSED_VALIDATION_NAME,
                },
            )
            focused_validation = json.loads(
                (output / REQUEST_FOCUSED_VALIDATION_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(focused_validation["valid"])
            self.assertTrue(
                all(
                    "rows" not in query
                    for query in focused_validation["queries"]
                )
            )
            self.assertLess(
                (output / REQUEST_FOCUSED_VALIDATION_NAME).stat().st_size,
                100_000,
            )

    def test_overwrite_and_input_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            output = family["runs"] / "existing-output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")
            config = PerfettoConversionConfig(
                run_directory=family["hybrid"],
                output_directory=output,
                trace_processor_path=_trace_processor_path(),
            )
            with self.assertRaises(FileExistsError):
                plan_perfetto_conversion(config)
            with self.assertRaises(FileExistsError):
                convert_perfetto(config)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

            linked = family["runs"] / "linked-hybrid"
            os.symlink(family["hybrid"], linked)
            with self.assertRaisesRegex(PerfettoInputError, "symlink"):
                plan_perfetto_conversion(
                    replace(
                        config,
                        run_directory=linked,
                        output_directory=family["runs"] / "unused",
                    )
                )

    def test_cli_dry_run_success_conversion_and_failure_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            dry_output = family["runs"] / "cli-dry"
            output = family["runs"] / "cli-output"
            base_args = [
                "convert",
                "perfetto",
                "--run",
                str(family["hybrid"]),
                "--trace-processor",
                str(_trace_processor_path()),
            ]

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                code = main(
                    [
                        *base_args,
                        "--output",
                        str(dry_output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "planned")
            self.assertFalse(dry_output.exists())

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                code = main([*base_args, "--output", str(output)])
            self.assertEqual(code, 0, stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "succeeded")
            self.assertTrue(output.is_dir())

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                code = main([*base_args, "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("conversion error:", stderr.getvalue())

    def test_rbln_native_profile_is_a_separate_unaligned_perfetto_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                rbln_profile=True,
            )
            output = family["runs"] / "perfetto-rbln"
            loaded = load_hybrid_run(family["hybrid"])
            self.assertEqual(len(loaded.native_envelopes), 1)
            envelope = loaded.native_envelopes[0]
            self.assertEqual(envelope.profiler_type, "npu_rbln")
            self.assertEqual(envelope.source_role, "npu")
            self.assertEqual(envelope.alignment_status, "partial")
            self.assertFalse(envelope.opaque_rbln_pb)

            deferred = plan_perfetto_conversion(
                PerfettoConversionConfig(
                    run_directory=family["hybrid"],
                    output_directory=family["runs"] / "perfetto-rbln-plan",
                    trace_processor_path=_trace_processor_path(),
                )
            )
            deferred_native = deferred["native_profiles"][0]
            self.assertFalse(deferred_native["opaque_rbln_pb"])
            self.assertEqual(
                deferred_native["rbln_pb_classification"],
                "perfetto_compatible_rbln_trace",
            )
            self.assertEqual(
                deferred_native["rbln_pb_structure_analysis"],
                "deferred_to_official_trace_processor",
            )
            self.assertEqual(deferred["separate_native_traces"], [])

            result = convert_perfetto(
                PerfettoConversionConfig(
                    run_directory=family["hybrid"],
                    output_directory=output,
                    trace_processor_path=_trace_processor_path(),
                    include_native_details=True,
                )
            )
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["clock_alignment_status"], "partial")
            self.assertEqual(len(result["native_profiles"]), 1)

            manifest = json.loads(
                (output / CONVERSION_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            native = manifest["native_profiles"][0]
            self.assertEqual(native["profiler_type"], "npu_rbln")
            self.assertEqual(native["alignment_status"], "partial_unaligned")
            self.assertEqual(
                native["rbln_pb_classification"],
                "perfetto_compatible_rbln_trace",
            )
            self.assertEqual(
                native["rbln_pb_structure_analysis"],
                "official_perfetto_protobuf_schema",
            )
            self.assertFalse(native["opaque_rbln_pb"])
            self.assertFalse(native["rbln_pb_raw_bytes_embedded"])
            self.assertFalse(native["native_details_emitted"])
            self.assertTrue(native["separate_native_trace_published"])
            self.assertEqual(
                manifest["rbln_pb_policy"]["classification"],
                "perfetto_compatible_rbln_trace",
            )
            self.assertEqual(
                manifest["rbln_pb_policy"]["structure_analysis"],
                "official_perfetto_protobuf_schema",
            )
            self.assertFalse(manifest["rbln_pb_policy"]["canonical_merge"])
            self.assertTrue(
                manifest["rbln_pb_policy"]["separate_native_trace_published"]
            )
            self.assertEqual(
                native["artifact_references"][0]["root_id"],
                "npu",
            )
            self.assertEqual(
                native["artifact_references"][0]["relative_path"],
                "raw/profiler/report.pb",
            )
            source_payload = (
                family["npu"] / "raw/profiler/report.pb"
            ).read_bytes()
            self.assertEqual(
                (output / RBLN_NATIVE_TRACE_NAME).read_bytes(),
                source_payload,
            )
            self.assertNotEqual(
                (output / TRACE_NAME).read_bytes(),
                source_payload,
            )

            native_validation = json.loads(
                (output / RBLN_NATIVE_VALIDATION_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(native_validation["valid"])
            self.assertEqual(native_validation["counts"]["slice_count"], 1)
            self.assertEqual(native_validation["counts"]["track_count"], 1)
            self.assertEqual(native_validation["counts"]["flow_count"], 1)
            self.assertFalse(
                native_validation["clock_policy"]["timestamp_rebased"]
            )
            self.assertFalse(
                native_validation["clock_policy"]["canonical_merge"]
            )
            self.assertEqual(len(result["separate_native_traces"]), 1)

            validation = json.loads(
                (output / TRACE_VALIDATION_NAME).read_text(encoding="utf-8")
            )
            self.assertTrue(validation["valid"])
            self.assertGreater(validation["counts"]["native_policy"], 0)
            native_query = next(
                query
                for query in validation["queries"]
                if query["name"] == "native_policy"
            )
            self.assertTrue(native_query["matched"])


class DetachedFamilyContractTests(unittest.TestCase):
    def test_synthetic_family_has_fresh_detached_closeout_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(Path(directory))
            loaded = load_hybrid_run(family["hybrid"])
            self.assertEqual(loaded.manifest.run_id, RUN_ID)
            self.assertGreater(loaded.closeout_artifact_count, 0)
            self.assertEqual(
                {item.root_id for item in loaded.root_fingerprints},
                {"coordinator", "gpu", "hybrid", "npu", "recovery"},
            )
            closeout_manifest = json.loads(
                (family["recovery"] / DETACHED_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            rebuilt = build_detached_artifact_manifest(
                {
                    "coordinator": family["coordinator"],
                    "gpu": family["gpu"],
                    "hybrid": family["hybrid"],
                    "npu": family["npu"],
                    "recovery": family["recovery"],
                },
                required_artifacts=_CLOSEOUT_REQUIRED,
            )
            self.assertEqual(closeout_manifest, rebuilt)

    def test_fresh_closeout_cannot_bless_an_incomplete_marker_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            family = _build_monitor_family(
                Path(directory),
                drop_hybrid_marker="prefill_end",
            )
            with self.assertRaisesRegex(
                PerfettoInputError,
                "canonical marker contract is not valid",
            ):
                load_hybrid_run(family["hybrid"])


if __name__ == "__main__":
    unittest.main()
