from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from perfetto_hetero_profiler.hybrid.layout import HybridRunLayout
from tools.evaluation.all_mode_overhead import (
    CONDITIONS,
    _CollectionOnlyHybridRunner,
    _metric_rows,
    _profile_operational_metrics,
    _record_block_failure,
    analyze_sampling_stream,
    build_all_mode_schedule,
    build_campaign_report,
    generate_campaign_report,
    validate_mode_artifacts,
)


def _row(sequence: int, timestamp_ns: int, role: str, metric: str) -> dict[str, object]:
    return {
        "metric_name": metric,
        "timestamp_ns": timestamp_ns,
        "attributes": {
            "telemetry.sample_sequence": sequence,
            "telemetry.sample_role": role,
        },
    }


class AllModeScheduleTests(unittest.TestCase):
    def test_schedule_is_deterministic_balanced_and_has_no_retries(self) -> None:
        first = build_all_mode_schedule(20260907)
        second = build_all_mode_schedule(20260907)
        self.assertEqual(first, second)
        self.assertEqual(first["formal_condition_blocks"], 30)
        self.assertEqual(first["automatic_retries"], 0)
        blocks = first["blocks"]
        assert isinstance(blocks, list)
        for round_index in range(1, 6):
            conditions = {
                item["condition"]
                for item in blocks
                if item["round"] == round_index
            }
            self.assertEqual(conditions, set(CONDITIONS))
        self.assertNotEqual(
            [item["condition"] for item in blocks[:6]],
            [item["condition"] for item in build_all_mode_schedule(7)["blocks"][:6]],
        )

    def test_invalid_seed_is_rejected(self) -> None:
        for seed in (True, -1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                build_all_mode_schedule(seed)  # type: ignore[arg-type]

    def test_three_round_schedule_has_exactly_eighteen_blocks(self) -> None:
        schedule = build_all_mode_schedule(20260909, formal_rounds=3)
        self.assertEqual(schedule["formal_rounds"], 3)
        self.assertEqual(schedule["formal_condition_blocks"], 18)
        blocks = schedule["blocks"]
        assert isinstance(blocks, list)
        for round_index in range(1, 4):
            self.assertEqual(
                {
                    item["condition"]
                    for item in blocks
                    if item["round"] == round_index
                },
                set(CONDITIONS),
            )

    def test_invalid_formal_round_count_is_rejected(self) -> None:
        for rounds in (True, 0, 21):
            with self.subTest(rounds=rounds), self.assertRaises(ValueError):
                build_all_mode_schedule(7, formal_rounds=rounds)  # type: ignore[arg-type]


class OneHertzSamplingTests(unittest.TestCase):
    def _valid_rows(self) -> list[dict[str, object]]:
        batches = [(1, 0, "baseline")]
        batches.extend((index + 2, (index + 1) * 1_000_000_000, "background") for index in range(12))
        batches.append((14, 13_000_000_000, "final"))
        rows: list[dict[str, object]] = []
        for sequence, timestamp, role in batches:
            rows.append(_row(sequence, timestamp, role, "resource.gpu.utilization"))
            rows.append(_row(sequence, timestamp, role, "resource.gpu.memory_used"))
        return rows

    def test_metric_fanout_is_one_batch_and_ten_request_samples_are_valid(self) -> None:
        result = analyze_sampling_stream(
            self._valid_rows(),
            request_start_ns=2_500_000_000,
            request_end_ns=12_500_000_000,
            configured_interval_ms=1000,
            minimum_background_samples=10,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["metric_record_count"], 28)
        self.assertEqual(result["sample_batch_count"], 14)
        self.assertEqual(result["request_window_background_sample_count"], 10)
        self.assertEqual(result["duplicate_count"], 0)
        self.assertEqual(result["drop_count"], 0)
        self.assertEqual(result["actual_interval_ns"]["median"], 1_000_000_000)

    def test_wrong_configuration_drop_duplicate_and_late_background_fail(self) -> None:
        rows = self._valid_rows()
        wrong = analyze_sampling_stream(
            rows,
            request_start_ns=2_500_000_000,
            request_end_ns=12_500_000_000,
            configured_interval_ms=999,
            minimum_background_samples=10,
        )
        self.assertFalse(wrong["valid"])

        dropped = [item for item in rows if item["attributes"]["telemetry.sample_sequence"] != 7]
        drop_result = analyze_sampling_stream(
            dropped,
            request_start_ns=2_500_000_000,
            request_end_ns=12_500_000_000,
            configured_interval_ms=1000,
            minimum_background_samples=10,
        )
        self.assertFalse(drop_result["valid"])
        self.assertEqual(drop_result["drop_count"], 1)

        duplicate = rows + [_row(5, 99_000_000_000, "background", "resource.gpu.power")]
        duplicate_result = analyze_sampling_stream(
            duplicate,
            request_start_ns=2_500_000_000,
            request_end_ns=12_500_000_000,
            configured_interval_ms=1000,
            minimum_background_samples=10,
        )
        self.assertFalse(duplicate_result["valid"])
        self.assertEqual(duplicate_result["duplicate_count"], 1)

        late = rows + [_row(15, 14_000_000_000, "background", "resource.gpu.utilization")]
        late_result = analyze_sampling_stream(
            late,
            request_start_ns=2_500_000_000,
            request_end_ns=12_500_000_000,
            configured_interval_ms=1000,
            minimum_background_samples=10,
        )
        self.assertFalse(late_result["valid"])
        self.assertTrue(late_result["background_after_final"])

    def test_stage_resource_aggregates_are_not_periodic_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = HybridRunLayout(Path(temporary), "run")
            for root in (layout.gpu, layout.npu):
                (root / "metrics").mkdir(parents=True)
            sample = _row(1, 100, "baseline", "resource.gpu.utilization")
            aggregate = {
                "metric_name": "resource.npu.memory_used",
                "timestamp_ns": 100,
                "attributes": {"aggregation.scope": "prefill"},
            }
            (layout.gpu / "metrics/metrics.jsonl").write_text(
                "\n".join(json.dumps(item) for item in (sample, aggregate)) + "\n",
                encoding="utf-8",
            )
            (layout.npu / "metrics/metrics.jsonl").write_text("", encoding="utf-8")
            rows = _metric_rows(layout)
            self.assertEqual(rows["gpu"], [sample])
            self.assertEqual(rows["npu"], [])

    def test_cpu_and_system_metrics_do_not_contaminate_npu_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = HybridRunLayout(Path(temporary), "run")
            for root in (layout.gpu, layout.npu):
                (root / "metrics").mkdir(parents=True)
            gpu = _row(1, 100, "baseline", "resource.gpu.utilization")
            cpu = _row(1, 100, "baseline", "resource.cpu.utilization")
            system = _row(1, 100, "baseline", "resource.system.memory_used")
            npu = _row(1, 101, "baseline", "resource.npu.utilization")
            (layout.gpu / "metrics/metrics.jsonl").write_text(
                "\n".join(json.dumps(item) for item in (gpu, cpu, system)) + "\n",
                encoding="utf-8",
            )
            (layout.npu / "metrics/metrics.jsonl").write_text(
                json.dumps(npu) + "\n",
                encoding="utf-8",
            )

            rows = _metric_rows(layout)

            self.assertEqual(rows["gpu"], [gpu])
            self.assertEqual(rows["npu"], [npu])
            self.assertEqual(rows["system"], [cpu, system])


class DeferredPostprocessTests(unittest.TestCase):
    def test_collection_runner_records_deferral_without_creating_products(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = object.__new__(_CollectionOnlyHybridRunner)
            runner.layout = HybridRunLayout(Path(temporary), "run")
            runner.layout.coordinator.mkdir(parents=True)

            runner._derive_products()

            self.assertFalse(
                (runner.layout.coordinator / "deferred_postprocess.json").exists()
            )
            self.assertFalse(runner.layout.perfetto.exists())
            self.assertFalse(runner.layout.overview.exists())


class AllModeReportTests(unittest.TestCase):
    def _write_campaign(
        self,
        root: Path,
        *,
        candidate_delta: float = 0.04,
        formal_rounds: int = 5,
        deferred: bool = False,
    ) -> None:
        schedule = build_all_mode_schedule(17, formal_rounds=formal_rounds)
        (root / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
        (root / "execution_plan.json").write_text(
            json.dumps(
                {
                    "formal_block_postprocessing": (
                        "deferred" if deferred else "per_block"
                    )
                }
            ),
            encoding="utf-8",
        )
        (root / "checkpoint.json").write_text(
            json.dumps({"status": "succeeded"}), encoding="utf-8"
        )
        for block in schedule["blocks"]:
            condition = block["condition"]
            round_index = block["round"]
            reference = condition == "reference"
            e2e = 100.0 if reference else 100.0 * (1 + candidate_delta)
            throughput = 80.0 if reference else 80.0 * (1 - candidate_delta)
            target = root / "raw" / f"round-{round_index:02d}" / condition
            target.mkdir(parents=True)
            result = {
                "metrics": {
                    "latency": {
                        "e2e_ns": {"median": e2e},
                        "ttft_ns": {"median": e2e / 2},
                        "tpot_ns": {"median": e2e / 8},
                    },
                    "throughput": {
                        "output_tokens_per_second": throughput,
                        "requests_per_second": throughput / 8,
                    },
                },
                "sampling_validation": {
                    "valid": True,
                    "streams": {
                        name: {"actual_interval_ns": {"median": 1_000_000_000}}
                        for name in ("gpu", "npu", "system")
                    },
                },
                "artifact_validation": {"valid": True},
                "operational_metrics": {"run_artifact_size_bytes": 123},
            }
            (target / "block_result.json").write_text(json.dumps(result), encoding="utf-8")

    def test_three_round_report_is_descriptive_without_confidence_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root, formal_rounds=3)
            report = build_campaign_report(root)
            self.assertTrue(report["complete"])
            self.assertEqual(report["formal_blocks_expected"], 18)
            self.assertIsNone(report["all_modes_supported"])
            for mode in CONDITIONS[1:]:
                value = report["modes"][mode]
                self.assertEqual(value["valid_pairs"], 3)
                self.assertAlmostEqual(value["e2e_median_percent"], 4.0)
                self.assertTrue(value["met"])
                self.assertIsNone(value["supported"])
                self.assertIsNone(
                    value["e2e_one_sided_95_ci_upper_percent"]
                )

    def test_deferred_report_requires_representative_postprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root, formal_rounds=3, deferred=True)
            self.assertFalse(build_campaign_report(root)["complete"])
            (root / "representative_postprocess.json").write_text(
                json.dumps({"status": "succeeded"}), encoding="utf-8"
            )
            self.assertTrue(build_campaign_report(root)["complete"])

    def test_reference_pairs_are_used_for_every_mode_and_gates_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            report = build_campaign_report(root)
            self.assertTrue(report["complete"])
            self.assertTrue(report["all_modes_met"])
            self.assertTrue(report["all_modes_supported"])
            for mode in CONDITIONS[1:]:
                value = report["modes"][mode]
                self.assertEqual(value["valid_pairs"], 5)
                self.assertAlmostEqual(value["e2e_median_percent"], 4.0)
                self.assertAlmostEqual(value["throughput_degradation_median_percent"], 4.0)
                self.assertTrue(value["met"])
                self.assertTrue(value["supported"])

    def test_negative_overhead_is_not_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root, candidate_delta=-0.10)
            report = build_campaign_report(root)
            value = report["modes"]["monitor"]
            self.assertAlmostEqual(value["e2e_median_percent"], -10.0)
            self.assertAlmostEqual(value["throughput_degradation_median_percent"], -10.0)

    def test_incomplete_campaign_is_not_claimed_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            (root / "raw/round-03/gpu-torch/block_result.json").unlink()
            report = build_campaign_report(root)
            self.assertFalse(report["complete"])
            self.assertFalse(report["all_modes_met"])
            self.assertEqual(report["modes"]["gpu-torch"]["status"], "incomplete")

    def test_reports_update_after_new_raw_evidence_and_are_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_campaign(root)
            for directory in (root / "report", root / "summary", root / "manifest"):
                directory.mkdir()
            missing = root / "raw/round-03/gpu-torch/block_result.json"
            preserved = missing.read_bytes()
            missing.unlink()
            first = generate_campaign_report(root)
            self.assertFalse(first["complete"])
            missing.write_bytes(preserved)
            second = generate_campaign_report(root)
            self.assertTrue(second["complete"])
            report_bytes = (root / "report/report.json").read_bytes()
            summary_bytes = {
                path.name: path.read_bytes()
                for path in sorted((root / "summary").iterdir())
            }
            generate_campaign_report(root)
            self.assertEqual((root / "report/report.json").read_bytes(), report_bytes)
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in sorted((root / "summary").iterdir())
                },
                summary_bytes,
            )
            self.assertEqual(
                sorted(summary_bytes),
                [
                    "artifact_validation.json",
                    "final_verdict.json",
                    "limitations.json",
                    "paired_results.json",
                    "per_mode_statistics.json",
                    "sampling_validation.json",
                ],
            )


class BlockFailureEvidenceTests(unittest.TestCase):
    class _Config:
        def load_hybrid(self) -> object:
            return type("Hybrid", (), {"rbln_cache_path": Path("/cache")})()

    def test_failure_evidence_preserves_error_cleanup_and_cache_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            block = {
                "round": 1,
                "condition": "monitor",
                "block_id": "round-01-monitor",
            }
            target = root / "raw/round-01/monitor"
            target.mkdir(parents=True)
            (target / "cache_fingerprint_before.json").write_text(
                json.dumps([{"relative_path": "same.rbln"}]), encoding="utf-8"
            )
            (target / "stdout.log").write_text("preserved", encoding="utf-8")
            environment = {"stage": "failed_condition_block", "ports": []}
            with (
                patch(
                    "tools.evaluation.all_mode_overhead.capture_environment",
                    return_value=environment,
                ),
                patch(
                    "tools.evaluation.all_mode_overhead.idle_reasons",
                    return_value=[],
                ),
                patch(
                    "tools.evaluation.all_mode_overhead._cache_fingerprint",
                    return_value=[{"relative_path": "same.rbln"}],
                ),
            ):
                path = _record_block_failure(
                    config=self._Config(),  # type: ignore[arg-type]
                    root=root,
                    block=block,
                    error=RuntimeError("native abort"),
                    interrupted=False,
                )
            self.assertEqual(path, target / "failure.json")
            evidence = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed")
            self.assertEqual(evidence["failure_type"], "RuntimeError")
            self.assertEqual(evidence["failure_message"], "native abort")
            self.assertEqual(evidence["automatic_retries"], 0)
            self.assertTrue(evidence["environment_after_failure_recorded"])
            self.assertTrue(evidence["cache_unchanged"])
            self.assertTrue(evidence["stdout_log_present"])
            self.assertEqual(evidence["post_failure_idle_reasons"], [])

    def test_diagnostic_failure_does_not_replace_original_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            block = {
                "round": 2,
                "condition": "gpu-nsys",
                "block_id": "round-02-gpu-nsys",
            }
            with patch(
                "tools.evaluation.all_mode_overhead.capture_environment",
                side_effect=OSError("inventory unavailable"),
            ):
                path = _record_block_failure(
                    config=self._Config(),  # type: ignore[arg-type]
                    root=root,
                    block=block,
                    error=ValueError("primary failure"),
                    interrupted=True,
                )
            assert path is not None
            evidence = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "interrupted")
            self.assertEqual(evidence["failure_message"], "primary failure")
            self.assertFalse(evidence["environment_after_failure_recorded"])
            self.assertTrue(
                any(
                    item.startswith("environment_after_failure: OSError")
                    for item in evidence["diagnostic_errors"]
                )
            )


class DetailedArtifactValidationTests(unittest.TestCase):
    def _layout(self, root: Path) -> HybridRunLayout:
        layout = HybridRunLayout(root, "run")
        layout.gpu.mkdir(parents=True)
        layout.npu.mkdir(parents=True)
        return layout

    @staticmethod
    def _summary(root: Path, value: dict[str, object]) -> None:
        target = root / "summary/detailed_profile.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")

    def test_gpu_torch_requires_parsed_cpu_cuda_runtime_and_kernel_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = self._layout(Path(temporary))
            trace = layout.gpu / "raw/capture.pt.trace.json.gz"
            trace.parent.mkdir(parents=True)
            trace.write_bytes(b"trace")
            summary = {
                "enabled": True,
                "kind": "gpu_torch",
                "event_count": 20,
                "activity_event_count": 10,
                "cpu_event_count": 5,
                "cuda_kernel_event_count": 2,
                "cuda_runtime_event_count": 3,
            }
            self._summary(layout.gpu, summary)
            self.assertTrue(validate_mode_artifacts(layout, "gpu-torch")["valid"])
            summary["cuda_kernel_event_count"] = 0
            self._summary(layout.gpu, summary)
            result = validate_mode_artifacts(layout, "gpu-torch")
            self.assertFalse(result["valid"])
            self.assertFalse(result["checks"]["cuda_kernel_events_present"])

    def test_nsys_requires_report_sqlite_and_official_cuda_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = self._layout(Path(temporary))
            raw = layout.gpu / "raw/nsys"
            raw.mkdir(parents=True)
            (raw / "capture.nsys-rep").write_bytes(b"report")
            self._summary(
                layout.gpu,
                {
                    "enabled": True,
                    "kind": "gpu_nsys",
                    "reports": {
                        "cuda_api_sum": {"available": True, "data_row_count": 2},
                        "cuda_gpu_kern_sum": {"available": True, "data_row_count": 3},
                    },
                },
            )
            self.assertFalse(validate_mode_artifacts(layout, "gpu-nsys")["valid"])
            (raw / "capture.sqlite").write_bytes(b"sqlite")
            self.assertTrue(validate_mode_artifacts(layout, "gpu-nsys")["valid"])

    def test_rbln_requires_perfetto_format_and_device_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = self._layout(Path(temporary))
            raw = layout.npu / "raw/rbln"
            raw.mkdir(parents=True)
            (raw / "capture.pb").write_bytes(b"perfetto")
            summary = {
                "enabled": True,
                "kind": "npu_rbln",
                "format": "perfetto_trace_protobuf",
                "device_timing_present": True,
                "host_timing_present": True,
            }
            self._summary(layout.npu, summary)
            self.assertTrue(validate_mode_artifacts(layout, "npu-rbln")["valid"])
            summary["device_timing_present"] = False
            self._summary(layout.npu, summary)
            self.assertFalse(validate_mode_artifacts(layout, "npu-rbln")["valid"])

    def test_operational_metrics_use_validated_api_and_postprocess_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = self._layout(Path(temporary))
            (layout.bundle / "artifact").write_bytes(b"1234")
            self._summary(
                layout.gpu,
                {
                    "enabled": True,
                    "kind": "gpu_torch",
                    "api": {
                        "start": {
                            "before_monotonic_ns": 120,
                            "after_monotonic_ns": 130,
                        },
                        "stop": {
                            "before_monotonic_ns": 180,
                            "after_monotonic_ns": 200,
                        },
                    },
                },
            )
            result = _profile_operational_metrics(
                layout,
                condition="gpu-torch",
                runner_started_ns=100,
                runner_ended_ns=300,
                request_window_start_ns=150,
                postprocessing_started_ns=210,
            )
            self.assertEqual(result["capture_duration_ns"], 80)
            self.assertEqual(result["profiler_start_api_rtt_ns"], 10)
            self.assertEqual(result["profiler_stop_and_finalize_api_rtt_ns"], 20)
            self.assertEqual(result["offline_postprocessing_duration_ns"], 90)
            self.assertEqual(
                result["startup_warmup_and_profiler_start_duration_ns"], 50
            )
            self.assertGreaterEqual(result["run_artifact_size_bytes"], 4)

    def test_recovered_operational_metrics_do_not_fabricate_runner_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = self._layout(Path(temporary))
            (layout.bundle / "artifact").write_bytes(b"1234")
            result = _profile_operational_metrics(
                layout,
                condition="npu-torch",
                runner_started_ns=None,
                runner_ended_ns=None,
                request_window_start_ns=150,
                postprocessing_started_ns=None,
            )
            self.assertIsNone(result["runner_wall_duration_ns"])
            self.assertIsNone(
                result["startup_warmup_and_profiler_start_duration_ns"]
            )
            self.assertIsNotNone(
                result["runner_wall_duration_unavailable_reason"]
            )


if __name__ == "__main__":
    unittest.main()
