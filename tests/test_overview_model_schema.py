"""CPU-only contract tests for deterministic Phase 6 Overview models."""

from __future__ import annotations

from dataclasses import fields, replace
import copy
import json
import math
import unittest

from perfetto_hetero_profiler.overview.model import (
    Comparability,
    ComparisonDelta,
    ComparisonKpi,
    ComparisonMetadata,
    ComparisonRun,
    ComparisonValue,
    DeltaValue,
    DisplayRule,
    KpiCalculation,
    KpiClock,
    KpiDirection,
    KpiScope,
    KpiSections,
    KpiSource,
    KpiValue,
    OverviewComparison,
    OverviewReport,
    ResourceSummary,
)
from perfetto_hetero_profiler.overview.schema import (
    OverviewSchemaError,
    canonical_json_bytes,
    canonical_sha256,
    load_json_schema,
    overview_report_from_dict,
    overview_document_from_json,
    overview_to_dict,
    validate_json_schema_contract,
    validate_kpi,
    validate_overview_comparison,
    validate_overview_report,
    validate_resource_summary,
)
from perfetto_hetero_profiler.schema import Availability
from perfetto_hetero_profiler.schema.constants import JSON_SCHEMA_DRAFT


RUN_ID = "phase6-run"
REQUEST_ID = "request-1"


def source(
    *,
    metric_name: str,
    record_ids: tuple[str, ...] = ("event-1",),
) -> KpiSource:
    return KpiSource(
        source_kind="normalized_metric",
        record_ids=record_ids,
        metric_names=(metric_name,),
        root_id="hybrid",
        relative_path="metrics/metrics.jsonl",
        details={"reconciled": True},
    )


def scope(
    *,
    scope_type: str = "request",
    observation_layer: str = "request_facing_client",
    request_id: str | None = REQUEST_ID,
    host_id: str | None = None,
    device_type: str | None = None,
    device_id: str | None = None,
    window: str | None = None,
) -> KpiScope:
    return KpiScope(
        run_id=RUN_ID,
        scope_type=scope_type,
        observation_layer=observation_layer,
        request_id=request_id,
        host_id=host_id,
        device_type=device_type,
        device_id=device_id,
        phase=None,
        window=window,
    )


def clock() -> KpiClock:
    return KpiClock(
        domain_ids=("hybrid-canonical",),
        alignment_status="canonical",
        alignment_method="same_clock_domain",
        offset_ns=0,
        uncertainty_ns=0,
    )


def display(unit: str) -> DisplayRule:
    values = {
        "ns": ("ms", 1, 1_000_000, 3),
        "tokens": ("tokens", 1, 1, 0),
        "requests": ("requests", 1, 1, 0),
        "bytes": ("MiB", 1, 1_048_576, 3),
        "percent": ("percent", 1, 1, 2),
    }
    display_unit, numerator, denominator, places = values[unit]
    return DisplayRule(
        unit=display_unit,
        scale_numerator=numerator,
        scale_denominator=denominator,
        decimal_places=places,
    )


def kpi(
    name: str = "latency.e2e",
    *,
    unit: str = "ns",
    value: int | float | None = 100,
    availability: Availability = Availability.AVAILABLE,
    reason: str | None = None,
    aggregation_method: str = "identity",
    kpi_scope: KpiScope | None = None,
) -> KpiValue:
    selected_scope = kpi_scope or scope()
    return KpiValue(
        name=name,
        canonical_unit=unit,
        availability=availability,
        value=value,
        unavailable_reason=reason,
        aggregation_method=aggregation_method,
        sample_count=1,
        sources=(source(metric_name=name),),
        scope=selected_scope,
        calculation=KpiCalculation(
            method_id=f"{name}.v1",
            formula="end_ns - start_ns",
        ),
        clock=clock(),
        quality_warnings=(),
        display=display(unit),
    )


def resource_summary() -> ResourceSummary:
    resource_scope = scope(
        scope_type="device",
        observation_layer="normalized_resource_metric",
        request_id=None,
        host_id="host-0",
        device_type="gpu",
        device_id="gpu-0",
    )
    values = {
        "max": ("maximum_v1", 20.0),
        "mean": ("arithmetic_mean_v1", 15.0),
        "min": ("minimum_v1", 10.0),
        "p50": ("percentile_r7_v1", 10.0),
        "p95": ("percentile_r7_v1", 20.0),
        "time_weighted_mean": (
            "trailing_interval_time_weighted_mean_v1",
            16.0,
        ),
    }
    aggregates = tuple(
        kpi(
            f"resource.gpu.utilization.{suffix}",
            unit="percent",
            value=value,
            aggregation_method=method,
            kpi_scope=resource_scope,
        )
        for suffix, (method, value) in values.items()
    )
    return ResourceSummary(
        metric_name="resource.gpu.utilization",
        canonical_unit="percent",
        scope=resource_scope,
        clock=clock(),
        total_sample_count=3,
        available_sample_count=2,
        unavailable_sample_count=1,
        availability_ratio=2 / 3,
        first_timestamp_ns=100,
        last_timestamp_ns=300,
        coverage_ns=200,
        aggregates=aggregates,
        quality_warnings=("one sample unavailable",),
    )


def report() -> OverviewReport:
    request_e2e = kpi()
    pipeline_e2e = replace(
        request_e2e,
        scope=scope(observation_layer="hybrid_pipeline"),
        sources=(
            source(
                metric_name="latency.e2e",
                record_ids=("marker-end", "marker-start"),
            ),
        ),
    )
    request_count = kpi(
        "request.count",
        unit="requests",
        value=1,
        kpi_scope=scope(
            scope_type="run",
            observation_layer="run",
            request_id=None,
            window="measured_smoke",
        ),
    )
    transfer_bytes = kpi(
        "transfer.bytes",
        unit="bytes",
        value=58_720_256,
        kpi_scope=scope(
            scope_type="transfer",
            observation_layer="hybrid_pipeline",
        ),
    )
    return OverviewReport(
        run={
            "run_id": RUN_ID,
            "mode": "hybrid",
            "profile_mode": "monitor",
            "status": "succeeded",
            "profiler_kind": "control",
            "canonical_clock_domain_id": "hybrid-canonical",
        },
        workload={
            "request_count": 1,
            "input_tokens": 5,
            "output_tokens": 8,
            "total_tokens": 13,
            "concurrency": None,
            "request_rate_per_s": None,
            "warmup_requests": 1,
            "max_output_tokens": 8,
            "temperature": 0.0,
            "retry_count": 0,
            "prompt_sha256": "d" * 64,
            "request_body_sha256": "e" * 64,
            "offline": True,
            "max_model_len": 512,
            "block_size": 512,
        },
        models=(
            {
                "role": "prefill_decode",
                "model_id": "Qwen3-0.6B",
                "revision": None,
                "dtype": None,
            },
        ),
        hardware=(
            {
                "device_type": "gpu",
                "device_id": "gpu-0",
                "model": "synthetic-gpu",
                "vendor": "synthetic",
                "memory_total_bytes": None,
            },
        ),
        kpis=KpiSections(
            request_facing_latency=(request_e2e,),
            pipeline_latency=(pipeline_e2e,),
            throughput_and_tokens=(request_count,),
            transfer=(transfer_bytes,),
        ),
        resources=(resource_summary(),),
        data_quality={
            "run_status": "succeeded",
            "canonical_marker_count": 44,
            "marker_validation": {
                "status": "valid",
                "missing_count": 0,
                "duplicate_count": 0,
                "pairing_violation_count": 0,
                "order_violation_count": 0,
            },
            "request_join": {
                "joined_count": 1,
                "unjoined_count": 0,
                "method": "correlation_id",
            },
            "alignment": {
                "status": "canonical",
                "method": "same_clock_domain",
                "offset_ns": 0,
                "uncertainty_ns": 0,
            },
            "resource_samples": {
                "total": 3,
                "available": 2,
                "unavailable": 1,
            },
            "profiler": {
                "kind": "control",
                "native_alignment_status": "not_applicable",
            },
            "source_artifact_validation": {
                "valid": True,
                "closeout_artifact_count": 3,
                "closeout_manifest_sha256": "f" * 64,
                "roots": [
                    {
                        "root_id": "hybrid",
                        "file_count": 3,
                        "fingerprint_sha256": "9" * 64,
                    }
                ],
            },
            "perfetto_sql_validation": {
                "valid": True,
                "query_count": 1,
                "mismatches": [],
            },
            "trace_sha256": "a" * 64,
            "per_sample_stream_preserved": True,
            "cleanup_complete": True,
            "rbln_pb_policy": {
                "classification": "not_applicable",
                "structure_analysis": "not_applicable",
                "raw_bytes_embedded": False,
            },
            "sample_limitations": ["single request"],
        },
        perfetto={
            "valid": True,
            "trace": {"size_bytes": 100, "sha256": "a" * 64},
            "counts": {
                "annotations": 1,
                "counters": 1,
                "dangling_flows": 0,
                "flows": 1,
                "import_errors": 0,
                "native_policy": 0,
                "process": 1,
                "slices": 1,
                "step_annotations": 0,
                "tracks": 1,
            },
            "query_count": 1,
            "queries": [
                {
                    "name": "process",
                    "row_count": 1,
                    "rows_sha256": "1" * 64,
                    "expected_row_count": 1,
                    "expected_rows_sha256": "1" * 64,
                    "matched": True,
                }
            ],
            "mismatches": [],
            "flow_endpoint_reconciliation": {
                "declared_flow_ids": [1],
                "source_endpoint_ids": [1],
                "destination_endpoint_ids": [1],
                "matched": True,
            },
            "artifact_validation": {
                "valid": True,
                "checked": 3,
                "mismatches": [],
                "manifest_sha256": "2" * 64,
            },
            "toolchain": {
                "filename": "trace_processor_shell",
                "version": "v56.1",
                "sha256": "3" * 64,
                "perfetto_package_version": "0.57.2",
                "protobuf_package_version": "6.33.6",
                "trace_processor_rpc_api_version": 14,
            },
        },
        native_profiles=(),
        interpretation={
            "comparison_scope": "single request diagnostic capture",
            "benchmark_claim_allowed": False,
            "limitations": ["single request"],
            "policies": {
                "request_observation_layers_separate": True,
                "timestamp_proximity_join": False,
                "unavailable_zero_fill": False,
                "native_clock_inference": False,
                "rbln_pb_parsing": False,
                "resource_device_aggregation": False,
            },
        },
    )


def comparison() -> OverviewComparison:
    run_control = ComparisonRun(
        run_id="run-control",
        run_mode="hybrid",
        profile_mode="monitor",
        profile_kind="control",
        overview_sha256="0" * 64,
        request_sample_count=1,
        model_identity_sha256="a" * 64,
        hardware_identity_sha256="b" * 64,
        workload_identity_sha256="c" * 64,
        canonical_clock_domain_id="hybrid-canonical",
        clock_alignment_status="canonical",
        source_integrity_valid=True,
        quality_warnings=("single request",),
    )
    run_torch = replace(
        run_control,
        run_id="run-torch",
        profile_mode="detailed_profile",
        profile_kind="gpu_torch",
        overview_sha256="1" * 64,
    )
    metric = ComparisonKpi(
        section="request_facing_latency",
        observation_layer="request_facing_client",
        name="latency.e2e",
        canonical_unit="ns",
        direction=KpiDirection.LOWER_IS_PREFERRED,
        values=(
            ComparisonValue(
                run_id="run-control",
                availability=Availability.AVAILABLE,
                value=100,
                unavailable_reason=None,
                sample_count=1,
            ),
            ComparisonValue(
                run_id="run-torch",
                availability=Availability.AVAILABLE,
                value=110,
                unavailable_reason=None,
                sample_count=1,
            ),
        ),
        deltas=(
            ComparisonDelta(
                run_id="run-torch",
                baseline_run_id="run-control",
                absolute=DeltaValue(
                    availability=Availability.AVAILABLE,
                    value=10,
                    unavailable_reason=None,
                ),
                percentage=DeltaValue(
                    availability=Availability.AVAILABLE,
                    value=10.0,
                    unavailable_reason=None,
                ),
            ),
        ),
        quality_warnings=("diagnostic only",),
    )
    return OverviewComparison(
        comparison=ComparisonMetadata(
            comparability=Comparability.DIAGNOSTIC_ONLY,
            comparability_reasons=(
                "profile modes differ",
                "single request per run",
            ),
            baseline_run_id="run-control",
        ),
        runs=(run_control, run_torch),
        metrics=(metric,),
        limitations=("not a generalized benchmark",),
    )


class OverviewSchemaContractTests(unittest.TestCase):
    def test_checked_in_draft_2020_12_contracts_match_models(self) -> None:
        validate_json_schema_contract()
        report_schema = load_json_schema("overview_report")
        comparison_schema = load_json_schema("overview_comparison")
        self.assertEqual(report_schema["$schema"], JSON_SCHEMA_DRAFT)
        self.assertEqual(comparison_schema["$schema"], JSON_SCHEMA_DRAFT)
        self.assertEqual(
            set(report_schema["properties"]),
            {item.name for item in fields(OverviewReport)},
        )
        self.assertEqual(
            set(comparison_schema["properties"]),
            {item.name for item in fields(OverviewComparison)},
        )

    def test_report_top_level_contract_and_round_trip(self) -> None:
        value = report()
        validate_overview_report(value)
        serialized = overview_to_dict(value)
        self.assertEqual(
            set(serialized),
            {
                "schema_version",
                "record_type",
                "run",
                "workload",
                "models",
                "hardware",
                "kpis",
                "resources",
                "data_quality",
                "perfetto",
                "native_profiles",
                "interpretation",
            },
        )
        restored = overview_document_from_json(canonical_json_bytes(value))
        self.assertEqual(restored, value)

    def test_canonical_json_is_stable_and_path_free(self) -> None:
        first = report()
        second = replace(
            first,
            workload={
                key: first.workload[key]
                for key in reversed(tuple(first.workload))
            },
            data_quality={
                key: first.data_quality[key]
                for key in reversed(tuple(first.data_quality))
            },
        )
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(canonical_sha256(first), canonical_sha256(second))
        self.assertNotIn(b"/home/", canonical_json_bytes(first))

    def test_actual_zero_and_unavailable_remain_distinct(self) -> None:
        zero = kpi(value=0)
        validate_kpi(zero)
        unavailable = replace(
            zero,
            availability=Availability.NOT_AVAILABLE,
            value=None,
            unavailable_reason="no token timestamp",
            sample_count=0,
        )
        validate_kpi(unavailable)
        self.assertEqual(overview_to_dict(replace(
            report(),
            kpis=KpiSections(
                request_facing_latency=(unavailable,),
                pipeline_latency=(),
                throughput_and_tokens=(),
                transfer=(),
            ),
            resources=(),
        ))["kpis"]["request_facing_latency"][0]["value"], None)

    def test_bool_nan_infinity_and_availability_mismatch_rejected(self) -> None:
        for bad in (True, math.nan, math.inf, -math.inf):
            with self.subTest(value=bad):
                with self.assertRaises(OverviewSchemaError):
                    validate_kpi(kpi(value=bad))
        with self.assertRaisesRegex(OverviewSchemaError, "must be null"):
            validate_kpi(
                kpi(
                    value=1,
                    availability=Availability.NOT_AVAILABLE,
                    reason="missing",
                )
            )
        with self.assertRaisesRegex(OverviewSchemaError, "non-empty"):
            validate_kpi(
                kpi(
                    value=None,
                    availability=Availability.NOT_AVAILABLE,
                    reason=None,
                )
            )

    def test_catalog_unit_integer_and_display_rules_rejected(self) -> None:
        with self.assertRaisesRegex(OverviewSchemaError, "catalog unit"):
            validate_kpi(replace(kpi(), canonical_unit="bytes"))
        request_count = kpi(
            "request.count",
            unit="requests",
            value=1.5,
            kpi_scope=scope(
                scope_type="run",
                observation_layer="run",
                request_id=None,
            ),
        )
        with self.assertRaisesRegex(OverviewSchemaError, "integer"):
            validate_kpi(request_count)
        with self.assertRaisesRegex(OverviewSchemaError, "unsupported"):
            validate_kpi(
                replace(
                    kpi(),
                    display=DisplayRule(
                        unit="ms",
                        scale_numerator=1,
                        scale_denominator=1_000,
                        decimal_places=3,
                    ),
                )
            )

    def test_absolute_paths_rejected_in_metadata_and_provenance(self) -> None:
        with self.assertRaisesRegex(OverviewSchemaError, "absolute path"):
            validate_overview_report(
                replace(
                    report(),
                    models=(
                        {
                            "role": "prefill_decode",
                            "model_id": "/home/user/model",
                            "revision": None,
                            "dtype": None,
                        },
                    ),
                )
            )
        with self.assertRaisesRegex(OverviewSchemaError, "normalized relative"):
            validate_kpi(
                replace(
                    kpi(),
                    sources=(
                        replace(
                            source(metric_name="latency.e2e"),
                            relative_path="../secret",
                        ),
                    ),
                )
            )

    def test_resource_counts_coverage_and_required_aggregates(self) -> None:
        value = resource_summary()
        validate_resource_summary(value)
        with self.assertRaisesRegex(OverviewSchemaError, "total_sample_count"):
            validate_resource_summary(
                replace(value, total_sample_count=4)
            )
        with self.assertRaisesRegex(OverviewSchemaError, "last_timestamp"):
            validate_resource_summary(
                replace(value, coverage_ns=199)
            )
        with self.assertRaisesRegex(OverviewSchemaError, "exactly"):
            validate_resource_summary(
                replace(value, aggregates=value.aggregates[:-1])
            )

    def test_unsorted_provenance_and_report_arrays_rejected(self) -> None:
        with self.assertRaisesRegex(OverviewSchemaError, "sorted"):
            validate_kpi(
                replace(
                    kpi(),
                    sources=(
                        source(metric_name="latency.e2e", record_ids=("z", "a")),
                    ),
                )
            )
        first = report()
        second_hardware = (
            {
                "device_type": "npu",
                "device_id": "npu-0",
                "vendor": "z",
                "model": "z",
                "memory_total_bytes": None,
            },
            {
                "device_type": "gpu",
                "device_id": "gpu-0",
                "vendor": "a",
                "model": "a",
                "memory_total_bytes": None,
            },
        )
        with self.assertRaisesRegex(OverviewSchemaError, "sorted"):
            validate_overview_report(
                replace(first, hardware=second_hardware)
            )

    def test_duplicate_json_keys_and_unknown_fields_rejected(self) -> None:
        with self.assertRaisesRegex(OverviewSchemaError, "duplicate"):
            overview_document_from_json(
                '{"record_type":"overview_report","record_type":"overview_report"}'
            )
        raw = overview_to_dict(report())
        raw["unexpected"] = True
        with self.assertRaisesRegex(OverviewSchemaError, "unknown"):
            overview_document_from_json(json.dumps(raw))

    def test_calculation_output_is_accepted_without_contract_adapter(self) -> None:
        from perfetto_hetero_profiler.overview.calculation import (
            calculate_overview_kpis,
        )
        from tests.test_overview_calculation import (
            RUN_ID as CALCULATION_RUN_ID,
            _fixture,
        )

        calculated = calculate_overview_kpis(_fixture())
        raw = overview_to_dict(report())
        raw["run"]["run_id"] = CALCULATION_RUN_ID
        raw["kpis"] = {
            name: calculated[name]
            for name in (
                "request_facing_latency",
                "pipeline_latency",
                "throughput_and_tokens",
                "transfer",
            )
        }
        raw["resources"] = calculated["resource_summaries"]

        parsed = overview_report_from_dict(raw)
        self.assertEqual(parsed.run["run_id"], CALCULATION_RUN_ID)
        self.assertEqual(
            parsed.kpis.request_facing_latency[0].scope.run_id,
            CALCULATION_RUN_ID,
        )


class OverviewComparisonContractTests(unittest.TestCase):
    def test_comparison_top_level_contract_and_round_trip(self) -> None:
        value = comparison()
        validate_overview_comparison(value)
        serialized = overview_to_dict(value)
        self.assertEqual(
            set(serialized),
            {
                "schema_version",
                "record_type",
                "comparison",
                "runs",
                "metrics",
                "limitations",
            },
        )
        self.assertEqual(
            overview_document_from_json(canonical_json_bytes(value)),
            value,
        )

    def test_zero_baseline_can_keep_absolute_but_percentage_unavailable(self) -> None:
        value = comparison()
        metric = value.metrics[0]
        delta = replace(
            metric.deltas[0],
            percentage=DeltaValue(
                availability=Availability.NOT_AVAILABLE,
                value=None,
                unavailable_reason="baseline denominator is zero",
            ),
        )
        validate_overview_comparison(
            replace(value, metrics=(replace(metric, deltas=(delta,)),))
        )

    def test_comparison_builder_output_preserves_latency_layers(self) -> None:
        from perfetto_hetero_profiler.overview.comparison import build_comparison

        control = overview_to_dict(report())
        detailed = copy.deepcopy(control)
        detailed["run"]["run_id"] = "phase6-run-detailed"
        detailed["run"]["profile_mode"] = "detailed_profile"
        detailed["run"]["profiler_kind"] = "gpu_torch"
        built = build_comparison([detailed, control])

        parsed = overview_document_from_json(
            json.dumps(built, allow_nan=False)
        )
        identities = {
            (metric.section, metric.observation_layer, metric.name)
            for metric in parsed.metrics
        }
        self.assertIn(
            (
                "request_facing_latency",
                "request_facing_client",
                "latency.e2e",
            ),
            identities,
        )
        self.assertIn(
            ("pipeline_latency", "hybrid_pipeline", "latency.e2e"),
            identities,
        )

    def test_comparison_run_coverage_and_baseline_rejected(self) -> None:
        value = comparison()
        with self.assertRaisesRegex(OverviewSchemaError, "every comparison run"):
            validate_overview_comparison(
                replace(
                    value,
                    metrics=(
                        replace(
                            value.metrics[0],
                            values=(value.metrics[0].values[0],),
                        ),
                    ),
                )
            )
        with self.assertRaisesRegex(OverviewSchemaError, "identify a compared run"):
            validate_overview_comparison(
                replace(
                    value,
                    comparison=replace(
                        value.comparison,
                        baseline_run_id="missing",
                    ),
                )
            )

    def test_not_comparable_must_not_emit_deltas(self) -> None:
        value = comparison()
        with self.assertRaisesRegex(OverviewSchemaError, "must not calculate"):
            validate_overview_comparison(
                replace(
                    value,
                    comparison=replace(
                        value.comparison,
                        comparability=Comparability.NOT_COMPARABLE,
                    ),
                )
            )
        validate_overview_comparison(
            replace(
                value,
                comparison=replace(
                    value.comparison,
                    comparability=Comparability.NOT_COMPARABLE,
                ),
                metrics=(
                    replace(value.metrics[0], deltas=()),
                ),
            )
        )

    def test_comparison_bool_and_nonfinite_values_rejected(self) -> None:
        value = comparison()
        metric = value.metrics[0]
        for bad in (True, math.nan, math.inf):
            with self.subTest(value=bad):
                bad_value = replace(metric.values[1], value=bad)
                with self.assertRaises(OverviewSchemaError):
                    validate_overview_comparison(
                        replace(
                            value,
                            metrics=(
                                replace(
                                    metric,
                                    values=(metric.values[0], bad_value),
                                ),
                            ),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
