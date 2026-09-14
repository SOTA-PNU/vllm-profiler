"""CPU-only contract tests for deterministic Overview models."""

from __future__ import annotations

import copy
import json
import math
import unittest
from dataclasses import fields, replace

from perfetto_hetero_profiler.overview.model import (
    DisplayRule,
    KpiSections,
    OverviewReport,
)
from perfetto_hetero_profiler.overview.schema import (
    OverviewSchemaError,
    canonical_json_bytes,
    canonical_sha256,
    load_json_schema,
    overview_document_from_json,
    overview_report_from_dict,
    overview_to_dict,
    validate_json_schema_contract,
    validate_kpi,
    validate_overview_report,
    validate_resource_summary,
)
from perfetto_hetero_profiler.schema import Availability
from perfetto_hetero_profiler.schema.constants import JSON_SCHEMA_DRAFT
from tests.support.overview_model import (
    comparison,
    kpi,
    report,
    resource_summary,
    scope,
    source,
)
from tools.evaluation.overview import (
    Comparability,
    DeltaValue,
    OverviewComparison,
    canonical_comparison_json_bytes,
    comparison_to_dict,
    load_comparison_schema,
    validate_overview_comparison,
)
from tools.evaluation.overview import (
    overview_document_from_json as comparison_document_from_json,
)


class OverviewSchemaContractTests(unittest.TestCase):
    def test_checked_in_draft_2020_12_contracts_match_models(self) -> None:
        validate_json_schema_contract()
        report_schema = load_json_schema("overview_report")
        comparison_schema = load_comparison_schema()
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

    def test_packaged_schema_structure_corpus_has_stable_paths(self) -> None:
        valid = overview_to_dict(report())
        cases = []
        unknown = copy.deepcopy(valid)
        unknown["run"]["unexpected"] = True
        cases.append((unknown, "overview.run.unexpected"))
        missing = copy.deepcopy(valid)
        del missing["workload"]["request_count"]
        cases.append((missing, "overview.workload.request_count"))
        wrong_type = copy.deepcopy(valid)
        wrong_type["perfetto"]["query_count"] = True
        cases.append((wrong_type, "overview.perfetto.query_count"))
        invalid_enum = copy.deepcopy(valid)
        invalid_enum["run"]["mode"] = "other"
        cases.append((invalid_enum, "overview.run.mode"))
        for value, expected_path in cases:
            with self.subTest(path=expected_path):
                with self.assertRaises(OverviewSchemaError) as caught:
                    overview_report_from_dict(value)
                self.assertEqual(caught.exception.field_path, expected_path)
                self.assertNotIn("/home/", str(caught.exception))

    def test_metric_stream_details_accept_compact_and_legacy_provenance(self) -> None:
        compact = overview_to_dict(report())
        details = compact["kpis"]["request_facing_latency"][0]["sources"][0][
            "details"
        ]
        details.clear()
        details.update(
            {
                "artifact_size_bytes": 123,
                "artifact_sha256": "a" * 64,
                "timestamp_evidence": (
                    "reconstruct_from_normalized_metric_stream_timestamp_ns"
                ),
                "stream_sample_count": 2,
            }
        )
        overview_report_from_dict(compact)

        legacy = copy.deepcopy(compact)
        legacy_details = legacy["kpis"]["request_facing_latency"][0]["sources"][
            0
        ]["details"]
        legacy_details["sample_timestamps_ns"] = [10, 20]
        overview_report_from_dict(legacy)

        malformed = copy.deepcopy(compact)
        malformed["kpis"]["request_facing_latency"][0]["sources"][0][
            "details"
        ]["artifact_sha256"] = "not-a-sha256"
        with self.assertRaises(OverviewSchemaError):
            overview_report_from_dict(malformed)

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
        from tests.support.overview_calculation import (
            RUN_ID as CALCULATION_RUN_ID,
        )
        from tests.support.overview_calculation import (
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
        serialized = comparison_to_dict(value)
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
            comparison_document_from_json(
                canonical_comparison_json_bytes(value)
            ),
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
        from tools.evaluation.overview import build_comparison

        control = overview_to_dict(report())
        detailed = copy.deepcopy(control)
        detailed["run"]["run_id"] = "overview-run-detailed"
        detailed["run"]["profile_mode"] = "detailed_profile"
        detailed["run"]["profiler_kind"] = "gpu_torch"
        built = build_comparison([detailed, control])

        parsed = comparison_document_from_json(
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
