"""Cross-check the explicit profiling contracts against their consumers."""

from __future__ import annotations

from dataclasses import MISSING, fields, replace
import importlib
from importlib.util import find_spec
import json
from pathlib import Path
import unittest

from perfetto_hetero_profiler.perfetto.validation_queries import (
    BASE_VALIDATION_QUERIES,
    NATIVE_VALIDATION_QUERIES,
    TIMELINE_VALIDATION_QUERIES,
)
from perfetto_hetero_profiler.overview import bundle, generator, model, render, schema
from perfetto_hetero_profiler.schema.catalog import (
    KPI_PRESENTATIONS,
    KPI_PRESENTATION_BY_IDENTITY,
    KPI_SECTION_METRICS,
    KPI_SECTION_ORDER,
    PIPELINE_STAGE_ORDER,
    RESOURCE_PRESENTATIONS,
    RESOURCE_TRACK_ORDER,
    STAGE_DEFINITIONS,
    TRACE_ATTRIBUTE_PRESENTATIONS,
    validate_catalog_contract,
)
from perfetto_hetero_profiler.schema.metric_catalog import (
    METRIC_DEFINITIONS,
    validate_metric_definitions,
)
from perfetto_hetero_profiler.schema.records import (
    ArtifactReference,
    ClockDomain,
    ClockTransform,
    DeviceDescriptor,
    EventRecord,
    HostDescriptor,
    MetricSample,
    ModelDescriptor,
    RunManifest,
    SoftwareDescriptor,
    SyncPoint,
    WorkloadDescriptor,
)


SCHEMA_ROOT = (
    Path(__file__).parents[1]
    / "src"
    / "perfetto_hetero_profiler"
    / "schema"
    / "json"
    / "v1"
)


def _schema_node(
    document: dict[str, object], definition: str | None
) -> dict[str, object]:
    if definition is None:
        return document
    definitions = document.get("$defs")
    assert isinstance(definitions, dict)
    node = definitions.get(definition)
    assert isinstance(node, dict)
    return node

_SCHEMA_RECORDS = (
    (RunManifest, "run_manifest.schema.json", None),
    (ModelDescriptor, "run_manifest.schema.json", "model"),
    (WorkloadDescriptor, "run_manifest.schema.json", "workload"),
    (HostDescriptor, "run_manifest.schema.json", "host"),
    (SoftwareDescriptor, "run_manifest.schema.json", "software"),
    (DeviceDescriptor, "run_manifest.schema.json", "device"),
    (EventRecord, "event_record.schema.json", None),
    (MetricSample, "metric_sample.schema.json", None),
    (ArtifactReference, "artifact_reference.schema.json", None),
    (ClockDomain, "clock_domain.schema.json", None),
    (SyncPoint, "sync_point.schema.json", None),
    (ClockTransform, "clock_transform.schema.json", None),
)


class RecordSchemaDriftTests(unittest.TestCase):
    def test_dataclass_fields_match_schema_properties(self):
        for record_class, filename, definition in _SCHEMA_RECORDS:
            with self.subTest(record=record_class.__name__):
                document = json.loads(
                    (SCHEMA_ROOT / filename).read_text(encoding="utf-8")
                )
                node = _schema_node(document, definition)
                self.assertEqual(
                    {item.name for item in fields(record_class)},
                    set(node["properties"]),
                )

    def test_top_level_record_type_constants_and_required_fields(self):
        for record_class, filename, definition in _SCHEMA_RECORDS:
            if definition is not None:
                continue
            with self.subTest(record=record_class.__name__):
                document = json.loads(
                    (SCHEMA_ROOT / filename).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    document["properties"]["record_type"]["const"],
                    record_class.__dataclass_fields__["record_type"].default.value,
                )
                self.assertEqual(
                    set(document["required"]),
                    {
                        item.name
                        for item in fields(record_class)
                        if (
                            item.default is MISSING
                            and item.default_factory is MISSING
                        ) or item.name in {"schema_version", "record_type"}
                    },
                )


class MetricAndPresentationContractTests(unittest.TestCase):
    def test_duplicate_metric_is_rejected(self):
        duplicate = replace(METRIC_DEFINITIONS[1], name=METRIC_DEFINITIONS[0].name)
        with self.assertRaisesRegex(RuntimeError, "duplicate official metric"):
            validate_metric_definitions((METRIC_DEFINITIONS[0], duplicate))

    def test_duplicate_stage_pair_and_order_are_rejected(self):
        duplicate_pair = replace(
            STAGE_DEFINITIONS[1], track_key="different_stage"
        )
        with self.assertRaisesRegex(RuntimeError, "stage marker pair"):
            validate_catalog_contract(stages=(STAGE_DEFINITIONS[1], duplicate_pair))
        duplicate_order = replace(
            STAGE_DEFINITIONS[2], pipeline_order=STAGE_DEFINITIONS[1].pipeline_order
        )
        with self.assertRaisesRegex(RuntimeError, "pipeline stage order"):
            validate_catalog_contract(stages=(STAGE_DEFINITIONS[1], duplicate_order))

    def test_noncontiguous_stage_order_is_rejected(self):
        stage = replace(STAGE_DEFINITIONS[1], pipeline_order=2)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            validate_catalog_contract(stages=(stage,))

    def test_unknown_metric_reference_is_rejected(self):
        stage = replace(STAGE_DEFINITIONS[0], metric_name="unknown.metric")
        with self.assertRaisesRegex(RuntimeError, "unknown metric"):
            validate_catalog_contract(stages=(stage,))

    def test_stage_metric_source_markers_are_consistent(self):
        stage = replace(STAGE_DEFINITIONS[0], end_event="wrong_end")
        with self.assertRaisesRegex(RuntimeError, "disagrees with metric"):
            validate_catalog_contract(stages=(stage,))

    def test_kpi_sections_and_trace_attribute_keys_share_catalog_identity(self):
        self.assertEqual(
            set(KPI_PRESENTATION_BY_IDENTITY),
            {
                (section, metric_name)
                for section in KPI_SECTION_ORDER
                for metric_name in KPI_SECTION_METRICS[section]
            },
        )
        keys = [item.attribute_key for item in TRACE_ATTRIBUTE_PRESENTATIONS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(key and not key.startswith("vllm_profiler.") for key in keys))

    def test_resource_order_is_declared_once(self):
        self.assertEqual(
            RESOURCE_TRACK_ORDER,
            {item.metric_name: item.order for item in RESOURCE_PRESENTATIONS},
        )
        self.assertEqual(
            PIPELINE_STAGE_ORDER,
            {
                item.track_key: item.pipeline_order
                for item in STAGE_DEFINITIONS
                if item.pipeline_order is not None
            },
        )


class ValidationQueryContractTests(unittest.TestCase):
    def test_query_registry_preserves_expected_identity_and_order(self):
        self.assertEqual(
            tuple(item.name for item in BASE_VALIDATION_QUERIES),
            (
                "process",
                "tracks",
                "slices",
                "annotations",
                "step_annotations",
                "counters",
                "flows",
                "dangling_flows",
                "import_errors",
                "native_policy",
            ),
        )
        self.assertEqual(
            tuple(item.name for item in NATIVE_VALIDATION_QUERIES),
            ("native_event_semantics",),
        )
        self.assertEqual(
            tuple(item.name for item in TIMELINE_VALIDATION_QUERIES),
            (
                "timeline_summary_hierarchy",
                "timeline_summary_slices",
                "timeline_summary_kpis",
                "timeline_summary_data_quality",
                "trace_attributes",
            ),
        )
        for query in (
            *BASE_VALIDATION_QUERIES,
            *NATIVE_VALIDATION_QUERIES,
            *TIMELINE_VALIDATION_QUERIES,
        ):
            self.assertTrue(query.sql.strip())


class EvaluationBoundaryTests(unittest.TestCase):
    FORBIDDEN_CORE_IDENTIFIERS = (
        "OverviewComparison",
        "OverviewComparisonConfig",
        "compare_overviews",
        "plan_overview_comparison",
        "_prepare_comparison",
        "LoadedComparisonBundle",
        "render_comparison_html",
        "build_comparison_validation",
        "COMPARISON_JSON_NAME",
        "COMPARISON_HTML_NAME",
        "COMPARISON_VALIDATION_NAME",
    )

    def test_comparison_module_and_schema_are_repository_only(self):
        root = Path(__file__).parents[1]
        self.assertFalse(
            (root / "src/perfetto_hetero_profiler/overview/comparison.py").exists()
        )
        self.assertFalse(
            (
                root
                / "src/perfetto_hetero_profiler/overview/json/v1/overview_comparison.schema.json"
            ).exists()
        )
        self.assertTrue((root / "tools/evaluation/overview_comparison.py").is_file())
        self.assertTrue(
            (
                root
                / "tools/evaluation/schema/overview_comparison.schema.json"
            ).is_file()
        )

    def test_comparison_symbols_are_not_core_public_api(self):
        for module in (bundle, generator, model, render, schema):
            with self.subTest(module=module.__name__):
                self.assertFalse(
                    any("comparison" in name.casefold() for name in module.__all__)
                )

    def test_core_contains_no_evaluation_dependency_or_comparison_contract(self):
        core = Path(__file__).parents[1] / "src/perfetto_hetero_profiler"
        source = {
            path.relative_to(core).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted(core.rglob("*.py"))
        }
        self.assertFalse(
            [path for path, text in source.items() if "tools.evaluation" in text]
        )
        for identifier in self.FORBIDDEN_CORE_IDENTIFIERS:
            with self.subTest(identifier=identifier):
                self.assertFalse(
                    [path for path, text in source.items() if identifier in text]
                )

    def test_core_modules_do_not_expose_moved_contract(self):
        locations = {
            "OverviewComparison": "perfetto_hetero_profiler.overview.model",
            "OverviewComparisonConfig": "perfetto_hetero_profiler.overview.generator",
            "LoadedComparisonBundle": "perfetto_hetero_profiler.overview.bundle",
            "render_comparison_html": "perfetto_hetero_profiler.overview.render",
            "build_comparison_validation": "perfetto_hetero_profiler.overview.validation",
        }
        for identifier, module_name in locations.items():
            with self.subTest(identifier=identifier):
                module = importlib.import_module(module_name)
                self.assertFalse(hasattr(module, identifier))

    def test_documentation_uses_repository_evaluation_command(self):
        root = Path(__file__).parents[1]
        for relative_path in ("README.md", "docs/usage.md"):
            with self.subTest(path=relative_path):
                text = (root / relative_path).read_text(encoding="utf-8")
                self.assertIn(
                    "python3 -m tools.evaluation overview compare",
                    text,
                )
                self.assertNotIn("hetero-profiler overview compare", text)


class DeadCodeBoundaryTests(unittest.TestCase):
    def test_reserved_synchronization_package_is_removed(self):
        self.assertIsNone(
            find_spec("perfetto_hetero_profiler.synchronization")
        )

    def test_removed_internal_schema_and_support_symbols_stay_absent(self):
        from perfetto_hetero_profiler.schema import catalog, constants
        from perfetto_hetero_profiler.support.config_fields import ConfigFields

        for name in (
            "RECORD_TYPES",
            "EXTENSION_NAMESPACES",
            "JSON_SCHEMA_FILES",
        ):
            with self.subTest(module=constants.__name__, name=name):
                self.assertFalse(hasattr(constants, name))
        self.assertIsNone(find_spec("perfetto_hetero_profiler.schema.field_contracts"))
        for name in ("STAGE_BY_TRACK", "TRACE_ATTRIBUTE_LATENCY_IDENTITIES"):
            with self.subTest(module=catalog.__name__, name=name):
                self.assertFalse(hasattr(catalog, name))
        self.assertFalse(hasattr(ConfigFields, "enum"))


if __name__ == "__main__":
    unittest.main()
