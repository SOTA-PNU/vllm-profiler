"""Cross-check the explicit profiling contracts against their consumers."""

from __future__ import annotations

from dataclasses import fields, replace
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
from perfetto_hetero_profiler.schema.field_contracts import (
    RECORD_FIELD_CONTRACTS,
    validate_field_contracts,
)
from perfetto_hetero_profiler.schema.metric_catalog import (
    METRIC_DEFINITIONS,
    validate_metric_definitions,
)


SCHEMA_ROOT = (
    Path(__file__).parents[1]
    / "src"
    / "perfetto_hetero_profiler"
    / "schema"
    / "json"
    / "v1"
)


def _schema_node(document: dict[str, object], contract: object) -> dict[str, object]:
    definition = getattr(contract, "schema_definition")
    if definition is None:
        return document
    definitions = document.get("$defs")
    assert isinstance(definitions, dict)
    node = definitions.get(definition)
    assert isinstance(node, dict)
    return node


def _resolve_property(
    document: dict[str, object], value: object
) -> dict[str, object]:
    assert isinstance(value, dict)
    reference = value.get("$ref")
    if not isinstance(reference, str):
        return value
    prefix = "#/$defs/"
    assert reference.startswith(prefix)
    definitions = document.get("$defs")
    assert isinstance(definitions, dict)
    resolved = definitions.get(reference[len(prefix):])
    assert isinstance(resolved, dict)
    return resolved


class RecordFieldContractTests(unittest.TestCase):
    def test_contract_fields_and_requiredness_match_dataclasses_and_schemas(self):
        validate_field_contracts()
        for contract in RECORD_FIELD_CONTRACTS:
            with self.subTest(record=contract.record_class.__name__):
                dataclass_fields = {item.name for item in fields(contract.record_class)}
                self.assertEqual(set(contract.field_names), dataclass_fields)
                document = json.loads(
                    (SCHEMA_ROOT / contract.schema_filename).read_text(encoding="utf-8")
                )
                node = _schema_node(document, contract)
                properties = node.get("properties")
                required = node.get("required")
                self.assertIsInstance(properties, dict)
                self.assertIsInstance(required, list)
                self.assertEqual(set(properties), dataclass_fields)
                self.assertEqual(set(required), set(contract.required_names))

    def test_contract_kinds_nullability_and_enums_match_schemas(self):
        for contract in RECORD_FIELD_CONTRACTS:
            document = json.loads(
                (SCHEMA_ROOT / contract.schema_filename).read_text(encoding="utf-8")
            )
            node = _schema_node(document, contract)
            properties = node["properties"]
            assert isinstance(properties, dict)
            for field_spec in contract.fields:
                with self.subTest(
                    record=contract.record_class.__name__, field=field_spec.name
                ):
                    property_schema = _resolve_property(
                        document, properties[field_spec.name]
                    )
                    enum_values = property_schema.get("enum")
                    constant = property_schema.get("const")
                    expected_values = field_spec.allowed_values
                    if field_spec.enum_type is not None:
                        expected_values = tuple(
                            item.value for item in field_spec.enum_type
                        )
                        if field_spec.nullable:
                            expected_values = (*expected_values, None)
                    if expected_values:
                        actual_values = (
                            tuple(enum_values)
                            if isinstance(enum_values, list)
                            else (constant,)
                        )
                        self.assertEqual(actual_values, expected_values)
                        continue
                    schema_type = property_schema.get("type")
                    actual_types = (
                        set(schema_type)
                        if isinstance(schema_type, list)
                        else {schema_type}
                    )
                    self.assertEqual("null" in actual_types, field_spec.nullable)
                    actual_types.discard("null")
                    if field_spec.value_kind == "number":
                        self.assertTrue(actual_types <= {"integer", "number"})
                        self.assertIn("number", actual_types)
                    else:
                        self.assertEqual(actual_types, {field_spec.value_kind})


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


if __name__ == "__main__":
    unittest.main()
