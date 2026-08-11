"""Official metric catalog tests."""

from pathlib import Path
import unittest

from perfetto_hetero_profiler.schema import METRIC_CATALOG, MetricKind, MetricScope


class MetricCatalogTests(unittest.TestCase):
    def test_catalog_has_expected_size(self) -> None:
        self.assertEqual(len(METRIC_CATALOG), 40)

    def test_required_metric_groups_exist(self) -> None:
        for name in (
            "latency.e2e",
            "latency.ttft",
            "latency.tpot",
            "throughput.requests",
            "resource.gpu.power",
            "resource.npu.power",
            "transfer.effective_bandwidth",
            "transfer.e2e_share",
            "latency.sampling",
            "hybrid.alignment_uncertainty",
        ):
            self.assertIn(name, METRIC_CATALOG)

    def test_latency_definition(self) -> None:
        definition = METRIC_CATALOG["latency.e2e"]
        self.assertEqual(definition.unit, "ns")
        self.assertIs(definition.kind, MetricKind.DURATION)
        self.assertEqual(definition.allowed_scopes, (MetricScope.REQUEST,))
        self.assertTrue(definition.derived)

    def test_percent_bounds(self) -> None:
        definition = METRIC_CATALOG["resource.gpu.utilization"]
        self.assertEqual((definition.minimum, definition.maximum), (0, 100))

    def test_metric_document_contains_every_name(self) -> None:
        text = Path("docs/metric_catalog_v1.md").read_text(encoding="utf-8")
        missing = [name for name in METRIC_CATALOG if name not in text]
        self.assertEqual(missing, [])
