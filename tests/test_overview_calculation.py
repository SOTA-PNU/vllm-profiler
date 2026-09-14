from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from perfetto_hetero_profiler.overview.calculation import (
    OverviewCalculationError,
    calculate_overview_kpis,
    union_duration_ns,
)
from perfetto_hetero_profiler.schema import (
    EventRecord,
    MetricSample,
)
from tests.support.overview_calculation import (
    ALIGNMENT,
    CLIENT_REQUEST_ID,
    CORRELATION_ID,
    _fixture,
    _metric,
    _section_by_name,
    _two_request_fixture,
)


class OverviewCalculationTests(unittest.TestCase):
    def test_complete_fixture_uses_distinct_observation_layers(self) -> None:
        result = calculate_overview_kpis(_fixture())

        self.assertEqual(
            list(result),
            [
                "request_facing_latency",
                "pipeline_latency",
                "throughput_and_tokens",
                "transfer",
                "resource_summaries",
            ],
        )
        request = _section_by_name(result, "request_facing_latency")
        pipeline = _section_by_name(result, "pipeline_latency")
        transfer = _section_by_name(result, "transfer")
        self.assertEqual(request["latency.e2e"]["value"], 110)
        self.assertEqual(pipeline["latency.e2e"]["value"], 100)
        self.assertEqual(
            request["latency.e2e"]["scope"]["observation_layer"],
            "request_facing_client",
        )
        self.assertEqual(
            pipeline["latency.e2e"]["scope"]["observation_layer"],
            "hybrid_pipeline",
        )
        self.assertEqual(pipeline["latency.sampling"]["value"], 8)
        self.assertEqual(pipeline["latency.sampling"]["sample_count"], 8)
        self.assertIn(
            "first repeated marker pair",
            pipeline["latency.sampling"]["quality_warnings"][0],
        )
        self.assertEqual(transfer["transfer.bytes"]["value"], 100)
        self.assertEqual(transfer["transfer.duration"]["value"], 10)
        self.assertEqual(
            transfer["transfer.effective_bandwidth"]["value"], 10_000_000_000
        )
        self.assertEqual(transfer["transfer.e2e_share"]["value"], 0.1)
        self.assertEqual(
            transfer["transfer.wait_duration"]["availability"], "not_available"
        )

    def test_explicit_transfer_id_join_is_pipeline_provenance(self) -> None:
        loaded = _fixture()
        loaded.metrics = tuple(
            replace(
                metric,
                dimensions={**metric.dimensions, "hybrid.join_method": "transfer_id"},
            )
            if metric.dimensions.get("hybrid.join_method") == "correlation_id"
            else metric
            for metric in loaded.metrics
        )
        result = calculate_overview_kpis(loaded)
        request = _section_by_name(result, "request_facing_latency")
        pipeline = _section_by_name(result, "pipeline_latency")
        self.assertEqual(request["latency.e2e"]["value"], 110)
        self.assertEqual(pipeline["latency.e2e"]["value"], 100)

    def test_runtime_boundary_kpis_preserve_values_samples_and_sources(self) -> None:
        loaded = _fixture()
        names_values = {
            "transfer.handoff_duration": 4,
            "transfer.setup_duration": 5,
            "transfer.wait_duration": 6,
            "decode.schedule_wait_duration": 7,
        }
        metrics = list(loaded.metrics)
        for name, value in names_values.items():
            metrics.append(
                _metric(
                    name,
                    value,
                    request_id=CORRELATION_ID,
                    interval_ns=value,
                    source_event_ids=[f"{name}-start", f"{name}-end"],
                )
            )
        result = calculate_overview_kpis(replace_metric_list(loaded, metrics))
        transfer = _section_by_name(result, "transfer")
        for name, value in names_values.items():
            with self.subTest(name=name):
                self.assertEqual(transfer[name]["availability"], "available")
                self.assertEqual(transfer[name]["value"], value)
                self.assertEqual(transfer[name]["sample_count"], 1)
                self.assertEqual(
                    transfer[name]["sources"][0]["record_ids"],
                    [f"{name}-end", f"{name}-start"],
                )
                self.assertTrue(transfer[name]["quality_warnings"])

    def test_multiple_requests_use_explicit_run_aggregates(self) -> None:
        result = calculate_overview_kpis(_two_request_fixture())
        request = _section_by_name(result, "request_facing_latency")
        pipeline = _section_by_name(result, "pipeline_latency")
        throughput = _section_by_name(result, "throughput_and_tokens")
        transfer = _section_by_name(result, "transfer")

        self.assertEqual(request["latency.e2e"]["value"], 110)
        self.assertEqual(request["latency.e2e"]["scope"]["scope_type"], "run")
        self.assertEqual(
            request["latency.e2e"]["aggregation_method"],
            "arithmetic_mean_across_measured_requests_v1",
        )
        self.assertEqual(pipeline["latency.sampling"]["value"], 8)
        self.assertEqual(throughput["request.count"]["value"], 2)
        self.assertEqual(throughput["request.input_tokens"]["value"], 6)
        self.assertEqual(throughput["request.output_tokens"]["value"], 4)
        self.assertEqual(throughput["request.total_tokens"]["value"], 10)
        self.assertEqual(transfer["transfer.bytes"]["availability"], "not_available")
        self.assertEqual(transfer["transfer.duration"]["value"], 10)
        self.assertEqual(transfer["transfer.duration"]["scope"]["scope_type"], "run")

        required = {
            "name",
            "canonical_unit",
            "availability",
            "value",
            "unavailable_reason",
            "aggregation_method",
            "sample_count",
            "sources",
            "scope",
            "calculation",
            "clock",
            "quality_warnings",
            "display",
        }
        for section in (
            "request_facing_latency",
            "pipeline_latency",
            "throughput_and_tokens",
            "transfer",
        ):
            for kpi in result[section]:
                self.assertEqual(set(kpi), required)

    def test_missing_duplicate_and_reversed_pairs_are_unavailable(self) -> None:
        cases = {}
        loaded = _fixture()
        cases["missing"] = replace_event_list(
            loaded,
            [
                event
                for event in loaded.events
                if event.event_name != "prefill_end"
            ],
        )
        loaded = _fixture()
        duplicate = next(
            event for event in loaded.events if event.event_name == "prefill_start"
        )
        cases["duplicate"] = replace_event_list(
            loaded, [*loaded.events, replace(duplicate, event_id="duplicate")]
        )
        loaded = _fixture()
        cases["reversed"] = replace_event_list(
            loaded,
            [
                replace(event, timestamp_ns=9)
                if event.event_name == "prefill_end"
                else event
                for event in loaded.events
            ],
        )
        expected_reason = {
            "missing": "missing",
            "duplicate": "ambiguous",
            "reversed": "reversed",
        }
        for label, fixture in cases.items():
            with self.subTest(label=label):
                result = calculate_overview_kpis(fixture)
                prefill = _section_by_name(result, "pipeline_latency")[
                    "latency.prefill"
                ]
                self.assertEqual(prefill["availability"], "not_available")
                self.assertIsNone(prefill["value"])
                self.assertIn(
                    expected_reason[label], prefill["unavailable_reason"]
                )

    def test_correlation_id_is_mandatory_and_not_inferred_by_time(self) -> None:
        loaded = _fixture()
        events = []
        for event in loaded.events:
            if event.event_name == "prefill_start":
                attributes = dict(event.attributes)
                attributes.pop("hybrid.correlation_id")
                event = replace(event, attributes=attributes)
            events.append(event)
        loaded = replace_event_list(loaded, events)
        with self.assertRaisesRegex(
            OverviewCalculationError, "explicit hybrid.correlation_id"
        ):
            calculate_overview_kpis(loaded)

    def test_transfer_id_and_byte_count_must_match(self) -> None:
        for field, replacement, message in (
            ("hybrid.transfer_id", "other-transfer", "transfer_id"),
            ("kv.transfer_bytes", 99, "byte counts disagree"),
        ):
            loaded = _fixture()
            events = []
            for event in loaded.events:
                if event.event_name == "kv_transfer_end":
                    attributes = dict(event.attributes)
                    attributes[field] = replacement
                    event = replace(event, attributes=attributes)
                events.append(event)
            with self.subTest(field=field):
                with self.assertRaisesRegex(OverviewCalculationError, message):
                    calculate_overview_kpis(replace_event_list(loaded, events))

    def test_bool_nonfinite_and_unit_mismatch_are_rejected(self) -> None:
        for bad_value, message in (
            (True, "non-bool"),
            (float("nan"), "finite"),
            (float("inf"), "finite"),
        ):
            loaded = _fixture()
            metrics = [
                replace(metric, value=bad_value)
                if metric.metric_name == "latency.ttft"
                else metric
                for metric in loaded.metrics
            ]
            with self.subTest(value=bad_value):
                with self.assertRaisesRegex(OverviewCalculationError, message):
                    calculate_overview_kpis(replace_metric_list(loaded, metrics))

        loaded = _fixture()
        metrics = [
            replace(metric, unit="ms")
            if metric.metric_name == "latency.ttft"
            else metric
            for metric in loaded.metrics
        ]
        with self.assertRaisesRegex(OverviewCalculationError, "unit mismatch"):
            calculate_overview_kpis(replace_metric_list(loaded, metrics))

    def test_tpot_is_unavailable_for_zero_or_one_output_token(self) -> None:
        for output_tokens in (0, 1):
            with self.subTest(output_tokens=output_tokens):
                result = calculate_overview_kpis(
                    _fixture(output_tokens=output_tokens)
                )
                tpot = _section_by_name(result, "request_facing_latency")[
                    "latency.tpot"
                ]
                self.assertEqual(tpot["availability"], "not_available")
                self.assertIsNone(tpot["value"])
                self.assertIn("at least two", tpot["unavailable_reason"])

        loaded = _fixture(output_tokens=2)
        metrics = [
            replace(metric, source_event_ids=None)
            if metric.metric_name == "latency.tpot"
            else metric
            for metric in loaded.metrics
        ]
        result = calculate_overview_kpis(replace_metric_list(loaded, metrics))
        tpot = _section_by_name(result, "request_facing_latency")[
            "latency.tpot"
        ]
        self.assertEqual(tpot["availability"], "not_available")
        self.assertIn("timestamp provenance", tpot["unavailable_reason"])

    def test_throughput_reconciliation_is_exact_and_window_scoped(self) -> None:
        loaded = _fixture()
        metrics = [
            replace(metric, value=metric.value + 1)
            if metric.metric_name == "throughput.total_tokens"
            else metric
            for metric in loaded.metrics
        ]
        with self.assertRaisesRegex(
            OverviewCalculationError, "count / measured_smoke"
        ):
            calculate_overview_kpis(replace_metric_list(loaded, metrics))

        loaded = _fixture()
        metrics = [
            replace(metric, dimensions={"window": "warmup"}, attributes={**ALIGNMENT})
            if metric.metric_name == "throughput.requests"
            else metric
            for metric in loaded.metrics
        ]
        with self.assertRaisesRegex(
            OverviewCalculationError, "exactly one measured_smoke"
        ):
            calculate_overview_kpis(replace_metric_list(loaded, metrics))

    def test_zero_denominators_produce_unavailable_derived_kpis(self) -> None:
        loaded = _fixture()
        events = [
            replace(event, timestamp_ns=30)
            if event.event_name == "kv_transfer_end"
            else event
            for event in loaded.events
        ]
        metrics = [
            replace(metric, value=0, interval_ns=0)
            if metric.metric_name == "latency.kv_transfer"
            and metric.request_id == CORRELATION_ID
            else metric
            for metric in loaded.metrics
        ]
        result = calculate_overview_kpis(
            replace_metric_list(replace_event_list(loaded, events), metrics)
        )
        transfer = _section_by_name(result, "transfer")
        self.assertEqual(transfer["transfer.duration"]["value"], 0)
        self.assertEqual(
            transfer["transfer.effective_bandwidth"]["availability"],
            "not_available",
        )

        loaded = _fixture()
        events = [
            replace(event, timestamp_ns=0)
            if event.event_name == "response_done"
            else event
            for event in loaded.events
        ]
        metrics = [
            replace(metric, value=0, interval_ns=0)
            if metric.metric_name == "latency.e2e"
            and metric.request_id == CORRELATION_ID
            else metric
            for metric in loaded.metrics
        ]
        result = calculate_overview_kpis(
            replace_metric_list(replace_event_list(loaded, events), metrics)
        )
        share = _section_by_name(result, "transfer")["transfer.e2e_share"]
        self.assertEqual(share["availability"], "not_available")
        self.assertIn("zero", share["unavailable_reason"])

    def test_clock_evidence_is_not_fabricated(self) -> None:
        loaded = _fixture()
        events = []
        for event in loaded.events:
            attributes = {
                key: value
                for key, value in event.attributes.items()
                if not key.startswith("hybrid.alignment_")
            }
            events.append(replace(event, attributes=attributes))
        metrics = []
        for metric in loaded.metrics:
            attributes = {
                key: value
                for key, value in metric.attributes.items()
                if not key.startswith("hybrid.alignment_")
            }
            metrics.append(replace(metric, attributes=attributes))
        result = calculate_overview_kpis(
            replace_metric_list(replace_event_list(loaded, events), metrics)
        )
        request_clock = result["request_facing_latency"][0]["clock"]
        self.assertEqual(request_clock["alignment_status"], "unknown")
        self.assertIsNone(request_clock["offset_ns"])
        self.assertIsNone(request_clock["uncertainty_ns"])
        pipeline = _section_by_name(result, "pipeline_latency")
        self.assertEqual(pipeline["latency.e2e"]["availability"], "not_available")

    def test_raw_request_is_reconciled_and_explicitly_linked(self) -> None:
        loaded = _fixture()
        row = {
            "client_request_id": CLIENT_REQUEST_ID,
            "client_request_hash": "explicit-hash",
            "start_monotonic_ns": 1_000,
            "end_monotonic_ns": 1_110,
            "e2e_ns": 110,
            "ttft_ns": 20,
            "tpot_ns": 5.0,
            "output_tokens": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw/client/measured_requests.jsonl"
            path.parent.mkdir(parents=True)
            payload = (json.dumps(row, sort_keys=True) + "\n").encode()
            path.write_bytes(payload)
            artifact = SimpleNamespace(
                artifact_id="measured-requests",
                relative_path="raw/client/measured_requests.jsonl",
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            source = SimpleNamespace(root=root, artifacts=(artifact,))
            loaded.sources = (source,)
            result = calculate_overview_kpis(loaded)
            sources = result["request_facing_latency"][0]["sources"]
            raw = next(
                source
                for source in sources
                if source["source_kind"] == "raw_measured_request"
            )
            self.assertEqual(
                raw["details"]["pipeline_link"]["method"], "client_request_hash"
            )

            row["ttft_ns"] = 21
            payload = (json.dumps(row, sort_keys=True) + "\n").encode()
            path.write_bytes(payload)
            artifact.size_bytes = len(payload)
            artifact.sha256 = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(
                OverviewCalculationError, "raw ttft_ns"
            ):
                calculate_overview_kpis(loaded)

    def test_non_hybrid_input_does_not_invent_pipeline_values(self) -> None:
        loaded = _fixture()
        loaded.events = ()
        loaded.metrics = tuple(
            metric
            for metric in loaded.metrics
            if metric.dimensions.get("hybrid.join_method") != "correlation_id"
        )
        result = calculate_overview_kpis(loaded)
        self.assertTrue(
            all(
                item["availability"] == "not_available"
                for item in result["pipeline_latency"]
            )
        )
        self.assertTrue(
            all(
                item["availability"] == "not_available"
                for item in result["transfer"]
            )
        )

    def test_missing_request_metrics_use_run_scope_without_inventing_request(self) -> None:
        loaded = _fixture()
        loaded.metrics = tuple(
            metric
            for metric in loaded.metrics
            if metric.metric_name
            not in {"latency.e2e", "latency.ttft", "latency.tpot"}
        )
        request = calculate_overview_kpis(loaded)["request_facing_latency"]
        self.assertTrue(
            all(item["availability"] == "not_available" for item in request)
        )
        self.assertTrue(
            all(item["scope"]["scope_type"] == "run" for item in request)
        )
        self.assertTrue(
            all(item["scope"]["request_id"] is None for item in request)
        )

    def test_wait_union_does_not_double_count_overlap(self) -> None:
        self.assertEqual(union_duration_ns([(0, 10), (5, 15), (20, 25)]), 20)
        self.assertEqual(union_duration_ns([]), 0)
        with self.assertRaisesRegex(OverviewCalculationError, "reversed"):
            union_duration_ns([(2, 1)])
        with self.assertRaisesRegex(OverviewCalculationError, "non-bool"):
            union_duration_ns([(False, 1)])


class IntervalUnionBoundaryTests(unittest.TestCase):
    """Pin the exact edge behaviour of the wait-interval union."""

    def test_adjacent_intervals_are_joined_without_double_counting(self) -> None:
        self.assertEqual(union_duration_ns([(0, 10), (10, 20)]), 20)

    def test_disjoint_intervals_exclude_the_gap(self) -> None:
        self.assertEqual(union_duration_ns([(0, 10), (30, 40)]), 20)

    def test_fully_contained_interval_does_not_extend_the_union(self) -> None:
        self.assertEqual(union_duration_ns([(0, 100), (10, 20)]), 100)

    def test_zero_length_interval_contributes_nothing(self) -> None:
        self.assertEqual(union_duration_ns([(5, 5)]), 0)
        self.assertEqual(union_duration_ns([(0, 10), (5, 5)]), 10)

    def test_unsorted_input_yields_the_same_union(self) -> None:
        self.assertEqual(
            union_duration_ns([(30, 40), (0, 10), (5, 15)]),
            union_duration_ns([(0, 10), (5, 15), (30, 40)]),
        )

    def test_bool_endpoints_are_rejected_on_either_side(self) -> None:
        with self.assertRaisesRegex(OverviewCalculationError, "non-bool"):
            union_duration_ns([(0, True)])
        with self.assertRaisesRegex(OverviewCalculationError, "non-bool"):
            union_duration_ns([(True, 5)])

    def test_float_endpoints_are_rejected(self) -> None:
        with self.assertRaisesRegex(OverviewCalculationError, "non-bool integer"):
            union_duration_ns([(0.0, 10.0)])


def replace_event_list(
    loaded: SimpleNamespace, events: list[EventRecord]
) -> SimpleNamespace:
    return SimpleNamespace(**{**loaded.__dict__, "events": tuple(events)})


def replace_metric_list(
    loaded: SimpleNamespace, metrics: list[MetricSample]
) -> SimpleNamespace:
    return SimpleNamespace(**{**loaded.__dict__, "metrics": tuple(metrics)})


if __name__ == "__main__":
    unittest.main()
