"""Explicit request join and marker-contract tests."""

import unittest

from perfetto_hetero_profiler.hybrid.join import (
    ITERATION_MARKERS,
    MARKER_ORDER,
    join_requests,
    validate_marker_order,
)
from perfetto_hetero_profiler.schema import DeviceType

from tests.hybrid_fixtures import event


def rows(
    names,
    *,
    request_id="request-1",
    start=100,
    step=10,
    role="gpu",
    attributes=None,
):
    result = []
    for index, name in enumerate(names):
        marker_attributes = dict(attributes or {})
        if (
            name in ITERATION_MARKERS
            and "decode.step_index" not in marker_attributes
        ):
            marker_attributes["decode.step_index"] = 0
        result.append(
            event(
                run_id=f"{role}-run",
                event_name=name,
                timestamp_ns=start + index * step,
                host_id=f"{role}-host",
                clock_domain_id="canonical",
                request_id=request_id,
                event_id=f"{role}-{name}-{index}",
                device_type=(
                    DeviceType.GPU if role == "gpu" else DeviceType.NPU
                ),
                attributes=marker_attributes,
            )
        )
    return result


def two_iteration_contract():
    source = rows(MARKER_ORDER)
    loop_end = next(item for item in source if item.event_name == "decode_loop_end")
    response = next(item for item in source if item.event_name == "response_done")
    loop_end.timestamp_ns = 400
    response.timestamp_ns = 410
    source.extend(
        rows(
            ITERATION_MARKERS,
            start=300,
            role="npu",
            attributes={"decode.step_index": 1},
        )
    )
    return source


class MarkerValidationTests(unittest.TestCase):
    def test_full_contract_is_valid(self):
        result = validate_marker_order(rows(MARKER_ORDER))
        self.assertEqual(result.status, "valid")

    def test_missing_marker_is_partial(self):
        result = validate_marker_order(
            rows(tuple(name for name in MARKER_ORDER if name != "kv_transform_start"))
        )
        self.assertEqual(result.status, "partial")
        self.assertIn("kv_transform_start", result.missing_markers)

    def test_duplicate_singleton_is_invalid(self):
        source = rows(MARKER_ORDER)
        source.append(source[0])
        result = validate_marker_order(source)
        self.assertEqual(result.status, "invalid")
        self.assertIn("request_received", result.duplicate_markers)

    def test_repeated_steps_with_distinct_indices_are_allowed(self):
        source = two_iteration_contract()
        result = validate_marker_order(source)
        self.assertEqual(result.status, "valid")
        self.assertNotIn("decode_step_start", result.duplicate_markers)
        self.assertFalse(result.pairing_issues)

    def test_duplicate_step_index_is_invalid(self):
        source = rows(MARKER_ORDER)
        duplicate = rows(
            ("decode_step_start",),
            start=205,
            role="npu",
            attributes={"decode.step_index": 0},
        )[0]
        source.append(duplicate)
        result = validate_marker_order(source)
        self.assertEqual(result.status, "invalid")
        self.assertIn("decode_step_start", result.duplicate_markers)

    def test_iteration_marker_without_step_index_is_invalid(self):
        source = rows(MARKER_ORDER)
        sampling = next(
            item for item in source if item.event_name == "sampling_start"
        )
        sampling.attributes.clear()
        result = validate_marker_order(source)
        self.assertEqual(result.status, "invalid")
        self.assertTrue(
            any("decode.step_index" in issue for issue in result.pairing_issues)
        )

    def test_legacy_hybrid_step_index_is_not_accepted(self):
        source = rows(MARKER_ORDER)
        sampling = next(
            item for item in source if item.event_name == "sampling_start"
        )
        sampling.attributes = {"hybrid.step_index": 0}
        result = validate_marker_order(source)
        self.assertEqual(result.status, "invalid")
        self.assertTrue(
            any("decode.step_index" in issue for issue in result.pairing_issues)
        )

    def test_boolean_decode_step_index_is_not_accepted(self):
        source = rows(MARKER_ORDER)
        sampling = next(
            item for item in source if item.event_name == "sampling_start"
        )
        sampling.attributes["decode.step_index"] = True
        result = validate_marker_order(source)
        self.assertEqual(result.status, "invalid")
        self.assertTrue(
            any("decode.step_index" in issue for issue in result.pairing_issues)
        )

    def test_unpaired_iteration_indices_are_invalid(self):
        source = rows(MARKER_ORDER)
        source.extend(
            rows(
                ("decode_step_start", "decode_step_end"),
                start=300,
                role="npu",
                attributes={"decode.step_index": 1},
            )
        )
        result = validate_marker_order(source)
        self.assertEqual(result.status, "invalid")
        self.assertTrue(any("indices" in issue for issue in result.pairing_issues))

    def test_sampling_pair_reversal_is_invalid(self):
        source = rows(MARKER_ORDER)
        sampling_start = next(
            item for item in source if item.event_name == "sampling_start"
        )
        sampling_end = next(
            item for item in source if item.event_name == "sampling_end"
        )
        sampling_start.timestamp_ns = sampling_end.timestamp_ns + 1
        result = validate_marker_order(source)
        self.assertEqual(result.status, "invalid")
        self.assertTrue(
            any(
                issue.before_name == "sampling_start"
                and issue.after_name == "sampling_end"
                for issue in result.ordering_issues
            )
        )

    def test_decode_loop_must_end_after_final_sampling(self):
        source = two_iteration_contract()
        loop_end = next(
            item for item in source if item.event_name == "decode_loop_end"
        )
        loop_end.timestamp_ns = 320
        result = validate_marker_order(source)
        self.assertEqual(result.status, "invalid")
        self.assertTrue(
            any(
                "ends after decode loop" in issue.reason
                for issue in result.ordering_issues
            )
        )

    def test_definite_ordering_violation(self):
        source = rows(("kv_transfer_end", "decode_loop_start"), start=100)
        source[0].timestamp_ns = 200
        source[1].timestamp_ns = 100
        issue = validate_marker_order(source).ordering_issues[0]
        self.assertEqual(issue.status, "definite_violation")

    def test_uncertainty_overlap_is_not_definite(self):
        source = rows(("kv_transfer_end", "decode_loop_start"), start=100)
        source[0].timestamp_ns = 110
        source[1].timestamp_ns = 100
        source[0].attributes["hybrid.alignment_uncertainty_ns"] = 6
        source[1].attributes["hybrid.alignment_uncertainty_ns"] = 6
        issue = validate_marker_order(source).ordering_issues[0]
        self.assertEqual(issue.status, "within_alignment_uncertainty")

    def test_equal_timestamps_are_valid_order(self):
        source = rows(("prefill_start", "prefill_end"), step=0)
        self.assertFalse(validate_marker_order(source).ordering_issues)

    def test_step_index_time_regression_is_invalid(self):
        source = two_iteration_contract()
        second_start = next(
            item
            for item in source
            if item.event_name == "decode_step_start"
            and item.attributes["decode.step_index"] == 1
        )
        second_start.timestamp_ns = 220
        issues = validate_marker_order(source).ordering_issues
        self.assertTrue(any("iteration index order" in issue.reason for issue in issues))

    def test_response_done_must_follow_optional_token_events(self):
        source = rows(MARKER_ORDER)
        response = next(item for item in source if item.event_name == "response_done")
        source.append(
            event(
                run_id="gpu-run",
                event_name="token_emitted",
                timestamp_ns=response.timestamp_ns + 1,
                host_id="gpu-host",
                clock_domain_id="canonical",
                request_id="request-1",
                event_id="late-token",
                device_type=DeviceType.GPU,
                attributes={"token.index": 0},
            )
        )
        result = validate_marker_order(source)
        self.assertEqual(result.status, "invalid")
        self.assertTrue(
            any(
                "final canonical response marker" in issue.reason
                for issue in result.ordering_issues
            )
        )


class RequestJoinTests(unittest.TestCase):
    def test_exact_request_id_join(self):
        result = join_requests(
            rows(("request_received",), role="gpu"),
            rows(("response_done",), role="npu"),
        )[0]
        self.assertEqual(result.join_method, "request_id")
        self.assertEqual(result.confidence, 1.0)

    def test_duplicate_request_id_is_ambiguous(self):
        gpu = [
            *rows(("request_received",), role="gpu"),
            *rows(("request_received",), role="gpu", start=200),
        ]
        npu = rows(("response_done",), role="npu")
        result = join_requests(gpu, npu)[0]
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.confidence, 0.0)

    def test_transfer_id_join(self):
        gpu = rows(
            ("request_received",),
            request_id="gpu-request",
            attributes={"hybrid.transfer_id": "transfer-1"},
        )
        npu = rows(
            ("response_done",),
            request_id="npu-request",
            role="npu",
            attributes={"hybrid.transfer_id": "transfer-1"},
        )
        self.assertEqual(join_requests(gpu, npu)[0].join_method, "transfer_id")

    def test_kv_transfer_id_join(self):
        gpu = rows(
            ("request_received",),
            request_id="gpu-request",
            attributes={"kv.transfer_id": "transfer-1"},
        )
        npu = rows(
            ("response_done",),
            request_id="npu-request",
            role="npu",
            attributes={"kv.transfer_id": "transfer-1"},
        )
        self.assertEqual(join_requests(gpu, npu)[0].join_method, "transfer_id")

    def test_nixl_transfer_id_join(self):
        gpu = rows(
            ("request_received",),
            request_id="gpu-request",
            attributes={"nixl.transfer_id": "transfer-1"},
        )
        npu = rows(
            ("response_done",),
            request_id="npu-request",
            role="npu",
            attributes={"nixl.transfer_id": "transfer-1"},
        )
        self.assertEqual(join_requests(gpu, npu)[0].join_method, "transfer_id")

    def test_correlation_id_join(self):
        gpu = rows(
            ("request_received",),
            request_id="gpu-request",
            attributes={"hybrid.correlation_id": "correlation-1"},
        )
        npu = rows(
            ("response_done",),
            request_id="npu-request",
            role="npu",
            attributes={"hybrid.correlation_id": "correlation-1"},
        )
        self.assertEqual(join_requests(gpu, npu)[0].join_method, "correlation_id")

    def test_duplicate_gpu_correlation_is_ambiguous(self):
        gpu = [
            *rows(
                ("request_received",),
                request_id="gpu-1",
                attributes={"hybrid.correlation_id": "shared"},
            ),
            *rows(
                ("request_received",),
                request_id="gpu-2",
                start=200,
                attributes={"hybrid.correlation_id": "shared"},
            ),
        ]
        npu = rows(
            ("response_done",),
            request_id="npu",
            role="npu",
            attributes={"hybrid.correlation_id": "shared"},
        )
        result = join_requests(gpu, npu)
        self.assertFalse(any(item.status == "joined" for item in result))
        self.assertTrue(any(item.status == "ambiguous" for item in result))

    def test_missing_identifier_is_not_joined(self):
        result = join_requests(
            rows(("request_received",), request_id="gpu"),
            rows(("response_done",), request_id="npu", role="npu"),
        )
        self.assertTrue(all(item.status == "not_joined" for item in result))

    def test_time_proximity_does_not_join(self):
        result = join_requests(
            rows(("request_received",), request_id="gpu", start=100),
            rows(("response_done",), request_id="npu", start=100, role="npu"),
        )
        self.assertFalse(any(item.status == "joined" for item in result))

    def test_ambiguous_transfer_is_not_auto_joined(self):
        gpu = rows(
            ("request_received",),
            request_id="gpu",
            attributes={"hybrid.transfer_id": "x"},
        )
        npu = [
            *rows(
                ("response_done",),
                request_id="npu-1",
                role="npu",
                attributes={"hybrid.transfer_id": "x"},
            ),
            *rows(
                ("response_done",),
                request_id="npu-2",
                role="npu",
                start=200,
                attributes={"hybrid.transfer_id": "x"},
            ),
        ]
        result = join_requests(gpu, npu)
        self.assertTrue(any(item.status == "ambiguous" for item in result))

    def test_missing_request_id_can_use_correlation(self):
        gpu = rows(
            ("request_received",),
            request_id=None,
            attributes={"hybrid.correlation_id": "x"},
        )
        npu = rows(
            ("response_done",),
            request_id=None,
            role="npu",
            attributes={"hybrid.correlation_id": "x"},
        )
        self.assertTrue(
            any(item.join_method == "correlation_id" for item in join_requests(gpu, npu))
        )

    def test_join_reports_missing_markers(self):
        result = join_requests(
            rows(("request_received",), role="gpu"),
            rows(("response_done",), role="npu"),
        )[0]
        self.assertEqual(result.status, "partial")
        self.assertIn("prefill_start", result.missing_markers)

    def test_join_preserves_both_event_sets(self):
        result = join_requests(
            rows(("request_received", "prefill_start"), role="gpu"),
            rows(("response_done",), role="npu"),
        )[0]
        self.assertEqual(len(result.events), 3)


if __name__ == "__main__":
    unittest.main()
