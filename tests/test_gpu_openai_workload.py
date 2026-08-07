"""Tests for text-free streaming observations and derived metrics."""

import json
import unittest

from perfetto_hetero_profiler.gpu.openai_client import (
    CompletionObservation,
    OpenAICompletionClient,
)
from perfetto_hetero_profiler.gpu.workload import (
    measured_window_metrics,
    observation_events,
    observation_metrics,
)
from perfetto_hetero_profiler.schema import record_to_dict


class FakeResponse:
    def __init__(self, chunks, status=200):
        self.status = status
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.chunks)


def sse(value) -> bytes:
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value)
    return f"data: {payload}\n\n".encode()


class CompletionClientTests(unittest.TestCase):
    def client(self, chunks, times):
        captured = {}

        def opener(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse(chunks)

        iterator = iter(times)
        return (
            OpenAICompletionClient(
                "http://127.0.0.1:18080",
                timeout_sec=3,
                monotonic_ns=lambda: next(iterator),
                opener=opener,
            ),
            captured,
        )

    def good_chunks(self):
        return [
            sse({"choices": [{"text": "secret", "token_ids": [1]}]}),
            sse({"choices": [{"text": "secret2", "token_ids": [2, 3]}]}),
            sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 3,
                        "total_tokens": 7,
                    },
                }
            ),
            sse("[DONE]"),
        ]

    def test_stream_observation_uses_exact_token_ids(self) -> None:
        client, _ = self.client(
            self.good_chunks(), [100, 110, 120, 150, 170, 190]
        )
        observation = client.complete(
            model="/models/qwen",
            request_id="request-1",
            prompt="do not retain",
            max_output_tokens=8,
        )
        self.assertEqual(observation.token_timestamps_ns, (120, 150, 150))
        self.assertEqual(observation.output_tokens, 3)
        self.assertEqual(observation.done_ns, 190)
        self.assertEqual(observation.response_started_ns, 110)

    def test_request_enables_stream_usage_and_token_ids(self) -> None:
        client, captured = self.client(
            self.good_chunks(), [100, 110, 120, 150, 170, 190]
        )
        client.complete(
            model="/models/qwen",
            request_id="request-1",
            prompt="private prompt",
            max_output_tokens=8,
        )
        body = json.loads(captured["request"].data)
        self.assertTrue(body["stream"])
        self.assertTrue(body["stream_options"]["include_usage"])
        self.assertTrue(body["return_token_ids"])
        self.assertEqual(captured["request"].get_header("X-request-id"), "request-1")

    def test_configurable_temperature_preserves_stream_contract(self) -> None:
        client, captured = self.client(
            self.good_chunks(), [100, 110, 120, 150, 170, 190]
        )
        client.complete(
            model="m", request_id="r", prompt="p", max_output_tokens=8,
            temperature=0.25, stream=True,
        )
        self.assertEqual(json.loads(captured["request"].data)["temperature"], 0.25)

    def test_non_streaming_has_no_fabricated_token_timestamps(self) -> None:
        body = json.dumps(
            {
                "choices": [{"text": "answer", "token_ids": [1, 2]}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            }
        ).encode()

        class NonStreamingResponse(FakeResponse):
            def read(self):
                return body

        captured = {}

        def opener(request, timeout):
            captured["request"] = request
            return NonStreamingResponse([])

        client = OpenAICompletionClient(
            "http://127.0.0.1:1", timeout_sec=1,
            monotonic_ns=iter([10, 15, 20]).__next__, opener=opener,
        )
        result = client.complete(
            model="m", request_id="r", prompt="p", max_output_tokens=2,
            stream=False,
        )
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.token_timestamps_ns, ())
        self.assertIsNone(result.ttft_ns)
        self.assertIsNone(result.tpot_ns)
        self.assertEqual(result.response_started_ns, 15)
        self.assertNotIn("stream_options", json.loads(captured["request"].data))

    def test_rejects_missing_usage(self) -> None:
        client, _ = self.client([sse("[DONE]")], [1, 2, 3])
        with self.assertRaisesRegex(RuntimeError, "exact usage"):
            client.complete(
                model="m", request_id="r", prompt="p", max_output_tokens=1
            )

    def test_rejects_missing_done_marker(self) -> None:
        chunks = [
            sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 0,
                        "total_tokens": 1,
                    },
                }
            )
        ]
        client, _ = self.client(chunks, [1, 2, 3])
        with self.assertRaisesRegex(RuntimeError, "without \\[DONE\\]"):
            client.complete(
                model="m", request_id="r", prompt="p", max_output_tokens=1
            )

    def test_rejects_usage_token_mismatch(self) -> None:
        chunks = [
            sse({"choices": [{"token_ids": [1]}]}),
            sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                }
            ),
            sse("[DONE]"),
        ]
        client, _ = self.client(chunks, [1, 2, 3, 4, 5])
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            client.complete(
                model="m", request_id="r", prompt="p", max_output_tokens=1
            )

    def test_rejects_empty_request_id(self) -> None:
        client, _ = self.client([], [])
        with self.assertRaisesRegex(ValueError, "request_id"):
            client.complete(
                model="m", request_id="", prompt="p", max_output_tokens=1
            )

    def test_rejects_excessive_output_limit(self) -> None:
        client, _ = self.client([], [])
        with self.assertRaisesRegex(ValueError, "16"):
            client.complete(
                model="m", request_id="r", prompt="p", max_output_tokens=17
            )


class WorkloadRecordTests(unittest.TestCase):
    def observation(self, request_id="r1", offset=0):
        return CompletionObservation(
            request_id=request_id,
            received_ns=100 + offset,
            token_timestamps_ns=(130 + offset, 150 + offset, 180 + offset),
            done_ns=200 + offset,
            input_tokens=4,
            output_tokens=3,
            total_tokens=7,
            http_status=200,
        )

    def test_only_allowed_external_events_are_emitted(self) -> None:
        names = [event.event_name for event in observation_events("run", self.observation())]
        self.assertEqual(
            names,
            [
                "request_received",
                "first_token_emitted",
                "token_emitted",
                "token_emitted",
                "response_done",
            ],
        )

    def test_events_use_one_request_and_clock_domain(self) -> None:
        records = observation_events("run", self.observation())
        self.assertEqual({record.request_id for record in records}, {"r1"})
        self.assertEqual(
            {record.clock_domain_id for record in records}, {"host-monotonic"}
        )

    def test_events_validate_against_schema(self) -> None:
        for record in observation_events("run", self.observation()):
            record_to_dict(record)

    def test_request_metrics_include_required_values(self) -> None:
        metrics = observation_metrics("run", self.observation())
        names = {metric.metric_name for metric in metrics}
        self.assertTrue(
            {
                "latency.e2e",
                "latency.ttft",
                "latency.tpot",
                "request.output_tokens",
            }.issubset(names)
        )

    def test_request_metrics_validate_against_schema(self) -> None:
        for record in observation_metrics("run", self.observation()):
            record_to_dict(record)

    def test_tpot_uses_inter_token_intervals(self) -> None:
        by_name = {
            metric.metric_name: metric
            for metric in observation_metrics("run", self.observation())
        }
        self.assertEqual(by_name["latency.tpot"].value, 25)
        self.assertEqual(
            by_name["latency.tpot"].attributes["vllm.timestamp_source"],
            "client_stream_arrival",
        )
        self.assertIn(
            "output_tokens-1",
            by_name["latency.tpot"].attributes["vllm.calculation"],
        )

    def test_tpot_omitted_for_one_token(self) -> None:
        observation = CompletionObservation(
            request_id="r",
            received_ns=10,
            token_timestamps_ns=(20,),
            done_ns=30,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            http_status=200,
        )
        names = {item.metric_name for item in observation_metrics("run", observation)}
        self.assertNotIn("latency.tpot", names)

    def test_measured_window_excludes_warmup_by_input(self) -> None:
        metrics = measured_window_metrics("run", [self.observation()])
        count = next(item for item in metrics if item.metric_name == "request.count")
        self.assertEqual(count.value, 1)
        self.assertEqual(count.dimensions, {"window": "measured_smoke"})

    def test_measured_window_throughput_values(self) -> None:
        observations = [self.observation("r1"), self.observation("r2", 100)]
        metrics = measured_window_metrics("run", observations)
        output = next(
            item for item in metrics if item.metric_name == "throughput.output_tokens"
        )
        self.assertEqual(output.value, 30_000_000)

    def test_empty_window_has_no_metrics(self) -> None:
        self.assertEqual(measured_window_metrics("run", []), [])


if __name__ == "__main__":
    unittest.main()
