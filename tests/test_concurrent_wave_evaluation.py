"""CPU-only tests for the repository concurrent-wave evaluation adapter."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from perfetto_hetero_profiler.gpu.openai_client import CompletionObservation
from tools.evaluation.concurrent_wave import (
    BlockSpec,
    ConcurrentWaveClient,
    ConcurrentWaveError,
    load_matrix_blocks,
)


class _State:
    def __init__(self, failures=(), timeouts=(), token_mismatches=()):
        self.failures = set(failures)
        self.timeouts = set(timeouts)
        self.token_mismatches = set(token_mismatches)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.maximum = 0
        self.call_times: dict[str, int] = {}
        self.closed = 0


class _FakeClient:
    def __init__(self, state: _State):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        with self.state.lock:
            self.state.closed += 1

    def complete(self, **request):
        request_id = request["request_id"]
        started = time.monotonic_ns()
        with self.state.lock:
            self.state.call_times[request_id] = started
            self.state.in_flight += 1
            self.state.maximum = max(self.state.maximum, self.state.in_flight)
        try:
            index = int(request_id.rsplit("-", 1)[1])
            time.sleep(0.004 * (4 - index % 4))
            if request_id in self.state.timeouts:
                raise TimeoutError("secret server timeout detail")
            if request_id in self.state.token_mismatches:
                raise RuntimeError("completion usage does not match secret token text")
            if request_id in self.state.failures:
                raise RuntimeError("secret generated response")
            ended = time.monotonic_ns()
            return CompletionObservation(
                request_id=request_id,
                received_ns=started,
                token_timestamps_ns=(started + 1,),
                done_ns=ended,
                input_tokens=256,
                output_tokens=1,
                total_tokens=257,
                http_status=200,
                response_started_ns=started + 1,
            )
        finally:
            with self.state.lock:
                self.state.in_flight -= 1


def _block(concurrency: int, measured_requests: int | None = None) -> BlockSpec:
    measured = measured_requests or concurrency * 2
    return BlockSpec(
        condition_id=f"m15-b{concurrency}-hybrid",
        round_index=2,
        concurrency=concurrency,
        topology="hybrid",
        warmup_requests=concurrency,
        measured_requests=measured,
        measured_waves=measured // concurrency,
    )


def _client(block: BlockSpec, state: _State) -> ConcurrentWaveClient:
    return ConcurrentWaveClient(
        "http://127.0.0.1:1",
        timeout_sec=1,
        block=block,
        client_factory=lambda *_args, **_kwargs: _FakeClient(state),
    )


def _complete_wave(client: ConcurrentWaveClient, concurrency: int, wave=0):
    start = wave * concurrency
    return [
        client.complete(
            model="model",
            request_id=f"run-measured-{index:03d}",
            prompt="TOP SECRET PROMPT",
            max_output_tokens=1,
            temperature=0,
            stream=True,
        )
        for index in range(start, start + concurrency)
    ]


class ConcurrentWaveTests(unittest.TestCase):
    def test_concurrency_one_two_four_reaches_maximum_and_overlap(self):
        for concurrency in (1, 2, 4):
            with self.subTest(concurrency=concurrency):
                state = _State()
                client = _client(_block(concurrency), state)
                observations = _complete_wave(client, concurrency)
                wave = client.wave_rows[0]
                self.assertEqual(len(observations), concurrency)
                self.assertEqual(state.maximum, concurrency)
                self.assertEqual(wave["max_in_flight"], concurrency)
                self.assertGreater(wave["common_overlap_ns"], 0)
                self.assertTrue(wave["expected_concurrency_reached"])
                self.assertTrue(wave["success"])
                self.assertEqual(client.active_task_count, 0)

    def test_no_transport_call_occurs_before_barrier_release(self):
        state = _State()
        client = _client(_block(4), state)
        _complete_wave(client, 4)
        release = client.wave_rows[0]["barrier_release_monotonic_ns"]
        self.assertTrue(client.wave_rows[0]["barrier_respected"])
        self.assertTrue(all(start >= release for start in state.call_times.values()))

    def test_failure_and_timeout_fail_wave_and_collect_every_task(self):
        for attribute in ("failures", "timeouts"):
            with self.subTest(attribute=attribute):
                failed = "run-measured-001"
                state = _State(**{attribute: (failed,)})
                client = _client(_block(4), state)
                with self.assertRaisesRegex(
                    ConcurrentWaveError, "failed concurrency validation"
                ):
                    _complete_wave(client, 4)
                self.assertFalse(client.wave_rows[0]["success"])
                self.assertEqual(len(client.request_rows), 4)
                self.assertEqual(client.active_task_count, 0)
                self.assertEqual(state.in_flight, 0)
                self.assertEqual(state.closed, 4)
                client.close()
                self.assertEqual(client.active_task_count, 0)

    def test_token_count_mismatch_fails_entire_wave(self):
        state = _State(token_mismatches=("run-measured-000",))
        client = _client(_block(2), state)
        with self.assertRaises(ConcurrentWaveError):
            _complete_wave(client, 2)
        failed = client.request_rows[0]
        self.assertFalse(client.wave_rows[0]["success"])
        self.assertEqual(failed["error"], "RuntimeError: token count mismatch")

    def test_invalid_concurrency_and_indivisible_counts_are_rejected(self):
        state = _State()
        with self.assertRaisesRegex(ConcurrentWaveError, "one of 1, 2, or 4"):
            _client(_block(3, 6), state)
        with self.assertRaisesRegex(ConcurrentWaveError, "divisible"):
            _client(_block(4, 6), state)

    def test_output_order_is_deterministic_and_text_free(self):
        state = _State()
        client = _client(_block(4), state)
        _complete_wave(client, 4)
        ids = [row["request_id"] for row in client.request_rows]
        self.assertEqual(ids, sorted(ids))
        serialized = json.dumps(client.request_rows)
        self.assertNotIn("TOP SECRET", serialized)
        self.assertNotIn("prompt", serialized.lower())
        self.assertNotIn("response", serialized.lower())
        self.assertEqual(
            set(client.request_rows[0]),
            {
                "condition_id",
                "round",
                "wave_id",
                "phase",
                "request_id",
                "concurrency",
                "request_start_monotonic_ns",
                "first_token_monotonic_ns",
                "stream_end_monotonic_ns",
                "http_status",
                "input_tokens",
                "output_tokens",
                "success",
                "error",
            },
        )

    def test_sanitized_failure_does_not_store_generated_text(self):
        state = _State(failures=("run-measured-000",))
        client = _client(_block(1), state)
        with self.assertRaises(ConcurrentWaveError):
            _complete_wave(client, 1)
        self.assertEqual(client.request_rows[0]["error"], "RuntimeError: request failed")
        self.assertNotIn("secret", json.dumps(client.request_rows).lower())


class MatrixTests(unittest.TestCase):
    def test_reads_predeclared_one_two_four_conditions(self):
        conditions = [
            "m05-b1-hybrid",
            "m15-b1-hybrid",
            "m3-b1-hybrid",
            "m15-b2-hybrid",
            "m15-b4-hybrid",
            "m15-b1-gpu",
            "m15-b1-npu",
        ]
        matrix = {
            "request_protocol": {
                "measured_requests_per_block": 52,
                "input_tokens": 256,
                "output_tokens": 32,
                "temperature": 0,
                "streaming": True,
            },
            "concurrency_protocol": [
                {
                    "concurrency": value,
                    "requests_per_wave": value,
                    "measured_waves": 52 // value,
                    "warmup_requests": value,
                }
                for value in (1, 2, 4)
            ],
            "recommended_ofat": {"schedule": [conditions] * 3},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            blocks = load_matrix_blocks(path)
        self.assertEqual(len(blocks), 21)
        self.assertEqual({block.concurrency for block in blocks}, {1, 2, 4})
        self.assertEqual(blocks[0].round_index, 1)
        self.assertEqual(blocks[-1].round_index, 3)

    def test_matrix_rejects_non_divisible_request_count(self):
        matrix = {
            "request_protocol": {
                "measured_requests_per_block": 53,
                "input_tokens": 256,
                "output_tokens": 32,
                "temperature": 0,
                "streaming": True,
            },
            "concurrency_protocol": [
                {
                    "concurrency": value,
                    "requests_per_wave": value,
                    "measured_waves": 53,
                    "warmup_requests": value,
                }
                for value in (1, 2, 4)
            ],
            "recommended_ofat": {"schedule": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            with self.assertRaisesRegex(ConcurrentWaveError, "protocol"):
                load_matrix_blocks(path)


if __name__ == "__main__":
    unittest.main()
