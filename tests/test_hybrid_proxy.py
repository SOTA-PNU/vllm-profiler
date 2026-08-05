"""Unit tests for the dependency-free hybrid proxy primitives."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest

from perfetto_hetero_profiler.hybrid.proxy import MarkerWriter, _block_count
from perfetto_hetero_profiler.hybrid.proxy import HybridProxyHandler, ProxyState
from perfetto_hetero_profiler.gpu.openai_client import OpenAICompletionClient


class _BackendHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.server.requests.append((self.headers["X-Request-Id"], body))
        if self.server.role == "prefill":
            value = {
                "kv_transfer_params": {
                    "do_remote_prefill": True,
                    "remote_request_id": "cmpl-request-1-0-deadbeef",
                    "remote_block_ids": [[1]],
                    "remote_engine_id": "engine",
                    "remote_host": "127.0.0.1",
                    "remote_port": 9999,
                }
            }
            payload = json.dumps(value).encode()
            content_type = "application/json"
        else:
            payload = (
                b'data: {"choices":[{"token_ids":[1]}]}\n\n'
                b'data: {"choices":[{"token_ids":[2]}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":3,'
                b'"completion_tokens":2,"total_tokens":5}}\n\n'
                b'data: [DONE]\n\n'
            )
            content_type = "text/event-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class HybridProxyTests(unittest.TestCase):
    def test_block_count_accepts_grouped_and_flat_ids(self) -> None:
        self.assertEqual(_block_count([[1, 2], [3], 4]), 4)
        self.assertEqual(_block_count(None), 0)

    def test_marker_writer_is_append_only_and_correlated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markers.jsonl"
            writer = MarkerWriter(path, host_id="localhost", clock_domain_id="clock")
            writer.emit(
                "request_received", "request-1", phase="request",
                source="test.proxy", attributes={"request.api": "completions"},
            )
            writer.emit(
                "response_done", "request-1", phase="response",
                source="test.proxy", attributes={},
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["sequence"] for row in rows], [1, 2])
            self.assertEqual({row["correlation_id"] for row in rows}, {"request-1"})
            self.assertEqual({row["request_id"] for row in rows}, {"request-1"})

    def test_concurrent_markers_have_unique_contiguous_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markers.jsonl"
            writer = MarkerWriter(path, host_id="localhost", clock_domain_id="clock")

            def emit(index: int) -> None:
                writer.emit(
                    "request_received", f"r-{index}", phase="request",
                    source="test.proxy", attributes={},
                )

            threads = [threading.Thread(target=emit, args=(index,)) for index in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(sorted(row["sequence"] for row in rows), list(range(1, 21)))
            self.assertEqual(len(rows), 20)

    def test_end_to_end_request_id_and_kv_metadata_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            servers = []
            threads = []
            try:
                for role in ("prefill", "decode"):
                    server = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
                    server.role = role
                    server.requests = []
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    servers.append(server)
                    threads.append(thread)
                marker_path = Path(directory) / "proxy.jsonl"
                state = ProxyState(
                    prefill_host="127.0.0.1",
                    prefill_port=servers[0].server_port,
                    decode_host="127.0.0.1",
                    decode_port=servers[1].server_port,
                    timeout_sec=2,
                    marker_writer=MarkerWriter(
                        marker_path,
                        host_id="localhost",
                        clock_domain_id="host-monotonic",
                    ),
                )
                proxy = ThreadingHTTPServer(("127.0.0.1", 0), HybridProxyHandler)
                proxy.proxy_state = state
                proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                proxy_thread.start()
                servers.append(proxy)
                threads.append(proxy_thread)

                observation = OpenAICompletionClient(
                    f"http://127.0.0.1:{proxy.server_port}", timeout_sec=3
                ).complete(
                    model="model", request_id="request-1", prompt="private",
                    max_output_tokens=2,
                )
                self.assertEqual(observation.output_tokens, 2)
                self.assertEqual(servers[0].requests[0][0], "request-1")
                self.assertEqual(servers[1].requests[0][0], "request-1")
                self.assertEqual(
                    servers[1].requests[0][1]["kv_transfer_params"]["remote_engine_id"],
                    "engine",
                )
                deadline = time.monotonic() + 1
                rows = []
                while time.monotonic() < deadline:
                    rows = [
                        json.loads(line)
                        for line in marker_path.read_text().splitlines()
                    ]
                    if len(rows) == 6:
                        break
                    time.sleep(0.01)
                self.assertEqual(
                    [row["event_name"] for row in rows],
                    [
                        "request_received", "prefill_start", "prefill_end",
                        "kv_export_start", "kv_export_end", "response_done",
                    ],
                )
                self.assertEqual({row["correlation_id"] for row in rows}, {"request-1"})
            finally:
                for server in reversed(servers):
                    server.shutdown()
                    server.server_close()
                for thread in threads:
                    thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
