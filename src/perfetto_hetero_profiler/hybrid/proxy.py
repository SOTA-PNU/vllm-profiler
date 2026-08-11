"""Small dependency-free prefill/decode proxy with canonical runtime markers."""

from __future__ import annotations

import argparse
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


class MarkerWriter:
    def __init__(self, path: Path, *, host_id: str, clock_domain_id: str) -> None:
        self.path = path
        self.host_id = host_id
        self.clock_domain_id = clock_domain_id
        self.lock = threading.Lock()
        self.sequence = 0

    def emit(
        self,
        event_name: str,
        request_id: str,
        *,
        phase: str,
        source: str,
        attributes: dict[str, Any],
        remote_request_id_suffix: str | None = None,
        transfer_id: str | None = None,
    ) -> None:
        with self.lock:
            self.sequence += 1
            row: dict[str, Any] = {
                "schema_version": "1.0.0",
                "marker_version": "1.1.0",
                "event_name": event_name,
                "timestamp_ns": time.monotonic_ns(),
                "host_id": self.host_id,
                "clock_domain_id": self.clock_domain_id,
                "process_role": "proxy",
                "pid": os.getpid(),
                "thread_id": threading.get_native_id(),
                "request_id": request_id,
                "correlation_id": request_id,
                "phase": phase,
                "source": source,
                "sequence": self.sequence,
                "attributes": attributes,
            }
            if remote_request_id_suffix:
                row["remote_request_id_suffix"] = remote_request_id_suffix[-64:]
            if transfer_id:
                row["transfer_id"] = transfer_id
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(
                        row,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()


class ProxyState:
    def __init__(
        self,
        *,
        prefill_host: str,
        prefill_port: int,
        decode_host: str,
        decode_port: int,
        timeout_sec: float,
        marker_writer: MarkerWriter,
    ) -> None:
        self.prefill_host = prefill_host
        self.prefill_port = prefill_port
        self.decode_host = decode_host
        self.decode_port = decode_port
        self.timeout_sec = timeout_sec
        self.marker_writer = marker_writer


def _post_json(
    host: str,
    port: int,
    path: str,
    body: dict[str, Any],
    request_id: str,
    timeout_sec: float,
) -> tuple[int, bytes, dict[str, str]]:
    connection = HTTPConnection(host, port, timeout=timeout_sec)
    payload = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
    connection.request(
        "POST",
        path,
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "X-Request-Id": request_id,
        },
    )
    response = connection.getresponse()
    data = response.read()
    headers = {name.lower(): value for name, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, data, headers


def _block_count(value: object) -> int:
    if not isinstance(value, list):
        return 0
    count = 0
    for group in value:
        count += len(group) if isinstance(group, list) else 1
    return count


class HybridProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "hetero-profiler-proxy/1"

    @property
    def state(self) -> ProxyState:
        return self.server.proxy_state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        print(f"proxy {self.address_string()} {format % args}", flush=True)

    def do_GET(self) -> None:
        if self.path != "/healthcheck":
            self.send_error(404)
            return
        payload = b'{"status":"ok"}\n'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        raw_length = self.headers.get("Content-Length")
        request_id = self.headers.get("X-Request-Id", "")
        if not request_id:
            self.send_error(400, "X-Request-Id is required")
            return
        try:
            length = int(raw_length or "")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("request JSON must be an object")
            self._handle_completion(body, request_id)
        except Exception as error:
            if not self.wfile.closed:
                payload = json.dumps({"error": str(error)}).encode()
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:
                    pass

    def _handle_completion(self, body: dict[str, Any], request_id: str) -> None:
        state = self.state
        markers = state.marker_writer
        markers.emit(
            "request_received",
            request_id,
            phase="request",
            source="hetero_profiler.proxy",
            attributes={"request.api": "completions"},
        )
        prefill_body = dict(body)
        prefill_body.update(
            {
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                },
                "stream": False,
                "max_tokens": 1,
            }
        )
        prefill_body.pop("stream_options", None)
        markers.emit(
            "prefill_start",
            request_id,
            phase="prefill",
            source="hetero_profiler.proxy.prefill",
            attributes={},
        )
        status, data, _headers = _post_json(
            state.prefill_host,
            state.prefill_port,
            "/v1/completions",
            prefill_body,
            request_id,
            state.timeout_sec,
        )
        markers.emit(
            "prefill_end",
            request_id,
            phase="prefill",
            source="hetero_profiler.proxy.prefill",
            attributes={"http.status_code": status},
        )
        if status != 200:
            raise RuntimeError(f"prefill returned HTTP {status}: {data[-500:]!r}")
        response = json.loads(data)
        if not isinstance(response, dict):
            raise RuntimeError("prefill response must be a JSON object")
        markers.emit(
            "kv_export_start",
            request_id,
            phase="kv_export",
            source="hetero_profiler.proxy.kv_metadata",
            attributes={},
        )
        kv_params = response.get("kv_transfer_params")
        if not isinstance(kv_params, dict) or not kv_params:
            raise RuntimeError("prefill response has no kv_transfer_params")
        remote_request_id = str(kv_params.get("remote_request_id", ""))
        markers.emit(
            "kv_export_end",
            request_id,
            phase="kv_export",
            source="hetero_profiler.proxy.kv_metadata",
            remote_request_id_suffix=remote_request_id,
            attributes={
                "kv.metadata_present": True,
                "kv.remote_block_count": _block_count(
                    kv_params.get("remote_block_ids")
                ),
            },
        )
        markers.emit(
            "kv_handoff_start",
            request_id,
            phase="kv_transfer",
            source="hetero_profiler.proxy.decode_handoff",
            remote_request_id_suffix=remote_request_id,
            transfer_id=f"{request_id}-handoff",
            attributes={"kv.handoff_state": "metadata_exported"},
        )
        decode_body = dict(body)
        decode_body["kv_transfer_params"] = kv_params
        payload = json.dumps(
            decode_body, allow_nan=False, separators=(",", ":")
        ).encode()
        connection = HTTPConnection(
            state.decode_host, state.decode_port, timeout=state.timeout_sec
        )
        connection.request(
            "POST",
            "/v1/completions",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
                "X-Request-Id": request_id,
            },
        )
        decode = connection.getresponse()
        if decode.status != 200:
            error = decode.read()
            connection.close()
            raise RuntimeError(
                f"decode returned HTTP {decode.status}: {error[-500:]!r}"
            )
        self.send_response(200)
        self.send_header(
            "Content-Type", decode.getheader("Content-Type", "text/event-stream")
        )
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = decode.read1(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        connection.close()
        markers.emit(
            "response_done",
            request_id,
            phase="response",
            source="hetero_profiler.proxy.response",
            attributes={"response.stream_exhausted": True},
        )
        self.close_connection = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--prefill-host", required=True)
    parser.add_argument("--prefill-port", required=True, type=int)
    parser.add_argument("--decode-host", required=True)
    parser.add_argument("--decode-port", required=True, type=int)
    parser.add_argument("--timeout-sec", required=True, type=float)
    parser.add_argument("--marker-file", required=True, type=Path)
    parser.add_argument("--host-id", default="localhost")
    parser.add_argument("--clock-domain-id", default="host-monotonic")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.marker_file.is_absolute():
        raise SystemExit("--marker-file must be absolute")
    state = ProxyState(
        prefill_host=args.prefill_host,
        prefill_port=args.prefill_port,
        decode_host=args.decode_host,
        decode_port=args.decode_port,
        timeout_sec=args.timeout_sec,
        marker_writer=MarkerWriter(
            args.marker_file,
            host_id=args.host_id,
            clock_domain_id=args.clock_domain_id,
        ),
    )
    server = ThreadingHTTPServer((args.host, args.port), HybridProxyHandler)
    server.proxy_state = state  # type: ignore[attr-defined]
    server.serve_forever(poll_interval=0.1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
