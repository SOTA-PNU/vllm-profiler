"""Evaluation-only 32-token OpenAI streaming client."""

from __future__ import annotations

import json
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from perfetto_hetero_profiler.gpu.openai_client import CompletionObservation


class FormalStreamingClient:
    """Keep the fixed formal request contract outside the packaged client."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_sec: float,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.monotonic_ns = monotonic_ns
        self.opener = opener

    def __enter__(self) -> "FormalStreamingClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """The stdlib transport owns no persistent resources."""

    def complete(
        self,
        *,
        model: str,
        request_id: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float = 0,
        stream: bool = True,
    ) -> CompletionObservation:
        if not request_id or max_output_tokens != 32:
            raise ValueError("formal requests require an ID and exactly 32 output tokens")
        if temperature != 0 or stream is not True:
            raise ValueError("formal requests require temperature=0 and streaming=true")
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": 32,
                "temperature": 0,
                "stream": True,
                "ignore_eos": True,
                "request_id": request_id,
                "return_token_ids": True,
                "stream_options": {"include_usage": True},
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/v1/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        )
        received_ns = self.monotonic_ns()
        timestamps: list[int] = []
        usage: dict[str, int] | None = None
        done_ns: int | None = None
        response_started_ns: int | None = None
        try:
            with self.opener(request, timeout=self.timeout_sec) as response:
                response_started_ns = self.monotonic_ns()
                status = int(response.status)
                if status != 200:
                    raise RuntimeError(f"completion returned HTTP {status}")
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="strict").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    arrival_ns = self.monotonic_ns()
                    if payload == "[DONE]":
                        done_ns = arrival_ns
                        break
                    chunk = json.loads(payload)
                    if "error" in chunk:
                        raise RuntimeError("completion stream returned an error")
                    if chunk.get("usage"):
                        raw = chunk["usage"]
                        usage = {
                            "prompt_tokens": int(raw["prompt_tokens"]),
                            "completion_tokens": int(raw["completion_tokens"]),
                            "total_tokens": int(raw["total_tokens"]),
                        }
                    for choice in chunk.get("choices", ()):
                        timestamps.extend(
                            arrival_ns for _ in (choice.get("token_ids") or ())
                        )
        except (HTTPError, URLError, TimeoutError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"completion request failed: {error}") from error
        if done_ns is None or usage is None:
            raise RuntimeError("completion stream lacks DONE or exact usage")
        if usage["completion_tokens"] != len(timestamps):
            raise RuntimeError("completion usage and streamed token count mismatch")
        return CompletionObservation(
            request_id=request_id,
            received_ns=received_ns,
            token_timestamps_ns=tuple(timestamps),
            done_ns=done_ns,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            http_status=status,
            response_started_ns=response_started_ns,
        )
