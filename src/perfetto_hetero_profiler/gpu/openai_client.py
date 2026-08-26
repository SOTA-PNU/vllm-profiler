"""Dependency-free OpenAI completions streaming client."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class CompletionObservation:
    request_id: str
    received_ns: int
    token_timestamps_ns: tuple[int, ...]
    done_ns: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    http_status: int
    response_started_ns: int | None = None

    @property
    def e2e_ns(self) -> int:
        return self.done_ns - self.received_ns

    @property
    def ttft_ns(self) -> int | None:
        if not self.token_timestamps_ns:
            return None
        return self.token_timestamps_ns[0] - self.received_ns

    @property
    def tpot_ns(self) -> float | None:
        if len(self.token_timestamps_ns) < 2:
            return None
        return (self.token_timestamps_ns[-1] - self.token_timestamps_ns[0]) / (
            len(self.token_timestamps_ns) - 1
        )


class OpenAICompletionClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_sec: float,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.monotonic_ns = monotonic_ns
        self.opener = opener

    def close(self) -> None:
        """Close client-owned resources (none for the stdlib implementation)."""

    def __enter__(self) -> "OpenAICompletionClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

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
        if not request_id:
            raise ValueError("request_id must not be empty")
        if not 1 <= max_output_tokens <= 16:
            raise ValueError("max_output_tokens must be in [1, 16]")
        if isinstance(temperature, bool) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be in [0, 2]")
        request_body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": stream,
            "request_id": request_id,
            "return_token_ids": True,
        }
        if stream:
            request_body["stream_options"] = {"include_usage": True}
        body = json.dumps(request_body).encode("utf-8")
        request = Request(
            f"{self.base_url}/v1/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": request_id,
            },
        )
        received_ns = self.monotonic_ns()
        token_timestamps: list[int] = []
        usage: dict[str, int] | None = None
        done_ns: int | None = None
        response_started_ns: int | None = None
        try:
            with self.opener(request, timeout=self.timeout_sec) as response:
                response_started_ns = self.monotonic_ns()
                status = int(response.status)
                if status != 200:
                    raise RuntimeError(f"completion returned HTTP {status}")
                if stream:
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
                            raise RuntimeError(f"completion stream error: {chunk['error']}")
                        chunk_usage = chunk.get("usage")
                        if chunk_usage:
                            usage = {
                                "prompt_tokens": int(chunk_usage["prompt_tokens"]),
                                "completion_tokens": int(
                                    chunk_usage["completion_tokens"]
                                ),
                                "total_tokens": int(chunk_usage["total_tokens"]),
                            }
                        for choice in chunk.get("choices", ()):
                            token_ids = choice.get("token_ids") or ()
                            token_timestamps.extend(arrival_ns for _ in token_ids)
                else:
                    document = json.loads(response.read())
                    if "error" in document:
                        raise RuntimeError(
                            f"completion response error: {document['error']}"
                        )
                    raw_usage = document.get("usage")
                    if raw_usage:
                        usage = {
                            "prompt_tokens": int(raw_usage["prompt_tokens"]),
                            "completion_tokens": int(raw_usage["completion_tokens"]),
                            "total_tokens": int(raw_usage["total_tokens"]),
                        }
                    done_ns = self.monotonic_ns()
        except (HTTPError, URLError, TimeoutError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"completion request failed: {error}") from error
        if stream and done_ns is None:
            raise RuntimeError("completion stream ended without [DONE]")
        if done_ns is None:  # pragma: no cover - defensive non-stream boundary
            raise RuntimeError("completion response did not finish")
        if usage is None:
            raise RuntimeError("completion stream did not include exact usage")
        if stream and usage["completion_tokens"] != len(token_timestamps):
            raise RuntimeError(
                "completion usage does not match streamed token_ids "
                f"({usage['completion_tokens']} != {len(token_timestamps)})"
            )
        return CompletionObservation(
            request_id=request_id,
            received_ns=received_ns,
            token_timestamps_ns=tuple(token_timestamps),
            done_ns=done_ns,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            http_status=status,
            response_started_ns=response_started_ns,
        )
