"""Evaluation-only concurrent request waves for the existing HybridRunner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable

from perfetto_hetero_profiler.gpu.openai_client import CompletionObservation, OpenAICompletionClient
from perfetto_hetero_profiler.hybrid.runner import HybridRunner
from perfetto_hetero_profiler.hybrid.runner_config import load_hybrid_runner_config
from perfetto_hetero_profiler.schema.records import RunStatus
from perfetto_hetero_profiler.support.files import sha256_file
from perfetto_hetero_profiler.support.json_io import write_jsonl_exclusive, write_pretty_json


SUPPORTED_CONCURRENCY = frozenset({1, 2, 4})
_REQUEST_ID = re.compile(r"^(?P<prefix>.+)-(?P<phase>warmup|measured)-(?P<index>\d+)$")
_CONDITION_ID = re.compile(
    r"^(?P<model>m05|m15|m3)-b(?P<concurrency>1|2|4)-(?P<topology>gpu|npu|hybrid)$"
)


class ConcurrentWaveError(RuntimeError):
    """A matrix, wave, or observed-concurrency gate failed."""


@dataclass(frozen=True, slots=True)
class BlockSpec:
    condition_id: str
    round_index: int
    concurrency: int
    topology: str
    warmup_requests: int
    measured_requests: int
    measured_waves: int
    input_tokens: int = 256
    output_tokens: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id, "round": self.round_index,
            "concurrency": self.concurrency, "topology": self.topology,
            "warmup_requests": self.warmup_requests,
            "measured_requests": self.measured_requests,
            "measured_waves": self.measured_waves,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True, slots=True)
class _CallResult:
    row: dict[str, object]
    observation: CompletionObservation | None


def load_matrix_blocks(path: Path) -> tuple[BlockSpec, ...]:
    """Read the predeclared schedule without inventing another config schema."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConcurrentWaveError("matrix root must be an object")
    protocols = value.get("concurrency_protocol")
    recommended = value.get("recommended_ofat")
    schedule = recommended.get("schedule") if isinstance(recommended, dict) else None
    request_protocol = value.get("request_protocol", {})
    total = request_protocol.get("measured_requests_per_block")
    input_tokens = request_protocol.get("input_tokens")
    output_tokens = request_protocol.get("output_tokens")
    if (
        input_tokens != 256 or output_tokens != 32
        or request_protocol.get("temperature") != 0
        or request_protocol.get("streaming") is not True
    ):
        raise ConcurrentWaveError("matrix does not match the fixed 256/32 stream contract")
    if not isinstance(protocols, list) or not isinstance(schedule, list):
        raise ConcurrentWaveError("matrix concurrency protocol or schedule is missing")
    by_concurrency: dict[int, dict[str, object]] = {}
    for item in protocols:
        if not isinstance(item, dict):
            raise ConcurrentWaveError("matrix concurrency protocol entry is invalid")
        concurrency, warmups = item.get("concurrency"), item.get("warmup_requests")
        valid = (
            concurrency in SUPPORTED_CONCURRENCY and not isinstance(concurrency, bool)
            and isinstance(total, int) and not isinstance(total, bool) and total > 0
            and isinstance(warmups, int) and not isinstance(warmups, bool) and warmups > 0
            and item.get("requests_per_wave") == concurrency
            and total % concurrency == 0 and warmups % concurrency == 0
            and item.get("measured_waves") == total // concurrency
        )
        if not valid or concurrency in by_concurrency:
            raise ConcurrentWaveError(f"invalid concurrency protocol for {concurrency!r}")
        by_concurrency[concurrency] = item
    if set(by_concurrency) != SUPPORTED_CONCURRENCY:
        raise ConcurrentWaveError("matrix must define concurrency 1, 2, and 4")
    blocks: list[BlockSpec] = []
    for round_index, condition_ids in enumerate(schedule, 1):
        if not isinstance(condition_ids, list):
            raise ConcurrentWaveError("matrix round must be a condition list")
        for condition_id in condition_ids:
            match = _CONDITION_ID.fullmatch(str(condition_id))
            if match is None:
                raise ConcurrentWaveError(f"invalid condition ID: {condition_id!r}")
            concurrency = int(match.group("concurrency"))
            protocol = by_concurrency[concurrency]
            blocks.append(BlockSpec(
                str(condition_id), round_index, concurrency, match.group("topology"),
                int(protocol["warmup_requests"]), total,
                int(protocol["measured_waves"]), input_tokens, output_tokens,
            ))
    if len(blocks) != 21:
        raise ConcurrentWaveError("recommended schedule must contain 21 blocks")
    return tuple(blocks)


def _safe_error(error: BaseException) -> str:
    message = str(error).lower()
    if isinstance(error, TimeoutError) or "timed out" in message or "timeout" in message:
        return f"{type(error).__name__}: request timed out"
    if "token" in message and ("match" in message or "count" in message):
        return f"{type(error).__name__}: token count mismatch"
    status = re.search(r"http(?: error)?\s+(\d{3})", message)
    if status:
        return f"{type(error).__name__}: HTTP {status.group(1)}"
    if "connection" in message or "url" in message:
        return f"{type(error).__name__}: request transport failed"
    return f"{type(error).__name__}: request failed"


def _max_in_flight(rows: list[dict[str, object]]) -> int:
    events = [event for row in rows for event in (
        (int(row["request_start_monotonic_ns"]), 1),
        (int(row["stream_end_monotonic_ns"]), -1),
    )]
    current = maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


class ConcurrentWaveClient:
    """Runner adapter backed by barrier-released, context-managed clients."""

    def __init__(
        self, base_url: str, *, timeout_sec: float, block: BlockSpec,
        client_factory: Callable[..., OpenAICompletionClient] = OpenAICompletionClient,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if (timeout_sec <= 0 or isinstance(block.concurrency, bool)
                or block.concurrency not in SUPPORTED_CONCURRENCY):
            raise ConcurrentWaveError("timeout must be positive; concurrency must be one of 1, 2, or 4")
        for label, count in (("warmup", block.warmup_requests),
                             ("measured", block.measured_requests)):
            if count <= 0 or count % block.concurrency:
                raise ConcurrentWaveError(
                    f"{label} request count must be positive and divisible by concurrency"
                )
        self.base_url, self.timeout_sec = base_url, timeout_sec
        self.block, self.client_factory = block, client_factory
        self.monotonic_ns = monotonic_ns
        self.request_rows: list[dict[str, object]] = []
        self.wave_rows: list[dict[str, object]] = []
        self._cached: dict[str, CompletionObservation] = {}
        self._active, self._closed = 0, False
        self._lock = threading.Lock()

    @property
    def active_task_count(self) -> int:
        with self._lock:
            return self._active

    def close(self) -> None:
        if self.active_task_count:
            raise ConcurrentWaveError("request tasks remain during client cleanup")
        self._cached.clear()
        self._closed = True

    def complete(self, **request: Any) -> CompletionObservation:
        if self._closed:
            raise ConcurrentWaveError("client is closed")
        request_id = str(request.get("request_id"))
        match = _REQUEST_ID.fullmatch(request_id)
        if match is None:
            raise ConcurrentWaveError("runner request ID does not expose phase/index")
        if request.get("stream") is not True:
            raise ConcurrentWaveError("concurrent evaluation requires streaming requests")
        if request_id in self._cached:
            return self._cached.pop(request_id)
        index, concurrency = int(match.group("index")), self.block.concurrency
        if index % concurrency:
            raise ConcurrentWaveError("runner requested a wave out of deterministic order")
        phase, wave_index = match.group("phase"), index // concurrency
        wave_id = f"{phase}-wave-{wave_index:03d}"
        calls = []
        for offset in range(concurrency):
            item = dict(request)
            item["request_id"] = f"{match.group('prefix')}-{phase}-{index + offset:03d}"
            calls.append(item)
        results, wave = self._dispatch(calls, phase, wave_id, wave_index)
        self.request_rows.extend(item.row for item in results)
        self.wave_rows.append(wave)
        if not wave["success"]:
            raise ConcurrentWaveError(f"{wave_id} failed concurrency validation")
        self._cached.update({str(item.row["request_id"]): item.observation
                             for item in results if item.observation is not None})
        return self._cached.pop(request_id)

    def _dispatch(self, calls: list[dict[str, Any]], phase: str,
                  wave_id: str, wave_index: int
                  ) -> tuple[list[_CallResult], dict[str, object]]:
        released: dict[str, int] = {}
        barrier = threading.Barrier(
            self.block.concurrency,
            action=lambda: released.setdefault("ns", self.monotonic_ns()),
        )
        with ThreadPoolExecutor(max_workers=self.block.concurrency,
                                thread_name_prefix="evaluation-wave") as executor:
            futures = [
                executor.submit(self._invoke, call, barrier, phase, wave_id)
                for call in calls
            ]
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        results.sort(key=lambda item: str(item.row["request_id"]))
        rows = [item.row for item in results]
        maximum = _max_in_flight(rows)
        overlap = max(0, min(int(row["stream_end_monotonic_ns"]) for row in rows)
                      - max(int(row["request_start_monotonic_ns"]) for row in rows))
        release_ns = released.get("ns")
        reached = maximum == self.block.concurrency and overlap > 0
        barrier_ok = release_ns is not None and all(
            int(row["request_start_monotonic_ns"]) >= release_ns for row in rows
        )
        success = (len(rows) == self.block.concurrency
                   and all(row["success"] for row in rows)
                   and reached and barrier_ok)
        wave = {
            "condition_id": self.block.condition_id, "round": self.block.round_index,
            "wave_id": wave_id, "phase": phase, "wave_index": wave_index,
            "concurrency": self.block.concurrency,
            "barrier_release_monotonic_ns": release_ns,
            "request_count": len(rows), "max_in_flight": maximum,
            "common_overlap_ns": overlap, "barrier_respected": barrier_ok,
            "expected_concurrency_reached": reached,
            "failed_request_ids": [row["request_id"] for row in rows
                                   if not row["success"]],
            "success": success,
        }
        return results, wave

    def _invoke(self, request: dict[str, Any], barrier: threading.Barrier,
                phase: str, wave_id: str) -> _CallResult:
        with self._lock:
            self._active += 1
        start_ns = end_ns = self.monotonic_ns()
        observation, error_text, error_status = None, None, None
        success = False
        try:
            barrier.wait(timeout=self.timeout_sec)
            start_ns = self.monotonic_ns()
            with self.client_factory(self.base_url, timeout_sec=self.timeout_sec) as client:
                observation = client.complete(**request)
            end_ns = observation.done_ns
            if (
                observation.input_tokens != self.block.input_tokens
                or observation.output_tokens != self.block.output_tokens
            ):
                raise ConcurrentWaveError("token count mismatch")
            success = True
        except Exception as error:
            end_ns, error_text = self.monotonic_ns(), _safe_error(error)
            status = re.search(r"http(?: error)?\s+(\d{3})", str(error).lower())
            error_status = int(status.group(1)) if status else None
        finally:
            with self._lock:
                self._active -= 1
        first = (observation.token_timestamps_ns[0]
                 if observation and observation.token_timestamps_ns else None)
        row = {
            "condition_id": self.block.condition_id, "round": self.block.round_index,
            "wave_id": wave_id, "phase": phase, "request_id": request["request_id"],
            "concurrency": self.block.concurrency,
            "request_start_monotonic_ns": start_ns,
            "first_token_monotonic_ns": first, "stream_end_monotonic_ns": end_ns,
            "http_status": observation.http_status if observation else error_status,
            "input_tokens": observation.input_tokens if observation else None,
            "output_tokens": observation.output_tokens if observation else None,
            "success": success, "error": error_text,
        }
        return _CallResult(row, observation if success else None)


def _select_block(path: Path, condition_id: str, round_index: int) -> BlockSpec:
    matches = [block for block in load_matrix_blocks(path)
               if block.condition_id == condition_id and block.round_index == round_index]
    if len(matches) != 1:
        raise ConcurrentWaveError("condition/round is not a unique scheduled block")
    return matches[0]


def plan(matrix_path: Path) -> dict[str, object]:
    blocks = load_matrix_blocks(matrix_path)
    return {"executes": False, "creates_output": False, "block_count": len(blocks),
            "blocks": [block.to_dict() for block in blocks]}


def write_block_artifacts(
    output: Path,
    block: BlockSpec,
    requests: list[dict[str, object]],
    waves: list[dict[str, object]],
    *,
    runner_status: str,
    runner_errors: tuple[str, ...] = (),
) -> dict[str, object]:
    """Publish the four deterministic evaluation artifacts for one block."""
    write_jsonl_exclusive(output / "requests.jsonl", requests)
    write_jsonl_exclusive(output / "waves.jsonl", waves)
    expected_waves = block.warmup_requests // block.concurrency + block.measured_waves
    validation = {
        "condition_id": block.condition_id, "round": block.round_index,
        "concurrency": block.concurrency, "expected_wave_count": expected_waves,
        "observed_wave_count": len(waves),
        "failed_wave_ids": [row["wave_id"] for row in waves if not row["success"]],
        "all_expected_concurrency_reached": bool(waves)
        and all(row["expected_concurrency_reached"] for row in waves),
        "valid": len(waves) == expected_waves and all(row["success"] for row in waves),
    }
    write_pretty_json(output / "concurrency_validation.json", validation)
    valid = validation["valid"] is True and runner_status == RunStatus.SUCCEEDED.value
    summary = {
        "condition_id": block.condition_id, "round": block.round_index,
        "status": "succeeded" if valid else "failed", "runner_status": runner_status,
        "request_count": len(requests), "wave_count": len(waves),
        "runner_errors": list(runner_errors), "stores_prompt_or_generated_text": False,
        "artifacts": {name: sha256_file(output / name) for name in
                      ("requests.jsonl", "waves.jsonl", "concurrency_validation.json")},
    }
    write_pretty_json(output / "summary.json", summary)
    if not valid:
        raise ConcurrentWaveError(f"block failed; diagnostics preserved in {output}")
    return summary


def run_block(*, matrix_path: Path, hybrid_config_path: Path, condition_id: str,
              round_index: int, block_root: Path,
              request_client_factory: Callable[..., object] = OpenAICompletionClient,
              runner_factory: Callable[..., HybridRunner] = HybridRunner,
              config_override: object | None = None) -> dict[str, object]:
    """Run one scheduled Hybrid block through the existing lifecycle."""
    block = _select_block(matrix_path, condition_id, round_index)
    if block.topology != "hybrid":
        raise ConcurrentWaveError("this adapter extends HybridRunner blocks only")
    config = (
        config_override
        if config_override is not None
        else load_hybrid_runner_config(Path(hybrid_config_path))
    )
    if config.max_num_seqs != block.concurrency:
        raise ConcurrentWaveError("max_num_seqs must equal the fixed cache capability")
    if config.workload.request_concurrency != block.concurrency:
        raise ConcurrentWaveError("core request concurrency does not match matrix protocol")
    if not config.workload.streaming:
        raise ConcurrentWaveError("hybrid workload must use streaming")
    if (config.workload.warmup_requests != block.warmup_requests
            or config.workload.measured_requests != block.measured_requests
            or config.workload.max_output_tokens != block.output_tokens):
        raise ConcurrentWaveError("hybrid request counts do not match matrix protocol")
    output = Path(block_root)
    if output.exists():
        raise ConcurrentWaveError("block output already exists")
    output.mkdir(parents=True)
    holder: list[ConcurrentWaveClient] = []

    def client_factory(base_url: str, *, timeout_sec: float) -> ConcurrentWaveClient:
        holder.append(ConcurrentWaveClient(
            base_url, timeout_sec=timeout_sec, block=block,
            client_factory=request_client_factory,
        ))
        return holder[-1]

    runner_status, runner_errors = "failed", ()
    try:
        result = runner_factory(
            config, run_root=output / "runner", run_id=f"r{round_index:02d}-{condition_id}",
            profile_mode="monitor", enable_telemetry=True, client_factory=client_factory,
        ).run()
        runner_status = result.status.value
        runner_errors = tuple(_safe_error(RuntimeError(item)) for item in result.errors)
    except Exception as error:
        runner_errors = (_safe_error(error),)
    return write_block_artifacts(
        output, block, holder[0].request_rows if holder else [],
        holder[0].wave_rows if holder else [], runner_status=runner_status,
        runner_errors=runner_errors,
    )
