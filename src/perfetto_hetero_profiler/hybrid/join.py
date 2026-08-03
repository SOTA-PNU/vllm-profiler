"""Request correlation and uncertainty-aware hybrid marker validation."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable

from ..schema import EventRecord


MARKER_ORDER = (
    "request_received",
    "prefill_start",
    "prefill_end",
    "kv_export_start",
    "kv_export_end",
    "kv_transfer_start",
    "kv_transfer_end",
    "kv_transform_start",
    "kv_transform_end",
    "decode_loop_start",
    "decode_step_start",
    "decode_step_end",
    "sampling_start",
    "sampling_end",
    "decode_loop_end",
    "response_done",
)

ITERATION_MARKERS = (
    "decode_step_start",
    "decode_step_end",
    "sampling_start",
    "sampling_end",
)
REPEATABLE_MARKERS = {*ITERATION_MARKERS, "token_emitted"}
TRANSFER_KEYS = (
    "hybrid.transfer_id",
    "kv.transfer_id",
    "nixl.transfer_id",
    "kv_transfer_id",
    "transfer_id",
)
CORRELATION_KEYS = ("hybrid.correlation_id", "correlation_id")


@dataclass(frozen=True)
class OrderingIssue:
    before_event_id: str
    after_event_id: str
    before_name: str
    after_name: str
    reversal_ns: int
    uncertainty_ns: int
    status: str
    reason: str


@dataclass(frozen=True)
class MarkerValidation:
    missing_markers: tuple[str, ...]
    duplicate_markers: tuple[str, ...]
    pairing_issues: tuple[str, ...]
    ordering_issues: tuple[OrderingIssue, ...]
    status: str


@dataclass(frozen=True)
class JoinResult:
    request_id: str | None
    gpu_request_ids: tuple[str, ...]
    npu_request_ids: tuple[str, ...]
    join_method: str
    confidence: float
    missing_markers: tuple[str, ...]
    duplicate_markers: tuple[str, ...]
    pairing_issues: tuple[str, ...]
    ordering_violations: tuple[OrderingIssue, ...]
    status: str
    reason: str | None
    events: tuple[EventRecord, ...]


def _uncertainty(event: EventRecord) -> int:
    value = event.attributes.get("hybrid.alignment_uncertainty_ns", 0)
    return value if isinstance(value, int) and value >= 0 else 0


def _step_index(event: EventRecord) -> int | None:
    value = event.attributes.get("decode.step_index")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _ordering_issue(
    before: EventRecord,
    after: EventRecord,
    *,
    context: str = "canonical marker order",
) -> OrderingIssue | None:
    if after.timestamp_ns >= before.timestamp_ns:
        return None
    reversal = before.timestamp_ns - after.timestamp_ns
    uncertainty = _uncertainty(before) + _uncertainty(after)
    within = reversal <= uncertainty
    return OrderingIssue(
        before_event_id=before.event_id,
        after_event_id=after.event_id,
        before_name=before.event_name,
        after_name=after.event_name,
        reversal_ns=reversal,
        uncertainty_ns=uncertainty,
        status=(
            "within_alignment_uncertainty"
            if within
            else "definite_violation"
        ),
        reason=(
            f"{context}; reversal is covered by alignment uncertainty"
            if within
            else f"{context}; reversal exceeds alignment uncertainty"
        ),
    )


def validate_marker_order(events: Iterable[EventRecord]) -> MarkerValidation:
    rows = tuple(events)
    by_name: dict[str, list[EventRecord]] = defaultdict(list)
    for event in rows:
        if event.event_name in MARKER_ORDER:
            by_name[event.event_name].append(event)
    missing = tuple(name for name in MARKER_ORDER if not by_name[name])
    duplicates: list[str] = []
    pairing_issues: list[str] = []
    for name, matches in by_name.items():
        if not matches:
            continue
        if len(matches) <= 1:
            if name in ITERATION_MARKERS and _step_index(matches[0]) is None:
                pairing_issues.append(
                    f"{name} requires a non-negative decode.step_index"
                )
            continue
        if name not in REPEATABLE_MARKERS:
            duplicates.append(name)
            continue
        indices = [_step_index(event) for event in matches]
        if any(index is None for index in indices):
            pairing_issues.append(
                f"every repeated {name} requires a non-negative decode.step_index"
            )
        concrete = [index for index in indices if index is not None]
        if len(set(concrete)) != len(concrete):
            duplicates.append(name)

    issues: list[OrderingIssue] = []
    representatives = [
        min(by_name[name], key=lambda event: event.timestamp_ns)
        for name in MARKER_ORDER
        if by_name[name]
    ]
    for before, after in zip(representatives, representatives[1:]):
        issue = _ordering_issue(before, after)
        if issue is not None:
            issues.append(issue)

    indexed: dict[str, dict[int, EventRecord]] = {}
    for name in ITERATION_MARKERS:
        matches: dict[int, EventRecord] = {}
        for event in by_name[name]:
            index = _step_index(event)
            if index is not None and index not in matches:
                matches[index] = event
        indexed[name] = matches

    if all(by_name[name] for name in ITERATION_MARKERS):
        expected_indices = set(indexed[ITERATION_MARKERS[0]])
        for name in ITERATION_MARKERS[1:]:
            actual = set(indexed[name])
            if actual != expected_indices:
                pairing_issues.append(
                    f"{name} indices {sorted(actual)} do not match "
                    f"decode step indices {sorted(expected_indices)}"
                )
        if expected_indices:
            ordered_indices = sorted(expected_indices)
            contiguous = list(
                range(ordered_indices[0], ordered_indices[-1] + 1)
            )
            if ordered_indices != contiguous:
                pairing_issues.append(
                    f"decode.step_index values must be contiguous: {ordered_indices}"
                )
            previous_sampling_end: EventRecord | None = None
            loop_start = (
                by_name["decode_loop_start"][0]
                if len(by_name["decode_loop_start"]) == 1
                else None
            )
            loop_end = (
                by_name["decode_loop_end"][0]
                if len(by_name["decode_loop_end"]) == 1
                else None
            )
            for index in ordered_indices:
                if not all(index in indexed[name] for name in ITERATION_MARKERS):
                    continue
                model_start = indexed["decode_step_start"][index]
                model_end = indexed["decode_step_end"][index]
                sampling_start = indexed["sampling_start"][index]
                sampling_end = indexed["sampling_end"][index]
                iteration = (
                    model_start,
                    model_end,
                    sampling_start,
                    sampling_end,
                )
                for before, after in zip(iteration, iteration[1:]):
                    issue = _ordering_issue(
                        before,
                        after,
                        context=f"decode iteration {index} order",
                    )
                    if issue is not None:
                        issues.append(issue)
                if previous_sampling_end is not None:
                    issue = _ordering_issue(
                        previous_sampling_end,
                        model_start,
                        context="decode iteration index order",
                    )
                    if issue is not None:
                        issues.append(issue)
                if loop_start is not None:
                    issue = _ordering_issue(
                        loop_start,
                        model_start,
                        context=f"decode iteration {index} starts before decode loop",
                    )
                    if issue is not None:
                        issues.append(issue)
                if loop_end is not None:
                    issue = _ordering_issue(
                        sampling_end,
                        loop_end,
                        context=f"decode iteration {index} ends after decode loop",
                    )
                    if issue is not None:
                        issues.append(issue)
                previous_sampling_end = sampling_end

    response_rows = by_name["response_done"]
    if len(response_rows) == 1:
        response = response_rows[0]
        optional_tokens = [
            event
            for event in rows
            if event.event_name in {"first_token_emitted", "token_emitted"}
        ]
        if optional_tokens:
            latest_token = max(
                optional_tokens,
                key=lambda event: (event.timestamp_ns, event.event_id),
            )
            issue = _ordering_issue(
                latest_token,
                response,
                context="response_done must be the final canonical response marker",
            )
            if issue is not None:
                issues.append(issue)

    definite = any(issue.status == "definite_violation" for issue in issues)
    if definite or duplicates or pairing_issues:
        status = "invalid"
    elif missing or issues:
        status = "partial"
    else:
        status = "valid"
    return MarkerValidation(
        missing_markers=missing,
        duplicate_markers=tuple(sorted(duplicates)),
        pairing_issues=tuple(dict.fromkeys(pairing_issues)),
        ordering_issues=tuple(issues),
        status=status,
    )


def _groups(events: Iterable[EventRecord]) -> dict[str, tuple[EventRecord, ...]]:
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        key = event.request_id or f"event:{event.event_id}"
        grouped[key].append(event)
    return {key: tuple(value) for key, value in grouped.items()}


def _identifier_values(
    events: tuple[EventRecord, ...], keys: tuple[str, ...]
) -> set[str]:
    values: set[str] = set()
    for event in events:
        for key in keys:
            value = event.attributes.get(key)
            if isinstance(value, str) and value:
                values.add(value)
    return values


def _joined(
    gpu_key: str,
    gpu_events: tuple[EventRecord, ...],
    npu_key: str,
    npu_events: tuple[EventRecord, ...],
    method: str,
    confidence: float,
) -> JoinResult:
    combined = tuple((*gpu_events, *npu_events))
    marker = validate_marker_order(combined)
    request_id = (
        gpu_events[0].request_id
        if gpu_events and gpu_events[0].request_id == npu_events[0].request_id
        else gpu_events[0].request_id or npu_events[0].request_id
    )
    status = "joined"
    reason = None
    if marker.status == "invalid":
        status = "invalid"
        reason = "marker contract is invalid"
    elif marker.status != "valid":
        status = "partial"
        reason = "marker contract is incomplete or uncertain"
    return JoinResult(
        request_id=request_id,
        gpu_request_ids=(gpu_key,),
        npu_request_ids=(npu_key,),
        join_method=method,
        confidence=confidence,
        missing_markers=marker.missing_markers,
        duplicate_markers=marker.duplicate_markers,
        pairing_issues=marker.pairing_issues,
        ordering_violations=marker.ordering_issues,
        status=status,
        reason=reason,
        events=combined,
    )


def join_requests(
    gpu_events: Iterable[EventRecord],
    npu_events: Iterable[EventRecord],
) -> tuple[JoinResult, ...]:
    gpu = _groups(gpu_events)
    npu = _groups(npu_events)
    results: list[JoinResult] = []
    used_gpu: set[str] = set()
    used_npu: set[str] = set()

    for key in sorted(set(gpu) & set(npu)):
        if key.startswith("event:"):
            continue
        gpu_boundaries = sum(
            event.event_name == "request_received" for event in gpu[key]
        )
        npu_boundaries = sum(
            event.event_name == "response_done" for event in npu[key]
        )
        if gpu_boundaries > 1 or npu_boundaries > 1:
            results.append(
                JoinResult(
                    request_id=key,
                    gpu_request_ids=(key,),
                    npu_request_ids=(key,),
                    join_method="request_id",
                    confidence=0.0,
                    missing_markers=(),
                    duplicate_markers=("request_id",),
                    pairing_issues=(),
                    ordering_violations=(),
                    status="ambiguous",
                    reason="duplicate request_id identifies multiple request boundaries",
                    events=tuple((*gpu[key], *npu[key])),
                )
            )
        else:
            results.append(
                _joined(key, gpu[key], key, npu[key], "request_id", 1.0)
            )
        used_gpu.add(key)
        used_npu.add(key)

    for method, keys, confidence in (
        ("transfer_id", TRANSFER_KEYS, 0.9),
        ("correlation_id", CORRELATION_KEYS, 0.8),
    ):
        available_gpu = {
            gpu_key: gpu_rows
            for gpu_key, gpu_rows in gpu.items()
            if gpu_key not in used_gpu
        }
        available_npu = {
            npu_key: npu_rows
            for npu_key, npu_rows in npu.items()
            if npu_key not in used_npu
        }
        candidates_by_gpu = {
            gpu_key: tuple(
                npu_key
                for npu_key, npu_rows in available_npu.items()
                if values and values & _identifier_values(npu_rows, keys)
            )
            for gpu_key, gpu_rows in available_gpu.items()
            for values in (_identifier_values(gpu_rows, keys),)
        }
        gpu_candidates_by_npu: dict[str, list[str]] = defaultdict(list)
        for gpu_key, candidates in candidates_by_gpu.items():
            for npu_key in candidates:
                gpu_candidates_by_npu[npu_key].append(gpu_key)

        for gpu_key, gpu_rows in available_gpu.items():
            candidates = candidates_by_gpu[gpu_key]
            one_to_one = (
                len(candidates) == 1
                and len(gpu_candidates_by_npu[candidates[0]]) == 1
            )
            if one_to_one:
                npu_key = candidates[0]
                results.append(
                    _joined(
                        gpu_key,
                        gpu_rows,
                        npu_key,
                        npu[npu_key],
                        method,
                        confidence,
                    )
                )
                used_gpu.add(gpu_key)
                used_npu.add(npu_key)
            elif candidates:
                results.append(
                    JoinResult(
                        request_id=gpu_rows[0].request_id,
                        gpu_request_ids=(gpu_key,),
                        npu_request_ids=tuple(sorted(candidates)),
                        join_method=method,
                        confidence=0.0,
                        missing_markers=(),
                        duplicate_markers=(),
                        pairing_issues=(),
                        ordering_violations=(),
                        status="ambiguous",
                        reason="multiple source candidates share the same identifier",
                        events=gpu_rows,
                    )
                )
                used_gpu.add(gpu_key)

    for gpu_key, rows in gpu.items():
        if gpu_key not in used_gpu:
            results.append(
                JoinResult(
                    request_id=rows[0].request_id,
                    gpu_request_ids=(gpu_key,),
                    npu_request_ids=(),
                    join_method="none",
                    confidence=0.0,
                    missing_markers=(),
                    duplicate_markers=(),
                    pairing_issues=(),
                    ordering_violations=(),
                    status="not_joined",
                    reason="no explicit shared request, transfer, or correlation id",
                    events=rows,
                )
            )
    for npu_key, rows in npu.items():
        if npu_key not in used_npu:
            results.append(
                JoinResult(
                    request_id=rows[0].request_id,
                    gpu_request_ids=(),
                    npu_request_ids=(npu_key,),
                    join_method="none",
                    confidence=0.0,
                    missing_markers=(),
                    duplicate_markers=(),
                    pairing_issues=(),
                    ordering_violations=(),
                    status="not_joined",
                    reason="no explicit shared request, transfer, or correlation id",
                    events=rows,
                )
            )
    return tuple(results)
