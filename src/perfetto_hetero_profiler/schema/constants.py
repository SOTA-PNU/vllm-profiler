"""Constants shared by schema v1 modules."""

from __future__ import annotations

import re

SCHEMA_VERSION = "1.0.0"
SCHEMA_MAJOR_VERSION = 1
JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

RECORD_TYPES = (
    "run_manifest",
    "event",
    "metric",
    "artifact",
    "clock_domain",
    "sync_point",
    "clock_transform",
)

CANONICAL_EVENT_NAMES = frozenset(
    {
        "request_received",
        "prefill_start",
        "prefill_end",
        "kv_export_start",
        "kv_export_end",
        "kv_transform_start",
        "kv_transform_end",
        "kv_transfer_start",
        "kv_transfer_end",
        "kv_handoff_start",
        "kv_handoff_end",
        "kv_transfer_setup_start",
        "kv_transfer_setup_end",
        "kv_transfer_wait_start",
        "kv_transfer_wait_end",
        "decode_loop_start",
        "decode_schedule_wait_start",
        "decode_schedule_wait_end",
        "decode_step_start",
        "decode_step_end",
        "decode_loop_end",
        "sampling_start",
        "sampling_end",
        "first_token_emitted",
        "token_emitted",
        "response_done",
    }
)

EXTENSION_NAMESPACES = ("vendor.", "vllm.", "torch.", "nsys.", "rbln.", "nixl.")
NAMESPACED_NAME_RE = re.compile(
    r"^(?:vendor|vllm|torch|nsys|rbln|nixl)\.[A-Za-z0-9_][A-Za-z0-9_.-]*$"
)
EVENT_NAMESPACED_NAME_RE = re.compile(
    r"^(?:collector|vendor|vllm|torch|nsys|rbln|nixl)"
    r"\.[A-Za-z0-9_][A-Za-z0-9_.-]*$"
)
ATTRIBUTE_NAME_RE = re.compile(
    r"^[a-z][a-z0-9_-]*\.[A-Za-z0-9_][A-Za-z0-9_.-]*$"
)
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

JSON_SCHEMA_FILES = (
    "run_manifest.schema.json",
    "event_record.schema.json",
    "metric_sample.schema.json",
    "artifact_reference.schema.json",
    "clock_domain.schema.json",
    "sync_point.schema.json",
    "clock_transform.schema.json",
)
