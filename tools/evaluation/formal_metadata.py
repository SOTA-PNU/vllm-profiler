"""Formal-campaign-only condition and environment metadata contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import subprocess
from typing import Any, Callable

from jsonschema import Draft202012Validator

from perfetto_hetero_profiler.collectors.npu.rbln_smi import (
    RblnSmiParseError,
    parse_rbln_smi_json,
)
from perfetto_hetero_profiler.support.files import sha256_file


CONDITION_SCHEMA_NAME = "formal_condition_metadata.schema.json"
_GIT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MODEL_SIZE_LABELS = {"m05": "0.5B", "m15": "1.5B", "m3": "3B"}
_PACKAGE_NAMES = (
    "vllm",
    "vllm-rbln",
    "torch",
    "nixl",
    "optimum-rbln",
    "rebel-compiler",
    "psutil",
)


class FormalMetadataError(RuntimeError):
    """Formal metadata is absent, inconsistent, or unsupported."""


def available(value: object) -> dict[str, object]:
    return {"availability": "available", "value": value, "reason": None}


def unavailable(reason: str) -> dict[str, object]:
    return {"availability": "not_available", "value": None, "reason": reason}


def _run(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str] | None:
    try:
        result = runner(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            shell=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result if result.returncode == 0 else None


def _python_environment(
    python: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, object]:
    program = (
        "import importlib.metadata as m,json,platform\n"
        f"names={list(_PACKAGE_NAMES)!r}\n"
        "packages={}\n"
        "for name in names:\n"
        " try: packages[name]={'availability':'available','value':m.version(name),'reason':None}\n"
        " except m.PackageNotFoundError: packages[name]={'availability':'not_available','value':None,'reason':'package is not installed in this server environment'}\n"
        "print(json.dumps({'python_version':platform.python_version(),'packages':packages},sort_keys=True))\n"
    )
    result = _run(runner, [str(python), "-c", program])
    if result is None:
        reason = "server Python environment query failed"
        return {
            "python_version": unavailable(reason),
            "packages": {name: unavailable(reason) for name in _PACKAGE_NAMES},
        }
    try:
        value = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        value = None
    if not isinstance(value, dict) or not isinstance(value.get("packages"), dict):
        reason = "server Python environment returned malformed metadata"
        return {
            "python_version": unavailable(reason),
            "packages": {name: unavailable(reason) for name in _PACKAGE_NAMES},
        }
    packages = value["packages"]
    if set(packages) != set(_PACKAGE_NAMES):
        raise FormalMetadataError("server package inventory is incomplete")
    python_version = value.get("python_version")
    if not isinstance(python_version, str) or not python_version:
        raise FormalMetadataError("server Python version is malformed")
    for name, record in packages.items():
        _validate_availability(record, f"server_environments.packages.{name}")
    return {"python_version": available(python_version), "packages": packages}


def _gpu_environment(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, object]:
    result = _run(
        runner,
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )
    if result is None:
        reason = "nvidia-smi inventory query failed"
        return {
            "query_status": unavailable(reason),
            "driver_version": unavailable(reason),
            "device_count": unavailable(reason),
            "devices": [],
            "memory_technology": unavailable(
                "memory technology was not reported by collected evidence"
            ),
        }
    rows = list(csv.reader(result.stdout.splitlines()))
    if not rows or any(len(row) != 4 for row in rows):
        raise FormalMetadataError("nvidia-smi inventory output is malformed")

    def numeric(raw: str, field: str, *, multiplier: int = 1) -> dict[str, object]:
        text = raw.strip()
        if text.lower() in {"", "n/a", "na", "not supported", "[not supported]"}:
            return unavailable(f"nvidia-smi did not report {field}")
        try:
            return available(int(float(text) * multiplier))
        except ValueError as error:
            raise FormalMetadataError(f"nvidia-smi {field} is malformed") from error

    pcie_result = _run(
        runner,
        [
            "nvidia-smi",
            "--query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,"
            "pcie.link.width.current,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ],
    )
    pcie_by_index: dict[int, dict[str, object]] = {}
    if pcie_result is not None:
        pcie_rows = list(csv.reader(pcie_result.stdout.splitlines()))
        if not pcie_rows or any(len(row) != 5 for row in pcie_rows):
            raise FormalMetadataError("nvidia-smi PCIe output is malformed")
        for row in pcie_rows:
            try:
                index = int(row[0].strip())
            except ValueError as error:
                raise FormalMetadataError("nvidia-smi PCIe device index is malformed") from error
            if index in pcie_by_index:
                raise FormalMetadataError("nvidia-smi PCIe device index is duplicated")
            pcie_by_index[index] = {
                "current_generation": numeric(row[1], "current PCIe generation"),
                "maximum_generation": numeric(row[2], "maximum PCIe generation"),
                "current_width": numeric(row[3], "current PCIe width"),
                "maximum_width": numeric(row[4], "maximum PCIe width"),
            }
    pcie_unavailable = {
        field: unavailable("nvidia-smi PCIe inventory query failed")
        for field in (
            "current_generation", "maximum_generation", "current_width", "maximum_width"
        )
    }
    devices = []
    device_indices: set[int] = set()
    drivers = set()
    for row in rows:
        try:
            index = int(row[0].strip())
        except ValueError as error:
            raise FormalMetadataError("nvidia-smi device index is malformed") from error
        if index in device_indices:
            raise FormalMetadataError("nvidia-smi device index is duplicated")
        device_indices.add(index)
        if row[3].strip():
            drivers.add(row[3].strip())
        devices.append(
            {
                "index": index,
                "model": (
                    available(row[1].strip())
                    if row[1].strip()
                    else unavailable("nvidia-smi did not report the GPU model")
                ),
                "memory_total_bytes": numeric(
                    row[2], "GPU total memory", multiplier=1024 * 1024
                ),
                "pcie": pcie_by_index.get(index, pcie_unavailable),
            }
        )
    if pcie_result is not None and set(pcie_by_index) != {
        device["index"] for device in devices
    }:
        raise FormalMetadataError("nvidia-smi PCIe inventory differs from GPU inventory")
    driver = next(iter(drivers)) if len(drivers) == 1 else None
    return {
        "query_status": available("nvidia-smi"),
        "driver_version": (
            available(driver)
            if driver is not None
            else unavailable("GPU driver versions were not uniform")
        ),
        "device_count": available(len(devices)),
        "devices": devices,
        "memory_technology": unavailable(
            "memory technology was not reported by collected evidence"
        ),
    }


def _npu_environment(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, object]:
    query = _run(runner, ["rbln-smi", "--json"])
    version = _run(runner, ["rbln-smi", "--version"])
    version_value = (
        available(version.stdout.strip())
        if version is not None and version.stdout.strip()
        else unavailable("rbln-smi version query failed")
    )
    if query is None:
        reason = "rbln-smi inventory query failed"
        return {
            "query_status": unavailable(reason),
            "rbln_smi_version": version_value,
            "kmd_driver_version": unavailable(reason),
            "device_count": unavailable(reason),
            "devices": [],
            "memory_technology": unavailable(
                "memory technology was not reported by collected evidence"
            ),
        }
    try:
        parsed = parse_rbln_smi_json(query.stdout)
    except RblnSmiParseError as error:
        raise FormalMetadataError("rbln-smi inventory output is malformed") from error
    devices = [
        {
            "index": row.index,
            "model": available(row.name),
            "memory_total_bytes": (
                available(row.memory_total_bytes.value)
                if row.memory_total_bytes.value is not None
                else unavailable(
                    row.memory_total_bytes.reason
                    or "rbln-smi did not report NPU total memory"
                )
            ),
            "firmware_version": (
                available(row.firmware_version)
                if row.firmware_version is not None
                else unavailable("rbln-smi did not report NPU firmware version")
            ),
        }
        for row in parsed.rows
    ]
    return {
        "query_status": available("rbln-smi"),
        "rbln_smi_version": version_value,
        "kmd_driver_version": (
            available(parsed.kmd_version)
            if parsed.kmd_version is not None
            else unavailable("rbln-smi did not report a KMD driver version")
        ),
        "device_count": available(len(devices)),
        "devices": devices,
        "memory_technology": unavailable(
            "memory technology was not reported by collected evidence"
        ),
    }


def collect_canonical_environment(
    config: object,
    *,
    profiler_head: str,
    vllm_rbln_head: str,
    query_devices: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, object]:
    """Collect path-free campaign environment evidence from the real runtimes."""

    command_runner = subprocess.run if runner is None else runner
    if query_devices:
        gpu = _gpu_environment(command_runner)
        npu = _npu_environment(command_runner)
    else:
        reason = "device inventory was disabled for this validation call"
        gpu = {
            "query_status": unavailable(reason),
            "driver_version": unavailable(reason),
            "device_count": unavailable(reason),
            "devices": [],
            "memory_technology": unavailable(reason),
        }
        npu = {
            "query_status": unavailable(reason),
            "rbln_smi_version": unavailable(reason),
            "kmd_driver_version": unavailable(reason),
            "device_count": unavailable(reason),
            "devices": [],
            "memory_technology": unavailable(reason),
        }
    value = {
        "schema_version": "1.0",
        "record_type": "formal_campaign_environment",
        "source_control": {
            "profiler_git_head": profiler_head,
            "vllm_rbln_git_head": vllm_rbln_head,
        },
        "operating_system": {
            "name": platform.system() or "not_available",
            "release": platform.release() or "not_available",
            "kernel_version": platform.version() or "not_available",
            "architecture": platform.machine() or "not_available",
        },
        "coordinator_python_version": available(platform.python_version()),
        "server_environments": {
            "gpu": _python_environment(config.gpu_vllm.parent / "python", command_runner),
            "npu": _python_environment(config.tokenizer_python, command_runner),
        },
        "gpu": gpu,
        "npu": npu,
        "privacy": {
            "hostname_recorded": False,
            "email_recorded": False,
            "credential_recorded": False,
            "absolute_paths_recorded": False,
        },
    }
    validate_environment(value)
    return value


def validate_environment(value: object) -> None:
    if not isinstance(value, dict):
        raise FormalMetadataError("environment metadata must be an object")
    expected = {
        "schema_version",
        "record_type",
        "source_control",
        "operating_system",
        "coordinator_python_version",
        "server_environments",
        "gpu",
        "npu",
        "privacy",
    }
    if set(value) != expected:
        raise FormalMetadataError("environment metadata fields differ from contract")
    if value.get("schema_version") != "1.0" or value.get("record_type") != "formal_campaign_environment":
        raise FormalMetadataError("environment metadata version or type is invalid")
    source = value.get("source_control")
    if not isinstance(source, dict) or set(source) != {
        "profiler_git_head", "vllm_rbln_git_head"
    } or any(not isinstance(item, str) or not _GIT_ID.fullmatch(item) for item in source.values()):
        raise FormalMetadataError("environment source-control identity is invalid")
    privacy = value.get("privacy")
    if (
        not isinstance(privacy, dict)
        or set(privacy) != {
            "hostname_recorded", "email_recorded", "credential_recorded",
            "absolute_paths_recorded",
        }
        or any(item is not False for item in privacy.values())
    ):
        raise FormalMetadataError("environment privacy flags must all be false")
    _validate_availability(value.get("coordinator_python_version"), "coordinator_python_version")
    servers = value.get("server_environments")
    if not isinstance(servers, dict) or set(servers) != {"gpu", "npu"}:
        raise FormalMetadataError("server environment inventory is invalid")
    for role, server in servers.items():
        if not isinstance(server, dict) or set(server) != {"python_version", "packages"}:
            raise FormalMetadataError(f"{role} server environment is invalid")
        _validate_availability(server["python_version"], f"{role}.python_version")
        packages = server["packages"]
        if not isinstance(packages, dict) or set(packages) != set(_PACKAGE_NAMES):
            raise FormalMetadataError(f"{role} package inventory is incomplete")
        for name, record in packages.items():
            _validate_availability(record, f"{role}.packages.{name}")
    operating_system = value.get("operating_system")
    if (
        not isinstance(operating_system, dict)
        or set(operating_system) != {"name", "release", "kernel_version", "architecture"}
        or any(not isinstance(item, str) or not item for item in operating_system.values())
    ):
        raise FormalMetadataError("operating-system metadata is invalid")
    for role, required in {
        "gpu": {"query_status", "driver_version", "device_count", "devices", "memory_technology"},
        "npu": {"query_status", "rbln_smi_version", "kmd_driver_version", "device_count", "devices", "memory_technology"},
    }.items():
        inventory = value.get(role)
        if not isinstance(inventory, dict) or set(inventory) != required:
            raise FormalMetadataError(f"{role} inventory fields differ from contract")
        for field in required - {"devices"}:
            _validate_availability(inventory[field], f"{role}.{field}")
        devices = inventory["devices"]
        if not isinstance(devices, list):
            raise FormalMetadataError(f"{role} devices must be a list")
        count = inventory["device_count"]
        if count["availability"] == "available" and count["value"] != len(devices):
            raise FormalMetadataError(f"{role} device count disagrees with inventory")

    def inspect(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key.lower() in {"hostname", "email", "credential", "password", "token"}:
                    raise FormalMetadataError(f"sensitive environment field: {path}.{key}")
                inspect(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                inspect(child, f"{path}[{index}]")
        elif isinstance(item, str):
            if PurePosixPath(item).is_absolute() or PureWindowsPath(item).is_absolute():
                raise FormalMetadataError(f"absolute path in environment metadata: {path}")
        elif item is not None and not isinstance(item, (bool, int, float)):
            raise FormalMetadataError(f"non-JSON environment value: {path}")

    inspect(value, "environment")


def _validate_availability(value: object, field: str) -> None:
    if not isinstance(value, dict) or set(value) != {"availability", "value", "reason"}:
        raise FormalMetadataError(f"availability record is malformed: {field}")
    status = value.get("availability")
    if status == "available":
        if value.get("value") is None or value.get("reason") is not None:
            raise FormalMetadataError(f"available value is incomplete: {field}")
    elif status == "not_available":
        if value.get("value") is not None or not isinstance(value.get("reason"), str) or not value["reason"]:
            raise FormalMetadataError(f"unavailable value lacks a reason: {field}")
    else:
        raise FormalMetadataError(f"availability status is invalid: {field}")


def _schema() -> Draft202012Validator:
    path = Path(__file__).parent / "schema" / CONDITION_SCHEMA_NAME
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


def validate_condition_schema(value: object) -> None:
    errors = sorted(_schema().iter_errors(value), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.path) or "$"
        raise FormalMetadataError(f"condition metadata schema failure at {location}")
    assert isinstance(value, dict)
    match = re.fullmatch(
        r"(?P<model>m05|m15|m3)-b(?P<concurrency>1|2|4)-(?P<topology>gpu|npu|hybrid)",
        value["condition_id"],
    )
    assert match is not None
    model = value["model"]
    workload = value["workload"]
    topology = value["topology"]
    concurrency = int(match.group("concurrency"))
    if (
        model["key"] != match.group("model")
        or model["size_label"] != _MODEL_SIZE_LABELS[model["key"]]
        or workload["request_concurrency"] != concurrency
        or workload["max_num_seqs"] != concurrency
        or workload["measured_waves"] != 52 // concurrency
    ):
        raise FormalMetadataError("condition identity disagrees with model or workload")
    expected_topology = {
        "gpu": ("gpu_only", "all_gpu", "gpu", "gpu", "not_applicable", None, None),
        "npu": ("npu_only", "all_npu", "npu", "npu", "not_applicable", None, None),
        "hybrid": (
            "hybrid", "gpu_prefill_npu_decode", "gpu", "npu", "gpu_to_npu",
            "kv_producer", "kv_consumer",
        ),
    }[match.group("topology")]
    observed_topology = (
        topology["mode"], topology["partition_strategy"], topology["prefill_device"],
        topology["decode_device"], topology["transfer_direction"],
        topology["producer_role"], topology["consumer_role"],
    )
    if observed_topology != expected_topology:
        raise FormalMetadataError("condition topology semantics are inconsistent")


def declared_model_size_label(model_key: str) -> str:
    try:
        return _MODEL_SIZE_LABELS[model_key]
    except KeyError as error:
        raise FormalMetadataError("unknown formal model key") from error


def build_condition_metadata(
    config: object,
    block: object,
    *,
    position: int,
    environment_sha256: str,
) -> dict[str, object]:
    model_key = block.condition_id.split("-", 1)[0]
    model = config.models[model_key]
    cache = config.caches[block.condition_id.rsplit("-", 1)[0]]
    topology = {
        "gpu": ("gpu_only", "all_gpu", "gpu", "gpu", "not_applicable", None, None),
        "npu": ("npu_only", "all_npu", "npu", "npu", "not_applicable", None, None),
        "hybrid": (
            "hybrid",
            "gpu_prefill_npu_decode",
            "gpu",
            "npu",
            "gpu_to_npu",
            "kv_producer",
            "kv_consumer",
        ),
    }[block.topology]
    value = {
        "schema_version": "1.0",
        "condition_id": block.condition_id,
        "position": position,
        "round": block.round_index,
        "model": {
            "key": model_key,
            "served_model_name": model.served_name,
            "revision": model.revision,
            "dtype": "bfloat16",
            "size_label": declared_model_size_label(model_key),
            "size_label_origin": "declared_experiment_contract",
            "exact_parameter_count": None,
            "snapshot_fingerprint": model.snapshot_fingerprint,
        },
        "workload": {
            "request_concurrency": block.concurrency,
            "batch_semantics": "concurrent_requests_per_wave",
            "max_num_seqs": cache.max_num_seqs,
            "input_tokens": block.input_tokens,
            "output_tokens": block.output_tokens,
            "measured_requests": block.measured_requests,
            "measured_waves": block.measured_waves,
        },
        "topology": {
            "mode": topology[0],
            "partition_strategy": topology[1],
            "prefill_device": topology[2],
            "decode_device": topology[3],
            "layer_partition_ratio": None,
            "layer_partition_ratio_status": "not_applicable",
            "transfer_direction": topology[4],
            "producer_role": topology[5],
            "consumer_role": topology[6],
        },
        "environment": {
            "relative_path": "../../environment.json",
            "sha256": environment_sha256,
        },
    }
    validate_condition_schema(value)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalMetadataError(f"required metadata is unreadable: {path.name}") from error
    if not isinstance(value, dict):
        raise FormalMetadataError(f"required metadata is not an object: {path.name}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise FormalMetadataError(f"required evidence is unreadable: {path.name}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise FormalMetadataError(f"required evidence has a non-object row: {path.name}")
    return rows


def validate_published_block(
    config: object,
    block: object,
    *,
    position: int,
    block_root: Path,
    environment_path: Path,
) -> dict[str, object]:
    """Cross-check one completed block against every formal evidence layer."""

    validate_environment(_load_json(environment_path))
    environment_sha = sha256_file(environment_path)
    condition_path = block_root / "condition_metadata.json"
    condition = _load_json(condition_path)
    expected = build_condition_metadata(
        config, block, position=position, environment_sha256=environment_sha
    )
    if condition != expected:
        raise FormalMetadataError("condition metadata does not match the scheduled block")
    reference = (block_root / condition["environment"]["relative_path"]).resolve()
    if reference != environment_path.resolve() or sha256_file(reference) != environment_sha:
        raise FormalMetadataError("condition environment reference SHA-256 mismatch")

    evidence_root = block_root / "evaluation"
    requests = _load_jsonl(evidence_root / "requests.jsonl")
    waves = _load_jsonl(evidence_root / "waves.jsonl")
    summary = _load_json(evidence_root / "summary.json")
    expected_request_count = block.warmup_requests + block.measured_requests
    if len(requests) != expected_request_count:
        raise FormalMetadataError("request evidence count mismatch")
    request_ids = [row.get("request_id") for row in requests]
    if len(set(request_ids)) != len(request_ids):
        raise FormalMetadataError("HTTP request IDs are not distinct")
    for row in requests:
        if (
            row.get("condition_id") != block.condition_id
            or row.get("round") != block.round_index
            or row.get("concurrency") != block.concurrency
            or row.get("success") is not True
            or row.get("http_status") != 200
            or row.get("input_tokens") != block.input_tokens
            or row.get("output_tokens") != block.output_tokens
            or not isinstance(row.get("request_start_monotonic_ns"), int)
            or not isinstance(row.get("stream_end_monotonic_ns"), int)
            or row["stream_end_monotonic_ns"] <= row["request_start_monotonic_ns"]
        ):
            raise FormalMetadataError("request evidence disagrees with condition metadata")
    phase_counts = {
        phase: sum(row.get("phase") == phase for row in requests)
        for phase in ("warmup", "measured")
    }
    if phase_counts != {
        "warmup": block.warmup_requests,
        "measured": block.measured_requests,
    }:
        raise FormalMetadataError("warmup/measured request evidence count mismatch")
    expected_wave_count = block.warmup_requests // block.concurrency + block.measured_waves
    if len(waves) != expected_wave_count:
        raise FormalMetadataError("wave evidence count mismatch")
    wave_ids = [row.get("wave_id") for row in waves]
    if len(set(wave_ids)) != len(wave_ids):
        raise FormalMetadataError("wave IDs are not distinct")
    for row in waves:
        if (
            row.get("condition_id") != block.condition_id
            or row.get("round") != block.round_index
            or row.get("concurrency") != block.concurrency
            or row.get("request_count") != block.concurrency
            or row.get("max_in_flight") != block.concurrency
            or not isinstance(row.get("barrier_release_monotonic_ns"), int)
            or not isinstance(row.get("common_overlap_ns"), int)
            or row["common_overlap_ns"] <= 0
            or row.get("barrier_respected") is not True
            or row.get("expected_concurrency_reached") is not True
            or row.get("success") is not True
            or row.get("failed_request_ids") != []
        ):
            raise FormalMetadataError("wave evidence disagrees with declared concurrency")
    wave_phase_counts = {
        phase: sum(row.get("phase") == phase for row in waves)
        for phase in ("warmup", "measured")
    }
    if wave_phase_counts != {
        "warmup": block.warmup_requests // block.concurrency,
        "measured": block.measured_waves,
    }:
        raise FormalMetadataError("warmup/measured wave evidence count mismatch")
    concurrency_validation = _load_json(evidence_root / "concurrency_validation.json")
    if (
        concurrency_validation.get("valid") is not True
        or concurrency_validation.get("condition_id") != block.condition_id
        or concurrency_validation.get("round") != block.round_index
        or concurrency_validation.get("concurrency") != block.concurrency
        or concurrency_validation.get("expected_wave_count") != expected_wave_count
        or concurrency_validation.get("observed_wave_count") != len(waves)
        or concurrency_validation.get("failed_wave_ids") != []
        or concurrency_validation.get("all_expected_concurrency_reached") is not True
    ):
        raise FormalMetadataError("concurrency validation artifact is inconsistent")
    if (
        summary.get("status") != "succeeded"
        or summary.get("runner_status") != "succeeded"
        or summary.get("condition_id") != block.condition_id
        or summary.get("round") != block.round_index
        or summary.get("request_count") != len(requests)
        or summary.get("wave_count") != len(waves)
        or summary.get("runner_errors") != []
        or summary.get("stores_prompt_or_generated_text") is not False
    ):
        raise FormalMetadataError("block summary terminal status is not succeeded")
    artifact_hashes = summary.get("artifacts")
    if not isinstance(artifact_hashes, dict) or any(
        artifact_hashes.get(name) != sha256_file(evidence_root / name)
        for name in ("requests.jsonl", "waves.jsonl", "concurrency_validation.json")
    ):
        raise FormalMetadataError("block summary artifact hashes are inconsistent")
    postflight = _load_json(block_root / "runtime-postflight.json")
    if (
        postflight.get("valid") is not True
        or postflight.get("busy_ports") != []
        or postflight.get("remaining_processes") != []
        or postflight.get("npu_contexts") != []
    ):
        raise FormalMetadataError("runtime cleanup postflight is invalid")

    if block.topology == "hybrid":
        run_id = f"r{block.round_index:02d}-{block.condition_id}"
        run_root = evidence_root / "runner" / run_id
        manifest = _load_json(run_root / "hybrid" / "manifest.json")
        workload = manifest.get("workload", {})
        runtime = manifest.get("configuration", {}).get("runtime_metadata", {})
        identity = runtime.get("model_identity", {})
        topology = runtime.get("topology", {})
        tokens = runtime.get("token_counts", {})
        if (
            manifest.get("mode") != "hybrid"
            or manifest.get("status") != "succeeded"
            or workload.get("request_count") != block.measured_requests
            or workload.get("concurrency") != block.concurrency
            or workload.get("input_tokens") != block.input_tokens
            or workload.get("output_tokens") != block.output_tokens
            or workload.get("max_model_len") != 512
            or runtime.get("max_num_seqs_capacity") != block.concurrency
            or tokens.get("input_tokens", {}).get("value") != block.input_tokens
            or tokens.get("input_tokens", {}).get("uniform") is not True
            or tokens.get("input_tokens", {}).get("minimum") != block.input_tokens
            or tokens.get("input_tokens", {}).get("maximum") != block.input_tokens
            or tokens.get("input_tokens", {}).get("sample_count") != block.measured_requests
            or tokens.get("output_tokens", {}).get("value") != block.output_tokens
            or tokens.get("output_tokens", {}).get("uniform") is not True
            or tokens.get("output_tokens", {}).get("minimum") != block.output_tokens
            or tokens.get("output_tokens", {}).get("maximum") != block.output_tokens
            or tokens.get("output_tokens", {}).get("sample_count") != block.measured_requests
            or identity.get("served_model_id") != condition["model"]["served_model_name"]
            or identity.get("revision") != condition["model"]["revision"]
            or identity.get("tokenizer_id") is not None
            or identity.get("dtype") != condition["model"]["dtype"]
            or identity.get("size_label") != condition["model"]["size_label"]
            or identity.get("size_label_origin") != "declared_experiment_contract"
            or identity.get("exact_parameter_count") is not None
            or identity.get("exact_parameter_count_status") != "not_available"
            or identity.get("metadata_origin") != "declared_experiment_contract"
            or identity.get("path_inference_used") is not False
            or topology.get("mode") != "hybrid"
            or topology.get("partition_strategy") != "gpu_prefill_npu_decode"
            or topology.get("prefill_device_type") != "gpu"
            or topology.get("decode_device_type") != "npu"
            or topology.get("layer_partition_ratio") is not None
            or topology.get("layer_partition_ratio_status") != "not_applicable"
            or topology.get("source_device_type") != "gpu"
            or topology.get("destination_device_type") != "npu"
            or topology.get("producer_role") != "kv_producer"
            or topology.get("consumer_role") != "kv_consumer"
            or topology.get("transfer_direction") != "gpu_to_npu"
        ):
            raise FormalMetadataError("core manifest disagrees with condition metadata")
        final = _load_json(run_root / "publication" / "final_result.json")
        lifecycle = final.get("lifecycle", {})
        started = lifecycle.get("started_at_unix_ns")
        finished = lifecycle.get("finished_at_unix_ns")
        if (
            final.get("status") != "succeeded"
            or final.get("measured_completed") != block.measured_requests
            or final.get("warmup_completed") != block.warmup_requests
            or lifecycle.get("terminal_status") != "succeeded"
            or lifecycle.get("shutdown_integrity") != "valid"
            or lifecycle.get("cleanup_status") != "complete"
            or lifecycle.get("measured_request_count") != block.measured_requests
            or not isinstance(started, int)
            or not isinstance(finished, int)
            or finished < started
            or lifecycle.get("duration_ns") != finished - started
        ):
            raise FormalMetadataError("core terminal status disagrees with cleanup evidence")
    else:
        shutdown = _load_json(evidence_root / "shutdown.json")
        if shutdown.get("killed") is not False or not isinstance(
            shutdown.get("return_code"), int
        ):
            raise FormalMetadataError("standalone shutdown cleanup is incomplete")

    return {
        "valid": True,
        "condition_metadata_sha256": sha256_file(condition_path),
        "environment_sha256": environment_sha,
        "request_count": len(requests),
        "wave_count": len(waves),
        "core_manifest_checked": block.topology == "hybrid",
        "terminal_cleanup_checked": True,
    }


__all__ = [
    "CONDITION_SCHEMA_NAME",
    "FormalMetadataError",
    "available",
    "build_condition_metadata",
    "collect_canonical_environment",
    "declared_model_size_label",
    "unavailable",
    "validate_condition_schema",
    "validate_environment",
    "validate_published_block",
]
