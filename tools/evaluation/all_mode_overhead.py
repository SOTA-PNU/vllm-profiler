"""Repository-only one-hertz overhead campaign orchestration.

This module deliberately stays outside the installable package.  It fixes the
experimental unit before any hardware result is observed, runs every condition
without automatic retry, and keeps the raw HybridRunner products as immutable
evidence.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time
from typing import Any, Iterable, Sequence

from perfetto_hetero_profiler.hybrid.layout import (
    HybridRunLayout,
    existing_collection_result_path,
    existing_final_result_path,
)
from perfetto_hetero_profiler.hybrid.runner import HybridRunner, _cache_fingerprint
from perfetto_hetero_profiler.hybrid.runner_config import (
    HybridRunnerConfig,
    load_hybrid_runner_config,
    validate_hybrid_invocation,
)
from perfetto_hetero_profiler.schema.records import RunStatus
from perfetto_hetero_profiler.support.files import sha256_file
from perfetto_hetero_profiler.support.json_io import canonical_json_bytes

from .environment import capture_environment, idle_reasons, wait_for_idle
from .statistics import OverheadDirection, paired_overhead, summarize_distribution
from .validation import validate_trial


SCHEMA_VERSION = "1.0"
CONDITIONS = (
    "reference",
    "monitor",
    "gpu-torch",
    "gpu-nsys",
    "npu-torch",
    "npu-rbln",
)
DEFAULT_FORMAL_ROUNDS = 5
AUTOMATIC_RETRIES = 0
ONE_HERTZ_INTERVAL_MS = 1000
MINIMUM_BACKGROUND_SAMPLES = 10
THRESHOLD_PERCENT = 5.0
ALLOWED_PROFILER_CAMPAIGN_CHANGES = frozenset(
    {
        "tools/evaluation/all_mode_overhead.py",
        "tools/evaluation/cli.py",
        "tools/evaluation/validation.py",
        "src/perfetto_hetero_profiler/hybrid/runner.py",
        "src/perfetto_hetero_profiler/overview/loader.py",
        "src/perfetto_hetero_profiler/perfetto/loader.py",
        "src/perfetto_hetero_profiler/perfetto/validation.py",
        "tests/test_all_mode_overhead.py",
        "tests/test_experiment_cli.py",
        "tests/test_evaluation_validation.py",
        "tests/test_hybrid_runner_lifecycle.py",
        "tests/test_overview_loader.py",
        "tests/test_perfetto_core.py",
    }
)
_ONE_SIDED_T_95_DF4 = 2.131846786326649
_TWO_SIDED_T_95_DF4 = 2.7764451051977987


class AllModeOverheadError(RuntimeError):
    """The campaign cannot continue without weakening its fixed protocol."""


class _TimedHybridRunner(HybridRunner):
    """Expose the runner's offline derivation boundary to the evaluation only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.postprocessing_started_monotonic_ns: int | None = None

    def _write_sources(self, *args: Any, **kwargs: Any) -> None:
        if self.postprocessing_started_monotonic_ns is None:
            self.postprocessing_started_monotonic_ns = time.monotonic_ns()
        super()._write_sources(*args, **kwargs)


class _CollectionOnlyHybridRunner(_TimedHybridRunner):
    """Keep formal blocks hardware-focused and defer heavy derived products."""

    def _derive_products(self) -> None:
        # The detached closeout inventory is created immediately before this
        # hook.  Writing into an inventoried source root here would invalidate
        # that immutable inventory.  Deferral is recorded outside the run
        # bundle by block_result.json and execution_plan.json.
        return None


def _canonical(value: object) -> bytes:
    return canonical_json_bytes(value)


def _atomic_write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    _atomic_write(path, _canonical(value), exclusive=exclusive)


def _write_json_once_or_verify(path: Path, value: object) -> None:
    data = _canonical(value)
    if path.is_file():
        if path.read_bytes() != data:
            raise AllModeOverheadError(f"immutable evidence mismatch: {path}")
        return
    _atomic_write(path, data, exclusive=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AllModeOverheadError(f"expected JSON object: {path}")
    return value


def _tree_fingerprint(root: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root": str(root),
        "files": files,
        "tree_sha256": hashlib.sha256(_canonical(files)).hexdigest(),
    }


def build_all_mode_schedule(
    seed: int,
    formal_rounds: int = DEFAULT_FORMAL_ROUNDS,
) -> dict[str, object]:
    """Build independently randomized, immutable six-condition rounds."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be an integer in [0, 2^32-1]")
    if (
        isinstance(formal_rounds, bool)
        or not isinstance(formal_rounds, int)
        or not 1 <= formal_rounds <= 20
    ):
        raise ValueError("formal_rounds must be an integer in [1, 20]")
    blocks: list[dict[str, object]] = []
    for round_index in range(1, formal_rounds + 1):
        order = sorted(
            CONDITIONS,
            key=lambda condition: hashlib.sha256(
                f"all-mode-overhead-1hz:{seed}:{round_index}:{condition}".encode("ascii")
            ).digest(),
        )
        for order_index, condition in enumerate(order, 1):
            block_id = f"round-{round_index:02d}-{condition}"
            blocks.append(
                {
                    "position": len(blocks),
                    "round": round_index,
                    "order_in_round": order_index,
                    "condition": condition,
                    "block_id": block_id,
                }
            )
    document = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "method": "sha256-keyed independent random permutation per round",
        "formal_rounds": formal_rounds,
        "conditions_per_round": len(CONDITIONS),
        "formal_condition_blocks": len(blocks),
        "automatic_retries": AUTOMATIC_RETRIES,
        "blocks": blocks,
    }
    document["sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return document


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    config_path: Path
    campaign_id: str
    hybrid_config_path: Path
    hybrid_config_sha256: str
    cache_source_config_path: Path
    cache_source_config_sha256: str
    profiler_root: Path
    profiler_head: str
    vllm_rbln_root: Path
    vllm_rbln_head: str
    prefill_python: Path
    decode_python: Path
    seed: int
    formal_rounds: int
    warmup_requests: int
    formal_requests_per_block: int
    expected_input_tokens: int
    expected_output_tokens: int
    minimum_background_samples: int
    sample_interval_ms: int
    historical_artifact_roots: dict[str, Path]

    @property
    def schedule(self) -> dict[str, object]:
        return build_all_mode_schedule(self.seed, self.formal_rounds)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "hybrid_config": {
                "path": str(self.hybrid_config_path),
                "sha256": self.hybrid_config_sha256,
            },
            "cache_source_config": {
                "path": str(self.cache_source_config_path),
                "sha256": self.cache_source_config_sha256,
            },
            "repositories": {
                "profiler": {"path": str(self.profiler_root), "head": self.profiler_head},
                "vllm_rbln": {"path": str(self.vllm_rbln_root), "head": self.vllm_rbln_head},
            },
            "python": {
                "prefill": str(self.prefill_python),
                "decode": str(self.decode_python),
            },
            "schedule": {
                "seed": self.seed,
                "formal_rounds": self.formal_rounds,
            },
            "measurement": {
                "warmup_requests": self.warmup_requests,
                "formal_requests_per_block": self.formal_requests_per_block,
                "expected_input_tokens": self.expected_input_tokens,
                "expected_output_tokens": self.expected_output_tokens,
                "minimum_request_window_background_samples": self.minimum_background_samples,
                "sample_interval_ms": self.sample_interval_ms,
            },
            "historical_artifact_roots": {
                name: str(path) for name, path in sorted(self.historical_artifact_roots.items())
            },
        }

    def load_hybrid(self) -> HybridRunnerConfig:
        if sha256_file(self.hybrid_config_path) != self.hybrid_config_sha256:
            raise AllModeOverheadError("hybrid config SHA-256 mismatch")
        return load_hybrid_runner_config(self.hybrid_config_path).with_overrides(
            warmup_requests=self.warmup_requests,
            measured_requests=self.formal_requests_per_block,
            max_output_tokens=self.expected_output_tokens,
        )


def _absolute_file(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a path string")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{field} must be an existing absolute regular file")
    return path


def _absolute_directory(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a path string")
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise ValueError(f"{field} must be an existing absolute directory")
    return path


def _absolute_executable(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a path string")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{field} must be an existing absolute executable")
    return path


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def load_campaign_config(path: Path) -> CampaignConfig:
    path = _absolute_file(str(path), "config")
    value = _read_json(path)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    campaign_id = value.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id or "/" in campaign_id:
        raise ValueError("campaign_id must be a safe non-empty path component")
    hybrid = value.get("hybrid_config")
    source = value.get("cache_source_config")
    repos = value.get("repositories")
    pythons = value.get("python")
    schedule = value.get("schedule")
    measurement = value.get("measurement")
    historical = value.get("historical_artifact_roots")
    if not all(isinstance(item, dict) for item in (hybrid, source, repos, pythons, schedule, measurement, historical)):
        raise ValueError("campaign config sections must be JSON objects")
    assert isinstance(hybrid, dict) and isinstance(source, dict)
    assert isinstance(repos, dict) and isinstance(pythons, dict)
    assert isinstance(schedule, dict) and isinstance(measurement, dict)
    assert isinstance(historical, dict)
    profiler = repos.get("profiler")
    vllm = repos.get("vllm_rbln")
    if not isinstance(profiler, dict) or not isinstance(vllm, dict):
        raise ValueError("repositories must define profiler and vllm_rbln")
    if set(historical) != set(CONDITIONS) - {"reference"}:
        raise ValueError("historical_artifact_roots must define the five candidate modes")
    result = CampaignConfig(
        config_path=path,
        campaign_id=campaign_id,
        hybrid_config_path=_absolute_file(hybrid.get("path"), "hybrid_config.path"),
        hybrid_config_sha256=_sha(hybrid.get("sha256"), "hybrid_config.sha256"),
        cache_source_config_path=_absolute_file(source.get("path"), "cache_source_config.path"),
        cache_source_config_sha256=_sha(source.get("sha256"), "cache_source_config.sha256"),
        profiler_root=_absolute_directory(profiler.get("path"), "repositories.profiler.path"),
        profiler_head=str(profiler.get("head")),
        vllm_rbln_root=_absolute_directory(vllm.get("path"), "repositories.vllm_rbln.path"),
        vllm_rbln_head=str(vllm.get("head")),
        prefill_python=_absolute_executable(pythons.get("prefill"), "python.prefill"),
        decode_python=_absolute_executable(pythons.get("decode"), "python.decode"),
        seed=schedule.get("seed"),
        formal_rounds=schedule.get("formal_rounds"),
        warmup_requests=measurement.get("warmup_requests"),
        formal_requests_per_block=measurement.get("formal_requests_per_block"),
        expected_input_tokens=measurement.get("expected_input_tokens"),
        expected_output_tokens=measurement.get("expected_output_tokens"),
        minimum_background_samples=measurement.get("minimum_request_window_background_samples"),
        sample_interval_ms=measurement.get("sample_interval_ms"),
        historical_artifact_roots={
            name: _absolute_directory(raw, f"historical_artifact_roots.{name}")
            for name, raw in historical.items()
        },
    )
    integer_contract = {
        "schedule.seed": (result.seed, 0, 0xFFFFFFFF),
        "schedule.formal_rounds": (result.formal_rounds, 1, 20),
        "measurement.warmup_requests": (result.warmup_requests, 1, 1),
        "measurement.formal_requests_per_block": (result.formal_requests_per_block, 1, 1000),
        "measurement.expected_input_tokens": (result.expected_input_tokens, 5, 5),
        "measurement.expected_output_tokens": (result.expected_output_tokens, 8, 8),
        "measurement.minimum_request_window_background_samples": (
            result.minimum_background_samples,
            MINIMUM_BACKGROUND_SAMPLES,
            MINIMUM_BACKGROUND_SAMPLES,
        ),
        "measurement.sample_interval_ms": (
            result.sample_interval_ms,
            ONE_HERTZ_INTERVAL_MS,
            ONE_HERTZ_INTERVAL_MS,
        ),
    }
    for field, (current, minimum, maximum) in integer_contract.items():
        if isinstance(current, bool) or not isinstance(current, int) or not minimum <= current <= maximum:
            raise ValueError(f"{field} must be in [{minimum}, {maximum}]")
    if sha256_file(result.cache_source_config_path) != result.cache_source_config_sha256:
        raise AllModeOverheadError("cache source config SHA-256 mismatch")
    result.load_hybrid()
    return result


def _command(argv: Sequence[str], *, env: dict[str, str] | None = None, timeout: float = 30.0) -> dict[str, object]:
    try:
        completed = subprocess.run(
            tuple(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=env,
        )
        return {
            "argv": list(argv),
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": list(argv), "error": f"{type(error).__name__}: {error}"}


def _git_snapshot(root: Path) -> dict[str, object]:
    return {
        "path": str(root),
        "branch": _command(("git", "-C", str(root), "branch", "--show-current")),
        "head": _command(("git", "-C", str(root), "rev-parse", "HEAD")),
        "status": _command(("git", "-C", str(root), "status", "--short")),
        "staged": _command(("git", "-C", str(root), "diff", "--cached", "--name-status")),
    }


def _import_snapshot(python: Path, pythonpath: Path) -> dict[str, object]:
    source = (
        "import importlib.metadata as m,json,platform,sys,vllm_rbln;"
        "d={x.metadata.get('Name','').lower():x.version for x in m.distributions()};"
        "import torch;"
        "print(json.dumps({'python':platform.python_version(),'executable':sys.executable,"
        "'vllm_rbln_file':vllm_rbln.__file__,'torch_profiler':hasattr(torch,'profiler'),"
        "'packages':{n:d.get(n) for n in ['vllm','torch','transformers','nixl','optimum-rbln','rebel-compiler']}},sort_keys=True))"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(pythonpath),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return _command((str(python), "-c", source), env=environment)


def _prompt_token_snapshot(config: CampaignConfig, hybrid: HybridRunnerConfig) -> dict[str, object]:
    prompt = hybrid.workload.prompt_text()
    source = (
        "from transformers import AutoTokenizer;import hashlib,json;"
        f"p={prompt!r};"
        f"t=AutoTokenizer.from_pretrained({str(hybrid.model_path)!r},local_files_only=True);"
        "ids=t.encode(p,add_special_tokens=True);"
        "print(json.dumps({'count':len(ids),'ids':ids,'prompt_sha256':hashlib.sha256(p.encode()).hexdigest()},sort_keys=True))"
    )
    environment = dict(os.environ)
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return _command((str(config.prefill_python), "-c", source), env=environment)


def _artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def _artifact_size_with_suffixes(path: Path, suffixes: tuple[str, ...]) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink() and item.name.endswith(suffixes)
    )


def estimate_disk_budget(config: CampaignConfig) -> dict[str, object]:
    rows: dict[str, dict[str, int]] = {}
    projected = 0
    for condition, path in sorted(config.historical_artifact_roots.items()):
        total = _artifact_size(path)
        runtime_scalable = _artifact_size_with_suffixes(
            path,
            (".pt.trace.json.gz", ".nsys-rep", ".sqlite", ".pb"),
        )
        deferred_trace = _artifact_size_with_suffixes(path, (".pftrace",))
        base = max(0, total - runtime_scalable - deferred_trace)
        per_block = base + runtime_scalable * config.formal_requests_per_block
        representative_postprocess = (
            deferred_trace * config.formal_requests_per_block
        )
        rows[condition] = {
            "historical_total_bytes": total,
            "historical_runtime_scalable_bytes": runtime_scalable,
            "historical_deferred_trace_bytes": deferred_trace,
            "historical_base_bytes": base,
            "projected_bytes_per_block": per_block,
            "formal_blocks": config.formal_rounds,
            "formal_collection_projected_bytes": (
                per_block * config.formal_rounds
            ),
            "representative_postprocess_projected_bytes": (
                representative_postprocess
            ),
            "projected_bytes": (
                per_block * config.formal_rounds
                + representative_postprocess
            ),
        }
        projected += per_block * config.formal_rounds + representative_postprocess
    reference = rows["monitor"]["historical_base_bytes"] * config.formal_rounds
    projected += reference
    required = math.ceil(projected * 1.30)
    free = shutil.disk_usage(config.config_path.parent).free
    return {
        "method": (
            "historical base plus native profiler artifacts scaled across formal "
            "blocks, with Perfetto trace size scaled once for each representative mode"
        ),
        "historical_request_count": 1,
        "formal_requests_per_block": config.formal_requests_per_block,
        "conditions": rows,
        "reference_projected_bytes": reference,
        "projected_bytes": projected,
        "margin_ratio": 0.30,
        "deferred_postprocessing": True,
        "representative_runs_per_candidate_mode": 1,
        "required_with_margin_bytes": required,
        "free_bytes": free,
        "valid": free >= required,
    }


def _config_contract(config: CampaignConfig, hybrid: HybridRunnerConfig) -> dict[str, object]:
    source = load_hybrid_runner_config(config.cache_source_config_path)
    current = {
        "model_path": str(hybrid.model_path),
        "cache_path": str(hybrid.rbln_cache_path),
        "prompt": hybrid.workload.prompt_text(),
        "warmup_requests": hybrid.workload.warmup_requests,
        "formal_requests_per_block": hybrid.workload.measured_requests,
        "output_tokens": hybrid.workload.max_output_tokens,
        "temperature": hybrid.workload.temperature,
        "streaming": hybrid.workload.streaming,
        "max_model_len": hybrid.max_model_len,
        "block_size": hybrid.block_size,
        "max_num_seqs": hybrid.max_num_seqs,
        "gpu_memory_utilization": hybrid.gpu_memory_utilization,
        "sample_interval_ms": hybrid.sample_interval_ms,
        "offline": hybrid.offline,
    }
    expected = {
        "warmup_requests": 1,
        "formal_requests_per_block": config.formal_requests_per_block,
        "output_tokens": 8,
        "temperature": 0.0,
        "streaming": True,
        "max_model_len": 512,
        "block_size": 512,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.2,
        "sample_interval_ms": 1000,
        "offline": True,
    }
    mismatches = [
        f"{field}: expected {value!r}, got {current.get(field)!r}"
        for field, value in expected.items()
        if current.get(field) != value
    ]
    source_contract = {
        "cache_path": str(source.rbln_cache_path),
        "max_model_len": source.max_model_len,
        "block_size": source.block_size,
        "max_num_seqs": source.max_num_seqs,
    }
    cache_compatible = (
        source.rbln_cache_path == hybrid.rbln_cache_path
        and source.max_model_len == hybrid.max_model_len
        and source.block_size == hybrid.block_size
        and source.max_num_seqs == hybrid.max_num_seqs
    )
    if not cache_compatible:
        mismatches.append(
            "existing cache source contract differs from requested runtime; persistent model compile is forbidden"
        )
    return {
        "current": current,
        "expected": expected,
        "cache_source": source_contract,
        "cache_contract_compatible": cache_compatible,
        "valid": not mismatches,
        "mismatches": mismatches,
    }


def run_preflight(config: CampaignConfig) -> dict[str, object]:
    hybrid = config.load_hybrid()
    profiler = _git_snapshot(config.profiler_root)
    vllm = _git_snapshot(config.vllm_rbln_root)
    repository_reasons: list[str] = []
    for name, snapshot, expected in (
        ("profiler", profiler, config.profiler_head),
        ("vllm-rbln", vllm, config.vllm_rbln_head),
    ):
        head = str(snapshot["head"].get("stdout", "")).strip()  # type: ignore[union-attr]
        # Preserve the first column of porcelain output (for example `` M``).
        # Stripping leading whitespace would turn the first path's initial
        # character into a status-column character and corrupt the allowlist.
        status = str(snapshot["status"].get("stdout", "")).rstrip()  # type: ignore[union-attr]
        if head != expected:
            repository_reasons.append(f"{name} HEAD mismatch: {head or 'unavailable'}")
        if status:
            changed_paths = {
                line[3:].strip()
                for line in status.splitlines()
                if len(line) >= 4
            }
            if name == "profiler" and changed_paths <= ALLOWED_PROFILER_CAMPAIGN_CHANGES:
                continue
            repository_reasons.append(
                f"{name} worktree has unexpected changes: {sorted(changed_paths)}"
            )
    contract = _config_contract(config, hybrid)
    prompt = _prompt_token_snapshot(config, hybrid)
    prompt_count: int | None = None
    if prompt.get("return_code") == 0:
        try:
            prompt_count = int(json.loads(str(prompt.get("stdout", "")))["count"])
        except (TypeError, ValueError, json.JSONDecodeError, KeyError):
            prompt_count = None
    environment = capture_environment(hybrid, stage="campaign_preflight")
    environment_reasons = idle_reasons(environment)
    imports = {
        "prefill": _import_snapshot(config.prefill_python, hybrid.prefill.pythonpath),
        "decode": _import_snapshot(config.decode_python, hybrid.decode.pythonpath),
    }
    import_reasons = [
        f"{name} import preflight failed"
        for name, value in imports.items()
        if value.get("return_code") != 0
    ]
    marker_names = (
        "kv_handoff_end",
        "kv_transfer_setup_start",
        "kv_transfer_setup_end",
        "decode_schedule_wait_start",
        "decode_schedule_wait_end",
    )
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(config.vllm_rbln_root.rglob("*.py"))
        if ".venv" not in path.parts
    )
    markers = {name: name in source_text for name in marker_names}
    disk = estimate_disk_budget(config)
    reasons = [*repository_reasons, *environment_reasons, *import_reasons]
    reasons.extend(str(item) for item in contract["mismatches"])
    if prompt_count != config.expected_input_tokens:
        reasons.append(
            f"tokenizer input count mismatch: expected {config.expected_input_tokens}, got {prompt_count}"
        )
    if not all(markers.values()):
        reasons.append("one or more required runtime marker capabilities are absent")
    if not disk["valid"]:
        reasons.append("disk budget including 30% margin exceeds available space")
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not reasons,
        "reasons": reasons,
        "config_contract": contract,
        "prompt_tokenization": prompt,
        "prompt_token_count": prompt_count,
        "repositories": {"profiler": profiler, "vllm_rbln": vllm},
        "allowed_profiler_campaign_changes": sorted(ALLOWED_PROFILER_CAMPAIGN_CHANGES),
        "environment": environment,
        "imports": imports,
        "runtime_marker_capability": markers,
        "disk_budget": disk,
    }


def _metric_rows(layout: HybridRunLayout) -> dict[str, list[dict[str, Any]]]:
    result = {"gpu": [], "npu": [], "system": []}
    for path in (layout.gpu / "metrics/metrics.jsonl", layout.npu / "metrics/metrics.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            name = row.get("metric_name")
            if not isinstance(name, str) or not name.startswith("resource."):
                continue
            attributes = row.get("attributes")
            if not isinstance(attributes, dict) or not (
                "telemetry.sample_sequence" in attributes
                or "telemetry.sample_role" in attributes
            ):
                # Stage aggregates intentionally reuse the resource.* metric
                # namespace, but they are not periodic collector samples.
                continue
            if name.startswith("resource.gpu."):
                stream = "gpu"
            elif name.startswith("resource.npu."):
                stream = "npu"
            else:
                # CPU and system-memory records are emitted by the GPU-side
                # monitor, but belong to the host/system sample stream.
                stream = "system"
            result[stream].append(row)
    return result


def analyze_sampling_stream(
    rows: Iterable[dict[str, Any]],
    *,
    request_start_ns: int,
    request_end_ns: int,
    configured_interval_ms: int,
    minimum_background_samples: int,
) -> dict[str, object]:
    """Validate sample batches without treating metric fan-out as duplicates."""

    batches: dict[int, dict[str, object]] = {}
    malformed = 0
    for row in rows:
        timestamp = row.get("timestamp_ns")
        attributes = row.get("attributes")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or not isinstance(attributes, dict):
            malformed += 1
            continue
        sequence = attributes.get("telemetry.sample_sequence")
        role = attributes.get("telemetry.sample_role")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or role not in {"baseline", "background", "final"}:
            malformed += 1
            continue
        batch = batches.setdefault(sequence, {"timestamps": set(), "roles": set(), "record_count": 0})
        batch["timestamps"].add(timestamp)  # type: ignore[union-attr]
        batch["roles"].add(role)  # type: ignore[union-attr]
        batch["record_count"] = int(batch["record_count"]) + 1
    duplicate_sequences = sum(
        len(batch["timestamps"]) != 1 or len(batch["roles"]) != 1
        for batch in batches.values()
    )
    ordered: list[tuple[int, int, str]] = []
    for sequence, batch in batches.items():
        if len(batch["timestamps"]) == 1 and len(batch["roles"]) == 1:
            ordered.append((sequence, next(iter(batch["timestamps"])), next(iter(batch["roles"]))))
    ordered.sort()
    missing_sequences = sum(max(0, current[0] - previous[0] - 1) for previous, current in zip(ordered, ordered[1:]))
    monotonic = all(current[1] > previous[1] for previous, current in zip(ordered, ordered[1:]))
    interval_ns = configured_interval_ms * 1_000_000
    intervals = [
        current[1] - previous[1]
        for previous, current in zip(ordered, ordered[1:])
        if previous[2] == current[2] == "background" and current[0] == previous[0] + 1
    ]
    request_background = [
        timestamp
        for _, timestamp, role in ordered
        if role == "background" and request_start_ns <= timestamp <= request_end_ns
    ]
    roles = {
        role: sum(item[2] == role for item in ordered)
        for role in ("baseline", "background", "final")
    }
    summary = summarize_distribution(intervals).to_dict() if intervals else None
    median = float(summary["median"]) if summary is not None else None
    final_timestamps = [timestamp for _, timestamp, role in ordered if role == "final"]
    background_after_final = bool(
        final_timestamps
        and any(timestamp > max(final_timestamps) for _, timestamp, role in ordered if role == "background")
    )
    valid = bool(
        configured_interval_ms == ONE_HERTZ_INTERVAL_MS
        and malformed == 0
        and duplicate_sequences == 0
        and missing_sequences == 0
        and monotonic
        and roles["baseline"] == 1
        and roles["final"] == 1
        and not background_after_final
        and len(request_background) >= minimum_background_samples
        and median is not None
        and 900_000_000 <= median <= 1_100_000_000
    )
    return {
        "valid": valid,
        "configured_interval_ms": configured_interval_ms,
        "metric_record_count": sum(int(batch["record_count"]) for batch in batches.values()),
        "sample_batch_count": len(ordered),
        "role_sample_count": roles,
        "request_window_background_sample_count": len(request_background),
        "request_window_start_ns": request_start_ns,
        "request_window_end_ns": request_end_ns,
        "actual_interval_ns": summary,
        "median_in_0_9_to_1_1_seconds": median is not None and 900_000_000 <= median <= 1_100_000_000,
        "drop_count": missing_sequences,
        "duplicate_count": duplicate_sequences,
        "malformed_record_count": malformed,
        "timestamps_strictly_increasing": monotonic,
        "background_after_final": background_after_final,
    }


def validate_sampling(layout: HybridRunLayout, config: CampaignConfig, condition: str) -> dict[str, object]:
    lifecycle = _read_json(layout.coordinator / "telemetry_lifecycle.json")
    if condition == "reference":
        rows = _metric_rows(layout)
        count = sum(len(items) for items in rows.values())
        return {
            "valid": count == 0,
            "condition": condition,
            "telemetry_enabled": False,
            "resource_metric_record_count": count,
        }
    if lifecycle.get("requested_interval_ms") != ONE_HERTZ_INTERVAL_MS:
        raise AllModeOverheadError("candidate telemetry was not configured at exactly 1000 ms")
    start = lifecycle.get("request_start_ns")
    end = lifecycle.get("request_end_ns")
    if not isinstance(start, int) or not isinstance(end, int):
        raise AllModeOverheadError("telemetry request window is unavailable")
    rows = _metric_rows(layout)
    streams = {
        name: analyze_sampling_stream(
            values,
            request_start_ns=start,
            request_end_ns=end,
            configured_interval_ms=config.sample_interval_ms,
            minimum_background_samples=config.minimum_background_samples,
        )
        for name, values in rows.items()
    }
    return {
        "valid": all(item["valid"] is True for item in streams.values()),
        "condition": condition,
        "telemetry_enabled": True,
        "streams": streams,
    }


def _primary_artifacts(layout: HybridRunLayout, condition: str) -> list[Path]:
    if condition in {"reference", "monitor"}:
        return []
    root = layout.gpu if condition.startswith("gpu-") else layout.npu
    patterns = {
        "gpu-torch": ("*.pt.trace.json.gz",),
        "gpu-nsys": ("*.nsys-rep", "*.sqlite"),
        "npu-torch": ("*.pt.trace.json.gz",),
        "npu-rbln": ("*.pb",),
    }[condition]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(found))


def validate_mode_artifacts(layout: HybridRunLayout, condition: str) -> dict[str, object]:
    files = _primary_artifacts(layout, condition)
    expected_minimum = 0 if condition in {"reference", "monitor"} else 1
    summaries = []
    for source in (layout.gpu, layout.npu):
        path = source / "summary/detailed_profile.json"
        if path.is_file():
            summaries.append(_read_json(path))
    rows = [
        {
            "relative_path": path.relative_to(layout.bundle).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    nonempty = all(item["size_bytes"] > 0 for item in rows)
    summary_kinds = sorted(str(item.get("kind")) for item in summaries if item.get("enabled") is True)
    checks: dict[str, bool] = {
        "minimum_artifact_count": len(rows) >= expected_minimum,
        "all_artifacts_nonempty": nonempty,
    }
    expected_kind = {
        "gpu-torch": "gpu_torch",
        "gpu-nsys": "gpu_nsys",
        "npu-torch": "npu_vllm",
        "npu-rbln": "npu_rbln",
    }.get(condition)
    enabled = [item for item in summaries if item.get("enabled") is True]
    if expected_kind is None:
        checks["detailed_profiler_disabled"] = len(enabled) == 0
    if expected_kind is not None:
        checks["exactly_one_detailed_profiler"] = len(enabled) == 1
        checks["expected_profiler_kind"] = (
            len(enabled) == 1 and enabled[0].get("kind") == expected_kind
        )
    if condition in {"gpu-torch", "npu-torch"} and len(enabled) == 1:
        summary = enabled[0]
        checks["torch_events_parsed"] = (
            isinstance(summary.get("event_count"), int)
            and not isinstance(summary.get("event_count"), bool)
            and int(summary["event_count"]) > 0
            and isinstance(summary.get("activity_event_count"), int)
            and int(summary["activity_event_count"]) > 0
        )
        checks["torch_cpu_events_present"] = (
            isinstance(summary.get("cpu_event_count"), int)
            and int(summary["cpu_event_count"]) > 0
        )
        if condition == "gpu-torch":
            checks["cuda_kernel_events_present"] = (
                isinstance(summary.get("cuda_kernel_event_count"), int)
                and int(summary["cuda_kernel_event_count"]) > 0
            )
            checks["cuda_runtime_events_present"] = (
                isinstance(summary.get("cuda_runtime_event_count"), int)
                and int(summary["cuda_runtime_event_count"]) > 0
            )
    if condition == "gpu-nsys" and len(enabled) == 1:
        summary = enabled[0]
        names = {path.suffix for path in files}
        checks["nsys_report_present"] = ".nsys-rep" in names
        checks["nsys_sqlite_present"] = ".sqlite" in names
        reports = summary.get("reports")
        checks["official_nsys_reports_parsed"] = bool(
            isinstance(reports, dict)
            and all(
                isinstance(reports.get(name), dict)
                and reports[name].get("available") is True
                and isinstance(reports[name].get("data_row_count"), int)
                and int(reports[name]["data_row_count"]) > 0
                for name in ("cuda_api_sum", "cuda_gpu_kern_sum")
            )
        )
    if condition == "npu-rbln" and len(enabled) == 1:
        summary = enabled[0]
        checks["rbln_perfetto_format"] = (
            summary.get("format") == "perfetto_trace_protobuf"
        )
        checks["rbln_device_timing_present"] = (
            summary.get("device_timing_present") is True
        )
        checks["rbln_host_timing_present"] = (
            summary.get("host_timing_present") is True
        )
    valid = all(checks.values())
    return {
        "valid": valid,
        "condition": condition,
        "artifact_count": len(rows),
        "artifacts": rows,
        "enabled_summary_kinds": summary_kinds,
        "checks": checks,
        "detailed_profile_summaries": enabled,
    }


def _profile_operational_metrics(
    layout: HybridRunLayout,
    *,
    condition: str,
    runner_started_ns: int | None,
    runner_ended_ns: int | None,
    request_window_start_ns: int | None,
    postprocessing_started_ns: int | None,
) -> dict[str, object]:
    summaries = []
    for source in (layout.gpu, layout.npu):
        path = source / "summary/detailed_profile.json"
        if path.is_file():
            value = _read_json(path)
            if value.get("enabled") is True:
                summaries.append(value)
    profile_enabled = condition not in {"reference", "monitor"}
    if len(summaries) > 1:
        raise AllModeOverheadError(
            "more than one detailed profiler was enabled in one condition"
        )
    summary = summaries[0] if summaries else None
    capture_duration_ns: int | None = None
    start_api_rtt_ns: int | None = None
    stop_api_rtt_ns: int | None = None
    if summary is not None:
        api = summary.get("api")
        if isinstance(api, dict):
            start = api.get("start")
            stop = api.get("stop")
            if isinstance(start, dict) and isinstance(stop, dict):
                fields = (
                    start.get("before_monotonic_ns"),
                    start.get("after_monotonic_ns"),
                    stop.get("before_monotonic_ns"),
                    stop.get("after_monotonic_ns"),
                )
                if all(isinstance(item, int) and not isinstance(item, bool) for item in fields):
                    start_before, start_after, stop_before, stop_after = fields
                    assert isinstance(start_before, int)
                    assert isinstance(start_after, int)
                    assert isinstance(stop_before, int)
                    assert isinstance(stop_after, int)
                    if start_before <= start_after <= stop_before <= stop_after:
                        start_api_rtt_ns = start_after - start_before
                        stop_api_rtt_ns = stop_after - stop_before
                        capture_duration_ns = stop_after - start_before
    postprocessing_duration_ns = (
        runner_ended_ns - postprocessing_started_ns
        if postprocessing_started_ns is not None
        and runner_ended_ns is not None
        and runner_ended_ns >= postprocessing_started_ns
        else None
    )
    runner_wall_duration_ns = (
        runner_ended_ns - runner_started_ns
        if runner_started_ns is not None
        and runner_ended_ns is not None
        and runner_ended_ns >= runner_started_ns
        else None
    )
    startup_duration_ns = (
        request_window_start_ns - runner_started_ns
        if runner_started_ns is not None
        and request_window_start_ns is not None
        and request_window_start_ns >= runner_started_ns
        else None
    )
    return {
        "profile_enabled": profile_enabled,
        "profile_kind": summary.get("kind") if summary is not None else None,
        "capture_duration_ns": capture_duration_ns,
        "capture_duration_unavailable_reason": (
            None
            if capture_duration_ns is not None
            else (
                "detailed profiling is disabled for this condition"
                if not profile_enabled
                else "validated profiler API timestamps are unavailable"
            )
        ),
        "profiler_start_api_rtt_ns": start_api_rtt_ns,
        "profiler_stop_and_finalize_api_rtt_ns": stop_api_rtt_ns,
        "offline_postprocessing_duration_ns": postprocessing_duration_ns,
        "offline_postprocessing_boundary": (
            "HybridRunner source normalization through publication return"
            if postprocessing_duration_ns is not None
            else None
        ),
        "runner_wall_duration_ns": runner_wall_duration_ns,
        "runner_wall_duration_unavailable_reason": (
            None
            if runner_wall_duration_ns is not None
            else "runner boundary was not persisted before postprocess failure"
        ),
        "startup_warmup_and_profiler_start_duration_ns": startup_duration_ns,
        "startup_duration_unavailable_reason": (
            None
            if startup_duration_ns is not None
            else "runner start boundary was not persisted before postprocess failure"
        ),
        "startup_excluded_from_inference_overhead": True,
        "run_artifact_size_bytes": _artifact_size(layout.bundle),
    }


def _block_metrics(layout: HybridRunLayout) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in (layout.gpu / "raw/client/measured_requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start = min(int(item["request_start_ns"]) for item in rows)
    end = max(int(item["stream_end_ns"]) for item in rows)
    duration = end - start
    if duration <= 0:
        raise AllModeOverheadError("formal request window duration is not positive")
    output_tokens = sum(int(item["output_tokens"]) for item in rows)
    return {
        "request_count": len(rows),
        "request_window_start_ns": start,
        "request_window_end_ns": end,
        "request_window_duration_ns": duration,
        "http_statuses": sorted({int(item["http_status"]) for item in rows}),
        "input_tokens": sum(int(item["input_tokens"]) for item in rows),
        "output_tokens": output_tokens,
        "latency": {
            "e2e_ns": summarize_distribution([float(item["e2e_ns"]) for item in rows]).to_dict(),
            "ttft_ns": summarize_distribution([float(item["ttft_ns"]) for item in rows]).to_dict(),
            "tpot_ns": summarize_distribution([float(item["tpot_ns"]) for item in rows]).to_dict(),
        },
        "throughput": {
            "requests_per_second": len(rows) * 1_000_000_000 / duration,
            "output_tokens_per_second": output_tokens * 1_000_000_000 / duration,
        },
    }


def _block_references(layout: HybridRunLayout) -> dict[str, object]:
    selected = {
        "measured_requests": layout.gpu / "raw/client/measured_requests.jsonl",
        "telemetry_lifecycle": layout.coordinator / "telemetry_lifecycle.json",
        "shutdown_integrity": layout.coordinator / "shutdown_integrity.json",
        "compile_gate": layout.coordinator / "compile_gate.json",
        "hybrid_manifest": layout.hybrid / "manifest.json",
        "gpu_manifest": layout.gpu / "manifest.json",
        "npu_manifest": layout.npu / "manifest.json",
        "perfetto_full": layout.perfetto / "trace.pftrace",
        "perfetto_request_focused": layout.request_perfetto / "trace.request-focused.pftrace",
        "overview_html": layout.overview / "overview.html",
    }
    return {
        name: {
            "relative_path": path.relative_to(layout.bundle).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in selected.items()
        if path.is_file()
    }


def _record_block_failure(
    *,
    config: CampaignConfig,
    root: Path,
    block: dict[str, Any],
    error: BaseException,
    interrupted: bool,
) -> Path | None:
    """Best-effort immutable evidence for a failed hardware block.

    The original exception must remain authoritative.  Every diagnostic step
    is therefore isolated so an unavailable device query or a partially
    written runner bundle cannot hide the failure that stopped the campaign.
    """

    block_root = (
        root
        / "raw"
        / f"round-{int(block['round']):02d}"
        / str(block["condition"])
    )
    try:
        block_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    diagnostic_errors: list[str] = []
    hybrid: HybridRunnerConfig | None = None
    try:
        hybrid = config.load_hybrid()
    except Exception as diagnostic_error:
        diagnostic_errors.append(
            "load_hybrid: "
            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        )

    environment_after: dict[str, object] | None = None
    if hybrid is not None:
        try:
            environment_after = capture_environment(
                hybrid, stage="failed_condition_block"
            )
            _write_json(
                block_root / "environment_after_failure.json",
                environment_after,
                exclusive=True,
            )
        except Exception as diagnostic_error:
            diagnostic_errors.append(
                "environment_after_failure: "
                f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            )

    cache_after: object | None = None
    if hybrid is not None:
        try:
            cache_after = _cache_fingerprint(hybrid.rbln_cache_path)
            _write_json(
                block_root / "cache_fingerprint_after_failure.json",
                cache_after,
                exclusive=True,
            )
        except Exception as diagnostic_error:
            diagnostic_errors.append(
                "cache_fingerprint_after_failure: "
                f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            )

    artifact_references: dict[str, object] = {}
    try:
        layout = HybridRunLayout(block_root / "runs", str(block["block_id"]))
        artifact_references = _block_references(layout)
    except Exception as diagnostic_error:
        diagnostic_errors.append(
            "artifact_references: "
            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        )

    cache_before_path = block_root / "cache_fingerprint_before.json"
    cache_unchanged: bool | None = None
    if cache_after is not None and cache_before_path.is_file():
        try:
            cache_unchanged = (
                json.loads(cache_before_path.read_text(encoding="utf-8"))
                == cache_after
            )
        except Exception as diagnostic_error:
            diagnostic_errors.append(
                "cache_comparison: "
                f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            )

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "block": block,
        "status": "interrupted" if interrupted else "failed",
        "automatic_retries": 0,
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "recorded_unix_ns": time.time_ns(),
        "recorded_monotonic_ns": time.monotonic_ns(),
        "environment_after_failure_recorded": environment_after is not None,
        "post_failure_idle_reasons": (
            idle_reasons(environment_after)
            if environment_after is not None
            else None
        ),
        "cache_after_failure_recorded": cache_after is not None,
        "cache_unchanged": cache_unchanged,
        "artifact_references": artifact_references,
        "stdout_log_present": (block_root / "stdout.log").is_file(),
        "stderr_log_present": (block_root / "stderr.log").is_file(),
        "diagnostic_errors": diagnostic_errors,
    }
    path = block_root / "failure.json"
    try:
        _write_json(path, evidence, exclusive=True)
    except Exception:
        return None
    return path


def _record_model_cache_after(
    root: Path,
    hybrid: HybridRunnerConfig,
    *,
    before: dict[str, object],
) -> dict[str, object]:
    after = {
        "model": _tree_fingerprint(hybrid.model_path),
        "cache": _tree_fingerprint(hybrid.rbln_cache_path),
    }
    _write_json_once_or_verify(root / "model_cache_fingerprint_after.json", after)
    return {
        "unchanged": after == before,
        "after": after,
    }


def _checkpoint(root: Path, config: CampaignConfig, schedule: dict[str, object]) -> dict[str, Any]:
    path = root / "checkpoint.json"
    if path.is_file():
        return _read_json(path)
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config.campaign_id,
        "config_sha256": config.sha256,
        "schedule_sha256": schedule["sha256"],
        "status": "initialized",
        "completed_blocks": [],
        "failed_block": None,
        "current_block": None,
        "automatic_retries": 0,
    }
    _write_json(path, value, exclusive=True)
    return value


def _update_checkpoint(root: Path, value: dict[str, Any]) -> None:
    value = dict(value)
    value["updated_unix_ns"] = time.time_ns()
    _write_json(root / "checkpoint.json", value)


def _initialize_campaign(config: CampaignConfig, root: Path) -> tuple[dict[str, object], dict[str, Any]]:
    if root.exists():
        raise FileExistsError(f"campaign root already exists: {root}")
    root.mkdir(parents=True)
    for subdir in ("raw", "summary", "report", "manifest"):
        (root / subdir).mkdir()
    schedule = config.schedule
    _atomic_write(
        root / "README.md",
        (
            "# All-mode one-hertz overhead campaign\n\n"
            "This directory preserves one immutable, fixed-seed comparison of a "
            "profiling-off reference with monitor, GPU Torch, GPU Nsight, NPU Torch, "
            "and NPU RBLN conditions. Each candidate enables GPU/NPU/System telemetry "
            "at exactly 1000 ms. Failed or interrupted blocks are never retried or "
            "overwritten.\n\n"
            "Use `python -m tools.evaluation all-mode-overhead status --campaign-root "
            f"{root}` to inspect the checkpoint.\n"
        ).encode("utf-8"),
        exclusive=True,
    )
    _write_json(root / "campaign_config.json", config.to_dict(), exclusive=True)
    _atomic_write(root / "hybrid_config.json", config.hybrid_config_path.read_bytes(), exclusive=True)
    _write_json(root / "schedule.json", schedule, exclusive=True)
    _write_json(
        root / "execution_plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "formal_rounds": config.formal_rounds,
            "formal_condition_blocks": len(schedule["blocks"]),
            "server_lifecycle_count_planned": len(schedule["blocks"]),
            "automatic_retries": 0,
            "warmup_requests_per_block": config.warmup_requests,
            "formal_requests_per_block": config.formal_requests_per_block,
            "reference": {"resource_telemetry": False, "detailed_profiler": False},
            "candidates": {
                name: {
                    "resource_telemetry": True,
                    "sample_interval_ms": ONE_HERTZ_INTERVAL_MS,
                    "detailed_profiler": None if name == "monitor" else name,
                }
                for name in CONDITIONS[1:]
            },
            "pairing": "each candidate with the same round reference",
            "outlier_exclusion": False,
            "formal_block_postprocessing": "deferred",
            "representative_postprocessing": {
                "count": len(CONDITIONS) - 1,
                "selection_policy": (
                    "first valid run per candidate mode in ascending round order"
                ),
                "after_all_formal_blocks": True,
            },
        },
        exclusive=True,
    )
    checkpoint = _checkpoint(root, config, schedule)
    return schedule, checkpoint


def _resume_campaign(config: CampaignConfig, root: Path) -> tuple[dict[str, object], dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    schedule = _read_json(root / "schedule.json")
    checkpoint = _read_json(root / "checkpoint.json")
    if checkpoint.get("config_sha256") != config.sha256 or checkpoint.get("schedule_sha256") != schedule.get("sha256"):
        raise AllModeOverheadError("resume identity mismatch")
    if checkpoint.get("status") == "cancelled":
        raise AllModeOverheadError("cancelled campaigns cannot be resumed")
    if checkpoint.get("failed_block") is not None:
        raise AllModeOverheadError("failed blocks are never retried; start a separately approved campaign")
    if checkpoint.get("current_block") is not None:
        checkpoint["status"] = "interrupted"
        checkpoint["failed_block"] = checkpoint["current_block"]
        _update_checkpoint(root, checkpoint)
        raise AllModeOverheadError("previous invocation stopped inside a block; automatic retry is forbidden")
    return schedule, checkpoint


def _run_one_block(
    *,
    config: CampaignConfig,
    root: Path,
    block: dict[str, Any],
) -> dict[str, object]:
    condition = str(block["condition"])
    block_id = str(block["block_id"])
    block_root = root / "raw" / f"round-{int(block['round']):02d}" / condition
    block_root.mkdir(parents=True)
    (block_root / "runs").mkdir()
    hybrid = config.load_hybrid()
    enable_telemetry = condition != "reference"
    profile_mode = "monitor" if condition == "reference" else condition
    validate_hybrid_invocation(
        hybrid,
        run_root=block_root / "runs",
        run_id=block_id,
        profile_mode=profile_mode,
    )
    _write_json(block_root / "config.json", {
        "campaign_config_sha256": config.sha256,
        "hybrid_config_sha256": config.hybrid_config_sha256,
        "block": block,
        "profile_mode": profile_mode,
        "resource_telemetry": enable_telemetry,
        "sample_interval_ms": config.sample_interval_ms if enable_telemetry else None,
    }, exclusive=True)
    _write_json(block_root / "command.json", {
        "execution": "in_process_HybridRunner",
        "profile_mode": profile_mode,
        "enable_telemetry": enable_telemetry,
        "run_root": str(block_root / "runs"),
        "run_id": block_id,
        "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    }, exclusive=True)
    before = wait_for_idle(hybrid)
    _write_json(block_root / "environment_before.json", before, exclusive=True)
    cache_before = _cache_fingerprint(hybrid.rbln_cache_path)
    _write_json(
        block_root / "cache_fingerprint_before.json",
        cache_before,
        exclusive=True,
    )
    started_mono = time.monotonic_ns()
    started_unix = time.time_ns()
    stdout_path = block_root / "stdout.log"
    stderr_path = block_root / "stderr.log"
    runner = _CollectionOnlyHybridRunner(
        hybrid,
        run_root=block_root / "runs",
        run_id=block_id,
        profile_mode=profile_mode,
        enable_telemetry=enable_telemetry,
    )
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = runner.run()
    ended_mono = time.monotonic_ns()
    ended_unix = time.time_ns()
    layout = HybridRunLayout(block_root / "runs", block_id)
    cache_after = _cache_fingerprint(hybrid.rbln_cache_path)
    if result.status is not RunStatus.SUCCEEDED:
        raise AllModeOverheadError("; ".join(result.errors) or "HybridRunner failed")
    if cache_before != cache_after:
        raise AllModeOverheadError("model cache fingerprint changed")
    trial = validate_trial(
        block_root,
        attempt_id=block_id,
        condition=condition,
        expected_requests=config.formal_requests_per_block,
        expected_input_tokens=config.expected_input_tokens,
        expected_output_tokens=config.expected_output_tokens,
        require_request_focused=False,
        require_derived_products=False,
    )
    if any(
        path.exists()
        for path in (layout.perfetto, layout.request_perfetto, layout.overview)
    ):
        raise AllModeOverheadError(
            "formal block unexpectedly generated deferred derived products"
        )
    sampling = validate_sampling(layout, config, condition)
    if sampling.get("valid") is not True:
        raise AllModeOverheadError("one-hertz sampling validation failed")
    artifacts = validate_mode_artifacts(layout, condition)
    if artifacts.get("valid") is not True:
        raise AllModeOverheadError("detailed profiler artifact validation failed")
    metrics = _block_metrics(layout)
    operational = _profile_operational_metrics(
        layout,
        condition=condition,
        runner_started_ns=started_mono,
        runner_ended_ns=ended_mono,
        request_window_start_ns=int(metrics["request_window_start_ns"]),
        postprocessing_started_ns=runner.postprocessing_started_monotonic_ns,
    )
    after = capture_environment(hybrid, stage="post_condition_block")
    remaining = idle_reasons(after)
    if remaining:
        raise AllModeOverheadError("post-block cleanup failed: " + "; ".join(remaining))
    _write_json(block_root / "environment_after.json", after, exclusive=True)
    _write_json(block_root / "requests.json", {"reference": _block_references(layout)["measured_requests"], "metrics": metrics}, exclusive=True)
    _write_json(block_root / "telemetry_lifecycle.json", sampling, exclusive=True)
    _write_json(block_root / "shutdown_integrity.json", _read_json(layout.coordinator / "shutdown_integrity.json"), exclusive=True)
    _write_json(block_root / "artifact_references.json", _block_references(layout), exclusive=True)
    block_result = {
        "schema_version": SCHEMA_VERSION,
        "block": block,
        "status": "succeeded",
        "profile_mode": profile_mode,
        "resource_telemetry": enable_telemetry,
        "derived_products_deferred": True,
        "representative_postprocess_selected": False,
        "runner_started_monotonic_ns": started_mono,
        "runner_ended_monotonic_ns": ended_mono,
        "runner_wall_duration_ns": ended_mono - started_mono,
        "runner_started_unix_ns": started_unix,
        "runner_ended_unix_ns": ended_unix,
        "trial_validation": trial,
        "sampling_validation": sampling,
        "artifact_validation": artifacts,
        "metrics": metrics,
        "operational_metrics": operational,
        "cache_unchanged": True,
        "cleanup_valid": True,
    }
    _write_json(block_root / "block_result.json", block_result, exclusive=True)
    return block_result


def recover_failed_postprocess(
    *,
    config_path: Path,
    campaign_root: Path,
) -> dict[str, object]:
    """Finish one failed block from its immutable successful collection.

    This path never starts a server or collector. It is limited to a
    checkpoint whose runner completed collection and failed only while
    deriving Perfetto/Overview products. Original failure evidence remains
    untouched.
    """

    config = load_campaign_config(config_path)
    root = Path(campaign_root)
    schedule = _read_json(root / "schedule.json")
    checkpoint = _read_json(root / "checkpoint.json")
    if (
        checkpoint.get("config_sha256") != config.sha256
        or checkpoint.get("schedule_sha256") != schedule.get("sha256")
    ):
        raise AllModeOverheadError("postprocess recovery identity mismatch")
    block_id = checkpoint.get("failed_block")
    if not isinstance(block_id, str):
        raise AllModeOverheadError("checkpoint has no failed block to recover")
    blocks = [
        item
        for item in schedule.get("blocks", [])
        if isinstance(item, dict) and item.get("block_id") == block_id
    ]
    if len(blocks) != 1:
        raise AllModeOverheadError("failed block is not uniquely present in schedule")
    block = blocks[0]
    condition = str(block["condition"])
    block_root = root / "raw" / f"round-{int(block['round']):02d}" / condition
    failure = _read_json(block_root / "failure.json")
    if not str(failure.get("failure_message", "")).startswith("postprocess "):
        raise AllModeOverheadError("only a postprocess-only failure can be recovered")
    layout = HybridRunLayout(block_root / "runs", block_id)
    runner_result = _read_json(existing_final_result_path(layout.publication))
    failures = runner_result.get("failures")
    if (
        runner_result.get("measured_completed") != config.formal_requests_per_block
        or not isinstance(failures, list)
        or not failures
        or any(
            not isinstance(item, dict) or item.get("failure_class") != "postprocess"
            for item in failures
        )
    ):
        raise AllModeOverheadError("runner failure was not limited to postprocessing")
    coordinator_result = _read_json(
        existing_collection_result_path(layout.coordinator)
    )
    shutdown = _read_json(layout.coordinator / "shutdown_integrity.json")
    if (
        coordinator_result.get("status") != "succeeded"
        or shutdown.get("status") != "valid"
    ):
        raise AllModeOverheadError("collection or shutdown did not succeed")
    hybrid = config.load_hybrid()
    current = {
        "model": _tree_fingerprint(hybrid.model_path),
        "cache": _tree_fingerprint(hybrid.rbln_cache_path),
    }
    if current != _read_json(root / "model_cache_fingerprint_before.json"):
        raise AllModeOverheadError("model or cache changed before recovery")
    before_environment = wait_for_idle(hybrid)

    required_products = [
        layout.perfetto / "trace.pftrace",
        layout.overview / "overview.json",
        layout.overview / "overview.html",
        layout.publication / "determinism.json",
    ]
    if config.formal_requests_per_block == 1:
        required_products.append(
            layout.perfetto / "trace.request-focused.pftrace"
        )
    present = [path.is_file() for path in required_products]
    recovery_started_ns = time.monotonic_ns()
    if not any(present):
        profile_mode = "monitor" if condition == "reference" else condition
        runner = _TimedHybridRunner(
            hybrid,
            run_root=block_root / "runs",
            run_id=block_id,
            profile_mode=profile_mode,
            enable_telemetry=condition != "reference",
        )
        runner._derive_products()
    elif (
        (layout.perfetto / "trace.pftrace").is_file()
        and (layout.perfetto / "trace_validation.json").is_file()
        and not layout.overview.exists()
        and not (layout.publication / "determinism.json").exists()
    ):
        from perfetto_hetero_profiler.overview.generator import (
            OverviewGenerationConfig,
            generate_overview,
        )

        overview = generate_overview(
            OverviewGenerationConfig(
                run_directory=layout.hybrid,
                perfetto_directory=layout.perfetto,
                output_directory=layout.overview,
                trace_processor_path=hybrid.trace_processor_path,
            )
        )
        if overview.get("status") != "succeeded":
            raise AllModeOverheadError("Overview recovery did not succeed")
        focused_hashes = (
            {
                "trace.request-focused.pftrace": sha256_file(
                    layout.perfetto / "trace.request-focused.pftrace"
                )
            }
            if config.formal_requests_per_block == 1
            else {}
        )
        _write_json(
            layout.publication / "determinism.json",
            {
                "verification_mode": "single_production_generation",
                "perfetto_byte_identical": None,
                "perfetto_sha256": {
                    "trace.pftrace": sha256_file(
                        layout.perfetto / "trace.pftrace"
                    )
                },
                "request_focused_perfetto_byte_identical": None,
                "request_focused_perfetto_sha256": focused_hashes,
                "request_focused_unavailable_reason": (
                    None
                    if focused_hashes
                    else "request-focused trace requires exactly one measured request"
                ),
                "overview_byte_identical": None,
                "overview_sha256": {
                    name: sha256_file(layout.overview / name)
                    for name in ("overview.json", "overview.html")
                },
                "temporary_repeat_preserved": False,
            },
            exclusive=True,
        )
    elif not all(present):
        raise AllModeOverheadError(
            "partial postprocess recovery products require manual inspection"
        )
    recovery_ended_ns = time.monotonic_ns()

    trial = validate_trial(
        block_root,
        attempt_id=block_id,
        condition=condition,
        expected_requests=config.formal_requests_per_block,
        expected_input_tokens=config.expected_input_tokens,
        expected_output_tokens=config.expected_output_tokens,
        require_request_focused=config.formal_requests_per_block == 1,
    )
    sampling = validate_sampling(layout, config, condition)
    if sampling.get("valid") is not True:
        raise AllModeOverheadError("one-hertz sampling validation failed")
    artifacts = validate_mode_artifacts(layout, condition)
    if artifacts.get("valid") is not True:
        raise AllModeOverheadError("detailed profiler artifact validation failed")
    metrics = _block_metrics(layout)
    operational = _profile_operational_metrics(
        layout,
        condition=condition,
        runner_started_ns=None,
        runner_ended_ns=None,
        request_window_start_ns=int(metrics["request_window_start_ns"]),
        postprocessing_started_ns=None,
    )
    operational["offline_postprocessing_recovery_duration_ns"] = (
        recovery_ended_ns - recovery_started_ns
    )
    after_environment = capture_environment(hybrid, stage="postprocess_recovery")
    remaining = idle_reasons(after_environment)
    if remaining:
        raise AllModeOverheadError(
            "post-recovery cleanup failed: " + "; ".join(remaining)
        )
    if current != {
        "model": _tree_fingerprint(hybrid.model_path),
        "cache": _tree_fingerprint(hybrid.rbln_cache_path),
    }:
        raise AllModeOverheadError("model or cache changed during recovery")

    references = _block_references(layout)
    _write_json(
        block_root / "environment_after.json", after_environment, exclusive=True
    )
    _write_json(
        block_root / "requests.json",
        {"reference": references["measured_requests"], "metrics": metrics},
        exclusive=True,
    )
    _write_json(
        block_root / "telemetry_lifecycle.json", sampling, exclusive=True
    )
    _write_json(
        block_root / "shutdown_integrity.json", shutdown, exclusive=True
    )
    _write_json(
        block_root / "artifact_references.json", references, exclusive=True
    )
    block_result = {
        "schema_version": SCHEMA_VERSION,
        "block": block,
        "status": "succeeded",
        "profile_mode": "monitor" if condition == "reference" else condition,
        "resource_telemetry": condition != "reference",
        "postprocess_recovered": True,
        "hardware_rerun": False,
        "original_failure_evidence": "failure.json",
        "original_failed_publication_result": (
            existing_final_result_path(layout.publication)
        ).relative_to(block_root).as_posix(),
        "trial_validation": trial,
        "sampling_validation": sampling,
        "artifact_validation": artifacts,
        "metrics": metrics,
        "operational_metrics": operational,
        "cache_unchanged": True,
        "cleanup_valid": True,
    }
    _write_json(block_root / "block_result.json", block_result, exclusive=True)
    _write_json(
        block_root / "postprocess_recovery.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "hardware_rerun": False,
            "original_failure_preserved": True,
            "before_environment_idle": idle_reasons(before_environment) == [],
            "after_environment_idle": True,
            "duration_ns": recovery_ended_ns - recovery_started_ns,
            "artifact_references": references,
        },
        exclusive=True,
    )
    completed = set(checkpoint.get("completed_blocks", []))
    completed.add(block_id)
    checkpoint["completed_blocks"] = sorted(completed)
    checkpoint["resolved_failed_block"] = block_id
    checkpoint["resolved_failure_evidence"] = checkpoint.get("failure_evidence")
    checkpoint["failed_block"] = None
    checkpoint["current_block"] = None
    checkpoint["status"] = "postprocess_recovered"
    _update_checkpoint(root, checkpoint)
    return {
        "status": "postprocess_recovered",
        "campaign_root": str(root),
        "block_id": block_id,
        "hardware_blocks_started": 0,
        "hardware_rerun": False,
        "completed_count": len(completed),
    }


def postprocess_representative_runs(
    *,
    config: CampaignConfig,
    root: Path,
    schedule: dict[str, object],
) -> dict[str, object]:
    """Generate heavyweight products for the first valid run of each mode.

    This function never starts hardware. Selection is deterministic by round
    number and happens only after every formal collection block has passed.
    """

    blocks = schedule.get("blocks")
    if not isinstance(blocks, list):
        raise AllModeOverheadError("schedule blocks are unavailable")
    expected_count = config.formal_rounds * len(CONDITIONS)
    checkpoint = _read_json(root / "checkpoint.json")
    completed = set(checkpoint.get("completed_blocks", []))
    if len(completed) != expected_count:
        raise AllModeOverheadError(
            "representative postprocessing requires all formal blocks"
        )

    selected: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS[1:]:
        candidates = sorted(
            (
                item
                for item in blocks
                if isinstance(item, dict) and item.get("condition") == condition
            ),
            key=lambda item: int(item["round"]),
        )
        for block in candidates:
            block_root = (
                root
                / "raw"
                / f"round-{int(block['round']):02d}"
                / condition
            )
            result_path = block_root / "block_result.json"
            if result_path.is_file() and _read_json(result_path).get("status") == "succeeded":
                selected[condition] = {
                    "block_id": str(block["block_id"]),
                    "round": int(block["round"]),
                    "relative_block_root": block_root.relative_to(root).as_posix(),
                }
                break
        if condition not in selected:
            raise AllModeOverheadError(
                f"no valid representative is available for {condition}"
            )

    selection = {
        "schema_version": SCHEMA_VERSION,
        "policy": "first valid run per candidate mode in ascending round order",
        "selection_uses_performance_results": False,
        "hardware_rerun": False,
        "selected": selected,
    }
    _write_json_once_or_verify(root / "representative_selection.json", selection)

    results: dict[str, object] = {}
    hybrid = config.load_hybrid()
    for condition in CONDITIONS[1:]:
        item = selected[condition]
        block_id = str(item["block_id"])
        block_root = root / str(item["relative_block_root"])
        layout = HybridRunLayout(block_root / "runs", block_id)
        required = (
            layout.perfetto / "trace.pftrace",
            layout.perfetto / "trace_validation.json",
            layout.overview / "overview.json",
            layout.overview / "overview.html",
            layout.overview / "overview_validation.json",
            layout.publication / "determinism.json",
        )
        present = [path.is_file() for path in required]
        started_ns = time.monotonic_ns()
        if not any(present):
            runner = _TimedHybridRunner(
                hybrid,
                run_root=block_root / "runs",
                run_id=block_id,
                profile_mode=condition,
                enable_telemetry=True,
            )
            runner._derive_products()
        elif not all(present):
            raise AllModeOverheadError(
                f"partial representative products require inspection: {block_id}"
            )
        trial = validate_trial(
            block_root,
            attempt_id=block_id,
            condition=condition,
            expected_requests=config.formal_requests_per_block,
            expected_input_tokens=config.expected_input_tokens,
            expected_output_tokens=config.expected_output_tokens,
            require_request_focused=False,
            require_derived_products=True,
        )
        artifact_validation = validate_mode_artifacts(layout, condition)
        if artifact_validation.get("valid") is not True:
            raise AllModeOverheadError(
                f"representative artifact validation failed: {block_id}"
            )
        trace_validation = _read_json(layout.perfetto / "trace_validation.json")
        overview_validation = _read_json(
            layout.overview / "overview_validation.json"
        )
        if (
            trace_validation.get("valid") is not True
            or overview_validation.get("valid") is not True
        ):
            raise AllModeOverheadError(
                f"representative derived validation failed: {block_id}"
            )
        evidence = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "condition": condition,
            "block_id": block_id,
            "round": item["round"],
            "selection_policy": selection["policy"],
            "hardware_rerun": False,
            "duration_ns": time.monotonic_ns() - started_ns,
            "request_focused_trace_generated": False,
            "request_focused_unavailable_reason": (
                "request-focused trace requires exactly one measured request; "
                f"this block has {config.formal_requests_per_block}"
            ),
            "trace_processor_validation": {
                "valid": True,
                "counts": trace_validation.get("counts"),
                "import_errors": next(
                    (
                        query.get("row_count")
                        for query in trace_validation.get("queries", [])
                        if isinstance(query, dict)
                        and query.get("name") == "import_errors"
                    ),
                    None,
                ),
            },
            "overview_validation": {
                "valid": True,
                "mismatches": overview_validation.get("mismatches", []),
            },
            "raw_profiler_artifact_validation": artifact_validation,
            "trial_validation": trial,
            "artifacts": _block_references(layout),
        }
        evidence_path = block_root / "representative_postprocess.json"
        if evidence_path.is_file():
            existing = _read_json(evidence_path)
            if any(
                existing.get(field) != evidence.get(field)
                for field in (
                    "status",
                    "condition",
                    "block_id",
                    "round",
                    "hardware_rerun",
                    "request_focused_trace_generated",
                    "trace_processor_validation",
                    "overview_validation",
                    "raw_profiler_artifact_validation",
                    "trial_validation",
                    "artifacts",
                )
            ):
                raise AllModeOverheadError(
                    f"representative evidence mismatch: {block_id}"
                )
            evidence = existing
        else:
            _write_json(evidence_path, evidence, exclusive=True)
        results[condition] = {
            "block_id": block_id,
            "round": item["round"],
            "evidence": evidence_path.relative_to(root).as_posix(),
            "artifacts": evidence["artifacts"],
        }

    document = {
        "schema_version": SCHEMA_VERSION,
        "status": "succeeded",
        "hardware_rerun": False,
        "selection": selection,
        "results": results,
    }
    _write_json_once_or_verify(root / "representative_postprocess.json", document)
    return document


def _paired_metric(
    by_block: dict[str, dict[str, Any]],
    *,
    condition: str,
    metric_path: tuple[str, ...],
    direction: OverheadDirection,
    formal_rounds: int,
) -> dict[str, object]:
    references: dict[int, float] = {}
    candidates: dict[int, float] = {}
    for round_index in range(1, formal_rounds + 1):
        for name, target in (("reference", references), (condition, candidates)):
            value: Any = by_block[f"round-{round_index:02d}-{name}"]["metrics"]
            for component in metric_path:
                value = value[component]
            target[round_index] = float(value)
    result = paired_overhead(
        references,
        candidates,
        direction=direction,
        expected_pair_count=formal_rounds,
    ).to_dict()
    ratios = [float(item["overhead_ratio"]) for item in result["pairs"]]
    result["percent_scale"] = 100.0
    if formal_rounds == DEFAULT_FORMAL_ROUNDS:
        mean = statistics.fmean(ratios)
        standard_error = statistics.stdev(ratios) / math.sqrt(len(ratios))
        result["two_sided_95_percent_confidence_interval"] = {
            "method": "Student t interval for paired mean, df=4",
            "lower_percent": 100 * (mean - _TWO_SIDED_T_95_DF4 * standard_error),
            "upper_percent": 100 * (mean + _TWO_SIDED_T_95_DF4 * standard_error),
        }
        result["one_sided_95_percent_upper"] = {
            "method": "one-sided Student t upper bound for paired mean, df=4",
            "upper_percent": 100 * (mean + _ONE_SIDED_T_95_DF4 * standard_error),
        }
    else:
        reason = (
            "not reported: this campaign has three fixed-condition repeats and "
            "does not claim confidence-interval generalization"
        )
        result["two_sided_95_percent_confidence_interval"] = {
            "method": None,
            "lower_percent": None,
            "upper_percent": None,
            "unavailable_reason": reason,
        }
        result["one_sided_95_percent_upper"] = {
            "method": None,
            "upper_percent": None,
            "unavailable_reason": reason,
        }
    return result


def build_campaign_report(root: Path) -> dict[str, object]:
    schedule = _read_json(root / "schedule.json")
    execution_plan_path = root / "execution_plan.json"
    execution_plan = (
        _read_json(execution_plan_path) if execution_plan_path.is_file() else {}
    )
    formal_rounds = schedule.get("formal_rounds")
    if (
        isinstance(formal_rounds, bool)
        or not isinstance(formal_rounds, int)
        or formal_rounds < 1
    ):
        raise AllModeOverheadError("schedule formal_rounds is invalid")
    expected_block_count = formal_rounds * len(CONDITIONS)
    checkpoint = _read_json(root / "checkpoint.json")
    by_block: dict[str, dict[str, Any]] = {}
    for block in schedule["blocks"]:
        block_id = str(block["block_id"])
        path = root / "raw" / f"round-{int(block['round']):02d}" / str(block["condition"]) / "block_result.json"
        if path.is_file():
            by_block[block_id] = _read_json(path)
    modes: dict[str, object] = {}
    for condition in CONDITIONS[1:]:
        expected = [f"round-{index:02d}-{condition}" for index in range(1, formal_rounds + 1)]
        references = [f"round-{index:02d}-reference" for index in range(1, formal_rounds + 1)]
        excluded_pairs = [
            {
                "round": round_index,
                "reason": "candidate block missing"
                if f"round-{round_index:02d}-{condition}" not in by_block
                else "same-round reference block missing",
            }
            for round_index in range(1, formal_rounds + 1)
            if f"round-{round_index:02d}-{condition}" not in by_block
            or f"round-{round_index:02d}-reference" not in by_block
        ]
        if excluded_pairs:
            modes[condition] = {
                "valid_pairs": formal_rounds - len(excluded_pairs),
                "excluded_pairs": excluded_pairs,
                "met": False,
                "supported": False,
                "status": "incomplete",
            }
            continue
        e2e = _paired_metric(
            by_block,
            condition=condition,
            metric_path=("latency", "e2e_ns", "median"),
            direction=OverheadDirection.INCREASE,
            formal_rounds=formal_rounds,
        )
        throughput = _paired_metric(
            by_block,
            condition=condition,
            metric_path=("throughput", "output_tokens_per_second"),
            direction=OverheadDirection.THROUGHPUT_DEGRADATION,
            formal_rounds=formal_rounds,
        )
        secondary = {
            "ttft_overhead": _paired_metric(
                by_block,
                condition=condition,
                metric_path=("latency", "ttft_ns", "median"),
                direction=OverheadDirection.INCREASE,
                formal_rounds=formal_rounds,
            ),
            "tpot_overhead": _paired_metric(
                by_block,
                condition=condition,
                metric_path=("latency", "tpot_ns", "median"),
                direction=OverheadDirection.INCREASE,
                formal_rounds=formal_rounds,
            ),
            "request_throughput_degradation": _paired_metric(
                by_block,
                condition=condition,
                metric_path=("throughput", "requests_per_second"),
                direction=OverheadDirection.THROUGHPUT_DEGRADATION,
                formal_rounds=formal_rounds,
            ),
        }
        e2e_median = 100 * float(e2e["overhead_ratio_summary"]["median"])
        throughput_median = 100 * float(throughput["overhead_ratio_summary"]["median"])
        e2e_upper_raw = e2e["one_sided_95_percent_upper"]["upper_percent"]
        throughput_upper_raw = throughput["one_sided_95_percent_upper"]["upper_percent"]
        e2e_upper = float(e2e_upper_raw) if e2e_upper_raw is not None else None
        throughput_upper = (
            float(throughput_upper_raw)
            if throughput_upper_raw is not None
            else None
        )
        sampling = [by_block[item]["sampling_validation"] for item in expected]
        interval_medians = [
            float(stream["actual_interval_ns"]["median"])
            for block_sampling in sampling
            for stream in block_sampling.get("streams", {}).values()
            if isinstance(stream.get("actual_interval_ns"), dict)
        ]
        modes[condition] = {
            "status": "complete",
            "valid_pairs": formal_rounds,
            "excluded_pairs": [],
            "e2e_latency_overhead": e2e,
            "output_token_throughput_degradation": throughput,
            "secondary_metrics": secondary,
            "operational_metrics_by_round": [
                by_block[item].get("operational_metrics") for item in expected
            ],
            "artifact_validation_by_round": [
                by_block[item].get("artifact_validation") for item in expected
            ],
            "e2e_median_percent": e2e_median,
            "e2e_one_sided_95_ci_upper_percent": e2e_upper,
            "throughput_degradation_median_percent": throughput_median,
            "throughput_one_sided_95_ci_upper_percent": throughput_upper,
            "sampling": sampling,
            "actual_interval_ns": summarize_distribution(
                interval_medians
            ).to_dict(),
            "met": e2e_median <= THRESHOLD_PERCENT and throughput_median <= THRESHOLD_PERCENT,
            "supported": (
                e2e_upper <= THRESHOLD_PERCENT
                and throughput_upper <= THRESHOLD_PERCENT
                if e2e_upper is not None and throughput_upper is not None
                else None
            ),
            "supported_unavailable_reason": (
                None
                if e2e_upper is not None and throughput_upper is not None
                else "confidence-interval support is not assessed for three repeats"
            ),
        }
    deferred = execution_plan.get("formal_block_postprocessing") == "deferred"
    representative_path = root / "representative_postprocess.json"
    representative_complete = (
        not deferred
        or (
            representative_path.is_file()
            and _read_json(representative_path).get("status") == "succeeded"
        )
    )
    complete = len(by_block) == expected_block_count and all(
        item.get("status") == "complete" for item in modes.values()
    ) and representative_complete
    all_supported = (
        all(item.get("supported") is True for item in modes.values())
        if complete and formal_rounds == DEFAULT_FORMAL_ROUNDS
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_root": str(root),
        "schedule_sha256": schedule["sha256"],
        "checkpoint_status": checkpoint.get("status"),
        "formal_blocks_completed": len(by_block),
        "formal_rounds": formal_rounds,
        "formal_blocks_expected": expected_block_count,
        "formal_block_postprocessing": execution_plan.get(
            "formal_block_postprocessing", "per_block"
        ),
        "representative_postprocessing_complete": representative_complete,
        "automatic_retries": 0,
        "complete": complete,
        "all_modes_met": complete and all(item.get("met") is True for item in modes.values()),
        "all_modes_supported": all_supported,
        "threshold_percent": THRESHOLD_PERCENT,
        "modes": modes,
        "limitations": [
            "One model, one GPU/NPU partition, one prompt, and concurrency one are measured.",
            (
                "These are fixed single-model, single-partition, single-prompt, "
                f"concurrency-one results from {formal_rounds} repeats; they are not "
                "a broader statistical generalization."
            ),
            "RBLN native-clock details remain separate when no canonical clock anchor exists.",
        ],
    }


def _summary_documents(report: dict[str, Any]) -> dict[str, object]:
    modes = report["modes"]
    complete_modes = {
        name: value
        for name, value in modes.items()
        if value.get("status") == "complete"
    }
    return {
        "paired_results.json": {
            "schema_version": SCHEMA_VERSION,
            "modes": {
                name: {
                    "e2e_latency_overhead": value["e2e_latency_overhead"]["pairs"],
                    "output_token_throughput_degradation": value[
                        "output_token_throughput_degradation"
                    ]["pairs"],
                    "secondary_metrics": {
                        metric: details["pairs"]
                        for metric, details in value["secondary_metrics"].items()
                    },
                }
                for name, value in complete_modes.items()
            },
        },
        "per_mode_statistics.json": {
            "schema_version": SCHEMA_VERSION,
            "modes": complete_modes,
        },
        "sampling_validation.json": {
            "schema_version": SCHEMA_VERSION,
            "modes": {
                name: value.get("sampling", []) for name, value in modes.items()
            },
        },
        "artifact_validation.json": {
            "schema_version": SCHEMA_VERSION,
            "modes": {
                name: value.get("artifact_validation_by_round", [])
                for name, value in modes.items()
            },
            "note": "Per-block artifact validations remain authoritative in raw/.",
        },
        "final_verdict.json": {
            "schema_version": SCHEMA_VERSION,
            "complete": report["complete"],
            "all_modes_met": report["all_modes_met"],
            "all_modes_supported": report["all_modes_supported"],
            "threshold_percent": report["threshold_percent"],
            "mode_verdicts": {
                name: {
                    "status": value.get("status"),
                    "met": value.get("met", False),
                    "supported": value.get("supported", False),
                }
                for name, value in modes.items()
            },
        },
        "limitations.json": {
            "schema_version": SCHEMA_VERSION,
            "limitations": report["limitations"],
        },
    }


def _render_markdown(report: dict[str, Any]) -> bytes:
    lines = [
        "# All-mode one-hertz profiler overhead",
        "",
        f"Completed blocks: {report['formal_blocks_completed']}/{report['formal_blocks_expected']}",
        "",
        "| Mode | Valid pairs | Actual interval | E2E median | E2E CI upper | Throughput degradation median | Throughput CI upper | met | supported |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, value in report["modes"].items():
        if value.get("status") != "complete":
            lines.append(
                f"| {name} | {value.get('valid_pairs', 0)} | n/a | n/a | "
                "n/a | n/a | n/a | false | false |"
            )
            continue
        e2e_upper = value["e2e_one_sided_95_ci_upper_percent"]
        throughput_upper = value[
            "throughput_one_sided_95_ci_upper_percent"
        ]
        lines.append(
            f"| {name} | {value['valid_pairs']} | "
            f"{value['actual_interval_ns']['median'] / 1_000_000:.6f} ms | "
            f"{value['e2e_median_percent']:.6f}% | "
            f"{f'{e2e_upper:.6f}%' if e2e_upper is not None else 'n/a'} | "
            f"{value['throughput_degradation_median_percent']:.6f}% | "
            f"{f'{throughput_upper:.6f}%' if throughput_upper is not None else 'n/a'} | "
            f"{str(value['met']).lower()} | {str(value['supported']).lower()} |"
        )
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {item}" for item in report["limitations"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _render_html(report: dict[str, Any]) -> bytes:
    rows = []
    for name, value in report["modes"].items():
        if value.get("status") != "complete":
            cells = (
                name,
                value.get("valid_pairs", 0),
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                False,
                False,
            )
        else:
            e2e_upper = value["e2e_one_sided_95_ci_upper_percent"]
            throughput_upper = value[
                "throughput_one_sided_95_ci_upper_percent"
            ]
            cells = (
                name,
                value["valid_pairs"],
                f"{value['actual_interval_ns']['median'] / 1_000_000:.6f} ms",
                f"{value['e2e_median_percent']:.6f}%",
                f"{e2e_upper:.6f}%" if e2e_upper is not None else "n/a",
                f"{value['throughput_degradation_median_percent']:.6f}%",
                (
                    f"{throughput_upper:.6f}%"
                    if throughput_upper is not None
                    else "n/a"
                ),
                value["met"],
                value["supported"],
            )
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in cells) + "</tr>")
    document = (
        "<!doctype html><meta charset=utf-8><title>All-mode 1Hz overhead</title>"
        "<h1>All-mode one-hertz profiler overhead</h1>"
        f"<p>Completed blocks: {report['formal_blocks_completed']}/{report['formal_blocks_expected']}</p>"
        "<table><thead><tr><th>Mode</th><th>Pairs</th><th>Actual interval</th>"
        "<th>E2E median</th><th>E2E CI upper</th>"
        "<th>Throughput degradation median</th><th>Throughput CI upper</th><th>met</th><th>supported</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return document.encode("utf-8")


def _artifact_manifest(root: Path) -> dict[str, object]:
    excluded = {"artifact_manifest.json", "artifact_manifest_validation.json"}
    files = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name not in excluded
    ]
    return {"schema_version": SCHEMA_VERSION, "files": files}


def generate_campaign_report(root: Path) -> dict[str, object]:
    report = build_campaign_report(root)
    report_json = _canonical(report)
    report_md = _render_markdown(report)
    report_html = _render_html(report)
    summaries = {
        name: _canonical(value)
        for name, value in _summary_documents(report).items()
    }
    # The second independent rendering must be byte-for-byte identical before publication.
    repeated_report = build_campaign_report(root)
    repeated_summaries = {
        name: _canonical(value)
        for name, value in _summary_documents(repeated_report).items()
    }
    if (
        report_json != _canonical(repeated_report)
        or report_md != _render_markdown(repeated_report)
        or report_html != _render_html(repeated_report)
        or summaries != repeated_summaries
    ):
        raise AllModeOverheadError("report regeneration was not deterministic")
    for name, data in (("report.json", report_json), ("report.md", report_md), ("report.html", report_html)):
        path = root / "report" / name
        _atomic_write(path, data)
    for name, data in summaries.items():
        _atomic_write(root / "summary" / name, data)
    manifest = _artifact_manifest(root)
    manifest_bytes = _canonical(manifest)
    _atomic_write(root / "manifest/artifact_manifest.json", manifest_bytes)
    current = _artifact_manifest(root)
    validation = {
        "schema_version": SCHEMA_VERSION,
        "valid": current == manifest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "deterministic_report_generation": True,
        "mismatches": [] if current == manifest else ["fresh manifest differs"],
    }
    _write_json(root / "manifest/artifact_manifest_validation.json", validation)
    if not validation["valid"]:
        raise AllModeOverheadError("campaign manifest fresh validation failed")
    return report


def run_all_mode_campaign(
    *,
    config_path: Path,
    campaign_root: Path,
    resume: bool = False,
    dry_run: bool = False,
    preflight_only: bool = False,
) -> dict[str, object]:
    config = load_campaign_config(config_path)
    if dry_run:
        return {
            "executes": False,
            "creates_output": False,
            "campaign_root": str(campaign_root),
            "config_sha256": config.sha256,
            "schedule": config.schedule,
            "automatic_retries": 0,
        }
    root = Path(campaign_root)
    if not root.is_absolute():
        raise ValueError("campaign_root must be absolute")
    schedule, checkpoint = (
        _resume_campaign(config, root) if resume else _initialize_campaign(config, root)
    )
    preflight = run_preflight(config)
    preflight_path = root / "preflight.json"
    if preflight_path.exists() and preflight_path.read_bytes() != _canonical(preflight):
        # Time-varying inventories are written separately on resume.
        _write_json(root / f"preflight-resume-{time.time_ns()}.json", preflight, exclusive=True)
    elif not preflight_path.exists():
        _write_json(preflight_path, preflight, exclusive=True)
    if not preflight["valid"]:
        checkpoint["status"] = "preflight_failed"
        checkpoint["failure_reasons"] = preflight["reasons"]
        _update_checkpoint(root, checkpoint)
        return {
            "status": "preflight_failed",
            "campaign_root": str(root),
            "hardware_blocks_started": 0,
            "automatic_retries": 0,
            "reasons": preflight["reasons"],
        }
    if preflight_only:
        checkpoint["status"] = "preflight_succeeded"
        _update_checkpoint(root, checkpoint)
        return {
            "status": "preflight_succeeded",
            "campaign_root": str(root),
            "hardware_blocks_started": 0,
            "automatic_retries": 0,
        }
    hybrid = config.load_hybrid()
    current_before = {
        "model": _tree_fingerprint(hybrid.model_path),
        "cache": _tree_fingerprint(hybrid.rbln_cache_path),
    }
    before_path = root / "model_cache_fingerprint_before.json"
    if before_path.is_file():
        model_cache_before = _read_json(before_path)
        if model_cache_before != current_before:
            raise AllModeOverheadError(
                "model or cache differs from the campaign's initial fingerprint"
            )
    else:
        model_cache_before = current_before
        _write_json(before_path, model_cache_before, exclusive=True)
    _write_json_once_or_verify(
        root / "environment_versions.json",
        {
            "imports": preflight["imports"],
            "tools": {
                "nsys": preflight["environment"]["nsys"],
                "trace_processor": preflight["environment"]["trace_processor"],
            },
        },
    )
    _write_json_once_or_verify(
        root / "repository_fingerprints.json", preflight["repositories"]
    )
    completed = set(checkpoint.get("completed_blocks", []))
    checkpoint["status"] = "running"
    _update_checkpoint(root, checkpoint)
    for block in schedule["blocks"]:
        block_id = str(block["block_id"])
        if block_id in completed:
            continue
        checkpoint["current_block"] = block_id
        _update_checkpoint(root, checkpoint)
        try:
            _run_one_block(config=config, root=root, block=block)
        except KeyboardInterrupt as error:
            failure_path = _record_block_failure(
                config=config,
                root=root,
                block=block,
                error=error,
                interrupted=True,
            )
            checkpoint["status"] = "interrupted"
            checkpoint["failed_block"] = block_id
            checkpoint["failure_evidence"] = (
                failure_path.relative_to(root).as_posix()
                if failure_path is not None
                else None
            )
            try:
                final_fingerprint = _record_model_cache_after(
                    root, hybrid, before=model_cache_before
                )
                checkpoint["model_cache_unchanged"] = final_fingerprint[
                    "unchanged"
                ]
            except Exception as diagnostic_error:
                checkpoint["model_cache_after_error"] = (
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                )
            checkpoint["current_block"] = None
            _update_checkpoint(root, checkpoint)
            raise
        except Exception as error:
            failure_path = _record_block_failure(
                config=config,
                root=root,
                block=block,
                error=error,
                interrupted=False,
            )
            checkpoint["status"] = "failed"
            checkpoint["failed_block"] = block_id
            checkpoint["failure_summary"] = f"{type(error).__name__}: {error}"
            checkpoint["failure_evidence"] = (
                failure_path.relative_to(root).as_posix()
                if failure_path is not None
                else None
            )
            try:
                final_fingerprint = _record_model_cache_after(
                    root, hybrid, before=model_cache_before
                )
                checkpoint["model_cache_unchanged"] = final_fingerprint[
                    "unchanged"
                ]
            except Exception as diagnostic_error:
                checkpoint["model_cache_after_error"] = (
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                )
            checkpoint["current_block"] = None
            _update_checkpoint(root, checkpoint)
            return {
                "status": "failed",
                "campaign_root": str(root),
                "failed_block": block_id,
                "automatic_retries": 0,
                "error": checkpoint["failure_summary"],
            }
        completed.add(block_id)
        checkpoint["completed_blocks"] = sorted(completed)
        checkpoint["current_block"] = None
        _update_checkpoint(root, checkpoint)
    final_fingerprint = _record_model_cache_after(
        root, hybrid, before=model_cache_before
    )
    if final_fingerprint["unchanged"] is not True:
        checkpoint["status"] = "failed"
        checkpoint["failure_summary"] = "model or cache fingerprint changed"
        _update_checkpoint(root, checkpoint)
        raise AllModeOverheadError("model or cache fingerprint changed")
    checkpoint["status"] = "collection_succeeded"
    _update_checkpoint(root, checkpoint)
    try:
        postprocess = postprocess_representative_runs(
            config=config,
            root=root,
            schedule=schedule,
        )
    except Exception as error:
        evidence = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "hardware_rerun": False,
            "formal_collection_blocks_preserved": len(completed),
            "automatic_retries": 0,
        }
        failure_path = root / "representative_postprocess_failure.json"
        _write_json_once_or_verify(failure_path, evidence)
        checkpoint["status"] = "postprocess_failed"
        checkpoint["failure_summary"] = f"{type(error).__name__}: {error}"
        checkpoint["failure_evidence"] = failure_path.relative_to(root).as_posix()
        checkpoint["current_block"] = None
        _update_checkpoint(root, checkpoint)
        generate_campaign_report(root)
        return {
            "status": "postprocess_failed",
            "campaign_root": str(root),
            "formal_blocks": len(completed),
            "hardware_rerun": False,
            "automatic_retries": 0,
            "error": checkpoint["failure_summary"],
        }
    checkpoint["representative_postprocess"] = (
        "representative_postprocess.json"
    )
    checkpoint["status"] = "succeeded"
    checkpoint["current_block"] = None
    checkpoint["failed_block"] = None
    checkpoint.pop("failure_summary", None)
    checkpoint.pop("failure_evidence", None)
    _update_checkpoint(root, checkpoint)
    report = generate_campaign_report(root)
    return {
        "status": "succeeded",
        "campaign_root": str(root),
        "formal_blocks": len(completed),
        "automatic_retries": 0,
        "representative_postprocess": postprocess,
        "all_modes_met": report["all_modes_met"],
        "all_modes_supported": report["all_modes_supported"],
    }


def campaign_status(root: Path) -> dict[str, object]:
    checkpoint = _read_json(Path(root) / "checkpoint.json")
    return {
        "campaign_root": str(root),
        "status": checkpoint.get("status"),
        "completed_blocks": checkpoint.get("completed_blocks", []),
        "completed_count": len(checkpoint.get("completed_blocks", [])),
        "current_block": checkpoint.get("current_block"),
        "failed_block": checkpoint.get("failed_block"),
        "automatic_retries": checkpoint.get("automatic_retries"),
    }


__all__ = [
    "AllModeOverheadError",
    "CampaignConfig",
    "analyze_sampling_stream",
    "build_all_mode_schedule",
    "build_campaign_report",
    "campaign_status",
    "estimate_disk_budget",
    "generate_campaign_report",
    "load_campaign_config",
    "postprocess_representative_runs",
    "recover_failed_postprocess",
    "run_all_mode_campaign",
    "run_preflight",
    "validate_mode_artifacts",
    "validate_sampling",
]
