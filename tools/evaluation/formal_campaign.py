"""Fail-closed execution of the fixed 21-block OFAT campaign."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Any

from perfetto_hetero_profiler.collectors.gpu import GpuTelemetryCollector
from perfetto_hetero_profiler.collectors.npu import NpuTelemetryCollector
from perfetto_hetero_profiler.collectors.process import ManagedProcess
from perfetto_hetero_profiler.collectors.command import CommandSpec
from perfetto_hetero_profiler.collectors.system import SystemTelemetryCollector
from perfetto_hetero_profiler.hybrid.runner import HybridRunner, _wait_http
from perfetto_hetero_profiler.hybrid.runner_config import (
    HybridRunnerConfig,
    ProfilerOutputConfig,
    ServerConfig,
    WorkloadConfig,
)
from perfetto_hetero_profiler.schema import RunStatus, write_jsonl
from perfetto_hetero_profiler.support.json_io import write_pretty_json

from .concurrent_wave import (
    BlockSpec,
    ConcurrentWaveClient,
    ConcurrentWaveError,
    load_matrix_blocks,
    run_block,
    write_block_artifacts,
)
from .formal_client import FormalStreamingClient


PROMPT_SHA256 = "928a5d427df9460d7b5f69178206f8d7bfd79c56cb4e54f21976456844a80c1f"
PROFILER_BASE_HEAD = "e24a75441d4329fe8b2098ee165e65fe8cc6f76a"
VLLM_RBLN_HEAD = "885cde4fc90073c96c159d13ac3db376869a1928"
_CACHE_KEYS = frozenset({"m05-b1", "m15-b1", "m15-b2", "m15-b4", "m3-b1"})
_ENV_ALLOWLIST = (
    "PATH", "LANG", "LC_ALL", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
    "RBLN_DEVICES", "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE", "VLLM_RBLN_USE_VLLM_MODEL",
    "VLLM_RBLN_COMPILE_MODEL", "VLLM_CACHE_ROOT",
)


class FormalCampaignError(RuntimeError):
    """The fixed campaign contract or one block failed."""


@dataclass(frozen=True, slots=True)
class Model:
    key: str
    snapshot: Path
    served_name: str
    revision: str
    snapshot_fingerprint: str


@dataclass(frozen=True, slots=True)
class Cache:
    key: str
    path: Path
    fingerprint: str
    max_num_seqs: int


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    path: Path
    matrix: Path
    prompt_file: Path
    profiler_root: Path
    vllm_rbln_root: Path
    profiler_python: Path
    tokenizer_python: Path
    gpu_vllm: Path
    npu_vllm_launcher: Path
    trace_processor: Path
    nsys: Path
    models: dict[str, Model]
    caches: dict[str, Cache]
    ports: dict[str, int]
    startup_sec: float
    request_sec: float
    shutdown_sec: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        ) + "\n"
    ).encode("utf-8")


def _snapshot_fingerprint(root: Path) -> str:
    selected = sorted(
        path for path in root.iterdir()
        if path.name in {"config.json", "generation_config.json", "model.safetensors.index.json"}
        or path.name.startswith("tokenizer")
        or path.suffix in {".safetensors", ".txt", ".model"}
    )
    rows = []
    for path in selected:
        resolved = path.resolve(strict=True)
        rows.append({
            "relative_path": path.name, "size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved), "is_symlink": path.is_symlink(),
            "symlink_target": path.readlink().as_posix() if path.is_symlink() else None,
        })
    return hashlib.sha256(_canonical(rows)).hexdigest()


def _cache_fingerprint(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            info = path.stat()
            rows.append({
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                "mode": info.st_mode & 0o7777, "sha256": _sha256(path),
            })
    return hashlib.sha256(_canonical(rows)).hexdigest()


def _path(value: object, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise FormalCampaignError(f"{label} must be absolute")
    return path


def load_config(path: Path) -> CampaignConfig:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "1.0" or value.get("automatic_retries") != 0:
        raise FormalCampaignError("config must be schema 1.0 with automatic_retries=0")
    protocol = value.get("protocol", {})
    expected = {
        "input_tokens": 256, "output_tokens": 32, "temperature": 0,
        "streaming": True, "ignore_eos": True, "max_model_len": 512,
        "block_size": 512, "sample_interval_ms": 1000,
        "gpu_memory_utilization": 0.2, "npu_memory_utilization": 0.92,
        "enable_prefix_caching": False,
    }
    if any(protocol.get(key) != item for key, item in expected.items()):
        raise FormalCampaignError("formal protocol differs from the fixed contract")
    models = {
        item["key"]: Model(
            item["key"], _path(item["snapshot"], "model snapshot"),
            item["served_name"], item["revision"], item["snapshot_fingerprint"],
        )
        for item in value.get("models", [])
    }
    caches = {
        item["key"]: Cache(
            item["key"], _path(item["path"], "cache"),
            item["fingerprint"], int(item["max_num_seqs"]),
        )
        for item in value.get("caches", [])
    }
    if set(models) != {"m05", "m15", "m3"} or set(caches) != _CACHE_KEYS:
        raise FormalCampaignError("config must map three models and five caches")
    paths = value["paths"]
    ports = {key: int(item) for key, item in value["ports"].items()}
    if set(ports) != {"gpu_http", "npu_http", "proxy_http", "gpu_nixl", "npu_nixl"}:
        raise FormalCampaignError("five fixed ports are required")
    if len(set(ports.values())) != 5:
        raise FormalCampaignError("formal ports must be distinct")
    timeouts = value["timeouts"]
    config = CampaignConfig(
        path.resolve(), _path(paths["matrix"], "matrix"),
        _path(paths["prompt_file"], "prompt_file"),
        _path(paths["profiler_root"], "profiler_root"),
        _path(paths["vllm_rbln_root"], "vllm_rbln_root"),
        _path(paths["profiler_python"], "profiler_python"),
        _path(paths["tokenizer_python"], "tokenizer_python"),
        _path(paths["gpu_vllm"], "gpu_vllm"),
        _path(paths["npu_vllm_launcher"], "npu_vllm_launcher"),
        _path(paths["trace_processor"], "trace_processor"),
        _path(paths["nsys"], "nsys"), models, caches, ports,
        float(timeouts["startup_sec"]), float(timeouts["request_sec"]),
        float(timeouts["shutdown_sec"]),
    )
    blocks = load_matrix_blocks(config.matrix)
    for block in blocks:
        cache = caches[block.condition_id.rsplit("-", 1)[0]]
        if cache.max_num_seqs != block.concurrency:
            raise FormalCampaignError(f"cache batch mismatch for {block.condition_id}")
    return config


def _git_state(root: Path) -> tuple[str, str]:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"], text=True
    )
    return head, status


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False, capture_output=True,
    )
    return result.returncode == 0


def _token_count(config: CampaignConfig, model: Model) -> int:
    program = (
        "from pathlib import Path\n"
        "from transformers import AutoTokenizer\n"
        "import sys\n"
        "prompt=Path(sys.argv[1]).read_text(encoding='utf-8')\n"
        "tokenizer=AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True)\n"
        "print(len(tokenizer(prompt, add_special_tokens=False).input_ids))\n"
    )
    result = subprocess.run(
        [str(config.tokenizer_python), "-c", program,
         str(config.prompt_file), str(model.snapshot)],
        check=False, capture_output=True, text=True, timeout=120,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    if result.returncode:
        raise FormalCampaignError(f"tokenizer preflight failed: {model.key}")
    try:
        return int(result.stdout.strip())
    except ValueError as error:
        raise FormalCampaignError(f"invalid tokenizer result: {model.key}") from error


def _import_version(python: Path, root: Path) -> tuple[str, str]:
    program = "import vllm, vllm_rbln; print(vllm.__version__); print(vllm_rbln.__file__)"
    result = subprocess.run(
        [str(python), "-c", program], check=False, capture_output=True, text=True,
        timeout=60,
        env={
            "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(root),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        },
    )
    lines = result.stdout.strip().splitlines()
    if result.returncode or len(lines) != 2:
        raise FormalCampaignError(f"vLLM/vllm-rbln import preflight failed: {python}")
    return lines[0], str(Path(lines[1]).resolve())


def preflight(config: CampaignConfig, *, query_devices: bool = True) -> dict[str, object]:
    required = (
        config.matrix, config.prompt_file, config.profiler_python,
        config.tokenizer_python, config.gpu_vllm, config.npu_vllm_launcher,
        config.trace_processor, config.nsys,
        *(model.snapshot for model in config.models.values()),
        *(cache.path for cache in config.caches.values()),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FormalCampaignError(f"required path is missing: {missing[0]}")
    profiler_head, profiler_status = _git_state(config.profiler_root)
    vllm_head, vllm_status = _git_state(config.vllm_rbln_root)
    if not _is_ancestor(config.profiler_root, PROFILER_BASE_HEAD, profiler_head):
        raise FormalCampaignError("profiler does not descend from the qualified base HEAD")
    if vllm_head != VLLM_RBLN_HEAD:
        raise FormalCampaignError("vllm-rbln HEAD mismatch")
    if profiler_status or vllm_status:
        raise FormalCampaignError("formal source worktree is not clean")
    if _sha256(config.prompt_file) != PROMPT_SHA256:
        raise FormalCampaignError("prompt fingerprint mismatch")
    imports = {}
    for role, python in (
        ("gpu", config.gpu_vllm.parent / "python"),
        ("npu", config.tokenizer_python),
    ):
        version, import_path = _import_version(python, config.vllm_rbln_root)
        expected_import = str((config.vllm_rbln_root / "vllm_rbln/__init__.py").resolve())
        if not version.startswith("0.22.") or import_path != expected_import:
            raise FormalCampaignError(f"{role} vLLM/import path mismatch")
        imports[role] = {"vllm_version": version, "vllm_rbln": import_path}
    for model in config.models.values():
        if model.snapshot.name != model.revision:
            raise FormalCampaignError(f"snapshot revision mismatch: {model.key}")
        if _snapshot_fingerprint(model.snapshot) != model.snapshot_fingerprint:
            raise FormalCampaignError(f"snapshot fingerprint mismatch: {model.key}")
        if _token_count(config, model) != 256:
            raise FormalCampaignError(f"prompt token count mismatch: {model.key}")
    for cache in config.caches.values():
        if _cache_fingerprint(cache.path) != cache.fingerprint:
            raise FormalCampaignError(f"cache fingerprint mismatch: {cache.key}")
    busy = [name for name, port in config.ports.items() if not _port_free(port)]
    if busy:
        raise FormalCampaignError(f"formal port is busy: {busy[0]}")
    process_check = subprocess.run(
        ["pgrep", "-af", "([v]llm|[E]ngineCore|[A]PIServer)"],
        check=False, capture_output=True, text=True,
    )
    if process_check.returncode not in {0, 1}:
        raise FormalCampaignError("process preflight command failed")
    if process_check.returncode == 0 and process_check.stdout.strip():
        raise FormalCampaignError("vLLM/EngineCore/APIServer process is already running")
    devices: dict[str, object] = {"queried": False}
    if query_devices:
        gpu = subprocess.run(
            ["nvidia-smi", "-L"], text=True, capture_output=True,
            check=False, timeout=30,
        )
        npu = subprocess.run(
            ["rbln-smi", "--json"], text=True, capture_output=True,
            check=False, timeout=30,
        )
        if gpu.returncode or npu.returncode:
            raise FormalCampaignError("GPU/NPU preflight command failed")
        parsed = json.loads(npu.stdout)
        if parsed.get("contexts"):
            raise FormalCampaignError("NPU contexts are not empty")
        devices = {"queried": True, "gpu": gpu.stdout.strip(), "npu_contexts": []}
    return {
        "valid": True, "executes": False, "creates_output": False,
        "profiler_head": profiler_head, "vllm_rbln_head": vllm_head,
        "block_count": 21, "cache_count": 5, "imports": imports,
        "devices": devices,
    }


def _port_free(port: int) -> bool:
    with socket.socket() as stream:
        try:
            stream.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _process_group_alive(process_group_id: int | None) -> bool:
    if process_group_id is None:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _runtime_postflight(config: CampaignConfig) -> dict[str, object]:
    try:
        busy_ports = [name for name, port in config.ports.items() if not _port_free(port)]
        processes = subprocess.run(
            ["pgrep", "-af", "([v]llm|[E]ngineCore|[A]PIServer)"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        process_rows = (
            processes.stdout.strip().splitlines() if processes.returncode == 0 else []
        )
        npu = subprocess.run(
            ["rbln-smi", "--json"], check=False, capture_output=True,
            text=True, timeout=30,
        )
        contexts = json.loads(npu.stdout).get("contexts") if not npu.returncode else None
    except Exception as error:
        return {"valid": False, "error": type(error).__name__}
    valid = (
        not busy_ports and not process_rows and contexts == []
        and processes.returncode in {0, 1}
    )
    return {
        "valid": valid, "busy_ports": busy_ports,
        "remaining_processes": process_rows, "npu_contexts": contexts,
    }


def _compile_evidence(stdout_path: Path, stderr_path: Path, concurrency: int) -> dict[str, object]:
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    expected = 5 if concurrency == 1 else 10
    markers = [int(item) for item in re.findall(r"Compile\(#(\d+)\)", stderr)]
    expected_markers = list(range(2, expected + 2))
    miss = "PersistentRBLNCacheMissError" in stdout + stderr or "Persistent RBLN model cache miss" in stdout + stderr
    valid = (
        markers == expected_markers and not miss
        and stdout.count("cache_policy=persistent_cache_hit_required") == 2
        and stdout.count("cache_policy=non_persistent_compile_allowed") == expected
        and stdout.count("Persistent RBLN cache-hit guard enabled: env=1 active=true mode=non-strict") == 2
    )
    return {
        "valid": valid, "persistent_compile_count": sum(item in {0, 1} for item in markers),
        "sampler_compile_ids": markers, "expected_sampler_compile_ids": expected_markers,
        "cache_miss_count": int(miss), "require_cache_hit": True,
        "compile_only": "unset", "strict_mode": "unset",
    }


class FormalHybridRunner(HybridRunner):
    """Use the verified non-strict guard and actual sampler warm-up contract."""

    def _compile_gate(self) -> None:
        raw = self.layout.coordinator / "raw"
        evidence = _compile_evidence(
            raw / "decode.stdout.log", raw / "decode.stderr.log", self.config.max_num_seqs
        )
        write_pretty_json(self.layout.coordinator / "compile_gate.json", evidence)
        if not evidence["valid"]:
            raise FormalCampaignError("NPU persistent-cache compile gate failed")


class _Telemetry:
    def __init__(self, block: BlockSpec, topology: str, pid_provider) -> None:
        device = (
            GpuTelemetryCollector(
                run_id=block.condition_id, host_id="localhost",
                clock_domain_id="host-monotonic", sample_interval_ms=1000,
                known_gpu_indices=(0,),
            )
            if topology == "gpu"
            else NpuTelemetryCollector(
                run_id=block.condition_id, host_id="localhost",
                clock_domain_id="host-monotonic", sample_interval_ms=1000,
                known_npu_indices=(0,),
            )
        )
        self.collectors = (
            device,
            SystemTelemetryCollector(
                run_id=block.condition_id, host_id="localhost",
                clock_domain_id="host-monotonic", pid_provider=pid_provider,
            ),
        )
        self.rows: list[Any] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name="formal-telemetry", daemon=True)
        self.started = False
        self.started_collectors = 0

    def start(self) -> None:
        try:
            for collector in self.collectors:
                collector.prepare(); collector.start()
                self.started_collectors += 1
            self.started = True
            self.sample("baseline")
            self.thread.start()
        except Exception:
            self.stop()
            raise

    def sample(self, role: str) -> None:
        with self.lock:
            for collector in self.collectors:
                for row in collector.sample():
                    self.rows.append(replace(row, attributes={**row.attributes, "telemetry.sample_role": role}))

    def stop(self) -> None:
        if not self.started and not self.started_collectors:
            return
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.errors.append("telemetry thread did not stop")
        if self.started:
            try:
                self.sample("final")
            except Exception as error:
                self.errors.append(f"final telemetry: {type(error).__name__}")
        for collector in reversed(self.collectors[:self.started_collectors]):
            try:
                collector.stop(); collector.finalize()
            except Exception as error:
                self.errors.append(f"telemetry cleanup: {type(error).__name__}")
        self.started = False
        self.started_collectors = 0

    def _loop(self) -> None:
        while not self.stop_event.wait(1.0):
            try:
                self.sample("background")
            except Exception as error:
                self.errors.append(f"telemetry sample: {type(error).__name__}")
                self.stop_event.set()


def _model_cache(config: CampaignConfig, block: BlockSpec) -> tuple[Model, Cache]:
    return config.models[block.condition_id.split("-", 1)[0]], config.caches[block.condition_id.rsplit("-", 1)[0]]


def _server_args(config: CampaignConfig, block: BlockSpec, topology: str) -> tuple[str, ...]:
    model, _ = _model_cache(config, block)
    executable = config.gpu_vllm if topology == "gpu" else config.npu_vllm_launcher
    port = config.ports[f"{topology}_http"]
    argv = [
        str(executable), "serve", str(model.snapshot), "--host", "127.0.0.1",
        "--port", str(port), "--block-size", "512", "--tensor-parallel-size", "1",
        "--served-model-name", model.served_name, "--max-model-len", "512",
        "--max-num-batched-tokens", "512", "--max-num-seqs", str(block.concurrency),
        "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching", "--seed", "20260909",
    ]
    if topology == "gpu":
        argv.extend(("--enforce-eager", "--gpu-memory-utilization", "0.2"))
    else:
        argv.extend(("--gpu-memory-utilization", "0.92"))
    return tuple(argv)


def _standalone_block(config: CampaignConfig, block: BlockSpec, output: Path) -> dict[str, object]:
    model, cache = _model_cache(config, block)
    output.mkdir(parents=True)
    raw = output / "raw"; raw.mkdir()
    env = {
        "PYTHONPATH": str(config.vllm_rbln_root), "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    if block.topology == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        env.update({
            "RBLN_DEVICES": "0", "VLLM_RBLN_USE_VLLM_MODEL": "1",
            "VLLM_RBLN_COMPILE_MODEL": "1", "VLLM_CACHE_ROOT": str(cache.path),
        })
    write_pretty_json(output / "execution_plan.json", {
        "argv": list(_server_args(config, block, block.topology)),
        "environment": env, "cwd": str(config.vllm_rbln_root),
        "require_cache_hit": block.topology == "npu", "compile_only": "unset",
    })
    before = _cache_fingerprint(cache.path)
    process = ManagedProcess(
        CommandSpec(
            argv=_server_args(config, block, block.topology), cwd=config.vllm_rbln_root,
            env_overrides=env, env_allowlist=_ENV_ALLOWLIST,
            terminate_grace_sec=config.shutdown_sec,
        ), raw / "server.stdout.log", raw / "server.stderr.log",
    )
    telemetry = _Telemetry(block, block.topology, lambda: process.process.pid if process.process else None)
    client: ConcurrentWaveClient | None = None
    errors: list[str] = []
    shutdown: dict[str, object] = {}
    try:
        process.start()
        _wait_http(
            f"http://127.0.0.1:{config.ports[block.topology + '_http']}",
            "/v1/models", process, config.startup_sec,
        )
        if block.topology == "npu":
            evidence = _compile_evidence(raw / "server.stdout.log", raw / "server.stderr.log", block.concurrency)
            write_pretty_json(output / "compile_gate.json", evidence)
            if not evidence["valid"]:
                raise FormalCampaignError("NPU startup compile gate failed")
        telemetry.start()
        client = ConcurrentWaveClient(
            f"http://127.0.0.1:{config.ports[block.topology + '_http']}",
            timeout_sec=config.request_sec, block=block, client_factory=FormalStreamingClient,
        )
        prompt = config.prompt_file.read_text(encoding="utf-8")
        prefix = f"r{block.round_index:02d}-{block.condition_id}"
        for phase, count in (("warmup", block.warmup_requests), ("measured", block.measured_requests)):
            for index in range(count):
                client.complete(
                    model=model.served_name, request_id=f"{prefix}-{phase}-{index:03d}",
                    prompt=prompt, max_output_tokens=32, temperature=0, stream=True,
                )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        if client is not None:
            try: client.close()
            except Exception as error: errors.append(f"client cleanup: {type(error).__name__}")
        telemetry.stop()
        errors.extend(telemetry.errors)
        if process.process is not None:
            stopped = process.stop_leader_first()
            shutdown = {"return_code": stopped.return_code, "killed": stopped.killed}
            if stopped.killed: errors.append("server required SIGKILL")
    write_jsonl(output / "telemetry.jsonl", telemetry.rows)
    write_pretty_json(output / "shutdown.json", shutdown)
    if _process_group_alive(process.process_group_id):
        errors.append("server process group remains after cleanup")
    if not _port_free(config.ports[block.topology + "_http"]):
        errors.append("server port remains occupied after cleanup")
    if block.topology == "npu":
        try:
            evidence = _compile_evidence(
                raw / "server.stdout.log", raw / "server.stderr.log", block.concurrency
            )
            write_pretty_json(output / "compile_gate.postflight.json", evidence)
            if not evidence["valid"]:
                errors.append("NPU postflight compile gate failed")
        except Exception as error:
            errors.append(f"NPU postflight evidence: {type(error).__name__}")
    if before != _cache_fingerprint(cache.path):
        errors.append("persistent cache fingerprint changed")
    requests = client.request_rows if client is not None else []
    waves = client.wave_rows if client is not None else []
    return write_block_artifacts(
        output, block, requests, waves,
        runner_status=RunStatus.SUCCEEDED.value if not errors else RunStatus.FAILED.value,
        runner_errors=tuple(errors),
    )


def _hybrid_config(config: CampaignConfig, block: BlockSpec) -> HybridRunnerConfig:
    model, cache = _model_cache(config, block)
    common = ("--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16", "--max-num-batched-tokens", "512", "--enable-chunked-prefill", "--no-enable-prefix-caching", "--seed", "20260909")
    return HybridRunnerConfig(
        config_path=config.path, model_path=model.snapshot, served_model_name=model.served_name,
        rbln_cache_path=cache.path,
        prefill=ServerConfig(config.gpu_vllm, config.vllm_rbln_root, config.vllm_rbln_root, "127.0.0.1", config.ports["gpu_http"], config.ports["gpu_nixl"], common),
        decode=ServerConfig(config.npu_vllm_launcher, config.vllm_rbln_root, config.vllm_rbln_root, "127.0.0.1", config.ports["npu_http"], config.ports["npu_nixl"], (*common, "--gpu-memory-utilization", "0.92")),
        proxy_python=config.profiler_python,
        proxy_entry_point="perfetto_hetero_profiler.hybrid.proxy", proxy_host="127.0.0.1",
        proxy_port=config.ports["proxy_http"],
        workload=WorkloadConfig(None, config.prompt_file, block.warmup_requests, block.measured_requests, 32, 0, True),
        prefill_connector={"kv_connector": "NixlConnector", "kv_role": "kv_producer", "kv_buffer_device": "cuda", "kv_load_failure_policy": "fail", "kv_connector_extra_config": {"kv_recompute_threshold": 0}},
        decode_connector={"kv_connector": "RblnNixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cpu", "kv_load_failure_policy": "fail", "kv_connector_extra_config": {"kv_recompute_threshold": 0, "remote_nixl_memory_type": "VRAM", "rbln_external_kv_format": "host_visible_hnd_to_runtime_private", "rbln_external_kv_source_dtype": "bfloat16"}},
        profiler_outputs=ProfilerOutputConfig(Path("raw/gpu/torch"), Path("raw/gpu/nsys/prefill"), Path("raw/npu/torch"), Path("raw/npu/rbln")),
        max_model_len=512, block_size=512, max_num_seqs=block.concurrency,
        gpu_memory_utilization=0.2, gpu_indices=(0,), npu_indices=(0,),
        sample_interval_ms=1000, startup_timeout_sec=config.startup_sec,
        request_timeout_sec=config.request_sec, shutdown_timeout_sec=config.shutdown_sec,
        trace_processor_path=config.trace_processor, nsys_executable=config.nsys, offline=True,
    )


def plan(config: CampaignConfig, campaign_root: Path) -> dict[str, object]:
    blocks = load_matrix_blocks(config.matrix)
    return {
        "executes": False, "creates_output": False, "campaign_root": str(campaign_root),
        "block_count": len(blocks), "automatic_retries": 0,
        "blocks": [{"position": index, **block.to_dict()} for index, block in enumerate(blocks, 1)],
    }


def run_campaign(config: CampaignConfig, campaign_root: Path, *, resume: bool = False) -> dict[str, object]:
    campaign_root = Path(campaign_root)
    if not campaign_root.is_absolute():
        raise FormalCampaignError("campaign root must be absolute")
    preflight_result = preflight(config)
    blocks = load_matrix_blocks(config.matrix)
    if campaign_root.exists() and not resume:
        raise FormalCampaignError("campaign root already exists")
    if not campaign_root.exists():
        campaign_root.mkdir(parents=True)
        shutil.copyfile(config.path, campaign_root / "campaign-config.json")
        write_pretty_json(campaign_root / "plan.json", plan(config, campaign_root))
        write_pretty_json(campaign_root / "preflight.json", preflight_result)
    else:
        snapshot = campaign_root / "campaign-config.json"
        saved_preflight = campaign_root / "preflight.json"
        if (
            not snapshot.is_file() or _sha256(snapshot) != _sha256(config.path)
            or not saved_preflight.is_file()
        ):
            raise FormalCampaignError("campaign snapshot differs from the fixed config")
        previous = json.loads(saved_preflight.read_text(encoding="utf-8"))
        if any(
            previous.get(key) != preflight_result.get(key)
            for key in ("profiler_head", "vllm_rbln_head", "imports")
        ):
            raise FormalCampaignError("source changed since the campaign began")
    completed = 0
    for position, block in enumerate(blocks, 1):
        root = campaign_root / "blocks" / f"{position:02d}-r{block.round_index:02d}-{block.condition_id}"
        result_path = root / "block-result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "succeeded":
                raise FormalCampaignError("failed block exists; automatic retry is forbidden")
            completed += 1
            continue
        if root.exists():
            raise FormalCampaignError("incomplete block exists; automatic retry is forbidden")
        root.mkdir(parents=True)
        write_pretty_json(root / "block.json", {"position": position, **block.to_dict()})
        try:
            if block.topology == "hybrid":
                cache_before = _cache_fingerprint(_model_cache(config, block)[1].path)
                summary = run_block(
                    matrix_path=config.matrix, hybrid_config_path=config.path,
                    condition_id=block.condition_id, round_index=block.round_index,
                    block_root=root / "evaluation", request_client_factory=FormalStreamingClient,
                    runner_factory=FormalHybridRunner, config_override=_hybrid_config(config, block),
                )
                if cache_before != _cache_fingerprint(_model_cache(config, block)[1].path):
                    raise FormalCampaignError("persistent cache fingerprint changed")
                raw = root / "evaluation/runner" / f"r{block.round_index:02d}-{block.condition_id}/coordinator/raw"
                evidence = _compile_evidence(raw / "decode.stdout.log", raw / "decode.stderr.log", block.concurrency)
                write_pretty_json(root / "compile-gate.postflight.json", evidence)
                if not evidence["valid"]:
                    raise FormalCampaignError("hybrid postflight compile gate failed")
            else:
                summary = _standalone_block(config, block, root / "evaluation")
            if summary.get("status") != "succeeded":
                raise FormalCampaignError("block did not succeed")
        except Exception:
            postflight = _runtime_postflight(config)
            write_pretty_json(root / "runtime-postflight.json", postflight)
            write_pretty_json(campaign_root / "status.json", {"status": "failed", "completed_blocks": completed, "failed_position": position, "automatic_retries": 0})
            raise
        postflight = _runtime_postflight(config)
        write_pretty_json(root / "runtime-postflight.json", postflight)
        if not postflight["valid"]:
            write_pretty_json(campaign_root / "status.json", {"status": "failed", "completed_blocks": completed, "failed_position": position, "automatic_retries": 0})
            raise FormalCampaignError("runtime postflight cleanup failed")
        completed += 1
        write_pretty_json(result_path, {
            "status": "succeeded", "position": position,
            "condition_id": block.condition_id, "round": block.round_index,
        })
        write_pretty_json(campaign_root / "status.json", {"status": "running", "completed_blocks": completed, "next_position": completed + 1 if completed < 21 else None, "automatic_retries": 0})
    result = {"status": "succeeded", "completed_blocks": completed, "automatic_retries": 0}
    write_pretty_json(campaign_root / "status.json", result)
    return result
