"""Read-only experiment environment snapshots and idle checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any

from ..hybrid.runner_config import HybridRunnerConfig


class EnvironmentNotIdleError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _command(argv: tuple[str, ...], timeout: float = 15.0) -> dict[str, object]:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return {
            "argv": list(argv),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": list(argv), "error": f"{type(error).__name__}: {error}"}


def _tree_stat_fingerprint(root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "root_name": root.name,
        "entries": rows,
        "sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
    }


def _proc_text(path: str) -> dict[str, object]:
    try:
        return {"path": path, "content": Path(path).read_text(encoding="utf-8")}
    except (OSError, UnicodeError) as error:
        return {"path": path, "error": f"{type(error).__name__}: {error}"}


def _port_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def capture_environment(config: HybridRunnerConfig, *, stage: str) -> dict[str, object]:
    wall_ns = time.time_ns()
    monotonic_ns = time.monotonic_ns()
    ports = {
        str(port): _port_free(host, port)
        for host, port in (
            (config.prefill.host, config.prefill.http_port),
            (config.decode.host, config.decode.http_port),
            (config.proxy_host, config.proxy_port),
            (config.prefill.host, config.prefill.nixl_port),
            (config.decode.host, config.decode.nixl_port),
        )
    }
    snapshot: dict[str, object] = {
        "stage": stage,
        "wall_clock_unix_ns": wall_ns,
        "monotonic_ns": monotonic_ns,
        "anchor_offset_ns": wall_ns - monotonic_ns,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()),
        "cpu_info": _proc_text("/proc/cpuinfo"),
        "system_memory": _proc_text("/proc/meminfo"),
        "ports_free": ports,
        "model_fingerprint": _tree_stat_fingerprint(config.model_path),
        "cache_fingerprint": _tree_stat_fingerprint(config.rbln_cache_path),
        "gpu": _command((
            "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        )),
        "gpu_processes": _command((
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )),
        "npu": _command(("rbln-smi", "--json")),
        "processes": _command(("ps", "-eo", "pid,ppid,pgid,user,state,cmd")),
        "trace_processor": _command((str(config.trace_processor_path), "--version"))
        if config.trace_processor_path is not None
        else {"availability": "not_configured"},
        "nsys": _command((str(config.nsys_executable), "--version")),
        "git_commit": _command(("git", "rev-parse", "HEAD")),
    }
    snapshot["fingerprint"] = hashlib.sha256(canonical_bytes(snapshot)).hexdigest()
    return snapshot


def idle_reasons(snapshot: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    ports = snapshot.get("ports_free")
    if not isinstance(ports, dict) or not all(value is True for value in ports.values()):
        reasons.append("one or more configured ports are in use")
    gpu = snapshot.get("gpu_processes")
    if isinstance(gpu, dict):
        if gpu.get("return_code") != 0:
            reasons.append("GPU process query failed")
        elif str(gpu.get("stdout", "")).strip():
            reasons.append("GPU compute process is present")
    npu = snapshot.get("npu")
    if isinstance(npu, dict) and npu.get("return_code") == 0:
        try:
            document = json.loads(str(npu.get("stdout", "")))
            if document.get("contexts"):
                reasons.append("NPU context is present")
            for device in document.get("devices", []):
                if int(device.get("memory", {}).get("used", "0")) != 0:
                    reasons.append(f"NPU {device.get('npu')} memory is not released")
        except (TypeError, ValueError, json.JSONDecodeError):
            reasons.append("NPU status is malformed")
    else:
        reasons.append("NPU status query failed")
    return reasons


def wait_for_idle(
    config: HybridRunnerConfig,
    *,
    timeout_sec: float = 60.0,
    interval_sec: float = 2.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_sec
    last: dict[str, object] | None = None
    while True:
        last = capture_environment(config, stage="pre_trial")
        reasons = idle_reasons(last)
        if not reasons:
            return last
        if time.monotonic() >= deadline:
            raise EnvironmentNotIdleError("; ".join(reasons))
        time.sleep(interval_sec)


__all__ = [
    "EnvironmentNotIdleError",
    "capture_environment",
    "canonical_bytes",
    "idle_reasons",
    "wait_for_idle",
]
