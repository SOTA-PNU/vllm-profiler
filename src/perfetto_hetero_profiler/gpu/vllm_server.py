"""Safe lifecycle management for a local vLLM OpenAI server."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class VllmServerConfig:
    model: Path
    host: str
    port: int
    gpu_memory_utilization: float
    max_model_len: int
    server_python: Path | None = None
    vllm_bin: Path | None = None
    torch_profiler_dir: Path | None = None
    nsys_output: Path | None = None
    offline: bool = True

    def __post_init__(self) -> None:
        if bool(self.server_python) == bool(self.vllm_bin):
            raise ValueError("exactly one of server_python or vllm_bin is required")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("collection server must bind to a loopback host")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if not 0 < self.gpu_memory_utilization <= 0.50:
            raise ValueError("gpu_memory_utilization must be in (0, 0.50]")
        if not 1 <= self.max_model_len <= 2048:
            raise ValueError("max_model_len must be in [1, 2048]")
        if self.torch_profiler_dir is not None:
            if not self.torch_profiler_dir.is_absolute():
                raise ValueError("torch_profiler_dir must be absolute")
            if self.nsys_output is not None:
                raise ValueError("torch and nsys profiling cannot be enabled together")
        if self.nsys_output is not None and not self.nsys_output.is_absolute():
            raise ValueError("nsys_output must be absolute")


def build_server_argv(config: VllmServerConfig) -> tuple[str, ...]:
    """Build the argv without executing or touching the filesystem."""
    if config.vllm_bin is not None:
        argv = [str(config.vllm_bin), "serve", str(config.model)]
    else:
        assert config.server_python is not None
        argv = [
            str(config.server_python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(config.model),
        ]
    argv.extend(
        [
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
            "--max-model-len",
            str(config.max_model_len),
            "--enforce-eager",
            "--no-async-scheduling",
        ]
    )
    if config.torch_profiler_dir is not None:
        argv.extend(
            [
                "--profiler-config.profiler=torch",
                f"--profiler-config.torch_profiler_dir={config.torch_profiler_dir}",
                "--profiler-config.torch_profiler_use_gzip=true",
            ]
        )
    if config.nsys_output is not None:
        argv = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=false",
            "--output",
            str(config.nsys_output),
            *argv,
        ]
    return tuple(argv)


def server_environment(config: VllmServerConfig) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    if config.offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def post_empty(base_url: str, endpoint: str, timeout_sec: float) -> None:
    request = Request(
        f"{base_url}{endpoint}",
        data=b"",
        method="POST",
        headers={"Content-Length": "0"},
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            if response.status != 200:
                raise RuntimeError(f"{endpoint} returned HTTP {response.status}")
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"{endpoint} failed: {error}") from error


class ManagedVllmServer:
    """Own one process group and clean up only that group."""

    def __init__(
        self,
        config: VllmServerConfig,
        stdout_path: Path,
        stderr_path: Path,
        *,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.popen = popen
        self.monotonic = monotonic
        self.sleep = sleep
        self.process: subprocess.Popen[bytes] | None = None
        self.started_monotonic_ns: int | None = None
        self.ready_monotonic_ns: int | None = None
        self._stdout = None
        self._stderr = None

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.config.host == "localhost" else self.config.host
        return f"http://{host}:{self.config.port}"

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("server was already started")
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout = self.stdout_path.open("wb")
        self._stderr = self.stderr_path.open("wb")
        try:
            self.process = self.popen(
                list(build_server_argv(self.config)),
                stdout=self._stdout,
                stderr=self._stderr,
                env=server_environment(self.config),
                start_new_session=True,
                shell=False,
            )
        except Exception:
            self._close_logs()
            raise
        self.started_monotonic_ns = time.monotonic_ns()

    def wait_ready(self, timeout_sec: float) -> int:
        if self.process is None:
            raise RuntimeError("server is not started")
        deadline = self.monotonic() + timeout_sec
        last_error = "health endpoint was not contacted"
        while self.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"vLLM exited before readiness with code {return_code}: "
                    f"{self._stderr_tail()}"
                )
            try:
                with urlopen(f"{self.base_url}/health", timeout=1.0) as response:
                    if response.status == 200:
                        self.ready_monotonic_ns = time.monotonic_ns()
                        return self.ready_monotonic_ns
                    last_error = f"HTTP {response.status}"
            except (HTTPError, URLError, TimeoutError) as error:
                last_error = str(error)
            self.sleep(0.25)
        raise TimeoutError(f"vLLM readiness timed out: {last_error}")

    def stop(self, timeout_sec: float) -> int:
        process = self.process
        if process is None:
            self._close_logs()
            return 0
        return_code = process.poll()
        if return_code is None:
            stop_signal = signal.SIGINT if self.config.nsys_output else signal.SIGTERM
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, stop_signal)
                else:
                    process.send_signal(stop_signal)
            except ProcessLookupError:
                pass
            try:
                return_code = process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    if os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                return_code = process.wait(timeout=5)
        self._close_logs()
        return int(return_code)

    def _stderr_tail(self, limit: int = 4000) -> str:
        if self._stderr is not None:
            self._stderr.flush()
        try:
            data = self.stderr_path.read_bytes()
        except OSError:
            return "stderr unavailable"
        return data[-limit:].decode("utf-8", errors="replace").strip()

    def _close_logs(self) -> None:
        for stream in (self._stdout, self._stderr):
            if stream is not None and not stream.closed:
                stream.close()
