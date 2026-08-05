"""Management for one explicitly launched child process."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Callable

from .command import CommandSpec, build_environment


@dataclass(frozen=True)
class CommandResult:
    return_code: int
    started_monotonic_ns: int
    ended_monotonic_ns: int
    timed_out: bool
    terminated: bool
    killed: bool


class ManagedProcess:
    """Own and stop only the subprocess created by this instance."""

    def __init__(
        self,
        spec: CommandSpec,
        stdout_path: Path,
        stderr_path: Path,
        *,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        get_process_group: Callable[[int], int] = os.getpgid,
        signal_process_group: Callable[[int, int], None] = os.killpg,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self.stdout_path = Path(stdout_path)
        self.stderr_path = Path(stderr_path)
        self._popen_factory = popen_factory
        self._monotonic_ns = monotonic_ns
        self._get_process_group = get_process_group
        self._signal_process_group = signal_process_group
        self._sleep = sleep
        self.process: subprocess.Popen[bytes] | None = None
        self.started_monotonic_ns: int | None = None
        self.process_group_id: int | None = None
        self._stdout = None
        self._stderr = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("child process has already been started")
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout = self.stdout_path.open("xb")
        try:
            self._stderr = self.stderr_path.open("xb")
            self.started_monotonic_ns = self._monotonic_ns()
            self.process = self._popen_factory(
                list(self.spec.argv),
                cwd=str(self.spec.cwd) if self.spec.cwd is not None else None,
                env=build_environment(self.spec),
                stdout=self._stdout,
                stderr=self._stderr,
                shell=False,
                start_new_session=True,
            )
            try:
                process_group_id = self._get_process_group(self.process.pid)
            except Exception:
                self.process.terminate()
                self.process.wait()
                raise
            if process_group_id != self.process.pid:
                self.process.terminate()
                self.process.wait()
                raise RuntimeError(
                    "child did not create the expected owned process group"
                )
            self.process_group_id = process_group_id
        except Exception:
            self._close_outputs()
            self.process = None
            self.process_group_id = None
            raise

    def poll(self) -> int | None:
        if self.process is None:
            raise RuntimeError("child process has not been started")
        return self.process.poll()

    def wait(self) -> CommandResult:
        if self.process is None or self.started_monotonic_ns is None:
            raise RuntimeError("child process has not been started")
        timed_out = terminated = killed = False
        try:
            return_code = self.process.wait(timeout=self.spec.timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminated = self._signal_owned_group(signal.SIGTERM)
            try:
                return_code = self.process.wait(timeout=self.spec.terminate_grace_sec)
            except subprocess.TimeoutExpired:
                killed = self._signal_owned_group(signal.SIGKILL)
                return_code = self.process.wait()
            else:
                remaining_terminated, remaining_killed = (
                    self._finish_remaining_group(term_already_sent=True)
                )
                terminated = terminated or remaining_terminated
                killed = killed or remaining_killed
        else:
            terminated, killed = self._finish_remaining_group(
                term_already_sent=False
            )
        finally:
            self._close_outputs()
        return CommandResult(
            return_code=return_code,
            started_monotonic_ns=self.started_monotonic_ns,
            ended_monotonic_ns=self._monotonic_ns(),
            timed_out=timed_out,
            terminated=terminated,
            killed=killed,
        )

    def stop(self, *, timed_out: bool = False) -> CommandResult:
        """Terminate the owned child if running, then wait for it."""
        if self.process is None:
            raise RuntimeError("child process has not been started")
        terminated = killed = False
        try:
            if self.process.poll() is None:
                terminated = self._signal_owned_group(signal.SIGTERM)
                try:
                    return_code = self.process.wait(
                        timeout=self.spec.terminate_grace_sec
                    )
                except subprocess.TimeoutExpired:
                    killed = self._signal_owned_group(signal.SIGKILL)
                    return_code = self.process.wait()
                else:
                    remaining_terminated, remaining_killed = (
                        self._finish_remaining_group(term_already_sent=True)
                    )
                    terminated = terminated or remaining_terminated
                    killed = killed or remaining_killed
            else:
                return_code = self.process.wait()
                terminated, killed = self._finish_remaining_group(
                    term_already_sent=False
                )
        finally:
            self._close_outputs()
        assert self.started_monotonic_ns is not None
        return CommandResult(
            return_code=return_code,
            started_monotonic_ns=self.started_monotonic_ns,
            ended_monotonic_ns=self._monotonic_ns(),
            timed_out=timed_out,
            terminated=terminated,
            killed=killed,
        )

    def stop_leader_first(
        self,
        *,
        leader_signal: signal.Signals | int = signal.SIGTERM,
        timed_out: bool = False,
    ) -> CommandResult:
        """Ask the verified leader to exit before using owned-group fallback.

        Servers that coordinate worker shutdown need a chance to run their
        normal signal handler.  Descendants are still cleaned up, but only
        after the leader exits or ignores the first signal.
        """
        if self.process is None:
            raise RuntimeError("child process has not been started")
        terminated = killed = False
        try:
            if self.process.poll() is None:
                try:
                    current_group = self._get_process_group(self.process.pid)
                except ProcessLookupError:
                    current_group = None
                if current_group is not None:
                    if current_group != self.process_group_id:
                        raise RuntimeError(
                            "owned process group identity changed; refusing to signal"
                        )
                    try:
                        self.process.send_signal(leader_signal)
                        terminated = True
                    except ProcessLookupError:
                        pass
                try:
                    return_code = self.process.wait(
                        timeout=self.spec.terminate_grace_sec
                    )
                except subprocess.TimeoutExpired:
                    terminated = (
                        self._signal_owned_group(signal.SIGTERM) or terminated
                    )
                    try:
                        return_code = self.process.wait(
                            timeout=self.spec.terminate_grace_sec
                        )
                    except subprocess.TimeoutExpired:
                        killed = self._signal_owned_group(signal.SIGKILL)
                        return_code = self.process.wait()
                    else:
                        remaining_terminated, remaining_killed = (
                            self._finish_remaining_group(term_already_sent=True)
                        )
                        terminated = terminated or remaining_terminated
                        killed = killed or remaining_killed
                else:
                    remaining_terminated, remaining_killed = (
                        self._finish_remaining_group(term_already_sent=False)
                    )
                    terminated = terminated or remaining_terminated
                    killed = killed or remaining_killed
            else:
                return_code = self.process.wait()
                terminated, killed = self._finish_remaining_group(
                    term_already_sent=False
                )
        finally:
            self._close_outputs()
        assert self.started_monotonic_ns is not None
        return CommandResult(
            return_code=return_code,
            started_monotonic_ns=self.started_monotonic_ns,
            ended_monotonic_ns=self._monotonic_ns(),
            timed_out=timed_out,
            terminated=terminated,
            killed=killed,
        )

    def _signal_owned_group(
        self, sig: signal.Signals | int, *, leader_may_have_exited: bool = False
    ) -> bool:
        """Signal only the process group created by ``start_new_session``."""
        if self.process is None or self.process_group_id is None:
            raise RuntimeError("child process has not been started")
        if not leader_may_have_exited:
            if self.process.poll() is not None:
                return False
            try:
                current_group = self._get_process_group(self.process.pid)
            except ProcessLookupError:
                return False
            if current_group != self.process_group_id:
                raise RuntimeError(
                    "owned process group identity changed; refusing to signal"
                )
        try:
            self._signal_process_group(self.process_group_id, sig)
        except ProcessLookupError:
            return False
        except PermissionError as error:
            raise RuntimeError(
                "permission denied for the verified owned process group"
            ) from error
        return True

    def _finish_remaining_group(
        self, *, term_already_sent: bool
    ) -> tuple[bool, bool]:
        """Terminate descendants left after the process-group leader exits."""
        if not self._signal_owned_group(0, leader_may_have_exited=True):
            return False, False
        terminated = True
        if not term_already_sent:
            self._signal_owned_group(signal.SIGTERM, leader_may_have_exited=True)
        if self.spec.terminate_grace_sec:
            self._sleep(self.spec.terminate_grace_sec)
        if self._signal_owned_group(0, leader_may_have_exited=True):
            self._signal_owned_group(signal.SIGKILL, leader_may_have_exited=True)
            return terminated, True
        return terminated, False

    def _close_outputs(self) -> None:
        for output in (self._stdout, self._stderr):
            if output is not None and not output.closed:
                output.close()
