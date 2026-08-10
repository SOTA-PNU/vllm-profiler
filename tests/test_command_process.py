"""Safe command specification and managed-process tests."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from perfetto_hetero_profiler.collectors import (
    CommandSpec,
    ManagedProcess,
    build_environment,
    mask_command,
    mask_environment,
)


class CommandTests(unittest.TestCase):
    @staticmethod
    def _process_is_active(pid):
        try:
            fields = (Path("/proc") / str(pid) / "stat").read_text().split()
        except FileNotFoundError:
            return False
        return len(fields) > 2 and fields[2] != "Z"

    def test_empty_argv_rejected(self):
        with self.assertRaises(ValueError):
            CommandSpec(argv=())

    def test_nonpositive_timeout_rejected(self):
        with self.assertRaises(ValueError):
            CommandSpec(argv=("true",), timeout_sec=0)

    def test_environment_uses_allowlist(self):
        spec = CommandSpec(argv=("true",), env_allowlist=("PATH",))
        self.assertEqual(
            build_environment(spec, {"PATH": "/bin", "SECRET": "hidden"}),
            {"PATH": "/bin"},
        )

    def test_disallowed_override_rejected(self):
        with self.assertRaisesRegex(ValueError, "not allowlisted"):
            CommandSpec(argv=("true",), env_overrides={"OTHER": "value"})

    def test_environment_masking(self):
        self.assertEqual(
            mask_environment({"API_KEY": "value", "LANG": "C"}),
            {"API_KEY": "***", "LANG": "C"},
        )

    def test_command_masking_separate_value(self):
        self.assertEqual(
            mask_command(("tool", "--token", "secret", "--flag")),
            ["tool", "--token", "***", "--flag"],
        )

    def test_command_masking_assignment(self):
        self.assertEqual(
            mask_command(("tool", "PASSWORD=secret")),
            ["tool", "PASSWORD=***"],
        )

    def test_stdout_and_stderr_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ManagedProcess(
                CommandSpec(
                    argv=(
                        sys.executable,
                        "-c",
                        "import sys; print('out'); print('err', file=sys.stderr)",
                    )
                ),
                root / "stdout.log",
                root / "stderr.log",
            )
            process.start()
            result = process.wait()
            self.assertEqual(result.return_code, 0)
            self.assertEqual((root / "stdout.log").read_text().strip(), "out")
            self.assertEqual((root / "stderr.log").read_text().strip(), "err")

    def test_nonzero_return_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ManagedProcess(
                CommandSpec(argv=(sys.executable, "-c", "raise SystemExit(7)")),
                root / "stdout.log",
                root / "stderr.log",
            )
            process.start()
            self.assertEqual(process.wait().return_code, 7)

    def test_timeout_terminates_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ManagedProcess(
                CommandSpec(
                    argv=(sys.executable, "-c", "import time; time.sleep(5)"),
                    timeout_sec=0.02,
                    terminate_grace_sec=0.2,
                ),
                root / "stdout.log",
                root / "stderr.log",
            )
            process.start()
            result = process.wait()
            self.assertTrue(result.timed_out)
            self.assertTrue(result.terminated)
            self.assertFalse(result.killed)

    def test_sigkill_fallback_for_child_ignoring_sigterm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ManagedProcess(
                CommandSpec(
                    argv=(
                        sys.executable,
                        "-c",
                        "import signal,time; "
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        "time.sleep(5)",
                    ),
                    timeout_sec=0.05,
                    terminate_grace_sec=0.05,
                ),
                root / "stdout.log",
                root / "stderr.log",
            )
            process.start()
            result = process.wait()
            self.assertTrue(result.timed_out)
            self.assertTrue(result.terminated)
            self.assertTrue(result.killed)

    def test_owned_grandchild_is_cleaned_after_parent_exits(self):
        script = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']); "
            "print(child.pid,flush=True); time.sleep(30)"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ManagedProcess(
                CommandSpec(
                    argv=(sys.executable, "-c", script),
                    timeout_sec=0.1,
                    terminate_grace_sec=0.05,
                ),
                root / "stdout.log",
                root / "stderr.log",
            )
            process.start()
            result = process.wait()
            grandchild_pid = int((root / "stdout.log").read_text().strip())
            deadline = time.monotonic() + 1
            while self._process_is_active(grandchild_pid) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(result.killed)
            self.assertFalse(self._process_is_active(grandchild_pid))

    def test_unowned_process_group_is_not_terminated(self):
        other = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            start_new_session=True,
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                process = ManagedProcess(
                    CommandSpec(
                        argv=(sys.executable, "-c", "import time; time.sleep(5)"),
                        timeout_sec=0.05,
                        terminate_grace_sec=0.05,
                    ),
                    root / "stdout.log",
                    root / "stderr.log",
                )
                process.start()
                process.wait()
                self.assertIsNone(other.poll())
        finally:
            other.terminate()
            other.wait(timeout=1)

    def test_stop_after_normal_exit_and_duplicate_stop_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ManagedProcess(
                CommandSpec(argv=(sys.executable, "-c", "pass")),
                root / "stdout.log",
                root / "stderr.log",
            )
            process.start()
            first = process.wait()
            second = process.stop()
            third = process.stop()
            self.assertEqual(
                (first.return_code, second.return_code, third.return_code),
                (0, 0, 0),
            )
            self.assertFalse(second.terminated)
            self.assertFalse(third.killed)

    def test_leader_first_stop_allows_graceful_signal_handler(self):
        script = (
            "import signal,sys,time; "
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
            "print('ready', flush=True); time.sleep(30)"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ManagedProcess(
                CommandSpec(
                    argv=(sys.executable, "-c", script),
                    terminate_grace_sec=0.5,
                ),
                root / "stdout.log",
                root / "stderr.log",
            )
            process.start()
            deadline = time.monotonic() + 1
            while not (root / "stdout.log").read_text().strip():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            result = process.stop_leader_first()
            repeated = process.stop_leader_first()
            self.assertEqual(result.return_code, 0)
            self.assertTrue(result.terminated)
            self.assertFalse(result.killed)
            self.assertEqual(repeated.return_code, 0)

    def test_start_failure_closes_output_handles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            process = ManagedProcess(
                CommandSpec(argv=("missing-command-process-test",)),
                stdout,
                stderr,
            )
            with self.assertRaises(FileNotFoundError):
                process.start()
            with stdout.open("ab") as output:
                output.write(b"closed")
            with stderr.open("ab") as output:
                output.write(b"closed")

    def test_invalid_process_group_is_rejected_without_group_signal(self):
        group_signals = []

        class FakeProcess:
            pid = 123

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ManagedProcess(
                CommandSpec(argv=("tool",)),
                root / "stdout.log",
                root / "stderr.log",
                popen_factory=lambda *args, **kwargs: FakeProcess(),
                get_process_group=lambda pid: pid + 1,
                signal_process_group=lambda group, sig: group_signals.append((group, sig)),
            )
            with self.assertRaisesRegex(RuntimeError, "expected owned process group"):
                process.start()
        self.assertEqual(group_signals, [])

    def test_safe_plan_does_not_expose_secret(self):
        spec = CommandSpec(
            argv=("tool", "--auth", "secret"),
            env_allowlist=("AUTH_TOKEN",),
            env_overrides={"AUTH_TOKEN": "secret"},
        )
        self.assertNotIn("secret", repr(spec.safe_plan()))

    def test_managed_process_does_not_use_shell(self):
        captured = {}

        class FakeProcess:
            pid = 123

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        def factory(argv, **kwargs):
            captured.update(kwargs)
            return FakeProcess()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = ManagedProcess(
                CommandSpec(argv=("tool",)),
                root / "stdout",
                root / "stderr",
                popen_factory=factory,
                get_process_group=lambda pid: pid,
                signal_process_group=lambda group, sig: (_ for _ in ()).throw(
                    ProcessLookupError()
                ),
            )
            process.start()
            process.wait()
        self.assertIs(captured["shell"], False)
        self.assertIs(captured["start_new_session"], True)


if __name__ == "__main__":
    unittest.main()
