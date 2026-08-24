from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from perfetto_hetero_profiler.experiments.checkpoint import (
    AttemptRecord,
    AttemptStatus,
    CheckpointIntegrityError,
    CheckpointStore,
    ExperimentCheckpoint,
)
from perfetto_hetero_profiler.experiments.config import canonical_config_bytes, load_experiment_config
from perfetto_hetero_profiler.experiments.experiment import RETRYABLE, _resume
from perfetto_hetero_profiler.experiments.failure import (
    ConnectionEvidence,
    FailureClass,
    FailurePhase,
    classify_connection_failure,
)
from perfetto_hetero_profiler.experiments.paths import ExperimentPaths
from perfetto_hetero_profiler.experiments.schedule import canonical_schedule_bytes

from tests.test_experiment_contract import write_config


class ResumeTests(unittest.TestCase):
    def fixture(self, root: Path):
        config = load_experiment_config(write_config(root))
        experiment = root / "experiment"
        experiment.mkdir()
        (experiment / "trials").mkdir()
        (experiment / "config.json").write_bytes(canonical_config_bytes(config))
        (experiment / "schedule.json").write_bytes(
            canonical_schedule_bytes(config.schedule)
        )
        (experiment / "hybrid_config.json").write_bytes(
            config.hybrid_config_path.read_bytes()
        )
        return config, ExperimentPaths.for_resume(experiment)

    def test_interrupted_running_attempt_is_finalized_and_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            config, paths = self.fixture(Path(directory))
            trial = config.schedule.trials[0]
            running = AttemptRecord(
                trial.attempt_id(1), trial.logical_trial_id, 1,
                AttemptStatus.RUNNING, trial.attempt_id(1),
            )
            initial = ExperimentCheckpoint.new(
                config_sha256=config.sha256,
                schedule_sha256=config.schedule.sha256,
            )
            store = CheckpointStore(paths.checkpoint)
            store.initialize(initial)
            store.update(initial.with_attempt(running))

            resumed = _resume(config, paths)
            final = resumed.attempts[-1]
            self.assertEqual(final.status, AttemptStatus.FAILED)
            self.assertEqual(final.failure_class, FailureClass.INTERRUPTED)
            self.assertIn(FailureClass.INTERRUPTED, RETRYABLE)

    def test_resume_rejects_config_snapshot_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            config, paths = self.fixture(Path(directory))
            CheckpointStore(paths.checkpoint).initialize(
                ExperimentCheckpoint.new(
                    config_sha256=config.sha256,
                    schedule_sha256=config.schedule.sha256,
                )
            )
            (paths.root / "config.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(CheckpointIntegrityError):
                _resume(config, paths)

    def test_resume_rejects_corrupt_success_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            config, paths = self.fixture(Path(directory))
            trial = config.schedule.trials[0]
            success = AttemptRecord(
                trial.attempt_id(1), trial.logical_trial_id, 1,
                AttemptStatus.SUCCEEDED, trial.attempt_id(1),
                artifact_validation_valid=True,
                environment_fingerprint="a" * 64,
            )
            initial = ExperimentCheckpoint.new(
                config_sha256=config.sha256,
                schedule_sha256=config.schedule.sha256,
            )
            store = CheckpointStore(paths.checkpoint)
            store.initialize(initial)
            store.update(initial.with_attempt(success))
            with mock.patch(
                "perfetto_hetero_profiler.experiments.experiment._fresh_attempt_validation",
                return_value=False,
            ):
                with self.assertRaises(CheckpointIntegrityError):
                    _resume(config, paths)


class FailureClassificationTests(unittest.TestCase):
    def evidence(self, **changes):
        fields = {
            "phase": FailurePhase.MEASURED_REQUEST,
            "process_role": "proxy",
            "expected_host": "127.0.0.1",
            "expected_port": 18192,
            "process_start_called": True,
            "process_ready": True,
            "process_returncode": None,
            "expected_listener_present": False,
            "required_roles": ("prefill", "decode", "proxy"),
            "ready_roles": ("prefill", "decode", "proxy"),
            "error_number": 111,
        }
        fields.update(changes)
        return ConnectionEvidence(**fields)

    def test_connection_refused_uses_structured_evidence(self):
        self.assertEqual(
            classify_connection_failure(self.evidence()),
            FailureClass.CLIENT_CONNECTION_REFUSED,
        )

    def test_readiness_timeout_uses_phase_and_timeout(self):
        value = self.evidence(
            phase=FailurePhase.HEALTH,
            process_ready=False,
            expected_listener_present=None,
            error_number=None,
            timed_out=True,
            ready_roles=(),
        )
        self.assertEqual(
            classify_connection_failure(value), FailureClass.HEALTH_TIMEOUT
        )


if __name__ == "__main__":
    unittest.main()
