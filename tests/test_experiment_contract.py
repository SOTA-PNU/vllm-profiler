from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib import resources
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from perfetto_hetero_profiler.schema.records import RunStatus
from tools.evaluation.checkpoint import (
    AttemptRecord,
    AttemptStatus,
    CheckpointIntegrityError,
    CheckpointStore,
    ExperimentCheckpoint,
)
from tools.evaluation.config import (
    ExperimentConfigError,
    load_experiment_config,
)
from tools.evaluation.compatibility import (
    LEGACY_SCHEDULE_SEED_DOMAIN,
)
from tools.evaluation.experiment import (
    CONDITION_MODE,
    _AttemptLifecycle,
    build_plan,
    run_experiment,
    validate_experiment,
)
from tools.evaluation.failure import FailureClass
from tools.evaluation.limitations import limitation_inventory
from tools.evaluation.paths import ExperimentPaths, ExperimentPathError
from tools.evaluation.schedule import (
    Condition,
    TrialKind,
    build_schedule,
    canonical_schedule_bytes,
)


def hybrid_document(root: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "model": {"path": str(root / "model"), "served_name": "Qwen3-0.6B", "rbln_cache_path": str(root / "cache")},
        "prefill": {"executable": str(root / "prefill"), "working_directory": str(root), "host": "127.0.0.1", "http_port": 18100, "nixl_port": 18559, "extra_args": []},
        "decode": {"executable": str(root / "decode"), "working_directory": str(root), "host": "127.0.0.1", "http_port": 18200, "nixl_port": 18659, "extra_args": []},
        "proxy": {"python": str(root / "python"), "entry_point": "perfetto_hetero_profiler.hybrid.proxy", "host": "127.0.0.1", "http_port": 18192},
        "workload": {"prompt": "Capital of South Korea is", "warmup_requests": 2, "measured_requests": 10, "max_output_tokens": 8, "temperature": 0, "streaming": True},
        "runtime": {"max_model_len": 512, "block_size": 512, "max_num_seqs": 1, "gpu_memory_utilization": 0.2, "gpu_indices": [0], "npu_indices": [0]},
        "connectors": {"prefill": {"kv_role": "kv_producer"}, "decode": {"kv_role": "kv_consumer"}},
        "profilers": {"gpu_torch_subdir": "raw/gpu/torch", "gpu_nsys_basename": "raw/gpu/nsys/gpu", "npu_torch_subdir": "raw/npu/torch", "npu_rbln_subdir": "raw/npu/rbln"},
        "telemetry": {"sample_interval_ms": 500},
        "timeouts": {"startup_sec": 300, "request_sec": 60, "shutdown_sec": 60},
        "tools": {"trace_processor": str(root / "trace_processor"), "nsys": str(root / "nsys")},
        "offline": True,
    }


def write_config(root: Path, *, mutate_hybrid=None):
    hybrid = hybrid_document(root)
    if mutate_hybrid:
        mutate_hybrid(hybrid)
    hybrid_path = root / "hybrid.json"
    hybrid_path.write_text(json.dumps(hybrid), encoding="utf-8")
    digest = hashlib.sha256(hybrid_path.read_bytes()).hexdigest()
    config_path = root / "experiment.json"
    config_path.write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": "test-repeatability",
        "hybrid_config": {"path": str(hybrid_path), "sha256": digest},
        "schedule": {"seed": 20260807, "max_hardware_attempts": 42},
    }), encoding="utf-8")
    return config_path


class ScheduleTests(unittest.TestCase):
    def test_schedule_is_deterministic_exact_and_uses_public_condition_name(self):
        first = build_schedule(seed=20260807)
        second = build_schedule(seed=20260807)
        other = build_schedule(seed=20260808)
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.sha256, other.sha256)
        self.assertEqual(len(first.trials), 36)
        self.assertEqual(len(first.pilot_trials), 6)
        self.assertEqual(len(first.formal_trials), 30)
        self.assertEqual(first.max_hardware_attempts, 42)
        self.assertIn(Condition.NPU_TORCH, {item.condition for item in first.trials})
        self.assertNotIn("npu_vllm", first.sha256 + json.dumps(first.to_dict()))
        for round_index in range(1, 6):
            self.assertEqual({item.condition for item in first.formal_round(round_index)}, set(Condition))

    def test_formal_is_locked_until_all_pilots_succeed(self):
        schedule = build_schedule(seed=1)
        ids = {item.logical_trial_id for item in schedule.pilot_trials[:-1]}
        self.assertFalse(schedule.formal_trials_unlocked(ids))
        ids.add(schedule.pilot_trials[-1].logical_trial_id)
        self.assertTrue(schedule.formal_trials_unlocked(ids))


class ConfigTests(unittest.TestCase):
    def test_packaged_schema_and_example_use_canonical_identity(self):
        schema_path = (
            resources.files("tools.evaluation")
            / "schema/profiler_experiment_config.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$id"],
            "https://sota-pnu.github.io/vllm-profiler/schema/"
            "profiler-experiment-config-v1.json",
        )
        self.assertEqual(
            schema["title"],
            "Hybrid profiler experiment configuration",
        )
        example = json.loads(
            (Path(__file__).parents[1] / "tools/evaluation/examples/profiler_experiment_config.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            example["experiment_id"],
            "fixed-hybrid-profiler-validation",
        )

    def test_fixed_config_and_dry_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_experiment_config(write_config(root))
            output = root / "experiment"
            plan = build_plan(config, output)
            self.assertFalse(output.exists())
            self.assertFalse(plan["executes"])
            self.assertFalse(plan["creates_output"])
            self.assertEqual(plan["logical_trial_count"], 36)
            self.assertEqual(plan["maximum_hardware_attempts"], 42)

    def test_workload_and_hybrid_hash_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = write_config(root, mutate_hybrid=lambda value: value["workload"].update({"measured_requests": 9}))
            with self.assertRaises(ExperimentConfigError):
                load_experiment_config(bad)
            good = write_config(root)
            document = json.loads(good.read_text())
            document["hybrid_config"]["sha256"] = "0" * 64
            good.write_text(json.dumps(document))
            with self.assertRaises(ExperimentConfigError):
                load_experiment_config(good)

    def test_dry_run_creates_no_output_or_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_config(root)
            output = root / "planned-experiment"
            with mock.patch(
                "tools.evaluation.experiment.HybridRunner"
            ) as runner:
                plan = run_experiment(
                    config_path=config_path,
                    experiment_root=output,
                    dry_run=True,
                )
            runner.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(plan["executes"])

    def test_condition_mapping_has_reference_without_telemetry_and_one_profiler(self):
        self.assertEqual(CONDITION_MODE[Condition.REFERENCE], ("monitor", False))
        self.assertEqual(CONDITION_MODE[Condition.MONITOR], ("monitor", True))
        modes = [mode for condition, (mode, _) in CONDITION_MODE.items() if condition not in {Condition.REFERENCE, Condition.MONITOR}]
        self.assertEqual(len(modes), len(set(modes)))

    def test_existing_schedule_identity_is_loaded_without_recalculation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_config(root)
            experiment = root / "existing-experiment"
            experiment.mkdir()
            stored_config = experiment / "config.json"
            stored_config.write_bytes(source.read_bytes())
            legacy_schedule = build_schedule(
                seed=20260807,
                seed_domain=LEGACY_SCHEDULE_SEED_DOMAIN,
            )
            (experiment / "schedule.json").write_bytes(
                canonical_schedule_bytes(legacy_schedule)
            )

            loaded = load_experiment_config(stored_config)
            self.assertEqual(loaded.schedule.sha256, legacy_schedule.sha256)
            self.assertEqual(
                canonical_schedule_bytes(loaded.schedule),
                (experiment / "schedule.json").read_bytes(),
            )


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_retry_and_terminal_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            store = CheckpointStore(path)
            checkpoint = ExperimentCheckpoint.new(config_sha256="a" * 64, schedule_sha256="b" * 64)
            store.initialize(checkpoint)
            trial = build_schedule(seed=0).trials[0]
            running = AttemptRecord(trial.attempt_id(1), trial.logical_trial_id, 1, AttemptStatus.RUNNING, trial.attempt_id(1))
            checkpoint = checkpoint.with_attempt(running)
            store.update(checkpoint)
            failed = replace(running, status=AttemptStatus.FAILED, failure_class=FailureClass.READINESS_TIMEOUT, failure_summary="timeout", artifact_validation_valid=False, environment_fingerprint="c" * 64)
            checkpoint = checkpoint.with_attempt(failed)
            store.update(checkpoint)
            retry = AttemptRecord(trial.attempt_id(2), trial.logical_trial_id, 2, AttemptStatus.RUNNING, trial.attempt_id(2))
            checkpoint = checkpoint.with_attempt(retry)
            store.update(checkpoint)
            success = replace(retry, status=AttemptStatus.SUCCEEDED, artifact_validation_valid=True, environment_fingerprint="d" * 64)
            checkpoint = checkpoint.with_attempt(success)
            store.update(checkpoint)
            self.assertEqual(store.load(), checkpoint)
            with self.assertRaises(CheckpointIntegrityError):
                checkpoint.with_attempt(replace(success, environment_fingerprint="e" * 64))
            with self.assertRaises(CheckpointIntegrityError):
                checkpoint.with_attempt(AttemptRecord(trial.attempt_id(2), trial.logical_trial_id, 2, AttemptStatus.RUNNING, trial.attempt_id(2)))

    def test_success_resume_requires_fresh_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            store = CheckpointStore(path)
            trial = build_schedule(seed=0).trials[0]
            success = AttemptRecord(trial.attempt_id(1), trial.logical_trial_id, 1, AttemptStatus.SUCCEEDED, trial.attempt_id(1), artifact_validation_valid=True, environment_fingerprint="d" * 64)
            checkpoint = ExperimentCheckpoint.new(config_sha256="a" * 64, schedule_sha256="b" * 64).with_attempt(success)
            store.initialize(replace(checkpoint, generation=0))
            decision = store.load_for_resume(expected_config_sha256="a" * 64, expected_schedule_sha256="b" * 64, validate_success=lambda _: True)
            self.assertEqual(decision.skip_logical_trial_ids, (trial.logical_trial_id,))
            with self.assertRaises(CheckpointIntegrityError):
                store.load_for_resume(expected_config_sha256="a" * 64, expected_schedule_sha256="b" * 64, validate_success=lambda _: False)


class ExperimentValidationTests(unittest.TestCase):
    def test_attempt_lifecycle_wraps_core_runner_with_evaluation_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs/sample-gpu/raw/client").mkdir(parents=True)
            (root / "runs/sample-gpu/raw/client/measured_requests.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )
            hybrid = SimpleNamespace()
            config = SimpleNamespace(load_hybrid=lambda: hybrid)
            runner_result = SimpleNamespace(
                status=RunStatus.SUCCEEDED,
                errors=(),
            )
            with mock.patch(
                "tools.evaluation.experiment.wait_for_idle",
                return_value={"stage": "before"},
            ), mock.patch(
                "tools.evaluation.experiment.validate_hybrid_invocation"
            ) as preflight, mock.patch(
                "tools.evaluation.experiment.HybridRunner"
            ) as runner, mock.patch(
                "tools.evaluation.experiment.validate_trial",
                return_value={"valid": True},
            ), mock.patch(
                "tools.evaluation.experiment.capture_environment",
                return_value={"stage": "after"},
            ), mock.patch(
                "tools.evaluation.experiment.idle_reasons",
                return_value=[],
            ):
                runner.return_value.run.return_value = runner_result
                outcome = _AttemptLifecycle(
                    config=config,
                    attempt_root=root,
                    attempt_id="sample",
                    condition=Condition.MONITOR,
                    profile_mode="monitor",
                    resource_telemetry=True,
                ).execute()

            self.assertTrue(outcome.valid)
            self.assertIsNone(outcome.failure)
            preflight.assert_called_once_with(
                hybrid,
                run_root=root / "runs",
                run_id="sample",
                profile_mode="monitor",
            )
            runner.assert_called_once_with(
                hybrid,
                run_root=root / "runs",
                run_id="sample",
                profile_mode="monitor",
                enable_telemetry=True,
            )
            self.assertEqual(
                json.loads((root / "environment_before.json").read_text()),
                {"stage": "before"},
            )
            self.assertEqual(
                json.loads((root / "environment_after.json").read_text()),
                {"stage": "after"},
            )
            independent = json.loads(
                (root / "independent_client.json").read_text()
            )
            self.assertEqual(
                independent["source_relative_path"],
                "runs/sample-gpu/raw/client/measured_requests.jsonl",
            )

    def test_fresh_validation_is_read_only_unless_persistence_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = build_schedule(seed=20260807)
            attempts = tuple(
                SimpleNamespace(
                    status=AttemptStatus.SUCCEEDED,
                    attempt_id=trial.attempt_id(1),
                    logical_trial_id=trial.logical_trial_id,
                )
                for trial in schedule.trials
            )
            config = SimpleNamespace(
                sha256="a" * 64,
                schedule=schedule,
            )
            checkpoint = SimpleNamespace(
                config_sha256=config.sha256,
                schedule_sha256=schedule.sha256,
                attempts=attempts,
            )
            with mock.patch(
                "tools.evaluation.experiment.load_experiment_config",
                return_value=config,
            ), mock.patch(
                "tools.evaluation.experiment.CheckpointStore.load",
                return_value=checkpoint,
            ), mock.patch(
                "tools.evaluation.experiment._fresh_attempt_validation",
                return_value=True,
            ), mock.patch(
                "tools.evaluation.experiment._write_json"
            ) as write_json:
                result = validate_experiment(root)
                write_json.assert_not_called()

                output = root / "fresh_validation.json"
                persisted = validate_experiment(root, output_path=output)
                write_json.assert_called_once_with(output, persisted)

            self.assertTrue(result["valid"])
            self.assertEqual(result["successful_trials_checked"], 36)


class SafetyAndLimitationsTests(unittest.TestCase):
    def test_symlink_and_nonempty_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "experiment"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                ExperimentPaths.plan_new(output)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ExperimentPathError):
                ExperimentPaths.for_resume(link)

    def test_limitation_inventory_is_current_and_complete(self):
        rows = limitation_inventory()
        ids = [row["limitation_id"] for row in rows]
        self.assertIn("cpu_power_measurement", ids)
        self.assertIn("rbln_canonical_clock_anchor", ids)
        self.assertIn("reference_runtime_markers", ids)
        self.assertIn("nixl_shutdown_integrity", ids)
        transfer = next(
            row for row in rows if row["limitation_id"] == "transfer_setup_wait_marker"
        )
        self.assertEqual(transfer["status"], "capability_gated")
        self.assertNotIn("rbln_pb_opaque", ids)


if __name__ == "__main__":
    unittest.main()
