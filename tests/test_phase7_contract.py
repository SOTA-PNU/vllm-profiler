from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from perfetto_hetero_profiler.phase7.checkpoint import (
    AttemptRecord,
    AttemptStatus,
    CheckpointIntegrityError,
    CheckpointStore,
    ExperimentCheckpoint,
)
from perfetto_hetero_profiler.phase7.config import (
    Phase7ConfigError,
    load_phase7_config,
)
from perfetto_hetero_profiler.phase7.experiment import CONDITION_MODE, build_plan
from perfetto_hetero_profiler.phase7.failure import FailureClass
from perfetto_hetero_profiler.phase7.limitations import limitation_inventory
from perfetto_hetero_profiler.phase7.paths import ExperimentPaths, Phase7PathError
from perfetto_hetero_profiler.phase7.schedule import (
    Condition,
    TrialPhase,
    build_schedule,
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
    def test_fixed_config_and_dry_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_phase7_config(write_config(root))
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
            with self.assertRaises(Phase7ConfigError):
                load_phase7_config(bad)
            good = write_config(root)
            document = json.loads(good.read_text())
            document["hybrid_config"]["sha256"] = "0" * 64
            good.write_text(json.dumps(document))
            with self.assertRaises(Phase7ConfigError):
                load_phase7_config(good)

    def test_condition_mapping_has_reference_without_telemetry_and_one_profiler(self):
        self.assertEqual(CONDITION_MODE[Condition.REFERENCE], ("monitor", False))
        self.assertEqual(CONDITION_MODE[Condition.MONITOR], ("monitor", True))
        modes = [mode for condition, (mode, _) in CONDITION_MODE.items() if condition not in {Condition.REFERENCE, Condition.MONITOR}]
        self.assertEqual(len(modes), len(set(modes)))


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
            with self.assertRaises(Phase7PathError):
                ExperimentPaths.for_resume(link)

    def test_limitation_inventory_is_current_and_complete(self):
        rows = limitation_inventory()
        ids = [row["limitation_id"] for row in rows]
        self.assertIn("rbln_canonical_clock_anchor", ids)
        self.assertIn("reference_runtime_markers", ids)
        self.assertNotIn("rbln_pb_opaque", ids)


if __name__ == "__main__":
    unittest.main()
