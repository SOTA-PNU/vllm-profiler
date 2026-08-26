from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluation.checkpoint import AttemptRecord, AttemptStatus, ExperimentCheckpoint
from tools.evaluation.report import build_report, canonical_json, render_report_html
from tools.evaluation.schedule import build_schedule


class ReportTests(unittest.TestCase):
    def test_pilot_is_excluded_and_formal_report_is_deterministic(self):
        schedule = build_schedule(seed=99)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = []
            for trial in schedule.trials:
                attempt_id = trial.attempt_id(1)
                trial_root = root / "trials" / attempt_id
                trial_root.mkdir(parents=True)
                base = 999_000_000 if trial.phase.value == "pilot" else trial.round_index * 1_000_000
                metrics = {
                    "latency.e2e": base + list(type(trial.condition)).index(trial.condition) * 100_000,
                    "latency.ttft": base / 2,
                    "latency.tpot": base / 20,
                    "throughput.requests": 10.0 + trial.round_index,
                    "throughput.output_tokens": 80.0 + trial.round_index,
                    "throughput.total_tokens": 130.0 + trial.round_index,
                }
                validation = {
                    "valid": True,
                    "metrics": metrics,
                    "resources": {} if trial.condition.value == "reference" else {
                        "resource.gpu.utilization": {"mean": 10.0, "maximum": 20.0, "sample_count": 2}
                    },
                }
                (trial_root / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
                if trial.condition.value != "reference":
                    npu_metrics = trial_root / "runs" / f"{attempt_id}-npu" / "metrics"
                    npu_metrics.mkdir(parents=True)
                    rows = [
                        {
                            "metric_name": "resource.npu.utilization",
                            "availability": "available",
                            "value": value,
                        }
                        for value in (1.0, 3.0)
                    ]
                    (npu_metrics / "metrics.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8",
                    )
                attempts.append(AttemptRecord(
                    attempt_id=attempt_id,
                    logical_trial_id=trial.logical_trial_id,
                    attempt_number=1,
                    status=AttemptStatus.SUCCEEDED,
                    relative_directory=attempt_id,
                    artifact_validation_valid=True,
                    environment_fingerprint="a" * 64,
                ))
            checkpoint = ExperimentCheckpoint(
                config_sha256="b" * 64,
                schedule_sha256=schedule.sha256,
                max_hardware_attempts=42,
                attempts=tuple(attempts),
            )
            first = build_report(root=root, config={"experiment_id": "test"}, schedule=schedule, checkpoint=checkpoint)
            second = build_report(root=root, config={"experiment_id": "test"}, schedule=schedule, checkpoint=checkpoint)
            self.assertEqual(canonical_json(first), canonical_json(second))
            self.assertEqual(first["formal_repeatability"]["reference"]["latency.e2e"]["sample_count"], 5)
            self.assertLess(first["formal_repeatability"]["reference"]["latency.e2e"]["mean"], 10_000_000)
            self.assertEqual(first["paired_overhead"]["monitor_vs_reference"]["latency.e2e"]["expected_pair_count"], 5)
            self.assertEqual(first["progress"]["successful_logical_trials"], 36)
            self.assertEqual(
                first["report_type"],
                "profiler_repeatability_overhead",
            )
            self.assertNotIn("phase", canonical_json(first).decode("utf-8").lower())
            self.assertEqual(
                first["resources"]["monitor"]["resource.npu.utilization"]
                ["trial_mean_distribution"]["mean"],
                2.0,
            )
            self.assertNotIn("resource.npu.utilization", first["resources"]["reference"])
            html = render_report_html(first)
            self.assertEqual(html, render_report_html(second))
            self.assertIn(b"not a Perfetto built-in Overview", html)
            self.assertNotIn(b"https://", html)


if __name__ == "__main__":
    unittest.main()
