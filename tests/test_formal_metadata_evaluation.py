"""CPU-only tests for formal condition and environment publication metadata."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from perfetto_hetero_profiler.support.files import sha256_file
from perfetto_hetero_profiler.support.json_io import (
    write_jsonl_exclusive,
    write_pretty_json,
)
from tools.evaluation.concurrent_wave import load_matrix_blocks
from tools.evaluation.formal_campaign import load_config
from tools.evaluation.formal_metadata import (
    FormalMetadataError,
    build_condition_metadata,
    collect_canonical_environment,
    validate_condition_schema,
    validate_environment,
    validate_published_block,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/evaluation/examples/final_ofat_campaign.json"


def _completed(argv, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def _failed_runner(argv, **_kwargs):
    return _completed(argv, returncode=1, stderr="unavailable")


def _environment() -> dict[str, object]:
    config = SimpleNamespace(
        gpu_vllm=Path("/private/gpu/bin/vllm"),
        tokenizer_python=Path("/private/npu/bin/python"),
    )
    return collect_canonical_environment(
        config,
        profiler_head="a" * 40,
        vllm_rbln_head="b" * 40,
        query_devices=False,
        runner=_failed_runner,
    )


class ConditionMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.blocks = load_matrix_blocks(cls.config.matrix)

    def test_all_formal_conditions_validate_with_truthful_semantics(self):
        values = [
            build_condition_metadata(
                self.config, block, position=index, environment_sha256="c" * 64
            )
            for index, block in enumerate(self.blocks, 1)
        ]
        self.assertEqual(len(values), 21)
        hybrid = next(value for value in values if value["condition_id"] == "m15-b4-hybrid")
        self.assertEqual(hybrid["model"]["size_label"], "1.5B")
        self.assertIsNone(hybrid["model"]["exact_parameter_count"])
        self.assertEqual(hybrid["workload"]["request_concurrency"], 4)
        self.assertEqual(hybrid["workload"]["max_num_seqs"], 4)
        self.assertEqual(hybrid["workload"]["measured_waves"], 13)
        self.assertIsNone(hybrid["topology"]["layer_partition_ratio"])
        self.assertEqual(hybrid["topology"]["transfer_direction"], "gpu_to_npu")
        standalone = next(value for value in values if value["condition_id"] == "m15-b1-gpu")
        self.assertEqual(standalone["topology"]["transfer_direction"], "not_applicable")
        self.assertIsNone(standalone["topology"]["producer_role"])

    def test_schema_rejects_cross_field_mismatches(self):
        block = next(item for item in self.blocks if item.condition_id == "m15-b4-hybrid")
        value = build_condition_metadata(
            self.config, block, position=3, environment_sha256="d" * 64
        )
        mutations = []
        wrong_size = deepcopy(value)
        wrong_size["model"]["size_label"] = "3B"
        mutations.append(wrong_size)
        wrong_concurrency = deepcopy(value)
        wrong_concurrency["workload"]["request_concurrency"] = 2
        mutations.append(wrong_concurrency)
        wrong_direction = deepcopy(value)
        wrong_direction["topology"]["transfer_direction"] = "not_applicable"
        mutations.append(wrong_direction)
        invented_ratio = deepcopy(value)
        invented_ratio["topology"]["layer_partition_ratio"] = 0.5
        mutations.append(invented_ratio)
        extra = deepcopy(value)
        extra["formal_claim"] = "unsupported"
        mutations.append(extra)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(FormalMetadataError):
                    validate_condition_schema(mutation)


class EnvironmentMetadataTests(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            gpu_vllm=Path("/private/gpu/bin/vllm"),
            tokenizer_python=Path("/private/npu/bin/python"),
        )

    def test_unavailable_queries_are_explicit_and_publish_no_paths(self):
        value = collect_canonical_environment(
            self._config(), profiler_head="a" * 40, vllm_rbln_head="b" * 40,
            query_devices=True, runner=_failed_runner,
        )
        validate_environment(value)
        self.assertEqual(value["gpu"]["query_status"]["availability"], "not_available")
        self.assertEqual(value["npu"]["query_status"]["availability"], "not_available")
        self.assertEqual(
            value["server_environments"]["gpu"]["packages"]["nixl"]["availability"],
            "not_available",
        )
        encoded = json.dumps(value)
        self.assertNotIn("/private/", encoded)
        self.assertNotIn("hostname", encoded.lower().replace("hostname_recorded", ""))

    def test_server_packages_drivers_devices_and_pcie_come_from_mock_evidence(self):
        packages = {
            name: {"availability": "available", "value": f"1.0-{name}", "reason": None}
            for name in (
                "vllm", "vllm-rbln", "torch", "nixl", "optimum-rbln",
                "rebel-compiler", "psutil",
            )
        }

        def runner(argv, **_kwargs):
            if "-c" in argv:
                return _completed(argv, stdout=json.dumps({
                    "python_version": "3.10.12", "packages": packages,
                }) + "\n")
            if argv[0] == "nvidia-smi":
                if "pcie.link" in argv[1]:
                    return _completed(argv, stdout="0, 4, 5, 8, 16\n")
                return _completed(argv, stdout="0, Mock GPU, 1024, 555.1\n")
            if argv == ["rbln-smi", "--version"]:
                return _completed(argv, stdout="rbln-smi 3.0.0\n")
            if argv == ["rbln-smi", "--json"]:
                return _completed(argv, stdout=json.dumps({
                    "KMD_version": "1.2.3",
                    "devices": [{
                        "npu": 0, "name": "Mock NPU", "status": "idle",
                        "memory": {"used": "0 MiB", "total": "2048 MiB"},
                        "util": "0 %", "card_power": "0 W",
                        "temperature": "30 C", "fw_ver": "9.9",
                    }],
                }))
            return _completed(argv, returncode=1)

        value = collect_canonical_environment(
            self._config(), profiler_head="a" * 40, vllm_rbln_head="b" * 40,
            query_devices=True, runner=runner,
        )
        self.assertEqual(value["gpu"]["driver_version"]["value"], "555.1")
        self.assertEqual(
            value["gpu"]["devices"][0]["pcie"]["maximum_generation"]["value"], 5
        )
        self.assertEqual(value["npu"]["kmd_driver_version"]["value"], "1.2.3")
        self.assertEqual(value["npu"]["devices"][0]["model"]["value"], "Mock NPU")
        self.assertEqual(value["gpu"]["memory_technology"]["availability"], "not_available")
        self.assertEqual(value["npu"]["memory_technology"]["availability"], "not_available")


class PublishedBlockValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.blocks = load_matrix_blocks(cls.config.matrix)

    def _publish(self, root: Path, block, *, position: int, hybrid: bool) -> tuple[Path, Path]:
        environment_path = root / "environment.json"
        write_pretty_json(environment_path, _environment())
        block_root = root / "blocks" / "block"
        evidence = block_root / "evaluation"
        evidence.mkdir(parents=True)
        write_pretty_json(
            block_root / "condition_metadata.json",
            build_condition_metadata(
                self.config, block, position=position,
                environment_sha256=sha256_file(environment_path),
            ),
        )
        requests = []
        for phase, count in (
            ("warmup", block.warmup_requests),
            ("measured", block.measured_requests),
        ):
            for index in range(count):
                requests.append({
                    "condition_id": block.condition_id, "round": block.round_index,
                    "phase": phase, "request_id": f"{phase}-{index}",
                    "concurrency": block.concurrency, "success": True,
                    "http_status": 200, "input_tokens": block.input_tokens,
                    "output_tokens": block.output_tokens,
                    "request_start_monotonic_ns": index * 10 + 1,
                    "stream_end_monotonic_ns": index * 10 + 9,
                })
        waves = []
        for phase, count in (
            ("warmup", block.warmup_requests // block.concurrency),
            ("measured", block.measured_waves),
        ):
            for index in range(count):
                waves.append({
                    "condition_id": block.condition_id, "round": block.round_index,
                    "phase": phase, "wave_id": f"{phase}-{index}",
                    "concurrency": block.concurrency,
                    "barrier_release_monotonic_ns": index * 10,
                    "request_count": block.concurrency,
                    "max_in_flight": block.concurrency, "common_overlap_ns": 1,
                    "barrier_respected": True,
                    "expected_concurrency_reached": True,
                    "failed_request_ids": [], "success": True,
                })
        write_jsonl_exclusive(evidence / "requests.jsonl", requests)
        write_jsonl_exclusive(evidence / "waves.jsonl", waves)
        write_pretty_json(evidence / "concurrency_validation.json", {
            "condition_id": block.condition_id, "round": block.round_index,
            "concurrency": block.concurrency,
            "expected_wave_count": len(waves), "observed_wave_count": len(waves),
            "failed_wave_ids": [], "all_expected_concurrency_reached": True,
            "valid": True,
        })
        artifacts = {
            name: sha256_file(evidence / name)
            for name in ("requests.jsonl", "waves.jsonl", "concurrency_validation.json")
        }
        write_pretty_json(evidence / "summary.json", {
            "condition_id": block.condition_id, "round": block.round_index,
            "status": "succeeded", "runner_status": "succeeded",
            "request_count": len(requests), "wave_count": len(waves),
            "runner_errors": [], "stores_prompt_or_generated_text": False,
            "artifacts": artifacts,
        })
        write_pretty_json(block_root / "runtime-postflight.json", {
            "valid": True, "busy_ports": [], "remaining_processes": [],
            "npu_contexts": [],
        })
        if hybrid:
            run_root = evidence / "runner" / f"r{block.round_index:02d}-{block.condition_id}"
            def token(value):
                return {
                    "value": value, "uniform": True, "minimum": value,
                    "maximum": value, "sample_count": block.measured_requests,
                }
            write_pretty_json(run_root / "hybrid" / "manifest.json", {
                "mode": "hybrid", "status": "succeeded",
                "workload": {
                    "request_count": block.measured_requests,
                    "concurrency": block.concurrency,
                    "input_tokens": block.input_tokens,
                    "output_tokens": block.output_tokens,
                    "max_model_len": 512,
                },
                "configuration": {"runtime_metadata": {
                    "max_num_seqs_capacity": block.concurrency,
                    "model_identity": {
                        "served_model_id": self.config.models["m15"].served_name,
                        "revision": self.config.models["m15"].revision,
                        "tokenizer_id": None,
                        "dtype": "bfloat16", "size_label": "1.5B",
                        "size_label_origin": "declared_experiment_contract",
                        "exact_parameter_count": None,
                        "exact_parameter_count_status": "not_available",
                        "metadata_origin": "declared_experiment_contract",
                        "path_inference_used": False,
                    },
                    "token_counts": {
                        "input_tokens": token(block.input_tokens),
                        "output_tokens": token(block.output_tokens),
                    },
                    "topology": {
                        "mode": "hybrid",
                        "partition_strategy": "gpu_prefill_npu_decode",
                        "prefill_device_type": "gpu", "decode_device_type": "npu",
                        "layer_partition_ratio": None,
                        "layer_partition_ratio_status": "not_applicable",
                        "source_device_type": "gpu", "destination_device_type": "npu",
                        "producer_role": "kv_producer", "consumer_role": "kv_consumer",
                        "transfer_direction": "gpu_to_npu",
                    },
                }},
            })
            write_pretty_json(run_root / "publication" / "final_result.json", {
                "status": "succeeded",
                "warmup_completed": block.warmup_requests,
                "measured_completed": block.measured_requests,
                "lifecycle": {
                    "started_at_unix_ns": 100, "finished_at_unix_ns": 200,
                    "duration_ns": 100,
                    "terminal_status": "succeeded", "shutdown_integrity": "valid",
                    "cleanup_status": "complete",
                    "measured_request_count": block.measured_requests,
                },
            })
        else:
            write_pretty_json(
                evidence / "shutdown.json", {"return_code": -15, "killed": False}
            )
        return block_root, environment_path

    def test_standalone_and_hybrid_evidence_cross_validate(self):
        cases = [
            (next(item for item in self.blocks if item.condition_id == "m15-b1-gpu"), False),
            (next(item for item in self.blocks if item.condition_id == "m15-b4-hybrid"), True),
        ]
        for block, hybrid in cases:
            with self.subTest(block=block.condition_id), tempfile.TemporaryDirectory() as directory:
                block_root, environment = self._publish(
                    Path(directory), block, position=1, hybrid=hybrid
                )
                result = validate_published_block(
                    self.config, block, position=1, block_root=block_root,
                    environment_path=environment,
                )
                self.assertTrue(result["valid"])
                self.assertEqual(result["core_manifest_checked"], hybrid)

    def test_mismatch_injections_fail_closed(self):
        block = next(item for item in self.blocks if item.condition_id == "m15-b4-hybrid")
        mutators = (
            lambda root: self._change_json(
                root / "condition_metadata.json", "condition_id", "m15-b2-hybrid"
            ),
            lambda root: self._change_json(
                root / "evaluation/runner/r01-m15-b4-hybrid/hybrid/manifest.json",
                "workload.concurrency", 2,
            ),
            lambda root: self._change_json(
                root / "evaluation/runner/r01-m15-b4-hybrid/publication/final_result.json",
                "lifecycle.cleanup_status", "incomplete",
            ),
        )
        for mutator in mutators:
            with tempfile.TemporaryDirectory() as directory:
                block_root, environment = self._publish(
                    Path(directory), block, position=1, hybrid=True
                )
                mutator(block_root)
                with self.assertRaises(FormalMetadataError):
                    validate_published_block(
                        self.config, block, position=1, block_root=block_root,
                        environment_path=environment,
                    )

    @staticmethod
    def _change_json(path: Path, dotted: str, value: object) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        target = document
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        write_pretty_json(path, document)


if __name__ == "__main__":
    unittest.main()
