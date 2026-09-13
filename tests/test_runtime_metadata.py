"""CPU-only tests for truthful, campaign-agnostic runtime metadata."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from perfetto_hetero_profiler.gpu.openai_client import CompletionObservation
from perfetto_hetero_profiler.hybrid.runner import HybridRunner
from perfetto_hetero_profiler.hybrid.runner_config import (
    HybridRunnerConfigError,
    load_hybrid_runner_config,
)
from perfetto_hetero_profiler.runtime_metadata import (
    RuntimeMetadataError,
    lifecycle_metadata,
    measured_request_concurrency,
    measured_token_metadata,
    merged_runtime_metadata,
    model_identity_metadata,
    topology_metadata,
    transfer_dimensions,
)
from perfetto_hetero_profiler.schema import (
    SCHEMA_VERSION,
    DeviceDescriptor,
    DeviceType,
    HostDescriptor,
    ModelDescriptor,
    ProfileMode,
    RunManifest,
    RunMode,
    RunStatus,
    SoftwareDescriptor,
    WorkloadDescriptor,
    validate_record,
)

from tests.support.runner_fakes import document


def observation(
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    *,
    received_ns: int = 1,
):
    return CompletionObservation(
        request_id=request_id,
        received_ns=received_ns,
        response_started_ns=received_ns + 1,
        token_timestamps_ns=tuple(
            range(received_ns + 1, received_ns + 1 + output_tokens)
        ),
        done_ns=received_ns + 10 + output_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        http_status=200,
    )


def source_manifest(
    mode: RunMode,
    *,
    runtime_metadata: dict | None,
    concurrency: int = 2,
    input_tokens: int | None = 256,
    output_tokens: int | None = 32,
    max_model_len: int = 512,
) -> RunManifest:
    device_type = DeviceType.GPU if mode is RunMode.GPU_ONLY else DeviceType.NPU
    role = "gpu" if device_type is DeviceType.GPU else "npu"
    configuration = {}
    if runtime_metadata is not None:
        configuration["runtime_metadata"] = runtime_metadata
    return RunManifest(
        run_id=f"source-{role}",
        mode=mode,
        profile_mode=ProfileMode.MONITOR,
        status=RunStatus.SUCCEEDED,
        created_at_unix_ns=1,
        models=[
            ModelDescriptor(
                role=role,
                model_id="served-model",
                revision="revision-1",
                tokenizer_id="tokenizer-1",
                dtype="bfloat16",
            )
        ],
        workload=WorkloadDescriptor(
            request_count=2,
            concurrency=concurrency,
            request_rate_per_s=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            max_model_len=max_model_len,
            warmup_requests=1,
        ),
        hosts=[
            HostDescriptor(
                host_id="host",
                role=role,
                hostname="host",
                operating_system="test",
                architecture="test",
            )
        ],
        software=[
            SoftwareDescriptor(
                name="server", version=None, role=f"{role}-server", path=None
            )
        ],
        devices=[
            DeviceDescriptor(
                host_id="host",
                device_type=device_type,
                device_id=f"{role}-0",
                vendor="test",
                model="test",
                status="available",
                memory_total_bytes=1,
            )
        ],
        configuration=configuration,
        attributes={
            "hybrid.source_role": role,
            "hybrid.transfer_role": (
                "kv_producer" if role == "gpu" else "kv_consumer"
            ),
            "hybrid.transfer_direction": "gpu_to_npu",
        },
    )


class RuntimeMetadataTests(unittest.TestCase):
    def test_config_defaults_actual_concurrency_and_keeps_capacity_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["runtime"]["max_num_seqs"] = 4
            path = root / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")

            config = load_hybrid_runner_config(path)

            self.assertEqual(config.workload.request_concurrency, 1)
            self.assertEqual(config.max_num_seqs, 4)

    def test_config_rejects_concurrency_above_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["workload"]["request_concurrency"] = 4
            value["runtime"]["max_num_seqs"] = 2
            path = root / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(HybridRunnerConfigError, "must not exceed"):
                load_hybrid_runner_config(path)

    def test_uniform_and_nonuniform_measured_tokens_are_truthful(self):
        uniform = measured_token_metadata(
            [observation("a", 256, 32), observation("b", 256, 32)]
        )
        self.assertEqual(uniform["input_tokens"]["value"], 256)
        self.assertEqual(uniform["output_tokens"]["value"], 32)
        self.assertTrue(uniform["input_tokens"]["uniform"])

        nonuniform = measured_token_metadata(
            [observation("a", 255, 31), observation("b", 256, 32)]
        )
        for field, minimum, maximum in (
            ("input_tokens", 255, 256),
            ("output_tokens", 31, 32),
        ):
            self.assertIsNone(nonuniform[field]["value"])
            self.assertFalse(nonuniform[field]["uniform"])
            self.assertEqual(nonuniform[field]["minimum"], minimum)
            self.assertEqual(nonuniform[field]["maximum"], maximum)
            self.assertEqual(nonuniform[field]["sample_count"], 2)

    def test_actual_concurrency_comes_from_measured_interval_overlap(self):
        sequential = [
            observation("a", 256, 8, received_ns=1),
            observation("b", 256, 8, received_ns=100),
        ]
        overlapping = [
            observation("a", 256, 8, received_ns=1),
            observation("b", 256, 8, received_ns=2),
        ]
        self.assertEqual(measured_request_concurrency(sequential), 1)
        self.assertEqual(measured_request_concurrency(overlapping), 2)
        self.assertIsNone(measured_request_concurrency([]))

    def test_declared_model_label_is_not_an_exact_or_path_inferred_count(self):
        metadata = model_identity_metadata(
            served_model_id="neutral-served-id",
            revision=None,
            tokenizer_id=None,
            dtype="bfloat16",
            size_label="0.5B",
            exact_parameter_count=None,
            metadata_origin="declared_config",
        )
        self.assertEqual(metadata["size_label"], "0.5B")
        self.assertEqual(metadata["size_label_origin"], "declared_config")
        self.assertIsNone(metadata["exact_parameter_count"])
        self.assertEqual(
            metadata["exact_parameter_count_status"], "not_available"
        )
        self.assertFalse(metadata["path_inference_used"])
        self.assertNotIn("model_path", metadata)

    def test_all_run_modes_have_explicit_non_layer_topology(self):
        expected = {
            RunMode.GPU_ONLY: ("all_gpu", "gpu", "gpu", "not_applicable"),
            RunMode.NPU_ONLY: ("all_npu", "npu", "npu", "not_applicable"),
            RunMode.HYBRID: (
                "gpu_prefill_npu_decode",
                "gpu",
                "npu",
                "gpu_to_npu",
            ),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode.value):
                metadata = topology_metadata(mode)
                self.assertEqual(metadata["partition_strategy"], values[0])
                self.assertEqual(metadata["prefill_device_type"], values[1])
                self.assertEqual(metadata["decode_device_type"], values[2])
                self.assertIsNone(metadata["layer_partition_ratio"])
                self.assertEqual(
                    metadata["layer_partition_ratio_status"], "not_applicable"
                )
                self.assertEqual(metadata["transfer_direction"], values[3])

    def test_transfer_dimensions_identify_gpu_producer_and_npu_consumer(self):
        self.assertEqual(
            transfer_dimensions(),
            {
                "transfer.source_device_type": "gpu",
                "transfer.destination_device_type": "npu",
                "transfer.direction": "gpu_to_npu",
                "transfer.producer_role": "kv_producer",
                "transfer.consumer_role": "kv_consumer",
            },
        )

    def test_merge_preserves_runtime_evidence_and_rejects_mismatch(self):
        tokens = measured_token_metadata(
            [observation("a", 256, 32), observation("b", 256, 32)]
        )
        identity = model_identity_metadata(
            served_model_id="served-model",
            revision="revision-1",
            tokenizer_id="tokenizer-1",
            dtype="bfloat16",
            size_label="declared-size",
            exact_parameter_count=None,
            metadata_origin="declared_config",
        )
        gpu_runtime = {
            "source_merge_contract": "truthful_runtime_metadata_v1",
            "model_identity": identity,
            "token_counts": tokens,
            "max_num_seqs_capacity": 4,
            "topology": topology_metadata(RunMode.GPU_ONLY),
        }
        npu_runtime = {
            **gpu_runtime,
            "topology": topology_metadata(RunMode.NPU_ONLY),
        }
        gpu = source_manifest(RunMode.GPU_ONLY, runtime_metadata=gpu_runtime)
        npu = source_manifest(RunMode.NPU_ONLY, runtime_metadata=npu_runtime)

        merged = merged_runtime_metadata(gpu, npu)

        assert merged is not None
        self.assertEqual(merged["token_counts"], tokens)
        self.assertEqual(merged["max_num_seqs_capacity"], 4)
        self.assertEqual(
            merged["topology"]["partition_strategy"],
            "gpu_prefill_npu_decode",
        )
        with self.assertRaisesRegex(RuntimeMetadataError, "actual concurrency"):
            merged_runtime_metadata(gpu, replace(npu, workload=replace(
                npu.workload, concurrency=1
            )))
        mismatched = {
            **npu_runtime,
            "model_identity": {**identity, "dtype": "float16"},
        }
        with self.assertRaisesRegex(RuntimeMetadataError, "model_identity"):
            merged_runtime_metadata(
                gpu,
                replace(
                    npu,
                    configuration={"runtime_metadata": mismatched},
                ),
            )

    def test_legacy_metadata_absence_is_compatible_but_partial_upgrade_fails(self):
        gpu = source_manifest(RunMode.GPU_ONLY, runtime_metadata=None)
        npu = source_manifest(RunMode.NPU_ONLY, runtime_metadata=None)
        self.assertIsNone(merged_runtime_metadata(gpu, npu))
        with self.assertRaisesRegex(RuntimeMetadataError, "both source"):
            merged_runtime_metadata(
                replace(
                    gpu,
                    configuration={
                        "runtime_metadata": {
                            "source_merge_contract": "truthful_runtime_metadata_v1"
                        }
                    },
                ),
                npu,
            )

    def test_runner_manifest_records_unavailable_hardware_and_software(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["runtime"]["max_num_seqs"] = 4
            value["model"].update(
                {
                    "revision": "revision-1",
                    "tokenizer_id": "tokenizer-1",
                    "dtype": "bfloat16",
                    "size_label": "declared-size",
                    "metadata_origin": "declared_config",
                }
            )
            path = root / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            runner = HybridRunner(
                load_hybrid_runner_config(path),
                run_root=root / "runs",
                run_id="metadata-test",
                profile_mode="monitor",
            )
            telemetry = SimpleNamespace(
                gpu=SimpleNamespace(discovered_rows=()),
                npu=SimpleNamespace(discovered_rows=()),
            )
            observations = [
                observation("a", 256, 32, received_ns=1),
                observation("b", 256, 32, received_ns=100),
            ]

            manifest = runner._manifest(
                role="gpu",
                status=RunStatus.SUCCEEDED,
                detailed=False,
                observations=observations,
                telemetry=telemetry,
            )

            validate_record(manifest)
            self.assertEqual(manifest.workload.concurrency, 1)
            self.assertEqual(manifest.workload.input_tokens, 256)
            self.assertEqual(manifest.workload.output_tokens, 32)
            self.assertEqual(manifest.configuration["max_num_seqs"], 4)
            self.assertEqual(manifest.devices[0].model, "not_available")
            self.assertEqual(manifest.devices[0].status, "not_available")
            availability = manifest.configuration["runtime_metadata"][
                "software_availability"
            ]
            self.assertEqual(
                availability["server_version"]["availability"], "not_available"
            )
            self.assertIsNone(manifest.software[0].version)

    def test_lifecycle_validation_and_schema_version_stability(self):
        lifecycle = lifecycle_metadata(
            started_at_unix_ns=10,
            finished_at_unix_ns=30,
            terminal_statuses=["succeeded", "succeeded"],
            measured_request_count=2,
            shutdown_integrity="valid",
            cleanup_status="complete",
        )
        self.assertEqual(lifecycle["duration_ns"], 20)
        self.assertEqual(lifecycle["terminal_status"], "succeeded")
        self.assertEqual(SCHEMA_VERSION, "1.0.0")
        with self.assertRaisesRegex(RuntimeMetadataError, "precedes"):
            lifecycle_metadata(
                started_at_unix_ns=30,
                finished_at_unix_ns=10,
                terminal_statuses=["failed"],
                measured_request_count=0,
                shutdown_integrity="invalid",
                cleanup_status="incomplete",
            )
        with self.assertRaisesRegex(RuntimeMetadataError, "status mismatch"):
            lifecycle_metadata(
                started_at_unix_ns=10,
                finished_at_unix_ns=30,
                terminal_statuses=["succeeded", "failed"],
                measured_request_count=2,
                shutdown_integrity="invalid",
                cleanup_status="complete",
            )


if __name__ == "__main__":
    unittest.main()
