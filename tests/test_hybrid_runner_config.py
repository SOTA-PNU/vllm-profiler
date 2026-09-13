"""CPU-only contract tests for the reusable hybrid runner."""

import contextlib
import io
from importlib import resources
import json
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.hybrid.layout import (
    COLLECTION_RESULT_NAME,
    FINAL_RESULT_NAME,
    HybridRunLayout,
    existing_collection_result_path,
    existing_final_result_path,
    existing_related_run_root,
    related_run_root,
)
from perfetto_hetero_profiler.hybrid.runner import build_hybrid_run_plan
from perfetto_hetero_profiler.hybrid.runner_config import (
    HYBRID_RUNNER_CONFIG_SCHEMA_NAME,
    HybridRunnerConfigError,
    load_hybrid_runner_config,
    validate_hybrid_invocation,
)


from tests.support.runner_fakes import document


class HybridRunnerConfigTests(unittest.TestCase):
    def load(self, root: Path, value: dict | None = None):
        path = root / "config.json"
        path.write_text(json.dumps(value or document(root)), encoding="utf-8")
        return load_hybrid_runner_config(path)

    def test_grouped_run_layout_has_one_top_level_directory(self) -> None:
        layout = HybridRunLayout(Path("/runs"), "example")
        self.assertEqual(layout.bundle, Path("/runs/example"))
        self.assertEqual(
            layout.all_roots,
            (
                Path("/runs/example/hybrid"),
                Path("/runs/example/sources/gpu"),
                Path("/runs/example/sources/npu"),
                Path("/runs/example/coordinator"),
                Path("/runs/example/perfetto"),
                Path("/runs/example/overview"),
                Path("/runs/example/recovery"),
                Path("/runs/example/publication"),
            ),
        )
        self.assertTrue(all(path.is_relative_to(layout.bundle) for path in layout.all_roots))

    def test_related_roots_support_grouped_and_legacy_layouts(self) -> None:
        self.assertEqual(
            related_run_root(Path("/runs/example/hybrid"), "example", "gpu"),
            Path("/runs/example/sources/gpu"),
        )
        self.assertEqual(
            related_run_root(Path("/runs/example"), "example", "gpu"),
            Path("/runs/example-gpu"),
        )

    def test_existing_perfetto_root_resolves_both_grouped_layout_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            historical = HybridRunLayout(runs, "old")
            historical.hybrid.mkdir(parents=True)
            old_full = historical.bundle / "perfetto/full"
            old_focused = historical.bundle / "perfetto/request-focused"
            old_full.mkdir(parents=True)
            old_focused.mkdir()
            (old_full / "trace.pftrace").write_bytes(b"full")
            (old_focused / "trace.request-focused.pftrace").write_bytes(
                b"focused"
            )
            self.assertEqual(
                existing_related_run_root(
                    historical.hybrid, "old", "perfetto"
                ),
                old_full,
            )
            self.assertEqual(
                existing_related_run_root(
                    historical.hybrid, "old", "request_perfetto"
                ),
                old_focused,
            )

            current = HybridRunLayout(runs, "new")
            current.hybrid.mkdir(parents=True)
            current.perfetto.mkdir()
            (current.perfetto / "trace.pftrace").write_bytes(b"full")
            (current.perfetto / "trace.request-focused.pftrace").write_bytes(
                b"focused"
            )
            self.assertEqual(
                existing_related_run_root(
                    current.hybrid, "new", "perfetto"
                ),
                current.perfetto,
            )
            self.assertEqual(
                existing_related_run_root(
                    current.hybrid, "new", "request_perfetto"
                ),
                current.perfetto,
            )

    def test_result_paths_prefer_canonical_names_and_accept_legacy_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = HybridRunLayout(Path(directory), "run")
            layout.coordinator.mkdir(parents=True)
            layout.publication.mkdir()
            legacy_collection = layout.coordinator / "result.json"
            legacy_final = layout.publication / "result.json"
            legacy_collection.write_text("{}\n", encoding="utf-8")
            legacy_final.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                existing_collection_result_path(layout.coordinator),
                legacy_collection,
            )
            self.assertEqual(
                existing_final_result_path(layout.publication), legacy_final
            )

            canonical_collection = layout.coordinator / COLLECTION_RESULT_NAME
            canonical_final = layout.publication / FINAL_RESULT_NAME
            canonical_collection.write_text("{}\n", encoding="utf-8")
            canonical_final.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                existing_collection_result_path(layout.coordinator),
                canonical_collection,
            )
            self.assertEqual(
                existing_final_result_path(layout.publication), canonical_final
            )

    def test_valid_config_and_plan_are_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            runs = root / "runs"
            plan = build_hybrid_run_plan(
                config, run_root=runs, run_id="example", profile_mode="monitor"
            )
            self.assertFalse(plan["executes"])
            self.assertFalse(runs.exists())
            self.assertEqual(
                plan["outputs"]["hybrid"], str(runs / "example/hybrid")
            )
            self.assertEqual(
                plan["outputs"]["gpu_source"],
                str(runs / "example/sources/gpu"),
            )
            self.assertEqual(
                plan["outputs"]["request_focused_perfetto"],
                str(runs / "example/perfetto"),
            )

    def test_versioned_schema_is_packaged_and_structural_corpus_matches(self) -> None:
        schema = (
            resources.files("perfetto_hetero_profiler.hybrid")
            .joinpath("json", "v1", HYBRID_RUNNER_CONFIG_SCHEMA_NAME)
        )
        self.assertEqual(json.loads(schema.read_text())["type"], "object")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = document(root)
            prompt_file = document(root)
            prompt_file["workload"]["prompt"] = None
            prompt_file["workload"]["prompt_file"] = str(root / "prompt.txt")
            cases = [
                ("valid", valid, True),
                ("explicit-null-xor", prompt_file, True),
                ("unknown", {**valid, "unknown": True}, False),
            ]
            missing = document(root)
            del missing["runtime"]["max_model_len"]
            cases.append(("missing", missing, False))
            wrong_type = document(root)
            wrong_type["runtime"]["max_num_seqs"] = True
            cases.append(("bool-integer", wrong_type, False))
            duplicate = document(root)
            duplicate["runtime"]["gpu_indices"] = [0, 0]
            cases.append(("duplicate-index", duplicate, False))
            for name, value, accepted in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.json"
                    path.write_text(json.dumps(value), encoding="utf-8")
                    if accepted:
                        load_hybrid_runner_config(path)
                    else:
                        with self.assertRaises(HybridRunnerConfigError):
                            load_hybrid_runner_config(path)

    def test_whitespace_only_strings_restore_the_previous_contract(self) -> None:
        fields = (
            ("workload-prompt", ("workload", "prompt")),
            ("workload-prompt-file", ("workload", "prompt_file")),
            ("model-path", ("model", "path")),
            ("model-served-name", ("model", "served_name")),
            ("model-cache", ("model", "rbln_cache_path")),
            ("prefill-executable", ("prefill", "executable")),
            ("prefill-working-directory", ("prefill", "working_directory")),
            ("prefill-pythonpath", ("prefill", "pythonpath")),
            ("proxy-python", ("proxy", "python")),
            ("gpu-torch-output", ("profilers", "gpu_torch_subdir")),
            ("gpu-nsys-output", ("profilers", "gpu_nsys_basename")),
            ("npu-torch-output", ("profilers", "npu_torch_subdir")),
            ("npu-rbln-output", ("profilers", "npu_rbln_subdir")),
            ("trace-processor", ("tools", "trace_processor")),
            ("nsys", ("tools", "nsys")),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, (section, field) in fields:
                with self.subTest(name=name):
                    value = document(root)
                    if field == "prompt_file":
                        value["workload"]["prompt"] = None
                    value[section][field] = " \t\n"
                    with self.assertRaises(HybridRunnerConfigError):
                        self.load(root, value)

    def test_non_whitespace_content_and_empty_extra_arg_remain_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["model"]["served_name"] = "  example model  "
            value["model"]["path"] = str(root / "model with spaces")
            value["workload"]["prompt"] = "\t Explain cache briefly. \n"
            value["profilers"]["gpu_torch_subdir"] = "raw/gpu/torch profile "
            value["prefill"]["extra_args"] = [""]

            config = self.load(root, value)

            self.assertEqual(config.served_model_name, "  example model  ")
            self.assertEqual(config.workload.prompt, "\t Explain cache briefly. \n")
            self.assertEqual(config.prefill.extra_args, ("",))

    def test_nonfinite_json_constants_are_rejected_before_schema_validation(self) -> None:
        replacements = (
            ("NaN", '"temperature": 0', '"temperature": NaN'),
            (
                "Infinity",
                '"gpu_memory_utilization": 0.2',
                '"gpu_memory_utilization": Infinity',
            ),
            (
                "-Infinity",
                '"startup_sec": 300',
                '"startup_sec": -Infinity',
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = json.dumps(document(root))
            for constant, before, after in replacements:
                with self.subTest(constant=constant):
                    path = root / f"{constant}.json"
                    path.write_text(original.replace(before, after), encoding="utf-8")
                    with self.assertRaises(HybridRunnerConfigError) as raised:
                        load_hybrid_runner_config(path)
                    self.assertEqual(
                        str(raised.exception),
                        "cannot read config: non-finite numeric constants are not valid JSON",
                    )
                    self.assertNotIn(str(root), str(raised.exception))

    def test_unknown_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["unexpected"] = True
            with self.assertRaisesRegex(HybridRunnerConfigError, "unknown config"):
                self.load(root, value)

    def test_relative_config_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(HybridRunnerConfigError, "--config"):
            load_hybrid_runner_config(Path("config.json"))

    def test_prompt_and_prompt_file_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["workload"]["prompt_file"] = str(root / "prompt.txt")
            with self.assertRaisesRegex(HybridRunnerConfigError, "exactly one"):
                self.load(root, value)

    def test_online_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["offline"] = False
            with self.assertRaisesRegex(HybridRunnerConfigError, "offline"):
                self.load(root, value)

    def test_duplicate_and_colliding_ports_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["decode"]["http_port"] = value["prefill"]["http_port"]
            with self.assertRaisesRegex(HybridRunnerConfigError, "ports must differ"):
                self.load(root, value)

    def test_twenty_millisecond_telemetry_interval_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = document(root)
            value["telemetry"]["sample_interval_ms"] = 20
            self.assertEqual(self.load(root, value).sample_interval_ms, 20)

            value["telemetry"]["sample_interval_ms"] = 19
            with self.assertRaisesRegex(
                HybridRunnerConfigError, "telemetry.sample_interval_ms"
            ):
                self.load(root, value)

    def test_existing_output_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            runs = root / "runs"
            target = runs / "same"
            target.mkdir(parents=True)
            marker = target / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                validate_hybrid_invocation(
                    config, run_root=runs, run_id="same", profile_mode="monitor"
                )
            self.assertEqual(marker.read_text(), "keep")

    def test_legacy_sibling_output_reserves_the_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            runs = root / "runs"
            legacy = runs / "same-gpu"
            legacy.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                validate_hybrid_invocation(
                    config, run_root=runs, run_id="same", profile_mode="monitor"
                )
            self.assertTrue(legacy.is_dir())

    def test_symlink_run_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            actual = root / "actual"
            actual.mkdir()
            linked = root / "linked"
            linked.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(HybridRunnerConfigError, "symlink"):
                validate_hybrid_invocation(
                    config, run_root=linked, run_id="run", profile_mode="monitor"
                )

    def test_broken_symlink_output_is_rejected_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            runs = root / "runs"
            runs.mkdir()
            target = runs / "same"
            target.symlink_to(root / "missing-output", target_is_directory=True)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                validate_hybrid_invocation(
                    config,
                    run_root=runs,
                    run_id="same",
                    profile_mode="monitor",
                )

            self.assertTrue(target.is_symlink())

    def test_cli_dry_run_uses_overrides_and_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(document(root)), encoding="utf-8")
            runs = root / "runs"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "collect", "hybrid", "--config", str(config_path),
                        "--run-root", str(runs), "--run-id", "dry-run",
                        "--profile-mode", "gpu-torch", "--warmup-requests", "0",
                        "--measured-requests", "1", "--max-output-tokens", "4",
                        "--prompt", "override", "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            value = json.loads(output.getvalue())
            self.assertEqual(value["profile_mode"], "gpu-torch")
            self.assertEqual(value["workload"]["warmup_requests"], 0)
            self.assertFalse(runs.exists())

    def test_all_modes_enable_only_one_profiler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(root)
            for mode in ("monitor", "gpu-torch", "gpu-nsys", "npu-torch", "npu-rbln"):
                plan = build_hybrid_run_plan(
                    config, run_root=root / "runs", run_id=mode, profile_mode=mode
                )
                prefill = plan["commands"]["prefill"]
                decode = plan["commands"]["decode"]
                torch_count = sum("profiler=torch" in arg for arg in [*prefill, *decode])
                rbln_count = int(mode == "npu-rbln")
                nsys_count = int(prefill[0] == str(config.nsys_executable))
                self.assertLessEqual(torch_count + rbln_count + nsys_count, 1)


if __name__ == "__main__":
    unittest.main()
