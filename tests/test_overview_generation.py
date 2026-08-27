"""CPU-only integration tests for deterministic Overview publication."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from perfetto_hetero_profiler.cli import main
from perfetto_hetero_profiler.overview.bundle import (
    load_overview_bundle,
)
from perfetto_hetero_profiler.overview.generator import (
    OVERVIEW_HTML_NAME,
    OVERVIEW_JSON_NAME,
    OVERVIEW_VALIDATION_NAME,
    OverviewGenerationConfig,
    generate_overview,
    plan_overview_generation,
)
from perfetto_hetero_profiler.overview.loader import OverviewInputError
from perfetto_hetero_profiler.overview.publication import (
    OVERVIEW_OUTPUT_ROOT_ID,
    OverviewPublicationError,
)
from perfetto_hetero_profiler.overview.render import (
    render_overview_html,
    validate_offline_html,
)
from perfetto_hetero_profiler.overview.schema import (
    canonical_json_bytes as canonical_model_json_bytes,
    overview_report_from_dict,
    overview_to_dict,
)
from perfetto_hetero_profiler.overview.validation import OverviewValidationError
from tools.evaluation.overview import (
    COMPARISON_HTML_NAME,
    COMPARISON_JSON_NAME,
    COMPARISON_VALIDATION_NAME,
    OverviewComparisonConfig,
    build_comparison,
    build_comparison_validation,
    compare_overviews,
    load_comparison_bundle,
    overview_input_evidence,
    plan_overview_comparison,
    render_comparison_html,
)
from tools.evaluation.cli import main as evaluation_main
from perfetto_hetero_profiler.perfetto.artifacts import verify_stored_sidecar
from perfetto_hetero_profiler.perfetto.converter import (
    PerfettoConversionConfig,
    convert_perfetto,
)
from perfetto_hetero_profiler.perfetto.tooling import (
    TRACE_PROCESSOR_FILENAME,
    TRACE_PROCESSOR_RELEASE,
)
from perfetto_hetero_profiler.schema import (
    DETACHED_MANIFEST_NAME,
    DETACHED_VALIDATION_NAME,
)

from tests.test_perfetto_conversion import (
    _build_monitor_family,
    _tree_state,
)
from tests.test_overview_model_schema import report as schema_valid_report


_OVERVIEW_NAMES = {
    OVERVIEW_JSON_NAME,
    OVERVIEW_HTML_NAME,
    OVERVIEW_VALIDATION_NAME,
    DETACHED_MANIFEST_NAME,
    DETACHED_VALIDATION_NAME,
}
_COMPARISON_NAMES = {
    COMPARISON_JSON_NAME,
    COMPARISON_HTML_NAME,
    COMPARISON_VALIDATION_NAME,
    DETACHED_MANIFEST_NAME,
    DETACHED_VALIDATION_NAME,
}


def _trace_processor_path() -> Path:
    return (
        Path(sys.prefix)
        / "bin"
        / f"{TRACE_PROCESSOR_FILENAME}-{TRACE_PROCESSOR_RELEASE}"
    )


def _contents(root: Path, names: tuple[str, ...]) -> dict[str, bytes]:
    return {name: (root / name).read_bytes() for name in names}


def _canonical_key(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _mode_fixture(mode: str) -> dict[str, object]:
    """Return a schema-valid display fixture for one supported run mode."""

    value = overview_to_dict(schema_valid_report())
    value["run"]["mode"] = mode
    gpu_hardware = copy.deepcopy(value["hardware"][0])
    gpu_resource = copy.deepcopy(value["resources"][0])
    npu_hardware = {
        **copy.deepcopy(gpu_hardware),
        "device_type": "npu",
        "device_id": "npu-0",
        "vendor": "Rebellions",
        "model": "RBLN-CA22",
    }
    npu_resource = copy.deepcopy(gpu_resource)
    npu_resource["metric_name"] = "resource.npu.utilization"
    npu_resource["scope"].update(
        {"device_type": "npu", "device_id": "npu-0"}
    )
    for aggregate in npu_resource["aggregates"]:
        aggregate["name"] = aggregate["name"].replace(
            "resource.gpu.",
            "resource.npu.",
        )
        aggregate["scope"].update(
            {"device_type": "npu", "device_id": "npu-0"}
        )

    if mode == "gpu_only":
        value["hardware"] = [gpu_hardware]
        value["resources"] = [gpu_resource]
    elif mode == "npu_only":
        value["hardware"] = [npu_hardware]
        value["resources"] = [npu_resource]
    elif mode == "hybrid":
        value["hardware"] = sorted(
            [gpu_hardware, npu_hardware],
            key=_canonical_key,
        )
        value["resources"] = sorted(
            [gpu_resource, npu_resource],
            key=lambda item: (
                str(item["metric_name"]),
                _canonical_key(item["scope"]),
            ),
        )
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unsupported fixture mode: {mode}")
    return overview_to_dict(overview_report_from_dict(value))


class OverviewRunModeGoldenTests(unittest.TestCase):
    """Pin schema-valid GPU, NPU, and hybrid JSON/HTML representations."""

    _GOLDEN_SHA256 = {
        "gpu_only": (
            "fb374e6752ad45f69c375aa5da512ea6dc482d389791b30b3696494ccb7583a8",
            "930a7213c0cd9c8b447066d68788fcb5f7137225976a414dc7b5127d3471d9c3",
        ),
        "npu_only": (
            "945c972608355fe7dcc926e4493da9eb4ea028ce2766f6456d8b2e76d29ab9cc",
            "f6edba6da521e6350234961b995066acdf7da9f062454b994740e4565ba0c346",
        ),
        "hybrid": (
            "365c5ae8ce946d0d3d152f4d92177f6875ffab8bfe08fd7f3c4979323c88cfea",
            "dc016886569376208bee11b016ca0e830ed4c6514a1dc75988a389b515fbe193",
        ),
    }

    def test_run_mode_fixtures_are_schema_valid_offline_and_golden(self) -> None:
        for mode, expected in self._GOLDEN_SHA256.items():
            with self.subTest(mode=mode):
                fixture = _mode_fixture(mode)
                model = overview_report_from_dict(fixture)
                canonical = canonical_model_json_bytes(model)
                html = render_overview_html(fixture)
                validation = validate_offline_html(html)
                self.assertTrue(validation["valid"], validation["issues"])
                self.assertEqual(
                    (
                        hashlib.sha256(canonical).hexdigest(),
                        hashlib.sha256(html.encode("utf-8")).hexdigest(),
                    ),
                    expected,
                )


@unittest.skipUnless(
    _trace_processor_path().is_file(),
    "dedicated pinned Trace Processor binary is unavailable",
)
class OverviewGenerationIntegrationTests(unittest.TestCase):
    """Exercise generation against genuine synthetic conversion outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        family_a_root = cls.base / "family-a"
        family_b_root = cls.base / "family-b"
        family_a_root.mkdir()
        family_b_root.mkdir()
        cls.family_a = _build_monitor_family(
            family_a_root,
            overview_metrics=True,
            run_id="overview-synthetic-a",
            gpu_run_id="overview-gpu-a",
            npu_run_id="overview-npu-a",
        )
        cls.family_b = _build_monitor_family(
            family_b_root,
            overview_metrics=True,
            run_id="overview-synthetic-b",
            gpu_run_id="overview-gpu-b",
            npu_run_id="overview-npu-b",
        )
        cls.perfetto_a = cls.family_a["runs"] / "perfetto-a"
        cls.perfetto_b = cls.family_b["runs"] / "perfetto-b"
        for family, output in (
            (cls.family_a, cls.perfetto_a),
            (cls.family_b, cls.perfetto_b),
        ):
            result = convert_perfetto(
                PerfettoConversionConfig(
                    run_directory=family["hybrid"],
                    output_directory=output,
                    trace_processor_path=_trace_processor_path(),
                )
            )
            if result["status"] != "succeeded":
                raise AssertionError(f"synthetic Perfetto conversion failed: {result}")

        cls.input_overview_a = cls.base / "input-overview-a"
        cls.input_overview_b = cls.base / "input-overview-b"
        for family, perfetto, output in (
            (cls.family_a, cls.perfetto_a, cls.input_overview_a),
            (cls.family_b, cls.perfetto_b, cls.input_overview_b),
        ):
            result = generate_overview(
                OverviewGenerationConfig(
                    run_directory=family["hybrid"],
                    perfetto_directory=perfetto,
                    output_directory=output,
                    trace_processor_path=_trace_processor_path(),
                )
            )
            if result["status"] != "succeeded":
                raise AssertionError(f"synthetic Overview generation failed: {result}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _generation_config(
        self,
        output: Path,
        *,
        family: dict[str, Path] | None = None,
        perfetto: Path | None = None,
    ) -> OverviewGenerationConfig:
        selected = self.family_a if family is None else family
        return OverviewGenerationConfig(
            run_directory=selected["hybrid"],
            perfetto_directory=self.perfetto_a if perfetto is None else perfetto,
            output_directory=output,
            trace_processor_path=_trace_processor_path(),
        )

    def test_dry_run_is_deterministic_and_does_not_publish(self) -> None:
        output = self.base / "dry-run-overview"
        roots = (
            self.family_a["hybrid"],
            self.family_a["gpu"],
            self.family_a["npu"],
            self.family_a["coordinator"],
            self.family_a["recovery"],
            self.perfetto_a,
        )
        before = _tree_state(roots)
        first = plan_overview_generation(self._generation_config(output))
        second = plan_overview_generation(self._generation_config(output))

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "planned")
        self.assertTrue(first["dry_run"])
        self.assertTrue(first["validation_valid"])
        self.assertFalse(first["hardware_execution"])
        self.assertFalse(output.exists())
        self.assertEqual(_tree_state(roots), before)

    def test_generation_is_deterministic_exact_and_freshly_loadable(self) -> None:
        first_output = self.base / "overview-first"
        second_output = self.base / "overview-second"
        roots = (
            self.family_a["hybrid"],
            self.family_a["gpu"],
            self.family_a["npu"],
            self.family_a["coordinator"],
            self.family_a["recovery"],
            self.perfetto_a,
        )
        before = _tree_state(roots)
        first = generate_overview(self._generation_config(first_output))
        second = generate_overview(self._generation_config(second_output))

        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(second["status"], "succeeded")
        self.assertEqual(
            {path.name for path in first_output.iterdir()},
            _OVERVIEW_NAMES,
        )
        self.assertEqual(
            {path.name for path in second_output.iterdir()},
            _OVERVIEW_NAMES,
        )
        self.assertEqual(
            _contents(
                first_output,
                (
                    OVERVIEW_JSON_NAME,
                    OVERVIEW_HTML_NAME,
                    OVERVIEW_VALIDATION_NAME,
                ),
            ),
            _contents(
                second_output,
                (
                    OVERVIEW_JSON_NAME,
                    OVERVIEW_HTML_NAME,
                    OVERVIEW_VALIDATION_NAME,
                ),
            ),
        )
        loaded = load_overview_bundle(first_output)
        self.assertEqual(loaded.run_id, "overview-synthetic-a")
        self.assertTrue(loaded.validation["valid"])
        self.assertTrue(loaded.artifact_validation["valid"])
        self.assertEqual(loaded.artifact_validation["mismatches"], [])
        self.assertEqual(_tree_state(roots), before)

    def test_overwrite_overlap_symlink_and_failed_staging_are_safe(self) -> None:
        existing = self.base / "existing-overview"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("keep\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            generate_overview(self._generation_config(existing))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

        with self.assertRaises(OverviewPublicationError):
            plan_overview_generation(
                self._generation_config(
                    self.family_a["hybrid"] / "nested-overview"
                )
            )

        real_parent = self.base / "real-output-parent"
        real_parent.mkdir()
        linked_parent = self.base / "linked-output-parent"
        os.symlink(real_parent, linked_parent)
        with self.assertRaisesRegex(OverviewPublicationError, "symlink"):
            plan_overview_generation(
                self._generation_config(linked_parent / "overview")
            )

        failed = self.base / "failed-overview"
        with mock.patch(
            "perfetto_hetero_profiler.overview.generator._generation_input_check",
            side_effect=RuntimeError("synthetic immutable-input failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                generate_overview(self._generation_config(failed))
        self.assertFalse(failed.exists())
        self.assertEqual(
            list(self.base.glob(".failed-overview.overview-staging-*")),
            [],
        )

    def test_input_ancestor_symlinks_are_rejected(self) -> None:
        linked_runs = self.base / "linked-runs"
        os.symlink(self.family_a["runs"], linked_runs)
        linked_run = linked_runs / self.family_a["hybrid"].name
        with self.assertRaisesRegex(OverviewInputError, "symlink"):
            plan_overview_generation(
                OverviewGenerationConfig(
                    run_directory=linked_run,
                    perfetto_directory=self.perfetto_a,
                    output_directory=self.base / "ancestor-linked-run-output",
                    trace_processor_path=_trace_processor_path(),
                )
            )

        linked_perfetto_parent = self.base / "linked-perfetto-parent"
        os.symlink(self.perfetto_a.parent, linked_perfetto_parent)
        linked_perfetto = linked_perfetto_parent / self.perfetto_a.name
        with self.assertRaisesRegex(OverviewInputError, "symlink"):
            plan_overview_generation(
                OverviewGenerationConfig(
                    run_directory=self.family_a["hybrid"],
                    perfetto_directory=linked_perfetto,
                    output_directory=self.base / "ancestor-linked-perfetto-output",
                    trace_processor_path=_trace_processor_path(),
                )
            )

        linked_overview_parent = self.base / "linked-overview-parent"
        os.symlink(self.input_overview_a.parent, linked_overview_parent)
        linked_overview = linked_overview_parent / self.input_overview_a.name
        with self.assertRaisesRegex(OverviewInputError, "symlink"):
            plan_overview_comparison(
                OverviewComparisonConfig(
                    input_directories=(linked_overview, self.input_overview_b),
                    output_directory=self.base / "ancestor-linked-compare-output",
                )
            )

    def test_omitted_trace_processor_uses_only_local_pinned_binary(self) -> None:
        output = self.base / "implicit-trace-processor"
        with mock.patch(
            "perfetto.trace_processor.platform.PlatformDelegate.get_shell_path",
            autospec=True,
            return_value=str(_trace_processor_path()),
        ) as resolver:
            plan = plan_overview_generation(
                OverviewGenerationConfig(
                    run_directory=self.family_a["hybrid"],
                    perfetto_directory=self.perfetto_a,
                    output_directory=output,
                    trace_processor_path=None,
                )
            )
        self.assertEqual(plan["status"], "planned")
        self.assertFalse(output.exists())
        self.assertGreater(resolver.call_count, 0)
        for call in resolver.call_args_list:
            self.assertEqual(
                call.kwargs,
                {
                    "bin_path": str(_trace_processor_path()),
                    "fetch_latest": False,
                },
            )

    def test_semantic_validation_failure_does_not_publish_or_leave_staging(
        self,
    ) -> None:
        output = self.base / "semantic-failure-overview"
        with mock.patch(
            "perfetto_hetero_profiler.overview.generator.build_overview_validation",
            side_effect=RuntimeError("synthetic semantic validation failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "semantic"):
                generate_overview(self._generation_config(output))
        self.assertFalse(output.exists())
        self.assertEqual(
            list(self.base.glob(".semantic-failure-overview.overview-staging-*")),
            [],
        )

    def test_corrupted_published_bundle_fails_fresh_loading(self) -> None:
        corrupted = self.base / "corrupted-overview"
        shutil.copytree(self.input_overview_a, corrupted)
        with (corrupted / OVERVIEW_JSON_NAME).open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(" ")
        with self.assertRaises(OverviewInputError):
            load_overview_bundle(corrupted)

    def test_generate_cli_dry_run_success_and_failure_exit_codes(self) -> None:
        dry_output = self.base / "cli-overview-dry"
        output = self.base / "cli-overview"
        base_args = [
            "overview",
            "generate",
            "--run",
            str(self.family_a["hybrid"]),
            "--perfetto",
            str(self.perfetto_a),
            "--trace-processor",
            str(_trace_processor_path()),
        ]

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(
                [
                    *base_args,
                    "--output",
                    str(dry_output),
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["status"], "planned")
        self.assertFalse(dry_output.exists())

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([*base_args, "--output", str(output)])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["status"], "succeeded")
        self.assertEqual(
            {path.name for path in output.iterdir()},
            _OVERVIEW_NAMES,
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([*base_args, "--output", str(output)])
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("overview error:", stderr.getvalue())

    def test_comparison_dry_run_publication_and_determinism(self) -> None:
        dry_output = self.base / "comparison-dry"
        first_output = self.base / "comparison-first"
        second_output = self.base / "comparison-second"

        dry_config = OverviewComparisonConfig(
            input_directories=(
                self.input_overview_b,
                self.input_overview_a,
            ),
            output_directory=dry_output,
            baseline_run_id="overview-synthetic-a",
        )
        first_plan = plan_overview_comparison(dry_config)
        second_plan = plan_overview_comparison(dry_config)
        self.assertEqual(first_plan, second_plan)
        self.assertEqual(first_plan["status"], "planned")
        self.assertEqual(
            first_plan["run_ids"],
            ["overview-synthetic-a", "overview-synthetic-b"],
        )
        self.assertFalse(dry_output.exists())

        common = {
            "input_directories": (
                self.input_overview_b,
                self.input_overview_a,
            ),
            "baseline_run_id": "overview-synthetic-a",
        }
        first = compare_overviews(
            OverviewComparisonConfig(
                **common,
                output_directory=first_output,
            )
        )
        second = compare_overviews(
            OverviewComparisonConfig(
                **common,
                output_directory=second_output,
            )
        )
        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(second["status"], "succeeded")
        self.assertEqual(
            {path.name for path in first_output.iterdir()},
            _COMPARISON_NAMES,
        )
        self.assertEqual(
            {path.name for path in second_output.iterdir()},
            _COMPARISON_NAMES,
        )
        self.assertEqual(
            _contents(
                first_output,
                (
                    COMPARISON_JSON_NAME,
                    COMPARISON_HTML_NAME,
                    COMPARISON_VALIDATION_NAME,
                ),
            ),
            _contents(
                second_output,
                (
                    COMPARISON_JSON_NAME,
                    COMPARISON_HTML_NAME,
                    COMPARISON_VALIDATION_NAME,
                ),
            ),
        )
        fresh = verify_stored_sidecar(
            first_output / DETACHED_MANIFEST_NAME,
            {OVERVIEW_OUTPUT_ROOT_ID: first_output},
            output_root_id=OVERVIEW_OUTPUT_ROOT_ID,
        )
        self.assertTrue(fresh["valid"])
        self.assertEqual(fresh["mismatches"], [])
        loaded = load_comparison_bundle(first_output)
        self.assertTrue(loaded.validation["valid"])
        self.assertTrue(loaded.artifact_validation["valid"])
        self.assertEqual(
            [item["run_id"] for item in loaded.comparison["runs"]],
            ["overview-synthetic-a", "overview-synthetic-b"],
        )

    def test_compare_cli_and_invalid_inputs(self) -> None:
        dry_output = self.base / "cli-comparison-dry"
        output = self.base / "cli-comparison"
        base_args = [
            "overview",
            "compare",
            "--input",
            str(self.input_overview_b),
            "--input",
            str(self.input_overview_a),
            "--baseline",
            "overview-synthetic-a",
        ]

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = evaluation_main(
                [
                    *base_args,
                    "--output",
                    str(dry_output),
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["status"], "planned")
        self.assertFalse(dry_output.exists())

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = evaluation_main([*base_args, "--output", str(output)])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["status"], "succeeded")
        self.assertEqual(
            {path.name for path in output.iterdir()},
            _COMPARISON_NAMES,
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = evaluation_main(
                [
                    "overview",
                    "compare",
                    "--input",
                    str(self.input_overview_a),
                    "--output",
                    str(self.base / "invalid-one-input"),
                ]
            )
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("evaluation error:", stderr.getvalue())

        with self.assertRaisesRegex(RuntimeError, "unique"):
            plan_overview_comparison(
                OverviewComparisonConfig(
                    input_directories=(
                        self.input_overview_a,
                        self.input_overview_a,
                    ),
                    output_directory=self.base / "duplicate-comparison",
                )
            )

        linked = self.base / "linked-overview-input"
        os.symlink(self.input_overview_a, linked)
        with self.assertRaisesRegex(OverviewInputError, "symlink"):
            plan_overview_comparison(
                OverviewComparisonConfig(
                    input_directories=(linked, self.input_overview_b),
                    output_directory=self.base / "linked-input-comparison",
                )
            )

    def test_comparison_validation_rejects_input_evidence_hash_mismatch(
        self,
    ) -> None:
        inputs = [
            load_overview_bundle(self.input_overview_a),
            load_overview_bundle(self.input_overview_b),
        ]
        comparison = build_comparison(
            [item.report for item in inputs],
            baseline_run_id="overview-synthetic-a",
        )
        html_validation = validate_offline_html(
            render_comparison_html(comparison)
        )
        evidence = [
            copy.deepcopy(overview_input_evidence(item)) for item in inputs
        ]
        evidence[0]["overview_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            OverviewValidationError,
            "mismatch",
        ):
            build_comparison_validation(
                comparison,
                input_evidence=evidence,
                html_validation=html_validation,
            )


if __name__ == "__main__":
    unittest.main()
