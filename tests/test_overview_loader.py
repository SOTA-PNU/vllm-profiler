"""Integration coverage for strict Overview input loading."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest import mock

from perfetto_hetero_profiler.overview.loader import (
    OverviewInputError,
    _exact_perfetto_files,
    _expected_query_count,
    assert_perfetto_unchanged,
    load_matching_perfetto,
    normalized_identity,
    perfetto_identity,
    phase_duration_reconciliation,
    reconciliation_summary,
)
from perfetto_hetero_profiler.perfetto.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_VALIDATION_NAME,
)
from perfetto_hetero_profiler.perfetto.converter import (
    CONVERSION_MANIFEST_NAME,
    TRACE_NAME,
    TRACE_ATTRIBUTE_VALIDATION_NAME,
    TRACE_VALIDATION_NAME,
    PerfettoConversionConfig,
    convert_perfetto,
)
from perfetto_hetero_profiler.perfetto.loader import load_hybrid_run
from perfetto_hetero_profiler.perfetto.timeline_summary import (
    TIMELINE_SUMMARY_MAPPING_VERSION,
)

from tests.test_perfetto_conversion import (
    _build_monitor_family,
    _trace_processor_path,
    _tree_state,
)


_PERFETTO_FILES = {
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_VALIDATION_NAME,
    CONVERSION_MANIFEST_NAME,
    TRACE_NAME,
    TRACE_VALIDATION_NAME,
    TRACE_ATTRIBUTE_VALIDATION_NAME,
}


class QueryInventoryTests(unittest.TestCase):
    def test_native_details_add_exactly_one_semantics_query(self) -> None:
        manifest = {
            "counts": {
                "native_detail_slice_count": 10,
                "native_detail_instant_count": 2,
            }
        }
        self.assertEqual(
            _expected_query_count(TIMELINE_SUMMARY_MAPPING_VERSION, manifest), 16
        )

    def test_separate_unaligned_native_trace_does_not_add_query(self) -> None:
        manifest = {
            "counts": {
                "native_detail_slice_count": 0,
                "native_detail_instant_count": 0,
                "separate_native_trace_count": 1,
            }
        }
        self.assertEqual(
            _expected_query_count(TIMELINE_SUMMARY_MAPPING_VERSION, manifest), 15
        )

    def test_exact_rbln_native_pair_is_an_allowed_bundle_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "artifact_manifest.json",
                "artifact_manifest_validation.json",
                "conversion_manifest.json",
                "trace.pftrace",
                "trace_validation.json",
                "trace.rbln-native.pftrace",
                "trace.rbln-native.validation.json",
            ):
                (root / name).write_bytes(b"test")
            self.assertEqual(len(_exact_perfetto_files(root)), 7)

    def test_partial_rbln_native_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "artifact_manifest.json",
                "artifact_manifest_validation.json",
                "conversion_manifest.json",
                "trace.pftrace",
                "trace_validation.json",
                "trace.rbln-native.pftrace",
            ):
                (root / name).write_bytes(b"test")
            with self.assertRaisesRegex(OverviewInputError, "exactly"):
                _exact_perfetto_files(root)


def _socket_creation_available() -> bool:
    """Detect managed sandboxes that prohibit Trace Processor's TCP socket."""

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return False
    probe.close()
    return True


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class PerfettoIdentityFilesystemTests(unittest.TestCase):
    """Filesystem rejection paths do not need to launch Trace Processor."""

    def _dummy_bundle(self, root: Path) -> None:
        root.mkdir()
        for index, name in enumerate(sorted(_PERFETTO_FILES)):
            (root / name).write_bytes(f"fixture-{index}\n".encode())

    def test_exact_current_file_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "perfetto"
            self._dummy_bundle(root)
            identity = perfetto_identity(root)
            self.assertEqual(len(identity.files), len(_PERFETTO_FILES))
            self.assertEqual(
                {item.relative_path for item in identity.files},
                _PERFETTO_FILES,
            )

            (root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(OverviewInputError, "exactly"):
                perfetto_identity(root)

    def test_missing_file_and_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            missing = base / "missing"
            self._dummy_bundle(missing)
            (missing / TRACE_VALIDATION_NAME).unlink()
            with self.assertRaisesRegex(OverviewInputError, "missing"):
                perfetto_identity(missing)

            real = base / "real"
            self._dummy_bundle(real)
            linked_root = base / "linked-root"
            os.symlink(real, linked_root)
            with self.assertRaisesRegex(OverviewInputError, "symlink"):
                perfetto_identity(linked_root)

            linked_file = base / "linked-file"
            shutil.copytree(real, linked_file, copy_function=shutil.copy2)
            trace = linked_file / TRACE_NAME
            trace.unlink()
            os.symlink(real / TRACE_NAME, trace)
            with self.assertRaisesRegex(OverviewInputError, "real regular file"):
                perfetto_identity(linked_file)


@unittest.skipUnless(
    _trace_processor_path().is_file() and _socket_creation_available(),
    "pinned Trace Processor or local TCP socket creation is unavailable",
)
class OverviewLoaderIntegrationTests(unittest.TestCase):
    """Official Trace Processor reconciliation against a generated bundle."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._fixture_directory = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls._fixture_directory.name)
        cls.family = _build_monitor_family(cls.fixture_root)
        cls.input_roots = (
            cls.family["hybrid"],
            cls.family["gpu"],
            cls.family["npu"],
            cls.family["coordinator"],
            cls.family["recovery"],
        )
        cls.output = cls.family["runs"] / "overview-loader-perfetto"
        cls.loaded = load_hybrid_run(cls.family["hybrid"])
        cls.normalized_identity_before = normalized_identity(cls.loaded)
        cls.source_state_before = _tree_state(cls.input_roots)

        result = convert_perfetto(
            PerfettoConversionConfig(
                run_directory=cls.family["hybrid"],
                output_directory=cls.output,
                trace_processor_path=_trace_processor_path(),
            )
        )
        if result["status"] != "succeeded":
            raise AssertionError(f"fixture conversion failed: {result}")
        cls.source_state_after_conversion = _tree_state(cls.input_roots)
        cls.perfetto_state_before = _tree_state((cls.output,))
        cls.perfetto_identity_before = perfetto_identity(cls.output)
        cls.bundle = load_matching_perfetto(
            cls.loaded,
            cls.output,
            trace_processor_path=_trace_processor_path(),
        )
        cls.perfetto_state_after = _tree_state((cls.output,))
        cls.source_state_after = _tree_state(cls.input_roots)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._fixture_directory.cleanup()
        super().tearDownClass()

    def setUp(self) -> None:
        self._case_directory = tempfile.TemporaryDirectory()
        self.case_root = Path(self._case_directory.name)

    def tearDown(self) -> None:
        self._case_directory.cleanup()

    def _copy_output(self, name: str) -> Path:
        destination = self.case_root / name
        shutil.copytree(
            self.output,
            destination,
            copy_function=shutil.copy2,
        )
        return destination

    def test_matching_source_and_perfetto_succeeds(self) -> None:
        bundle = self.bundle
        self.assertEqual(bundle.root, self.output)
        self.assertEqual(
            bundle.stored_trace_validation,
            bundle.fresh_trace_validation,
        )
        self.assertEqual(bundle.identity, self.perfetto_identity_before)
        self.assertEqual(len(bundle.identity.files), len(_PERFETTO_FILES))
        self.assertEqual(
            {item.relative_path for item in bundle.identity.files},
            _PERFETTO_FILES,
        )

        summary = reconciliation_summary(bundle)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["query_count"], 15)
        self.assertEqual(summary["mismatches"], [])
        self.assertTrue(all(query["matched"] for query in summary["queries"]))
        self.assertTrue(summary["artifact_validation"]["valid"])

    def test_source_and_perfetto_identity_are_unchanged(self) -> None:
        self.assertEqual(
            self.source_state_before,
            self.source_state_after_conversion,
        )
        self.assertEqual(self.source_state_before, self.source_state_after)
        self.assertEqual(
            self.perfetto_state_before,
            self.perfetto_state_after,
        )
        reloaded = load_hybrid_run(self.family["hybrid"])
        self.assertEqual(
            normalized_identity(reloaded),
            self.normalized_identity_before,
        )
        self.assertEqual(
            perfetto_identity(self.output),
            self.perfetto_identity_before,
        )
        assert_perfetto_unchanged(self.bundle, self.bundle)

    def test_phase_duration_reconciliation_matches_integer_ns(self) -> None:
        values = phase_duration_reconciliation(self.bundle)
        self.assertEqual(
            [item["kpi_name"] for item in values],
            [
                "latency.e2e",
                "latency.prefill",
                "latency.kv_export",
                "latency.kv_transfer",
                "latency.kv_transform",
                "latency.decode",
                "latency.sampling",
            ],
        )
        self.assertTrue(all(item["matched"] for item in values))
        self.assertTrue(all(item["slice_count"] == 1 for item in values))
        for item in values:
            self.assertIsInstance(item["event_duration_ns"], int)
            self.assertEqual(
                item["event_duration_ns"],
                item["perfetto_duration_ns"],
            )
            self.assertGreater(item["event_duration_ns"], 0)

    def test_exact_five_files_are_required_for_matching_load(self) -> None:
        extra = self._copy_output("extra")
        (extra / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(OverviewInputError, "exactly"):
            load_matching_perfetto(
                self.loaded,
                extra,
                trace_processor_path=_trace_processor_path(),
            )

        missing = self._copy_output("missing")
        (missing / ARTIFACT_VALIDATION_NAME).unlink()
        with self.assertRaisesRegex(OverviewInputError, "missing"):
            load_matching_perfetto(
                self.loaded,
                missing,
                trace_processor_path=_trace_processor_path(),
            )

    def test_wrong_run_and_source_fingerprint_are_rejected(self) -> None:
        wrong_run = self._copy_output("wrong-run")
        manifest_path = wrong_run / CONVERSION_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_id"] = "different-run"
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(OverviewInputError, "run_id"):
            load_matching_perfetto(
                self.loaded,
                wrong_run,
                trace_processor_path=_trace_processor_path(),
            )

        with tempfile.TemporaryDirectory() as other_directory:
            other_family = _build_monitor_family(Path(other_directory))
            other_loaded = load_hybrid_run(other_family["hybrid"])
            self.assertNotEqual(
                normalized_identity(other_loaded),
                self.normalized_identity_before,
            )
            with self.assertRaisesRegex(OverviewInputError, "input_validation"):
                load_matching_perfetto(
                    other_loaded,
                    self.output,
                    trace_processor_path=_trace_processor_path(),
                )

    def test_trace_tamper_and_perfetto_symlinks_are_rejected(self) -> None:
        tampered = self._copy_output("tampered")
        with (tampered / TRACE_NAME).open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(OverviewInputError, "size/SHA-256"):
            load_matching_perfetto(
                self.loaded,
                tampered,
                trace_processor_path=_trace_processor_path(),
            )

        linked_root = self.case_root / "linked-root"
        os.symlink(self.output, linked_root)
        with self.assertRaisesRegex(OverviewInputError, "symlink"):
            load_matching_perfetto(
                self.loaded,
                linked_root,
                trace_processor_path=_trace_processor_path(),
            )

        linked_file = self._copy_output("linked-file")
        trace = linked_file / TRACE_NAME
        trace.unlink()
        os.symlink(self.output / TRACE_NAME, trace)
        with self.assertRaisesRegex(OverviewInputError, "real regular file"):
            load_matching_perfetto(
                self.loaded,
                linked_file,
                trace_processor_path=_trace_processor_path(),
            )

    def test_stored_and_fresh_validation_mismatches_are_rejected(self) -> None:
        invalid_stored = self._copy_output("invalid-stored")
        validation_path = invalid_stored / TRACE_VALIDATION_NAME
        stored = json.loads(validation_path.read_text(encoding="utf-8"))
        stored["run_id"] = "different-run"
        _write_json(validation_path, stored)
        with self.assertRaisesRegex(
            OverviewInputError, "stored Perfetto trace validation run mismatch"
        ):
            load_matching_perfetto(
                self.loaded,
                invalid_stored,
                trace_processor_path=_trace_processor_path(),
            )

        fresh_mismatch = self._copy_output("fresh-mismatch")
        fresh = copy.deepcopy(self.bundle.stored_trace_validation)
        fresh["counts"]["slices"] += 1
        with mock.patch(
            "perfetto_hetero_profiler.overview.loader.validate_trace",
            return_value=fresh,
        ):
            with self.assertRaisesRegex(
                OverviewInputError, "fresh official Trace Processor result"
            ):
                load_matching_perfetto(
                    self.loaded,
                    fresh_mismatch,
                    trace_processor_path=_trace_processor_path(),
                )


if __name__ == "__main__":
    unittest.main()
