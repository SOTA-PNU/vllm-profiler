"""Run layout and artifact path safety tests."""

from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.schema import (
    ArtifactKind,
    ArtifactReference,
    RunPaths,
    SchemaValidationError,
    validate_record,
)


def artifact(path: str) -> ArtifactReference:
    return ArtifactReference(
        run_id="run-1",
        artifact_id="artifact-1",
        artifact_kind=ArtifactKind.RAW_LOG,
        relative_path=path,
        format="text",
        producer="test",
        created_at_unix_ns=1,
        attributes={},
    )


class RunPathTests(unittest.TestCase):
    def test_path_computation_does_not_create(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = RunPaths(Path(directory) / "runs", "run-1")
            self.assertEqual(paths.manifest, paths.root / "manifest.json")
            self.assertFalse(paths.root.exists())

    def test_parent_run_id_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "run_id"):
            RunPaths(Path("runs"), "../escape")

    def test_absolute_run_id_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "run_id"):
            RunPaths(Path("runs"), "/escape")

    def test_artifact_parent_escape_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "relative_path"):
            validate_record(artifact("../outside.log"))

    def test_artifact_absolute_path_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "relative_path"):
            validate_record(artifact("/tmp/outside.log"))

    def test_valid_artifact_relative_path(self) -> None:
        validate_record(artifact("raw/gpu/telemetry.jsonl"))

    def test_create_builds_reserved_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = RunPaths(Path(directory) / "runs", "run-1")
            paths.create()
            for path in (
                paths.events.parent,
                paths.metrics.parent,
                paths.clock_domains.parent,
                paths.perfetto_trace.parent,
                paths.overview.parent,
                paths.root / "raw" / "gpu",
                paths.root / "raw" / "npu",
            ):
                self.assertTrue(path.is_dir())

    def test_nonempty_run_reuse_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = RunPaths(Path(directory) / "runs", "run-1")
            paths.create()
            paths.manifest.write_text("occupied", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                paths.create()
