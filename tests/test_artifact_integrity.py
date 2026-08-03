"""Detached closeout artifact-integrity regression tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.schema import (
    DETACHED_MANIFEST_NAME,
    DETACHED_VALIDATION_NAME,
    RECOVERY_RESULT_NAME,
    RECOVERY_ROOT_ID,
    ArtifactIntegrityError,
    create_detached_recovery,
    validate_detached_artifact_manifest,
)


def fingerprint(path: Path) -> tuple[int, str, int]:
    data = path.read_bytes()
    return (
        len(data),
        hashlib.sha256(data).hexdigest(),
        path.stat().st_mtime_ns,
    )


def write_source(root: Path, name: str = "artifact.json") -> Path:
    root.mkdir()
    path = root / name
    path.write_text('{"value":1}\n', encoding="utf-8")
    return path


def recovery_result() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "record_type": "closeout_recovery_result",
        "source_run_id": "source-run",
        "success": True,
        "hardware_rerun": False,
        "postprocess_only": True,
    }


class DetachedArtifactIntegrityTests(unittest.TestCase):
    def test_recovery_result_is_immutable_and_validation_is_detached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = write_source(root / "source")
            source_before = fingerprint(source_file)
            output, manifest, first = create_detached_recovery(
                root / "closeout",
                {"source": source_file.parent},
                recovery_result(),
                required_artifacts=(("source", source_file.name),),
            )
            recovery_path = output / RECOVERY_RESULT_NAME
            recovery_before = fingerprint(recovery_path)
            manifest_path = output / DETACHED_MANIFEST_NAME
            report_path = output / DETACHED_VALIDATION_NAME
            report_mtime = report_path.stat().st_mtime_ns
            roots = {
                "source": source_file.parent,
                RECOVERY_ROOT_ID: output,
            }
            second = validate_detached_artifact_manifest(
                manifest_path,
                roots,
                report_path=report_path,
            )
            third = validate_detached_artifact_manifest(
                manifest_path,
                roots,
                report_path=report_path,
            )

            self.assertTrue(first["valid"])
            self.assertEqual(first, second)
            self.assertEqual(second, third)
            self.assertEqual(fingerprint(source_file), source_before)
            self.assertEqual(fingerprint(recovery_path), recovery_before)
            self.assertEqual(report_path.stat().st_mtime_ns, report_mtime)
            self.assertNotIn("artifact_manifest_validation", recovery_path.read_text())
            entries = {
                (item["root_id"], item["relative_path"])
                for item in manifest["artifacts"]
            }
            self.assertIn((RECOVERY_ROOT_ID, RECOVERY_RESULT_NAME), entries)
            self.assertNotIn((RECOVERY_ROOT_ID, DETACHED_MANIFEST_NAME), entries)
            self.assertNotIn((RECOVERY_ROOT_ID, DETACHED_VALIDATION_NAME), entries)
            self.assertEqual(
                manifest["artifact_count"],
                first["checked"],
            )

    def test_changed_artifact_is_detected_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = write_source(root / "source")
            output, _, _ = create_detached_recovery(
                root / "closeout",
                {"source": source_file.parent},
                recovery_result(),
                required_artifacts=(("source", source_file.name),),
            )
            source_file.write_text('{"value":2}\n', encoding="utf-8")
            report = validate_detached_artifact_manifest(
                output / DETACHED_MANIFEST_NAME,
                {"source": source_file.parent, RECOVERY_ROOT_ID: output},
            )
            self.assertFalse(report["valid"])
            self.assertEqual(report["mismatches"][0]["reason"], "changed")
            self.assertIn("sha256", report["mismatches"][0]["fields"])

    def test_missing_and_unexpected_artifacts_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = write_source(root / "source")
            output, _, _ = create_detached_recovery(
                root / "closeout",
                {"source": source_file.parent},
                recovery_result(),
                required_artifacts=(("source", source_file.name),),
            )
            source_file.unlink()
            (source_file.parent / "unexpected.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            report = validate_detached_artifact_manifest(
                output / DETACHED_MANIFEST_NAME,
                {"source": source_file.parent, RECOVERY_ROOT_ID: output},
            )
            self.assertFalse(report["valid"])
            self.assertEqual(
                {item["reason"] for item in report["mismatches"]},
                {"missing", "unexpected"},
            )

    def test_output_collision_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = write_source(root / "source")
            output = root / "closeout"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                create_detached_recovery(
                    output,
                    {"source": source_file.parent},
                    recovery_result(),
                    required_artifacts=(("source", source_file.name),),
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

    def test_missing_required_artifact_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = write_source(root / "source")
            output = root / "closeout"
            with self.assertRaisesRegex(
                ArtifactIntegrityError,
                "required artifact is missing",
            ):
                create_detached_recovery(
                    output,
                    {"source": source_file.parent},
                    recovery_result(),
                    required_artifacts=(("source", "missing.json"),),
                )
            self.assertFalse(output.exists())

    def test_failed_or_hardware_recovery_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = write_source(root / "source")
            for field, value, pattern in (
                ("success", False, "failed recovery"),
                ("hardware_rerun", True, "metadata-only"),
            ):
                with self.subTest(field=field):
                    payload = recovery_result()
                    payload[field] = value
                    output = root / f"closeout-{field}"
                    with self.assertRaisesRegex(ArtifactIntegrityError, pattern):
                        create_detached_recovery(
                            output,
                            {"source": source_file.parent},
                            payload,
                        )
                    self.assertFalse(output.exists())

    def test_output_cannot_overlap_a_source_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = write_source(root / "source")
            with self.assertRaisesRegex(ArtifactIntegrityError, "outside"):
                create_detached_recovery(
                    source_file.parent / "closeout",
                    {"source": source_file.parent},
                    recovery_result(),
                )

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            target = root / "target.txt"
            target.write_text("target\n", encoding="utf-8")
            os.symlink(target, source / "linked.txt")
            with self.assertRaisesRegex(ArtifactIntegrityError, "symlinks"):
                create_detached_recovery(
                    root / "closeout",
                    {"source": source},
                    recovery_result(),
                )

    def test_unsafe_and_duplicate_manifest_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = write_source(root / "source")
            output, _, _ = create_detached_recovery(
                root / "closeout",
                {"source": source_file.parent},
                recovery_result(),
            )
            manifest_path = output / DETACHED_MANIFEST_NAME
            roots = {"source": source_file.parent, RECOVERY_ROOT_ID: output}
            original = json.loads(manifest_path.read_text(encoding="utf-8"))
            for mutation, pattern in (
                (
                    lambda value: value["artifacts"][0].update(
                        {"relative_path": "."}
                    ),
                    "safe relative path",
                ),
                (
                    lambda value: value["artifacts"][0].update(
                        {"relative_path": "../escape"}
                    ),
                    "safe relative path",
                ),
                (
                    lambda value: value["artifacts"].append(
                        dict(value["artifacts"][0])
                    ),
                    "duplicate artifact",
                ),
            ):
                with self.subTest(pattern=pattern):
                    value = json.loads(json.dumps(original))
                    mutation(value)
                    value["artifact_count"] = len(value["artifacts"])
                    manifest_path.write_text(
                        json.dumps(value, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ArtifactIntegrityError, pattern):
                        validate_detached_artifact_manifest(
                            manifest_path,
                            roots,
                        )


if __name__ == "__main__":
    unittest.main()
