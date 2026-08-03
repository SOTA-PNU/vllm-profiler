"""CPU-only safety tests for transactional Overview publication."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.overview.publication import (
    OverviewPublicationError,
    canonical_json_bytes,
    publish_bundle,
    validate_output_path,
)
from perfetto_hetero_profiler.perfetto.artifacts import verify_stored_sidecar


class OverviewPublicationTests(unittest.TestCase):
    def _payloads(self) -> dict[str, bytes]:
        return {
            "overview.json": canonical_json_bytes(
                {"record_type": "overview_report", "value": 0}
            ),
            "overview.html": b"<!doctype html><title>Overview</title>\n",
            "overview_validation.json": canonical_json_bytes(
                {"valid": True, "mismatches": []}
            ),
        }

    def test_publishes_exact_five_file_detached_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = validate_output_path(
                root / "result",
                immutable_roots=(source,),
            )
            result = publish_bundle(output, payloads=self._payloads())
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [
                    "artifact_manifest.json",
                    "artifact_manifest_validation.json",
                    "overview.html",
                    "overview.json",
                    "overview_validation.json",
                ],
            )
            manifest = json.loads(
                (output / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            inventoried = {
                item["relative_path"] for item in manifest["artifacts"]
            }
            self.assertEqual(
                inventoried,
                {
                    "overview.html",
                    "overview.json",
                    "overview_validation.json",
                },
            )
            self.assertNotIn("artifact_manifest.json", inventoried)
            self.assertNotIn("artifact_manifest_validation.json", inventoried)
            fresh = verify_stored_sidecar(
                output / "artifact_manifest.json",
                {"overview": output},
                output_root_id="overview",
            )
            self.assertTrue(fresh["valid"])
            self.assertEqual(fresh["mismatches"], [])
            self.assertEqual(len(result["files"]), 5)

    def test_existing_empty_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "result"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                validate_output_path(output, immutable_roots=(source,))
            self.assertEqual(list(output.iterdir()), [])

    def test_source_output_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            with self.assertRaises(OverviewPublicationError):
                validate_output_path(
                    source / "result",
                    immutable_roots=(source,),
                )

    def test_symlink_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(OverviewPublicationError):
                validate_output_path(
                    link / "result",
                    immutable_roots=(),
                )

    def test_failure_before_publish_removes_only_owned_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = validate_output_path(
                root / "result",
                immutable_roots=(source,),
            )

            def fail() -> None:
                raise RuntimeError("input changed")

            with self.assertRaisesRegex(RuntimeError, "input changed"):
                publish_bundle(
                    output,
                    payloads=self._payloads(),
                    before_publish=fail,
                )
            self.assertFalse(output.exists())
            self.assertEqual(
                [
                    path.name
                    for path in root.iterdir()
                    if path.name.startswith(".result.overview-staging-")
                ],
                [],
            )
            self.assertTrue(source.is_dir())

    def test_publish_race_preserves_competing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = validate_output_path(
                root / "result",
                immutable_roots=(source,),
            )

            def race() -> None:
                output.mkdir()
                (output / "owner.txt").write_text("other\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                publish_bundle(
                    output,
                    payloads=self._payloads(),
                    before_publish=race,
                )
            self.assertEqual(
                (output / "owner.txt").read_text(encoding="utf-8"),
                "other\n",
            )

    def test_canonical_json_rejects_non_finite_numbers(self):
        with self.assertRaises(ValueError):
            canonical_json_bytes({"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
