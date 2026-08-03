"""Schema CLI tests."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from perfetto_hetero_profiler.cli import main


class SchemaCliTests(unittest.TestCase):
    def run_cli(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_schema_version(self) -> None:
        exit_code, stdout, stderr = self.run_cli(["schema", "version"])
        self.assertEqual((exit_code, stdout.strip(), stderr), (0, "1.0.0", ""))

    def test_schema_list(self) -> None:
        exit_code, stdout, _ = self.run_cli(["schema", "list"])
        self.assertEqual(exit_code, 0)
        self.assertIn("run_manifest", stdout.splitlines())
        self.assertIn("clock_transform", stdout.splitlines())

    def test_valid_json_exit_zero(self) -> None:
        exit_code, stdout, stderr = self.run_cli(
            ["schema", "validate", "examples/schema_v1/manifest_hybrid.json"]
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("1 record", stdout)
        self.assertEqual(stderr, "")

    def test_valid_jsonl_exit_zero(self) -> None:
        exit_code, stdout, _ = self.run_cli(
            ["schema", "validate", "examples/schema_v1/events.jsonl"]
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("14 records", stdout)

    def test_invalid_json_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"schema_version":"1.0.0"}\n', encoding="utf-8")
            exit_code, _, stderr = self.run_cli(["schema", "validate", str(path)])
        self.assertEqual(exit_code, 2)
        self.assertIn("record_type", stderr)

    def test_jsonl_error_includes_line_number(self) -> None:
        valid = Path("examples/schema_v1/events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(valid + "\n{}\n", encoding="utf-8")
            exit_code, _, stderr = self.run_cli(["schema", "validate", str(path)])
        self.assertEqual(exit_code, 2)
        self.assertIn("line 2", stderr)

    def test_unsupported_extension_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.txt"
            path.write_text("{}\n", encoding="utf-8")
            exit_code, _, stderr = self.run_cli(["schema", "validate", str(path)])
        self.assertEqual(exit_code, 2)
        self.assertIn(".json or .jsonl", stderr)
