"""Phase 0 CLI tests."""

import contextlib
import io
import unittest

from perfetto_hetero_profiler.cli import main


class CliTests(unittest.TestCase):
    def test_help(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(SystemExit, "0"):
                main(["--help"])
        self.assertIn("hetero-profiler", output.getvalue())

    def test_version(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["version"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().strip(), "hetero-profiler 0.1.0")


if __name__ == "__main__":
    unittest.main()
