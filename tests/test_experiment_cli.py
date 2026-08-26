from __future__ import annotations

import contextlib
import io
import unittest

from perfetto_hetero_profiler.cli import build_parser as build_core_parser
from tools.evaluation.cli import build_parser
from tools.evaluation.compatibility import LEGACY_CLI_COMMAND


class ExperimentCliTests(unittest.TestCase):
    def test_command_and_subcommand_help(self):
        parser = build_parser()
        for arguments in (
            ["--help"],
            ["run", "--help"],
            ["status", "--help"],
            ["validate", "--help"],
            ["report", "--help"],
            ["overview", "compare", "--help"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as caught:
                with contextlib.redirect_stdout(io.StringIO()):
                    parser.parse_args(arguments)
            self.assertEqual(caught.exception.code, 0)

    def test_development_stage_command_is_rejected(self):
        parser = build_core_parser()
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args([LEGACY_CLI_COMMAND])
        self.assertEqual(caught.exception.code, 2)

    def test_installed_cli_does_not_register_evaluation(self):
        parser = build_core_parser()
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["experiment"])
        self.assertEqual(caught.exception.code, 2)

    def test_installed_cli_rejects_overview_comparison(self):
        parser = build_core_parser()
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["overview", "compare"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
