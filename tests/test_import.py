"""Package import tests."""

import unittest

import perfetto_hetero_profiler


class ImportTests(unittest.TestCase):
    def test_package_version(self) -> None:
        self.assertEqual(perfetto_hetero_profiler.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
