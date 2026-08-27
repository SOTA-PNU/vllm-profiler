"""Package import tests."""

from importlib.util import find_spec
import unittest

import perfetto_hetero_profiler
from perfetto_hetero_profiler import gpu, npu
from perfetto_hetero_profiler.gpu.vllm_collection import (
    GpuVllmCollectionConfig,
    GpuVllmCollectionRunner,
)
from perfetto_hetero_profiler.npu.runtime_collection import (
    NpuRuntimeCollectionConfig,
    NpuRuntimeCollectionRunner,
)


class ImportTests(unittest.TestCase):
    def test_package_version(self) -> None:
        self.assertEqual(perfetto_hetero_profiler.__version__, "0.1.0")

    def test_collection_modules_and_public_names_replace_old_names(self) -> None:
        self.assertIsNotNone(
            find_spec("perfetto_hetero_profiler.gpu.vllm_collection")
        )
        self.assertIsNotNone(
            find_spec("perfetto_hetero_profiler.npu.runtime_collection")
        )
        self.assertIsNone(find_spec("perfetto_hetero_profiler.gpu.smoke"))
        self.assertIsNone(
            find_spec("perfetto_hetero_profiler.npu.runtime_smoke")
        )
        self.assertIs(gpu.GpuVllmCollectionConfig, GpuVllmCollectionConfig)
        self.assertIs(gpu.GpuVllmCollectionRunner, GpuVllmCollectionRunner)
        self.assertIs(npu.NpuRuntimeCollectionConfig, NpuRuntimeCollectionConfig)
        self.assertIs(npu.NpuRuntimeCollectionRunner, NpuRuntimeCollectionRunner)
        for module, names in (
            (gpu, ("GpuVllmSmokeConfig", "GpuVllmSmokeRunner")),
            (npu, ("NpuRuntimeSmokeConfig", "NpuRuntimeSmokeRunner")),
        ):
            for name in names:
                with self.subTest(module=module.__name__, name=name):
                    self.assertFalse(hasattr(module, name))


if __name__ == "__main__":
    unittest.main()
