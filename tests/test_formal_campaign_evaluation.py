"""CPU-only checks for the fixed formal-campaign evaluation contract."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.evaluation.concurrent_wave import BlockSpec
from tools.evaluation.formal_campaign import (
    FormalCampaignError,
    _compile_evidence,
    _hybrid_config,
    _server_args,
    load_config,
    plan,
    run_campaign,
)
from tools.evaluation.formal_client import FormalStreamingClient


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/evaluation/examples/final_ofat_campaign.json"


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        chunk = {
            "choices": [{"token_ids": list(range(32)), "text": "SECRET"}],
            "usage": {"prompt_tokens": 256, "completion_tokens": 32,
                      "total_tokens": 288},
        }
        return iter([
            ("data: " + json.dumps(chunk) + "\n").encode(),
            b"data: [DONE]\n",
        ])


class FormalClientTests(unittest.TestCase):
    def test_exact_request_contract_and_text_free_observation(self):
        captured = []

        def opener(request, **_kwargs):
            captured.append(json.loads(request.data))
            return _Response()

        ticks = iter(range(1000, 2000))
        observation = FormalStreamingClient(
            "http://127.0.0.1:1", timeout_sec=1,
            monotonic_ns=lambda: next(ticks), opener=opener,
        ).complete(
            model="model", request_id="request-1", prompt="TOP SECRET",
            max_output_tokens=32, temperature=0, stream=True,
        )
        self.assertEqual(captured[0]["max_tokens"], 32)
        self.assertTrue(captured[0]["ignore_eos"])
        self.assertTrue(captured[0]["stream"])
        self.assertEqual(observation.input_tokens, 256)
        self.assertEqual(observation.output_tokens, 32)
        self.assertNotIn("SECRET", repr(observation))


class CompileGateTests(unittest.TestCase):
    def _evidence(self, concurrency: int, *, persistent=False):
        expected = 5 if concurrency == 1 else 10
        stdout = (
            "cache_policy=persistent_cache_hit_required\n" * 2
            + "cache_policy=non_persistent_compile_allowed\n" * expected
            + "Persistent RBLN cache-hit guard enabled: env=1 active=true mode=non-strict\n" * 2
        )
        markers = list(range(2, expected + 2))
        if persistent:
            markers.insert(0, 0)
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        (root / "stdout").write_text(stdout, encoding="utf-8")
        (root / "stderr").write_text(
            "\n".join(f"Compile(#{item})" for item in markers), encoding="utf-8"
        )
        return directory, _compile_evidence(root / "stdout", root / "stderr", concurrency)

    def test_accepts_only_exact_sampler_sequence(self):
        for concurrency in (1, 2, 4):
            with self.subTest(concurrency=concurrency):
                directory, evidence = self._evidence(concurrency)
                self.addCleanup(directory.cleanup)
                self.assertTrue(evidence["valid"])
                self.assertEqual(evidence["persistent_compile_count"], 0)

    def test_rejects_persistent_compile(self):
        directory, evidence = self._evidence(1, persistent=True)
        self.addCleanup(directory.cleanup)
        self.assertFalse(evidence["valid"])
        self.assertEqual(evidence["persistent_compile_count"], 1)


class CampaignTests(unittest.TestCase):
    def test_fixed_plan_has_all_21_blocks_in_declared_order(self):
        result = plan(load_config(CONFIG), Path("/unused"))
        self.assertEqual(result["block_count"], 21)
        self.assertEqual(result["blocks"][0]["condition_id"], "m3-b1-hybrid")
        self.assertEqual(result["blocks"][-1]["condition_id"], "m15-b1-npu")
        self.assertEqual(
            [sum(row["topology"] == name for row in result["blocks"])
             for name in ("hybrid", "gpu", "npu")],
            [15, 3, 3],
        )

    def test_server_contract_locks_memory_prefix_cache_and_guard_launcher(self):
        config = load_config(CONFIG)
        block = BlockSpec("m15-b1-npu", 1, 1, "npu", 1, 52, 52, 256, 32)
        argv = _server_args(config, block, "npu")
        self.assertIn("--no-enable-prefix-caching", argv)
        self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.92")
        hybrid = _hybrid_config(
            config,
            BlockSpec("m15-b1-hybrid", 1, 1, "hybrid", 1, 52, 52, 256, 32),
        )
        self.assertEqual(hybrid.gpu_memory_utilization, 0.2)
        self.assertIn("--gpu-memory-utilization", hybrid.decode.extra_args)
        launcher = config.npu_vllm_launcher.read_text(encoding="utf-8")
        self.assertIn("unset VLLM_RBLN_COMPILE_ONLY", launcher)
        self.assertIn("unset VLLM_RBLN_COMPILE_STRICT_MODE", launcher)
        self.assertIn("VLLM_RBLN_REQUIRE_CACHE_HIT=1", launcher)

    def test_first_failure_stops_campaign_and_resume_refuses_retry(self):
        config = load_config(CONFIG)
        blocks = (
            BlockSpec("m15-b1-gpu", 1, 1, "gpu", 1, 1, 1),
            BlockSpec("m15-b1-npu", 1, 1, "npu", 1, 1, 1),
        )
        calls = []

        def fake_block(_config, block, output):
            calls.append(block.condition_id)
            output.mkdir(parents=True)
            if block.topology == "npu":
                raise FormalCampaignError("injected failure")
            return {"status": "succeeded"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            with (
                patch("tools.evaluation.formal_campaign.preflight", return_value={"valid": True}),
                patch("tools.evaluation.formal_campaign._runtime_postflight", return_value={"valid": True}),
                patch("tools.evaluation.formal_campaign.load_matrix_blocks", return_value=blocks),
                patch("tools.evaluation.formal_campaign._standalone_block", side_effect=fake_block),
            ):
                with self.assertRaisesRegex(FormalCampaignError, "injected"):
                    run_campaign(config, root)
                self.assertEqual(calls, ["m15-b1-gpu", "m15-b1-npu"])
                self.assertEqual(json.loads((root / "status.json").read_text())["failed_position"], 2)
                with self.assertRaisesRegex(FormalCampaignError, "incomplete block"):
                    run_campaign(config, root, resume=True)


if __name__ == "__main__":
    unittest.main()
