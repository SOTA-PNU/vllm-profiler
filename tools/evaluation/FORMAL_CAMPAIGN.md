# Fixed 21-block formal campaign

This repository-only command binds the qualified model snapshots and five persistent
caches to the predeclared 21-block OFAT schedule. It uses 256 input tokens, exactly 32
output tokens, streaming, `ignore_eos=true`, `temperature=0`, BF16,
`enable_prefix_caching=false`, 1 Hz telemetry, and zero automatic retries.

The NPU launcher unsets compile-only and strict mode, sets
`VLLM_RBLN_REQUIRE_CACHE_HIT=1`, and uses the qualified `vllm-rbln` worktree. Startup
is rejected unless the persistent graphs are cache hits and the only compile markers
are the expected non-persistent sampler graphs. Cache fingerprints are compared before
and after every NPU-using block.

Run these commands from the evaluation worktree. Dry-run and preflight do not create
the campaign directory. Preflight queries device state but starts no server and sends
no inference request.

```bash
cd /home/yewon/perfetto-hetero-profiler/profiler-concurrent-wave-evaluation

CONFIG=/home/yewon/perfetto-hetero-profiler/profiler-concurrent-wave-evaluation/tools/evaluation/examples/final_ofat_campaign.json
RESULT=/home/yewon/perfetto-hetero-profiler/formal-ofat-20260913
PYTHON=/home/yewon/perfetto-hetero-profiler/.venvs/perfetto-tools/bin/python

PYTHONPATH=src:. "$PYTHON" -m tools.evaluation formal-campaign \
  --config "$CONFIG" --campaign-root "$RESULT" --dry-run

PYTHONPATH=src:. "$PYTHON" -m tools.evaluation formal-campaign \
  --config "$CONFIG" --campaign-root "$RESULT" --preflight-only

PYTHONPATH=src:. "$PYTHON" -m tools.evaluation formal-campaign \
  --config "$CONFIG" --campaign-root "$RESULT"
```

The final command is the only hardware-running command. It starts one block at a time,
uses the fixed order, and stops at the first failure. There is no automatic retry.
`--resume` skips only blocks with a completed success sentinel; it refuses to rerun an
incomplete or failed block. Use a new result path for any separately approved rerun.
