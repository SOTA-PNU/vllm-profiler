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

Hybrid decode additionally uses the vendor P/D contract: device tensors are enabled,
the source GPU is visible to UCX, layout reorder is fixed to zero, and the RBLN runtime
addresses are loopback. NPU-only validation keeps its separate non-P/D launcher.

The campaign publishes one path-free `environment.json`. Every block publishes
`condition_metadata.json`, references the campaign environment by relative path and
SHA-256, and receives a `condition_validation.json` success record only after request,
wave, core-manifest, token, topology, and terminal-cleanup evidence agree. Unsupported
hardware or software fields are explicit `not_available` records with reasons. These
formal condition concepts remain repository-only and are excluded from distributions.

Run these commands from the evaluation worktree. Dry-run and preflight do not create
the campaign directory. Preflight queries device state but starts no server and sends
no inference request.

```bash
cd /home/yewon/perfetto-hetero-profiler/profiler-formal-experiment-metadata

CONFIG=/home/yewon/perfetto-hetero-profiler/profiler-formal-experiment-metadata/tools/evaluation/examples/final_ofat_campaign.json
RESULT=/home/yewon/perfetto-hetero-profiler/experiment-data/active/model-batch-topology/runs/formal-ofat-next
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
