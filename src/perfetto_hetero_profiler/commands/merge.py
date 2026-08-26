"""Hybrid bundle merge command parser and handler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..hybrid import (
    AlignmentMethod,
    HybridBundleMerger,
    HybridMergeConfig,
    build_hybrid_plan,
)
from ..schema import RunStatus


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "merge", help="Merge immutable normalized source bundles."
    )
    parser.set_defaults(merge_parser=parser)
    commands = parser.add_subparsers(dest="merge_target")
    hybrid = commands.add_parser(
        "hybrid", help="Align and merge fake GPU/NPU source bundles."
    )
    hybrid.add_argument("--run-root", type=Path, required=True)
    hybrid.add_argument("--run-id", required=True)
    hybrid.add_argument("--gpu-run", type=Path, required=True)
    hybrid.add_argument("--npu-run", type=Path, required=True)
    hybrid.add_argument(
        "--alignment-method",
        choices=tuple(method.value for method in AlignmentMethod),
        default=AlignmentMethod.SAME_CLOCK_DOMAIN.value,
    )
    hybrid.add_argument("--max-uncertainty-ns", type=int, default=1_000_000)
    hybrid.add_argument("--host-id", default="hybrid-coordinator")
    hybrid.add_argument("--probe-count", type=int, default=7)
    hybrid.add_argument("--minimum-probe-samples", type=int, default=5)
    hybrid.add_argument("--fake-offset-ns", type=int, default=0)
    hybrid.add_argument("--fake-delay-ns", type=int, default=100_000)
    hybrid.add_argument("--fake-jitter-ns", type=int, default=0)
    hybrid.add_argument("--fake-asymmetry-ns", type=int, default=0)
    hybrid.add_argument("--dry-run", action="store_true")


def handle(args: argparse.Namespace, _: argparse.ArgumentParser) -> int:
    if args.merge_target != "hybrid":
        args.merge_parser.print_help()
        return 0
    try:
        config = HybridMergeConfig(
            run_root=args.run_root,
            run_id=args.run_id,
            gpu_run=args.gpu_run,
            npu_run=args.npu_run,
            alignment_method=AlignmentMethod(args.alignment_method),
            max_uncertainty_ns=args.max_uncertainty_ns,
            coordinator_host_id=args.host_id,
            probe_count=args.probe_count,
            minimum_probe_samples=args.minimum_probe_samples,
            fake_offset_ns=args.fake_offset_ns,
            fake_delay_ns=args.fake_delay_ns,
            fake_jitter_ns=args.fake_jitter_ns,
            fake_asymmetry_ns=args.fake_asymmetry_ns,
        )
        if args.dry_run:
            print(
                json.dumps(
                    build_hybrid_plan(config),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = HybridBundleMerger(config).merge()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"merge error: {error}", file=sys.stderr)
        return 2
    print(
        f"{result.status.value}: {result.run_directory} "
        f"(joined={result.joined_request_count}, "
        f"unjoined={result.unjoined_request_count}, "
        f"uncertainty_ns={result.uncertainty_ns}, "
        f"reasons={'; '.join(result.reasons) or 'none'})"
    )
    return 0 if result.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL} else 1
