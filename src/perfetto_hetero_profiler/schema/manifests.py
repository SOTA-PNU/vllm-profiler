"""Run-manifest publication shared by device collectors."""

from __future__ import annotations

import os
from pathlib import Path

from .records import RunManifest
from .serialization import write_json


def publish_run_manifest(
    path: str | Path,
    manifest: RunManifest,
    *,
    initial: bool = False,
) -> None:
    """Create an initial manifest or atomically replace its running state."""

    output = Path(path)
    if initial:
        write_json(output, manifest)
        return
    temporary = output.with_name(f".{output.name}.tmp")
    write_json(temporary, manifest)
    os.replace(temporary, output)
