"""Run-manifest publication shared by device collectors."""

from __future__ import annotations

from pathlib import Path
import uuid

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
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    write_json(temporary, manifest)
    temporary.replace(output)
