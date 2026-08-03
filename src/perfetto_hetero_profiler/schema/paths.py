"""Safe path helpers for the run artifact layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .validation import validate_run_id


@dataclass(frozen=True)
class RunPaths:
    """Compute or explicitly create paths for one run."""

    runs_root: Path
    run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs_root", Path(self.runs_root))
        validate_run_id(self.run_id)

    @property
    def root(self) -> Path:
        return self.runs_root / self.run_id

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def clock_domains(self) -> Path:
        return self.root / "clocks" / "clock_domains.jsonl"

    @property
    def sync_points(self) -> Path:
        return self.root / "clocks" / "sync_points.jsonl"

    @property
    def transforms(self) -> Path:
        return self.root / "clocks" / "transforms.jsonl"

    @property
    def events(self) -> Path:
        return self.root / "events" / "events.jsonl"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics" / "metrics.jsonl"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts" / "artifacts.jsonl"

    @property
    def perfetto_trace(self) -> Path:
        return self.root / "trace" / "merged.pftrace"

    @property
    def overview(self) -> Path:
        return self.root / "summary" / "overview.json"

    def create(self, *, allow_nonempty: bool = False) -> None:
        """Create the layout, refusing a non-empty existing run by default."""
        if self.root.exists() and any(self.root.iterdir()) and not allow_nonempty:
            raise FileExistsError(f"run directory is not empty: {self.root}")
        directories = (
            self.root / "clocks",
            self.root / "events",
            self.root / "metrics",
            self.root / "artifacts",
            self.root / "raw" / "client",
            self.root / "raw" / "gpu",
            self.root / "raw" / "npu",
            self.root / "raw" / "system",
            self.root / "trace",
            self.root / "summary",
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
