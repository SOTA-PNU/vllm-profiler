"""Single source of names for a reusable hybrid execution's output roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HybridRunLayout:
    run_root: Path
    run_id: str

    def _named(self, suffix: str = "") -> Path:
        return self.run_root / f"{self.run_id}{suffix}"

    @property
    def hybrid(self) -> Path:
        return self._named()

    @property
    def gpu(self) -> Path:
        return self._named("-gpu")

    @property
    def npu(self) -> Path:
        return self._named("-npu")

    @property
    def coordinator(self) -> Path:
        return self._named("-coordinator")

    @property
    def perfetto(self) -> Path:
        return self._named("-perfetto")

    @property
    def request_perfetto(self) -> Path:
        return self._named("-perfetto-request-focused")

    @property
    def overview(self) -> Path:
        return self._named("-overview")

    @property
    def recovery(self) -> Path:
        return self._named("-closeout-recovery")

    @property
    def publication(self) -> Path:
        return self._named("-publication")

    @property
    def all_roots(self) -> tuple[Path, ...]:
        return (
            self.hybrid, self.gpu, self.npu, self.coordinator, self.perfetto,
            self.request_perfetto, self.overview, self.recovery, self.publication,
        )


__all__ = ["HybridRunLayout"]
