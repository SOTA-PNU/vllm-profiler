"""Single source of paths for one grouped hybrid execution bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_GROUPED_PATHS = {
    "hybrid": Path("hybrid"),
    "gpu": Path("sources/gpu"),
    "npu": Path("sources/npu"),
    "coordinator": Path("coordinator"),
    "perfetto": Path("perfetto"),
    "request_perfetto": Path("perfetto"),
    "overview": Path("overview"),
    "recovery": Path("recovery"),
    "publication": Path("publication"),
}
_LEGACY_GROUPED_PATHS = {
    "perfetto": Path("perfetto/full"),
    "request_perfetto": Path("perfetto/request-focused"),
}
_LEGACY_SUFFIXES = {
    "hybrid": "",
    "gpu": "-gpu",
    "npu": "-npu",
    "coordinator": "-coordinator",
    "perfetto": "-perfetto",
    "request_perfetto": "-perfetto-request-focused",
    "overview": "-overview",
    "recovery": "-closeout-recovery",
    "publication": "-publication",
}


def related_run_root(hybrid_root: Path, run_id: str, product: str) -> Path:
    """Derive a trusted related root for a grouped or legacy hybrid bundle."""

    root = Path(hybrid_root)
    if product not in _GROUPED_PATHS:
        raise ValueError(f"unknown hybrid product: {product}")
    if root.name == "hybrid" and root.parent.name == run_id:
        return root.parent / _GROUPED_PATHS[product]
    if root.name == run_id:
        return root.parent / f"{run_id}{_LEGACY_SUFFIXES[product]}"
    raise ValueError("hybrid root does not match a supported run layout")


def existing_related_run_root(
    hybrid_root: Path,
    run_id: str,
    product: str,
) -> Path:
    """Resolve a produced root across canonical and historical layouts.

    New grouped runs publish both Perfetto views in one ``perfetto/`` bundle.
    Historical grouped runs used ``perfetto/full`` and
    ``perfetto/request-focused``.  File sentinels avoid mistaking the shared
    historical parent directory for a canonical bundle.
    """

    canonical = related_run_root(hybrid_root, run_id, product)
    if product not in _LEGACY_GROUPED_PATHS:
        return canonical
    sentinel = {
        "perfetto": "trace.pftrace",
        "request_perfetto": "trace.request-focused.pftrace",
    }[product]
    if (canonical / sentinel).is_file():
        return canonical
    root = Path(hybrid_root)
    if root.name == "hybrid" and root.parent.name == run_id:
        historical = root.parent / _LEGACY_GROUPED_PATHS[product]
        if (historical / sentinel).is_file():
            return historical
    return canonical


@dataclass(frozen=True, slots=True)
class HybridRunLayout:
    run_root: Path
    run_id: str

    @property
    def bundle(self) -> Path:
        """Top-level directory that owns every product of one execution."""

        return self.run_root / self.run_id

    @property
    def hybrid(self) -> Path:
        return self.bundle / _GROUPED_PATHS["hybrid"]

    @property
    def gpu(self) -> Path:
        return self.bundle / _GROUPED_PATHS["gpu"]

    @property
    def npu(self) -> Path:
        return self.bundle / _GROUPED_PATHS["npu"]

    @property
    def coordinator(self) -> Path:
        return self.bundle / _GROUPED_PATHS["coordinator"]

    @property
    def perfetto(self) -> Path:
        return self.bundle / _GROUPED_PATHS["perfetto"]

    @property
    def request_perfetto(self) -> Path:
        return self.bundle / _GROUPED_PATHS["request_perfetto"]

    @property
    def overview(self) -> Path:
        return self.bundle / _GROUPED_PATHS["overview"]

    @property
    def recovery(self) -> Path:
        return self.bundle / _GROUPED_PATHS["recovery"]

    @property
    def publication(self) -> Path:
        return self.bundle / _GROUPED_PATHS["publication"]

    @property
    def all_roots(self) -> tuple[Path, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.hybrid,
                    self.gpu,
                    self.npu,
                    self.coordinator,
                    self.perfetto,
                    self.request_perfetto,
                    self.overview,
                    self.recovery,
                    self.publication,
                )
            )
        )

    @property
    def legacy_roots(self) -> tuple[Path, ...]:
        """Flat roots checked only to prevent reuse of an old run identity."""

        return tuple(
            self.run_root / f"{self.run_id}{suffix}"
            for suffix in _LEGACY_SUFFIXES.values()
        )


__all__ = [
    "HybridRunLayout",
    "existing_related_run_root",
    "related_run_root",
]
