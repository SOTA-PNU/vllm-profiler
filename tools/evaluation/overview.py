"""Repository-only Overview comparison API.

Single-run Overview generation remains in the installed profiler package.
This module owns the comparison surface while reusing core report loading,
publication, rendering, and validation primitives.
"""

from perfetto_hetero_profiler.overview.bundle import (
    LoadedComparisonBundle,
    load_comparison_bundle,
)
from perfetto_hetero_profiler.overview.generator import (
    OverviewComparisonConfig,
    compare_overviews,
    plan_overview_comparison,
)
from perfetto_hetero_profiler.overview.model import (
    Comparability,
    ComparisonDelta,
    ComparisonKpi,
    ComparisonMetadata,
    ComparisonRun,
    ComparisonValue,
    DeltaValue,
    KpiDirection,
    OverviewComparison,
)
from perfetto_hetero_profiler.overview.render import render_comparison_html
from perfetto_hetero_profiler.overview.schema import (
    overview_comparison_from_dict,
    validate_comparison_delta,
    validate_comparison_kpi,
    validate_comparison_metadata,
    validate_comparison_run,
    validate_comparison_value,
    validate_overview_comparison,
)
from perfetto_hetero_profiler.overview.validation import build_comparison_validation

from .overview_comparison import (
    OverviewComparisonError,
    build_comparison,
    load_comparison_schema,
)


__all__ = [
    "Comparability",
    "ComparisonDelta",
    "ComparisonKpi",
    "ComparisonMetadata",
    "ComparisonRun",
    "ComparisonValue",
    "DeltaValue",
    "KpiDirection",
    "LoadedComparisonBundle",
    "OverviewComparison",
    "OverviewComparisonConfig",
    "OverviewComparisonError",
    "build_comparison",
    "build_comparison_validation",
    "compare_overviews",
    "load_comparison_bundle",
    "load_comparison_schema",
    "overview_comparison_from_dict",
    "plan_overview_comparison",
    "render_comparison_html",
    "validate_comparison_delta",
    "validate_comparison_kpi",
    "validate_comparison_metadata",
    "validate_comparison_run",
    "validate_comparison_value",
    "validate_overview_comparison",
]
