"""Deterministic KPI Overview calculation, rendering, and publication.

The package initializer intentionally stays dependency-light.  Phase 6 product
commands import the official Perfetto runtime lazily, so schema-only users and
CPU tests can still import this package without that optional execution path.
"""

from importlib import import_module


__all__ = [
    "OverviewCalculationError",
    "OverviewComparisonConfig",
    "OverviewComparisonError",
    "OverviewGenerationConfig",
    "OverviewGenerationError",
    "OverviewRenderError",
    "OverviewSchemaError",
    "build_comparison",
    "calculate_overview_kpis",
    "compare_overviews",
    "generate_overview",
    "overview_comparison_from_dict",
    "overview_report_from_dict",
    "plan_overview_comparison",
    "plan_overview_generation",
    "render_comparison_html",
    "render_overview_html",
    "validate_offline_html",
]

_EXPORT_MODULES = {
    "OverviewCalculationError": ".calculation",
    "calculate_overview_kpis": ".calculation",
    "OverviewComparisonError": ".comparison",
    "build_comparison": ".comparison",
    "OverviewComparisonConfig": ".generator",
    "OverviewGenerationConfig": ".generator",
    "OverviewGenerationError": ".generator",
    "compare_overviews": ".generator",
    "generate_overview": ".generator",
    "plan_overview_comparison": ".generator",
    "plan_overview_generation": ".generator",
    "OverviewRenderError": ".render",
    "render_comparison_html": ".render",
    "render_overview_html": ".render",
    "validate_offline_html": ".render",
    "OverviewSchemaError": ".schema",
    "overview_comparison_from_dict": ".schema",
    "overview_report_from_dict": ".schema",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
