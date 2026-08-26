"""Deterministic KPI Overview calculation, rendering, and publication.

The package initializer intentionally stays dependency-light. Overview product
commands import the official Perfetto runtime lazily, so schema-only users and
CPU tests can still import this package without that optional execution path.
"""

from importlib import import_module


__all__ = [
    "OverviewCalculationError",
    "OverviewGenerationConfig",
    "OverviewGenerationError",
    "OverviewRenderError",
    "OverviewSchemaError",
    "calculate_overview_kpis",
    "generate_overview",
    "overview_report_from_dict",
    "plan_overview_generation",
    "render_overview_html",
    "validate_offline_html",
]

_EXPORT_MODULES = {
    "OverviewCalculationError": ".calculation",
    "calculate_overview_kpis": ".calculation",
    "OverviewGenerationConfig": ".generator",
    "OverviewGenerationError": ".generator",
    "generate_overview": ".generator",
    "plan_overview_generation": ".generator",
    "OverviewRenderError": ".render",
    "render_overview_html": ".render",
    "validate_offline_html": ".render",
    "OverviewSchemaError": ".schema",
    "overview_report_from_dict": ".schema",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
