"""Repeatability, accuracy, and overhead experiment support."""

from .accuracy import (
    AccuracyCheck,
    AccuracyError,
    ClientLatencyMetric,
    client_latency_accuracy,
    exact_count_accuracy,
    exact_marker_accuracy,
)
from .config import ExperimentConfig, ExperimentConfigError, load_experiment_config
from .experiment import (
    ExperimentError,
    build_plan,
    experiment_status,
    generate_report,
    run_experiment,
    validate_experiment,
)
from .schedule import Condition, ExperimentSchedule, TrialKind, build_schedule
from .statistics import paired_overhead, summarize_distribution

__all__ = [
    "AccuracyCheck",
    "AccuracyError",
    "ClientLatencyMetric",
    "Condition",
    "ExperimentConfig",
    "ExperimentConfigError",
    "ExperimentError",
    "ExperimentSchedule",
    "TrialKind",
    "build_plan",
    "build_schedule",
    "client_latency_accuracy",
    "exact_count_accuracy",
    "exact_marker_accuracy",
    "experiment_status",
    "generate_report",
    "load_experiment_config",
    "paired_overhead",
    "run_experiment",
    "summarize_distribution",
    "validate_experiment",
]
