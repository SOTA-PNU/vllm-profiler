"""Input-only identifiers for experiments created by older releases."""

LEGACY_SCHEDULE_SEED_DOMAIN = "phase7b"
LEGACY_REPORT_TYPE = "phase7b_repeatability_overhead"
LEGACY_REPORT_CONFIG_KEY = "phase7"
LEGACY_CONFIG_SCHEMA_ID = (
    "https://sota-pnu.github.io/vllm-profiler/schema/phase7b-config-v1.json"
)
LEGACY_CLI_COMMAND = "phase7"

__all__ = [
    "LEGACY_CLI_COMMAND",
    "LEGACY_CONFIG_SCHEMA_ID",
    "LEGACY_REPORT_CONFIG_KEY",
    "LEGACY_REPORT_TYPE",
    "LEGACY_SCHEDULE_SEED_DOMAIN",
]
