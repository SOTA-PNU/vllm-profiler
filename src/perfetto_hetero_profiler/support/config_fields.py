"""Strict, reusable field checks for JSON and CLI configuration values."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigFields:
    """Validate scalar and object fields with one caller-owned error type."""

    def __init__(self, error_type: type[ValueError]) -> None:
        self.error_type = error_type

    def _error(self, message: str) -> ValueError:
        return self.error_type(message)

    def object(self, value: object, field: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise self._error(f"{field} must be an object")
        return dict(value)

    def reject_unknown(
        self, value: dict[str, Any], allowed: set[str], field: str
    ) -> None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise self._error(f"unknown {field} field: {unknown[0]}")

    def exact_object(
        self, value: object, field: str, fields: set[str]
    ) -> dict[str, Any]:
        result = self.object(value, field)
        self.reject_unknown(result, fields, field)
        missing = sorted(fields - set(result))
        if missing:
            raise self._error(f"missing {field} field: {missing[0]}")
        return result

    def string(self, value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise self._error(f"{field} must be a non-empty string")
        return value

    def integer(
        self, value: object, field: str, minimum: int, maximum: int
    ) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise self._error(f"{field} must be an integer")
        if not minimum <= value <= maximum:
            raise self._error(f"{field} must be in [{minimum}, {maximum}]")
        return value

    def number(
        self, value: object, field: str, minimum: float, maximum: float
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._error(f"{field} must be a number")
        result = float(value)
        if not minimum <= result <= maximum:
            raise self._error(f"{field} must be in [{minimum:g}, {maximum:g}]")
        return result

    def boolean(self, value: object, field: str) -> bool:
        if not isinstance(value, bool):
            raise self._error(f"{field} must be a boolean")
        return value

    def absolute_path(
        self, value: object, field: str, *, no_symlink: bool = False
    ) -> Path:
        path = Path(self.string(value, field))
        if not path.is_absolute():
            raise self._error(f"{field} must be an absolute path")
        if no_symlink and path.exists() and path.is_symlink():
            raise self._error(f"{field} must not be a symlink")
        return path

    def relative_path(self, value: object, field: str) -> Path:
        path = Path(self.string(value, field))
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise self._error(f"{field} must be a safe relative output path")
        return path
