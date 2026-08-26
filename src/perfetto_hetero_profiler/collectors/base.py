"""Stateful collector lifecycle shared by monitor implementations."""

from __future__ import annotations

from enum import Enum
from typing import Any


class CollectorState(str, Enum):
    CREATED = "created"
    PREPARED = "prepared"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


class CollectorError(RuntimeError):
    """Raised when a collector lifecycle contract is violated."""


class BaseCollector:
    """Template lifecycle with strict start/sample ordering and safe stop."""

    def __init__(self) -> None:
        self.state = CollectorState.CREATED
        self.last_error: Exception | None = None
        self._stop_completed = False

    def prepare(self) -> None:
        if self.state is not CollectorState.CREATED:
            raise CollectorError(f"prepare is invalid while {self.state.value}")
        try:
            self._prepare()
        except Exception as error:
            self._fail(error)
            raise
        self.state = CollectorState.PREPARED

    def start(self) -> None:
        if self.state is not CollectorState.PREPARED:
            raise CollectorError(f"start is invalid while {self.state.value}")
        try:
            self._start()
        except Exception as error:
            self._fail(error)
            raise
        self.state = CollectorState.RUNNING

    def sample(self) -> Any:
        if self.state is not CollectorState.RUNNING:
            raise CollectorError(f"sample is invalid while {self.state.value}")
        try:
            return self._sample()
        except Exception as error:
            self._fail(error)
            raise

    def stop(self) -> None:
        if self.state is CollectorState.STOPPED or self._stop_completed:
            return
        if self.state is CollectorState.CREATED:
            raise CollectorError("stop is invalid before prepare")
        previous = self.state
        try:
            if previous in {CollectorState.RUNNING, CollectorState.FAILED}:
                self._stop()
                self._stop_completed = True
        except Exception as error:
            self._fail(error)
            raise
        if previous is not CollectorState.FAILED:
            self.state = CollectorState.STOPPED

    def finalize(self) -> Any:
        if self.state is CollectorState.RUNNING:
            self.stop()
        if self.state not in {CollectorState.STOPPED, CollectorState.FAILED}:
            raise CollectorError(f"finalize is invalid while {self.state.value}")
        return self._finalize()

    def __enter__(self) -> "BaseCollector":
        self.prepare()
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
        self.finalize()

    def _fail(self, error: Exception) -> None:
        self.last_error = error
        self.state = CollectorState.FAILED

    def _prepare(self) -> None:
        pass

    def _start(self) -> None:
        pass

    def _sample(self) -> Any:
        raise NotImplementedError

    def _stop(self) -> None:
        pass

    def _finalize(self) -> Any:
        return None
