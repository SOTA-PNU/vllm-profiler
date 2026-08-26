"""Deterministic pilot and formal evaluation scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Collection, Iterable


SCHEDULE_VERSION = "1.0.0"
SCHEDULE_SEED_DOMAIN = "profiler-experiment"
PILOT_REPETITIONS = 1
FORMAL_ROUNDS = 5
MAX_HARDWARE_ATTEMPTS = 42


class ScheduleError(ValueError):
    """An experiment schedule or attempt request is invalid."""


class Condition(str, Enum):
    REFERENCE = "reference"
    MONITOR = "monitor"
    GPU_TORCH = "gpu_torch"
    GPU_NSYS = "gpu_nsys"
    NPU_TORCH = "npu_torch"
    NPU_RBLN = "npu_rbln"


CONDITIONS: tuple[Condition, ...] = tuple(Condition)


class TrialKind(str, Enum):
    PILOT = "pilot"
    FORMAL = "formal"


_LOGICAL_ID_RE = re.compile(
    r"^(?:pilot-r00|formal-r0[1-5])-"
    r"(?:reference|monitor|gpu_torch|gpu_nsys|npu_torch|npu_rbln)$"
)
_ATTEMPT_ID_RE = re.compile(
    r"^(?:pilot-r00|formal-r0[1-5])-"
    r"(?:reference|monitor|gpu_torch|gpu_nsys|npu_torch|npu_rbln)-a(?:01|02)$"
)


def validate_logical_trial_id(value: str) -> str:
    if not isinstance(value, str) or _LOGICAL_ID_RE.fullmatch(value) is None:
        raise ScheduleError(f"invalid logical trial id: {value!r}")
    return value


def validate_attempt_id(value: str) -> str:
    if not isinstance(value, str) or _ATTEMPT_ID_RE.fullmatch(value) is None:
        raise ScheduleError(f"invalid attempt id: {value!r}")
    return value


def make_attempt_id(logical_trial_id: str, attempt_number: int) -> str:
    """Build the stable initial/retry identifier for a logical trial."""

    validate_logical_trial_id(logical_trial_id)
    if (
        isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number not in {1, 2}
    ):
        raise ScheduleError("attempt_number must be 1 or 2")
    return f"{logical_trial_id}-a{attempt_number:02d}"


@dataclass(frozen=True)
class TrialSpec:
    """One predeclared logical trial, independent of hardware attempts."""

    position: int
    phase: TrialKind
    round_index: int
    condition: Condition

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or self.position < 0:
            raise ScheduleError("position must be a non-negative integer")
        if not isinstance(self.phase, TrialKind):
            object.__setattr__(self, "phase", TrialKind(self.phase))
        if not isinstance(self.condition, Condition):
            object.__setattr__(self, "condition", Condition(self.condition))
        if isinstance(self.round_index, bool) or not isinstance(
            self.round_index, int
        ):
            raise ScheduleError("round_index must be an integer")
        if self.phase is TrialKind.PILOT and self.round_index != 0:
            raise ScheduleError("pilot round_index must be zero")
        if self.phase is TrialKind.FORMAL and not 1 <= self.round_index <= 5:
            raise ScheduleError("formal round_index must be between 1 and 5")

    @property
    def logical_trial_id(self) -> str:
        return (
            f"{self.phase.value}-r{self.round_index:02d}-"
            f"{self.condition.value}"
        )

    @property
    def repetition_index(self) -> int:
        return 1 if self.phase is TrialKind.PILOT else self.round_index

    def attempt_id(self, attempt_number: int) -> str:
        return make_attempt_id(self.logical_trial_id, attempt_number)

    def to_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "phase": self.phase.value,
            "round_index": self.round_index,
            "repetition_index": self.repetition_index,
            "condition": self.condition.value,
            "logical_trial_id": self.logical_trial_id,
        }


@dataclass(frozen=True)
class ExperimentSchedule:
    """The immutable schedule committed before results are observed."""

    seed: int
    trials: tuple[TrialSpec, ...]
    max_retries_per_trial: int = 1
    max_hardware_attempts: int = MAX_HARDWARE_ATTEMPTS
    schedule_version: str = SCHEDULE_VERSION

    def __post_init__(self) -> None:
        if self.schedule_version != SCHEDULE_VERSION:
            raise ScheduleError(
                f"unsupported schedule_version: {self.schedule_version!r}"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 0xFFFFFFFF
        ):
            raise ScheduleError("seed must be an integer between 0 and 2^32-1")
        if self.max_retries_per_trial != 1:
            raise ScheduleError("the experiment permits one retry per logical trial")
        if (
            isinstance(self.max_hardware_attempts, bool)
            or not isinstance(self.max_hardware_attempts, int)
            or not len(self.trials)
            <= self.max_hardware_attempts
            <= MAX_HARDWARE_ATTEMPTS
        ):
            raise ScheduleError(
                "max_hardware_attempts must cover the base schedule and be <= 42"
            )
        self._validate_trials()

    def _validate_trials(self) -> None:
        expected_count = len(CONDITIONS) * (PILOT_REPETITIONS + FORMAL_ROUNDS)
        if len(self.trials) != expected_count:
            raise ScheduleError(
                f"the experiment requires exactly {expected_count} logical trials"
            )
        if tuple(trial.position for trial in self.trials) != tuple(
            range(expected_count)
        ):
            raise ScheduleError("trial positions must be contiguous and ordered")
        logical_ids = tuple(trial.logical_trial_id for trial in self.trials)
        if len(set(logical_ids)) != len(logical_ids):
            raise ScheduleError("logical trial ids must be unique")

        pilots = self.pilot_trials
        if len(pilots) != len(CONDITIONS) or {
            trial.condition for trial in pilots
        } != set(CONDITIONS):
            raise ScheduleError("the pilot must contain every condition once")
        if any(
            trial.phase is TrialKind.PILOT
            for trial in self.trials[len(CONDITIONS) :]
        ):
            raise ScheduleError("all pilot trials must precede formal trials")

        for round_index in range(1, FORMAL_ROUNDS + 1):
            round_trials = self.formal_round(round_index)
            if len(round_trials) != len(CONDITIONS) or {
                trial.condition for trial in round_trials
            } != set(CONDITIONS):
                raise ScheduleError(
                    f"formal round {round_index} must contain every condition once"
                )

    @property
    def pilot_trials(self) -> tuple[TrialSpec, ...]:
        return tuple(
            trial for trial in self.trials if trial.phase is TrialKind.PILOT
        )

    @property
    def formal_trials(self) -> tuple[TrialSpec, ...]:
        return tuple(
            trial for trial in self.trials if trial.phase is TrialKind.FORMAL
        )

    def formal_round(self, round_index: int) -> tuple[TrialSpec, ...]:
        return tuple(
            trial
            for trial in self.trials
            if trial.phase is TrialKind.FORMAL
            and trial.round_index == round_index
        )

    def formal_trials_unlocked(
        self,
        successful_logical_trial_ids: Collection[str],
    ) -> bool:
        """Return true only after all six pilots have succeeded."""

        successes = set(successful_logical_trial_ids)
        return all(
            trial.logical_trial_id in successes for trial in self.pilot_trials
        )

    def assert_attempt_available(self, completed_attempt_count: int) -> None:
        if (
            isinstance(completed_attempt_count, bool)
            or not isinstance(completed_attempt_count, int)
            or completed_attempt_count < 0
        ):
            raise ScheduleError(
                "completed_attempt_count must be a non-negative integer"
            )
        if completed_attempt_count >= self.max_hardware_attempts:
            raise ScheduleError(
                f"hardware attempt limit reached: {self.max_hardware_attempts}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule_version": self.schedule_version,
            "seed": self.seed,
            "max_retries_per_trial": self.max_retries_per_trial,
            "max_hardware_attempts": self.max_hardware_attempts,
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_schedule_bytes(self)).hexdigest()


def _randomized_conditions(
    seed: int,
    label: str,
    *,
    seed_domain: str,
) -> tuple[Condition, ...]:
    """Return a stable pseudorandom order without Python RNG version coupling."""

    return tuple(
        sorted(
            CONDITIONS,
            key=lambda condition: hashlib.sha256(
                f"{seed_domain}:{seed}:{label}:{condition.value}".encode("ascii")
            ).digest(),
        )
    )


def build_schedule(
    *,
    seed: int,
    max_hardware_attempts: int = MAX_HARDWARE_ATTEMPTS,
    seed_domain: str = SCHEDULE_SEED_DOMAIN,
) -> ExperimentSchedule:
    """Build six pilots followed by five cyclic, balanced formal rounds."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 0xFFFFFFFF
    ):
        raise ScheduleError("seed must be an integer between 0 and 2^32-1")

    trials: list[TrialSpec] = []
    if not isinstance(seed_domain, str) or not seed_domain:
        raise ScheduleError("seed_domain must be a non-empty string")

    for condition in _randomized_conditions(seed, "pilot", seed_domain=seed_domain):
        trials.append(
            TrialSpec(
                position=len(trials),
                phase=TrialKind.PILOT,
                round_index=0,
                condition=condition,
            )
        )

    for round_index in range(1, FORMAL_ROUNDS + 1):
        for condition in _randomized_conditions(
            seed,
            f"formal:{round_index}",
            seed_domain=seed_domain,
        ):
            trials.append(
                TrialSpec(
                    position=len(trials),
                    phase=TrialKind.FORMAL,
                    round_index=round_index,
                    condition=condition,
                )
            )

    return ExperimentSchedule(
        seed=seed,
        trials=tuple(trials),
        max_hardware_attempts=max_hardware_attempts,
    )


def canonical_schedule_bytes(schedule: ExperimentSchedule) -> bytes:
    """Serialize a schedule deterministically for identity and manifests."""

    return (
        json.dumps(
            schedule.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def schedule_by_logical_id(
    schedule: ExperimentSchedule,
) -> dict[str, TrialSpec]:
    return {trial.logical_trial_id: trial for trial in schedule.trials}


def condition_order(trials: Iterable[TrialSpec]) -> tuple[str, ...]:
    """Expose condition order without leaking enum implementation details."""

    return tuple(trial.condition.value for trial in trials)
