"""Config for per camera dark and bias calibration sequencer"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from burr.task_source.flats.products import (
    UnsafeIdentityError,
    validate_component,
)

CalibrationFrameType = Literal["dark", "bias"]


class CalibrationSequencePlan(BaseModel):
    """One required camera calibration sequence"""

    camera: str = Field(
        description=(
            "Camera id expected in FITS UNITID header, e.g. C0"
        )
    )

    instrument: str = Field(
        description=(
            "Sensorkit instruement that must take the frames (e.g C0-ASI183MM)"
        )
    )

    filter: str = Field(
        description=("Opaque filter-wheel position used to block incident light, normally blank")
    )

    frame_type: CalibrationFrameType
    exposure_seconds: float = Field(
        ge=0,
        description=("Requested exposure in seconds, DARK exposures must be greater than zero, BIAS exposures may be zero")
    )

    frames: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Frames required for this sequence, defaults to the plan's frames_per_sequence"
        )
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    @field_validator("camera", "filter")
    @classmethod
    def _safe_path_identity(cls, value:str, info) -> str:
        """Reject identities that cannot safely appear in product paths."""
        if value != value.strip():
            raise ValueError(
                f"{info.field_name} {value!r} has leading or trailing "
                "whitespace"
            )
        try:
            return validate_component(info.field_name, value)
        except UnsafeIdentityError as error:
            raise ValueError(str(error)) from error

    @field_validator("instrument")
    @classmethod
    def _valid_instrument(cls, value: str) -> str:
        """Require one nonempty SensorKit instrument key."""
        if (
            not value
            or value != value.strip()
            or any(
                character.isspace()
                or ord(character) < 32
                or character in "/\\"
                for character in value
            )
        ):
            raise ValueError(
                f"instrument {value!r} is not a valid SensorKit device key"
            )
        return value

    @model_validator(mode="after")
    def _valid_exposure(self) -> CalibrationSequencePlan:
        """Require a finite exposure and a positive DARK exposure."""
        if not math.isfinite(self.exposure_seconds):
            raise ValueError("exposure_seconds must be finite")
        if self.frame_type == "dark" and self.exposure_seconds <= 0:
            raise ValueError(
                "DARK exposure_seconds must be greater than zero"
            )
        return self

    def identity(self) -> tuple[str, str, str, float]:
        """Case-insensitive identity used to reject duplicate sequences."""
        return (
            self.camera.casefold(),
            self.filter.casefold(),
            self.frame_type,
            self.exposure_seconds,
        )

    def label(self) -> str:
        """Human-readable sequence identity for validation and logs."""
        return (
            f"{self.camera}/{self.frame_type}/"
            f"{self.exposure_seconds:g}s/{self.filter}"
        )


class CalibrationPlanConfig(BaseModel):
    """Complete set of camera calibration sequences"""

    frames_per_sequence: int = Field(
        default = 20,
        ge=1,
        description="Default number of frames required for each sequence"
    )
    sequences: list[CalibrationSequencePlan] = Field(
        description=(
            "Every camera, frame-type and exposure sequence to collect."
        )
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )

    @model_validator(mode="after")
    def _consistent(self) -> CalibrationPlanConfig:
        """Reject empty, duplicate or inconsistently mapped plans"""

        if not self.sequences:
            raise ValueError("Calibration plan needs at least one sequence")

        seen: dict[tuple[str, str, str, float], CalibrationSequencePlan] = {}
        cameras: dict[str, tuple[str, str]] = {}
        instruments: dict[str, tuple[str, str]] = {}

        for sequence in self.sequences:
            identity = sequence.identity()
            previous = seen.get(identity)
            if previous is not None:
                raise ValueError(
                    f"duplicate calibration sequence {sequence.label()}"
                    f"(already listed as {previous.label()})"
                )
            seen[identity]=sequence

            _same_mapping(
                cameras,
                sequence.camera,
                sequence.instrument,
                "camera",
                "instrument",
            )
            _same_mapping(
                instruments,
                sequence.instrument,
                sequence.camera,
                "instrument",
                "camera",
            )

        return self

    def target(self, sequence: CalibrationSequencePlan) -> int:
        """Required frame count for one sequence"""
        return sequence.frames or self.frames_per_sequence


def _same_mapping(
    mapping: dict[str, tuple[str, str]],
    name: str,
    other: str,
    kind: str,
    other_kind: str,
) -> None:
    """Require a stable one-to-one camera/instrument mapping."""
    folded = name.casefold()
    existing = mapping.get(folded)

    if existing is None:
        mapping[folded] = (name, other)
        return

    existing_name, existing_other = existing
    if existing_name != name or existing_other != other:
        raise ValueError(
            f"{kind} {name!r} conflicts with existing {kind} "
            f"{existing_name!r}: mapped to {other_kind} {other!r}, "
            f"previously {existing_other!r}"
        )
