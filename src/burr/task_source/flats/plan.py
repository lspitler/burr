"""Twilight-flat collection plan: which camera/filter pairs need flats.

A plan lists every required pair, the SensorKit instrument that takes it, and
how many accepted frames it needs. With a plan, flat collection:

- tracks every planned pair from the start, including pairs that never
  return an image;
- sends each still-incomplete pair its own exposure, pinned to its
  instrument and filter;
- is complete only when every planned pair reaches its target;
- never counts a camera or filter that is not in the plan.

See ``docs/flats.md`` for the planning and matching rules.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from burr.task_source.flats.products import (
    FlatSequenceKey,
    UnsafeIdentityError,
    validate_component,
)


class FlatPairPlan(BaseModel):
    """One required camera/filter pair."""

    camera: str = Field(
        description="Camera unit, as written to the frame's UNITID header (e.g. C0)"
    )
    instrument: str = Field(
        description="SensorKit instrument device key that takes this pair (e.g. C0-ASI183MM)"
    )
    filter: str = Field(
        description="Filter name, exactly as the camera's wheel reports it (e.g. Lum)"
    )
    frames: int | None = Field(
        default=None,
        ge=1,
        description="Accepted frames required; defaults to the plan's",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("camera", "filter")
    @classmethod
    def _safe_name(cls, value: str, info) -> str:
        """Camera and filter become directory and filename fields: reject rather than sanitise."""
        if value != value.strip():
            raise ValueError(
                f"{info.field_name} {value!r} has leading or trailing whitespace"
            )
        try:
            return validate_component(info.field_name, value)
        except UnsafeIdentityError as e:
            raise ValueError(str(e)) from e

    @field_validator("instrument")
    @classmethod
    def _device_key(cls, value: str) -> str:
        """A SensorKit device key: one token, no whitespace, control characters or separators."""
        if not value or any(
            ch.isspace() or ord(ch) < 32 or ch in "/\\" for ch in value
        ):
            raise ValueError(f"instrument {value!r} is not a valid device key")
        return value

    @property
    def key(self) -> FlatSequenceKey:
        return FlatSequenceKey(self.camera, self.filter)


class FlatPlanConfig(BaseModel):
    """Explicit list of camera/filter pairs to collect flats for."""

    frames_per_pair: int = Field(
        default=10, ge=1, description="Accepted frames required for each pair"
    )
    pairs: list[FlatPairPlan] = Field(
        description="Every camera/filter pair that needs flats"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _consistent(self) -> FlatPlanConfig:
        """Reject an empty plan, duplicate pairs and conflicting camera/instrument mappings."""
        if not self.pairs:
            raise ValueError("a flat plan needs at least one camera/filter pair")
        seen: dict[tuple[str, str], FlatPairPlan] = {}
        instruments: dict[str, str] = {}
        cameras: dict[str, str] = {}
        for pair in self.pairs:
            folded = pair.key.folded()
            if folded in seen:
                raise ValueError(
                    f"duplicate plan entry {pair.key.label()} "
                    f"(already listed as {seen[folded].key.label()}; names are compared case-insensitively)"
                )
            seen[folded] = pair
            _same_mapping(
                instruments, pair.instrument, pair.camera, "instrument", "camera"
            )
            _same_mapping(cameras, pair.camera, pair.instrument, "camera", "instrument")
        return self

    def target(self, pair: FlatPairPlan) -> int:
        """Accepted frames required for ``pair``."""
        return pair.frames or self.frames_per_pair

    def by_key(self) -> dict[FlatSequenceKey, FlatPairPlan]:
        return {pair.key: pair for pair in self.pairs}


def _same_mapping(
    mapping: dict[str, str], name: str, other: str, kind: str, other_kind: str
) -> None:
    """Each camera has one instrument and each instrument one camera."""
    if mapping.setdefault(name, other) != other:
        raise ValueError(
            f"{kind} {name!r} is mapped to both {other_kind} {mapping[name]!r} and {other!r}"
        )
