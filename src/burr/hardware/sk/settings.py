"""Burr-owned camera settings for SensorKit task construction."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)


class CameraSettings(BaseModel):
    filter_name: str | None = None
    readout_mode: int | None = None
    gain: float | None = None
    binning: int | None = None

    def with_defaults(self, shared: CameraSettings) -> CameraSettings:
        """Inherit omitted fields while preserving explicit null overrides."""
        return shared.model_copy(
            update=self.model_dump(exclude_unset=True)
        )


class CollectConfig(CameraSettings):
    instrument: str | None = Field(default=None, min_length=1)
    instruments: dict[
        Annotated[str, Field(min_length=1)], CameraSettings
    ] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_selection(self) -> Self:
        if self.instrument is not None and self.instruments is not None:
            raise ValueError("set either instrument or instruments, not both")
        return self

    @field_serializer("instruments")
    def _dump_instruments(
        self,
        instruments: dict[str, CameraSettings] | None,
        info: SerializationInfo,
    ) -> dict[str, dict] | None:
        if instruments is None:
            return None

        return {
            name: settings.model_dump(
                mode=info.mode,
                exclude_unset=True,
            )
            for name, settings in instruments.items()
        }

    def instrument_settings(
        self,
    ) -> list[tuple[str | None, CameraSettings]]:
        shared = CameraSettings.model_validate(
            self.model_dump(include=set(CameraSettings.model_fields))
        )

        if self.instruments is None:
            return [(self.instrument, shared)]

        return [
            (name, settings.with_defaults(shared))
            for name, settings in self.instruments.items()
        ]