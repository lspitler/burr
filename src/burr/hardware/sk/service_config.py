"""SensorKit configuration bridge owned by Burr."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Self

import sensorkit.api as sk
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sensorkit.astro.common import SitePosition
from sensorkit.config import get_config_section
from sensorkit.config.parser import SensorKitConfig

from burr.core.config import (
    AppConfig,
    PlottingConfig,
    RuntimeConfig,
    ScheduleConfig,
    SKConfig,
)
from burr.hardware.sk.settings import CollectConfig
from burr.models.site import SiteMetadata

SERVICE_MODULE = "burr.hardware.sk.burr_program"
LEGACY_SERVICE_MODULE = "sensorkit.burr.program"


class BurrServiceConfig(BaseModel):
    """The existing burr: deployment fields, now consumed by Burr."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    controller: str = Field(min_length=1)
    plotting: PlottingConfig = Field(
        default_factory=lambda: PlottingConfig(debug=False, review=False)
    )
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    collect: CollectConfig = Field(default_factory=CollectConfig)
    minimum_altitude_degrees: float = Field(default=0, ge=0, le=90)
    raw_data_root: Path | None = None

    @model_validator(mode="after")
    def _require_plan_readback(self) -> Self:
        flats = self.schedule.twilight_flats
        calibration = self.schedule.calibration_frames

        if flats.collect and flats.plan is not None and self.raw_data_root is None:
            raise ValueError("raw_data_root is required for planned SensorKit flats")
        if calibration.collect and self.raw_data_root is None:
            raise ValueError("raw_data_root is required for SensorKit calibration frames")

        return self

    async def to_app_config(self, kit: sk.SensorKit) -> AppConfig:
        controller = kit.controller(self.controller)
        position = await controller.kv_get_model(SitePosition)
        return AppConfig(
            stage="stage",
            plotting=self.plotting,
            runtime=self.runtime,
            schedule=self.schedule,
            site=SiteMetadata(
                name=self.controller,
                latitude=position.latitude_degrees,
                longitude=position.longitude_degrees,
                altitude_km=position.altitude_km,
                minimum_altitude_degrees=self.minimum_altitude_degrees,
            ),
            sk=SKConfig(
                burr_controller_entity=self.controller,
                data_root=self.raw_data_root,
            ),
        )


def _register_section() -> None:
    existing = get_config_section("burr")
    if existing is not None:
        raise RuntimeError(
            "The burr: section is already registered by "
            f"{existing.service_path!r}. Import only "
            "burr.hardware.sk.burr_program; remove sensorkit.burr.program."
        )
    sk.declare_config_section(
        "burr",
        # The functional program currently owns one runtime context.
        Annotated[list[BurrServiceConfig], Field(max_length=1)],
        id_source="by_subkey",
        service_path=SERVICE_MODULE,
    )


def service_config_for(
    config: SensorKitConfig, service_name: str
) -> BurrServiceConfig | None:
    """Reject conflicting providers or duplicate program/service definitions."""
    modules = [
        (service, (service.python_module or "").partition(":")[0])
        for service in config.services
    ]
    if any(module == LEGACY_SERVICE_MODULE for _, module in modules):
        raise ValueError(
            "Remove the sensorkit.burr.program service before using Burr's service"
        )

    own_services = [service for service, module in modules if module == SERVICE_MODULE]
    if len(own_services) > 1:
        raise ValueError("Configure only one Burr-owned service")

    models = [
        model
        for values in config.entity_kv.values()
        for model in values
        if isinstance(model, BurrServiceConfig)
    ]
    if not models:
        return None
    if len(models) != 1 or models[0].id != service_name:
        raise ValueError(f"Expected one burr: entry with id {service_name!r}")
    if len(own_services) != 1 or own_services[0].id != service_name:
        raise ValueError("The burr: entry must launch Burr's service implementation")
    if sum(service.id == service_name for service in config.services) != 1:
        raise ValueError("Remove the duplicate explicit services: entry for Burr")
    return models[0]


_register_section()
