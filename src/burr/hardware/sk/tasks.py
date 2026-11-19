"""Burr-side helpers for building SensorKit target types.

Burr emits canonical ``StandardCollectTask`` instances to the SensorKit
controller. The per-frame tracking mode is encoded directly in the task's
``target`` field — ``ICRSTarget`` for sidereal, ``RateTarget`` for custom
rate-follow, ``TLETarget`` for TLE-driven follow. Everything a downstream
SK client needs to render the task lives in standard fields.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timezone

import astropy.units as u
import sensorkit.api as sk
from astropy.coordinates import AltAz as AstropyAltAz, EarthLocation, SkyCoord
from astropy.time import Time
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sensorkit.astro.common import ReferenceFrame, TLE
from sensorkit.astro.coords import Equatorial, Horizontal
from sensorkit.astro.target import (
    AltAzTarget as SKAltAzTarget,
    ICRSTarget,
    RateTarget,
    TLETarget as SKTLETarget,
    Target,
)
from sensorkit.data.filesys import FileNameTemplate
from sensorkit.std.collect import CameraParameterSet, StandardCollectTask
from sensorkit.std.instrument import FrameType

from burr.core.config import AppConfig
from burr.hardware.sk.settings import CameraSettings, CollectConfig
from burr.models.observation import (
    AltAzTarget,
    LunarTarget,
    ObservationRequest,
    RADecTarget,
    TargetUnion,
    TLETarget,
)
from burr.models.run import BurrRun
from burr.models.site import SiteMetadata
from burr.models.tracking import Rates, TrackingMode


@sk.declare_keyword(key="Burr")
class BurrContext(BaseModel):
    """Sequence identity carried in TaskSubmission.context under the Burr key.

    Split tracking tasks from one exposure share seq; each exposure gets a
    fresh seq. Filename expressions use Burr.seq, Burr.targ and Burr.mode.
    """

    seq: str
    targ: str
    mode: str


_FILE_NAME_TEMPLATE = (
    'f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{Burr.mode}_{Burr.targ}'
    '_{Task.id}_{Instrument.name.encode().hex()}'
    '_f{Collect.frame_number}.fits"'
)


class CalibrationTaskEntry(BaseModel):
    """One camera-only DARK or BIAS sequence sent to SensorKit."""

    camera: str = Field(
        min_length=1,
        description="Camera identity expected in the UNITID FITS header",
    )
    instrument: str = Field(
        min_length=1,
        description="SensorKit instrument name that must take the frames",
    )
    filter_name: str = Field(
        alias="filter",
        min_length=1,
        description="Opaque filter-wheel position, normally Blank",
    )
    frame_type: FrameType
    exposure_seconds: float = Field(ge=0)
    frame_count: int = Field(ge=1)

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    @field_validator("camera", "instrument", "filter_name")
    @classmethod
    def _nonblank_name(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError(
                "value must not have leading or trailing whitespace: "
                f"{value!r}"
            )
        return value

    @field_validator("frame_type")
    @classmethod
    def _calibration_frame_type(
        cls,
        value: FrameType,
    ) -> FrameType:
        if value not in (FrameType.DARK, FrameType.BIAS):
            raise ValueError(
                "calibration frame_type must be DARK or BIAS"
            )
        return value

    @model_validator(mode="after")
    def _dark_needs_positive_exposure(self):
        if (
            self.frame_type == FrameType.DARK
            and self.exposure_seconds <= 0
        ):
            raise ValueError(
                "DARK exposure_seconds must be greater than zero"
            )
        return self


def to_sk_target(burr_target) -> Target:
    """Convert a burr target to the corresponding sensorkit astro target."""
    match burr_target:
        case RADecTarget():
            return ICRSTarget(
                coords=Equatorial(
                    ra=burr_target.right_ascension_deg,
                    dec=burr_target.declination_deg,
                )
            )
        # LunarTarget extends AltAzTarget - match before AltAzTarget.
        case LunarTarget():
            return SKAltAzTarget(
                coords=Horizontal(
                    az=burr_target.azimuth_deg,
                    alt=burr_target.altitude_deg,
                )
            )
        case AltAzTarget():
            return SKAltAzTarget(
                coords=Horizontal(
                    az=burr_target.azimuth_deg,
                    alt=burr_target.altitude_deg,
                )
            )
        case TLETarget():
            # Synthesize a line0 ("name line") from the NORAD catalog number
            # in line1[2:7]. SK's newer std controller (device-module-refactors
            # branch) reads target_id via ``line0.split()[1]``, which crashes
            # with AttributeError on line0=None. The old controller read
            # NORAD directly from line1[2:7], which would be more robust —
            # flagged upstream. For now, giving line0 a two-token string
            # (``NORAD <id>``) makes split()[1] yield the NORAD number.
            norad_id = (
                burr_target.tle_line1[2:7].strip()
                if burr_target.tle_line1 and len(burr_target.tle_line1) >= 7
                else "unknown"
            )
            return SKTLETarget(
                tle=TLE(
                    line0=f"NORAD {norad_id}",
                    line1=burr_target.tle_line1,
                    line2=burr_target.tle_line2,
                )
            )
        case _:
            raise ValueError(
                f"Unsupported burr target type: {type(burr_target).__name__}"
            )


def to_sk_sidereal_target(
    burr_target: TargetUnion,
    initial_time: datetime | None = None,
    site: SiteMetadata | None = None,
) -> Target:
    """SK target for sidereal-tracked frames, freezing AltAz to ICRS.

    RADec and TLE targets pass straight through ``to_sk_target``. AltAz/Lunar
    targets are *frozen* to ICRS at ``initial_time`` (default now) using
    ``site``: sensorkit tracks an ``AltAzTarget`` as **fixed** (alt/az held
    constant), which is wrong for a sidereal collect — the mount would hold the
    horizon coordinate and let the sky drift through the field. Converting the
    alt/az pointing to the RA/Dec it maps to *at emit time* makes the mount lock
    onto that sky position and track it sidereally.

    Coverage and lunar both map in alt/az but want sidereal tracking, so they
    flow through here. Flats (tracking OFF) keep their alt/az target via
    ``to_sk_target`` and are never passed to this function.
    """
    match burr_target:
        case LunarTarget() | AltAzTarget():
            if site is None:
                raise ValueError(
                    "site is required to freeze an AltAz/Lunar target to ICRS "
                    "for sidereal tracking"
                )
            if initial_time is None:
                initial_time = datetime.now(tz=timezone.utc)
            return ICRSTarget(
                coords=_altaz_to_equatorial(
                    az_deg=burr_target.azimuth_deg,
                    alt_deg=burr_target.altitude_deg,
                    time=initial_time,
                    site=site,
                )
            )
        case _:
            # RADec → ICRSTarget, TLE → SKTLETarget: already sidereal-correct.
            return to_sk_target(burr_target)


def to_sk_ratetarget(
    burr_target: TargetUnion,
    rates: Rates,
    initial_time: datetime | None = None,
    site: SiteMetadata | None = None,
    frame: ReferenceFrame = ReferenceFrame.ICRF,
) -> RateTarget:
    """Build a sensorkit ``RateTarget`` from a burr target + concrete rates.

    Initial position comes from ``burr_target``; rates are converted from
    arcsec/s (burr's unit, stored in ``Rates``) to deg/s (sensorkit's
    ``Coordinates`` unit convention). The frame label on the returned
    ``RateTarget`` is set by ``frame``: ICRF is what pwi4 accepts, CIRF
    is what node_platform accepts. Picking the wrong one sends the
    target through sensorkit's BaseTarget.adapt() → to_ephemeris_target()
    path, which is unimplemented for RateTarget. AltAz/Lunar inputs are
    converted using the provided ``site`` and ``initial_time``.

    Parameters
    ----------
    burr_target : TargetUnion
        The burr target whose *position* to use as the rate-follow
        origin. RADec is passed through directly; AltAz/Lunar are
        converted to ICRS using ``site`` + ``initial_time``.
    rates : Rates
        RA/Dec rate vector in arcsec/s.
    initial_time : datetime, optional
        UTC time at which ``initial_coords`` is valid. Defaults to
        ``datetime.now(UTC)``.
    site : SiteMetadata, optional
        Observer location. Required for AltAz/Lunar conversion; ignored
        for RADec inputs. When omitted for AltAz, raises ValueError.

    TLE targets raise: their motion is intrinsic to the orbit and
    should not be combined with custom rates.
    """
    if initial_time is None:
        initial_time = datetime.now(tz=timezone.utc)

    rates_coords = Equatorial(
        ra=rates.right_ascension_rate_arcsec_s / 3600.0,
        dec=rates.declination_rate_arcsec_s / 3600.0,
    )

    match burr_target:
        case RADecTarget():
            initial_coords = Equatorial(
                ra=burr_target.right_ascension_deg,
                dec=burr_target.declination_deg,
            )
        case LunarTarget() | AltAzTarget():
            if site is None:
                raise ValueError(
                    "site is required to convert AltAz/Lunar target to ICRS "
                    "for RateTarget construction"
                )
            initial_coords = _altaz_to_equatorial(
                az_deg=burr_target.azimuth_deg,
                alt_deg=burr_target.altitude_deg,
                time=initial_time,
                site=site,
            )
        case TLETarget():
            raise ValueError(
                "cannot combine custom rates with a TLE target — use TLETarget alone"
            )
        case _:
            raise ValueError(
                f"Unsupported burr target type: {type(burr_target).__name__}"
            )

    return RateTarget(
        frame=frame,
        rates=rates_coords,
        initial_time=initial_time,
        initial_frame=frame,
        initial_coords=initial_coords,
    )


def _altaz_to_equatorial(
    az_deg: float,
    alt_deg: float,
    time: datetime,
    site: SiteMetadata,
) -> Equatorial:
    """Convert an (az, alt) position at a given time + site to ICRS RA/Dec."""
    location = EarthLocation(
        lat=site.latitude * u.deg,
        lon=site.longitude * u.deg,
        height=(site.altitude_km or 0) * u.km,
    )
    altaz_coord = SkyCoord(
        az=az_deg * u.deg,
        alt=alt_deg * u.deg,
        frame=AstropyAltAz(obstime=Time(time), location=location),
    )
    icrs_coord = altaz_coord.icrs
    return Equatorial(
        ra=float(icrs_coord.ra.deg),
        dec=float(icrs_coord.dec.deg),
    )

def build_requested_tasks(
    run: BurrRun,
    config: AppConfig,
    request: ObservationRequest,
    collect: CollectConfig | None = None,
) -> list[sk.TaskSubmission]:
    if "calibration_plan" in request.metadata:
        return [build_calibration_task(run, request, collect)]

    if "flat_plan" in request.metadata:
        return [build_planned_flat_task(run, request, collect)]

    return list(build_sk_tasks(run, config, request, collect))


def build_sk_tasks(
    run: BurrRun,
    config: AppConfig,
    request: ObservationRequest,
    collect: CollectConfig | None = None,
) -> Iterator[sk.TaskSubmission]:
    """Translate ordinary requests; execution identity is assigned by SensorKit."""
    if request.target is None:
        raise ValueError(
            "ordinary SensorKit requests require a target; camera-only "
            "requests must use calibration_plan"
        )

    collect = collect or CollectConfig()
    parts = _tracking_parts(
        [spec.tracking_mode for spec in request.exposure_map]
    )
    frame_type = (
        FrameType.FLAT
        if request.source_name == "twilight_flats"
        else FrameType.LIGHT
    )
    end_time = run.lighting_schedule.schedule_end_utc

    for seconds in request.exposure_seconds:
        # Sample a rate distribution once per requested exposure.
        rates = _concrete_rates(request, seconds)
        context = _exposure_context(request)

        for mode, count, sidereal_frames in parts:
            task = StandardCollectTask(
                target=_tracking_target(request, config, mode, rates),
                camera_params=_camera_params(
                    collect, frame_type, seconds, count
                ),
                end_time=end_time,
                sidereal_frames=sidereal_frames,
            )
            yield task.submit(
                context=context,
                expiry_time=end_time,
            )


def build_calibration_task(
    run: BurrRun,
    request: ObservationRequest,
    collect: CollectConfig | None = None,
) -> sk.TaskSubmission:
    """Build one camera-only DARK/BIAS SensorKit task.

    Calibration tasks deliberately use ``target=None``. SensorKit therefore
    sends no FollowTarget or Stop commands to the mount. Each metadata entry
    is pinned to an explicit SensorKit instrument and an opaque filter-wheel
    position, normally Blank.
    """
    if request.source_name != "calibration_frames":
        raise ValueError(
            "calibration_plan is only supported for calibration_frames "
            "requests"
        )
    if request.target is not None:
        raise ValueError(
            "calibration_frames requests must have target=None"
        )
    if request.exposure_seconds or request.exposure_map:
        raise ValueError(
            "calibration_frames requests describe exposures only through "
            "metadata.calibration_plan"
        )

    raw_entries = request.metadata.get("calibration_plan")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(
            "calibration_plan must contain at least one sequence"
        )

    entries = [
        CalibrationTaskEntry.model_validate(entry)
        for entry in raw_entries
    ]
    collect = collect or CollectConfig()
    settings_by_instrument = dict(collect.instrument_settings())
    params = []

    for entry in entries:
        inherited = settings_by_instrument.get(
            entry.instrument,
            collect,
        )

        # The plan owns the opaque wheel position. Other camera settings
        # continue to inherit from shared/per-instrument collect config.
        settings = inherited.model_copy(
            update={"filter_name": entry.filter_name}
        )
        params.append(
            _parameter_set(
                entry.instrument,
                settings,
                entry.frame_type,
                entry.exposure_seconds,
                entry.frame_count,
            )
        )

    end_time = run.lighting_schedule.schedule_end_utc
    task = StandardCollectTask(
        target=None,
        target_id="calibration",
        camera_params=params,
        end_time=end_time,
    )
    return task.submit(
        context=_exposure_context(request),
        expiry_time=end_time,
    )


def build_planned_flat_task(
    run: BurrRun,
    request: ObservationRequest,
    collect: CollectConfig | None = None,
) -> sk.TaskSubmission:
    """Keep the plan's pair order, filters and independent exposures."""
    if request.target is None:
        raise ValueError(
            "planned twilight flats require a pointing target"
        )

    if request.source_name != "twilight_flats":
        raise ValueError(
            "flat_plan is only supported for twilight_flats requests"
        )

    if not request.exposure_map or any(
        spec.tracking_mode != TrackingMode.OFF
        for spec in request.exposure_map
    ):
        raise ValueError(
            "planned flats require a nonempty, untracked exposure_map"
        )

    entries = request.metadata["flat_plan"]
    if not entries:
        raise ValueError(
            "flat_plan must contain at least one camera/filter pair"
        )

    collect = collect or CollectConfig()
    settings_by_instrument = dict(collect.instrument_settings())
    params = []

    for entry in entries:
        instrument = entry["instrument"]
        settings = settings_by_instrument.get(instrument, collect)

        # Preserve camera-setting inheritance while the plan controls filter.
        settings = settings.model_copy(
            update={"filter_name": entry["filter"]}
        )
        params.append(
            _parameter_set(
                instrument,
                settings,
                FrameType.FLAT,
                float(entry["exposure_seconds"]),
                1,
            )
        )

    end_time = run.lighting_schedule.schedule_end_utc
    task = StandardCollectTask(
        target=to_sk_target(request.target),
        camera_params=params,
        end_time=end_time,
    )
    return task.submit(
        context=_exposure_context(request),
        expiry_time=end_time,
    )


def _tracking_parts(
    modes: list[TrackingMode],
) -> list[tuple[TrackingMode, int, list[int]]]:
    """Preserve uniform and two-phase tracking patterns."""
    if not modes:
        raise ValueError("exposure_map must not be empty")

    count = len(modes)

    if TrackingMode.RATE not in modes:
        mode = (
            TrackingMode.SIDEREAL
            if TrackingMode.SIDEREAL in modes
            else TrackingMode.OFF
        )
        return [(mode, count, [])]

    if all(mode == TrackingMode.RATE for mode in modes):
        return [(TrackingMode.RATE, count, [])]

    first_rate = modes.index(TrackingMode.RATE)

    if first_rate == 0:
        split = next(
            i
            for i, mode in enumerate(modes)
            if mode != TrackingMode.RATE
        )
        if all(
            mode != TrackingMode.RATE for mode in modes[split:]
        ):
            return [
                (
                    TrackingMode.RATE,
                    count,
                    list(range(split, count)),
                )
            ]

    elif all(
        mode == TrackingMode.RATE for mode in modes[first_rate:]
    ):
        before = (
            TrackingMode.SIDEREAL
            if TrackingMode.SIDEREAL in modes[:first_rate]
            else TrackingMode.OFF
        )
        return [
            (before, first_rate, []),
            (TrackingMode.RATE, count - first_rate, []),
        ]

    raise ValueError("unsupported tracking interleaving")


def _concrete_rates(
    request: ObservationRequest,
    seconds: float,
) -> Rates | None:
    if isinstance(request.target, TLETarget) or not any(
        spec.tracking_mode == TrackingMode.RATE
        for spec in request.exposure_map
    ):
        return None

    if request.rates is None:
        raise ValueError(
            "RATE frames require tracking rates or a TLE target"
        )

    if isinstance(request.rates, Rates):
        return request.rates

    return request.rates.to_concrete_rates(seconds)


def _tracking_target(
    request: ObservationRequest,
    config: AppConfig,
    mode: TrackingMode,
    rates: Rates | None,
) -> Target:
    if mode == TrackingMode.SIDEREAL:
        return to_sk_sidereal_target(
            request.target,
            site=config.site,
        )

    if mode == TrackingMode.OFF or isinstance(request.target, TLETarget):
        return to_sk_target(request.target)

    if rates is None:
        raise ValueError("RATE frames require concrete tracking rates")

    return to_sk_ratetarget(
        request.target,
        rates,
        site=config.site,
        frame=ReferenceFrame(config.sk.rate_target_frame),
    )


def _parameter_set(
    instrument: str | None,
    settings: CameraSettings,
    frame_type: FrameType,
    seconds: float,
    count: int,
) -> CameraParameterSet:
    if (
        instrument is not None
        and "instrument" not in CameraParameterSet.model_fields
    ):
        raise ValueError(
            "this SensorKit cannot pin an exposure to an instrument"
        )

    return CameraParameterSet(
        instrument=instrument,
        frame_type=frame_type,
        integration_time_seconds=seconds,
        frame_count=count,
        filter_name=settings.filter_name,
        readout_mode=settings.readout_mode,
        gain=settings.gain,
        binning_x=settings.binning,
        binning_y=settings.binning,
    )


def _camera_params(
    collect: CollectConfig,
    frame_type: FrameType,
    seconds: float,
    count: int,
) -> CameraParameterSet | list[CameraParameterSet]:
    params = [
        _parameter_set(
            instrument, settings, frame_type, seconds, count
        )
        for instrument, settings in collect.instrument_settings()
    ]
    return params if collect.instruments is not None else params[0]


def _exposure_context(
    request: ObservationRequest,
) -> sk.KeywordDict:
    metadata = request.metadata
    target = next(
        (
            metadata[key]
            for key in (
                "target_name",
                "star_name",
                "satellite_name",
                "pixel_id",
                "norad_id",
            )
            if metadata.get(key)
        ),
        request.task_id,
    )
    safe_target = (
        str(target)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    return sk.KeywordDict(
        BurrContext(
            seq=str(uuid.uuid4()),
            targ=safe_target,
            mode=request.source_name,
        ),
        FileNameTemplate(template=_FILE_NAME_TEMPLATE),
    )
