"""
Flat field collection management during twilight.

This module provides tools for managing the collection of flat field calibration
frames during twilight, with automatic exposure time adjustment based on sky
brightness changes.

Exposure control follows the huntsman-pocs ``FlatFieldSequence`` scheme: the
next exposure scales the last one to hit the count target, then corrects for
the sky brightening/darkening by ``2 ** (elapsed / sky_doubling_seconds)``
since that frame was taken. Counts are a sigma-clipped mean over a central
cutout, which keeps vignetted corners out of the statistic.

Sequences: state is kept per camera/filter pair (``FlatSequenceKey``),
from the FITS ``UNITID`` and ``FILTER`` headers. Each pair counts, numbers
and restores independently; see ``products.py`` for naming, publication and
restart, and ``docs/flats.md`` for the recovery rules.

Planned collection (``twilight_flats.plan``, SensorKit only): every
planned camera/filter pair is tracked from the start and gets its own
exposure in each request, pinned to its instrument and filter. Frames are
matched to the request by the SensorKit task ID (``SKTASKID``) and to the
pair by ``UNITID``/``FILTER``; anything else is reported and never counted.
Collection is complete only when every planned pair reaches its target.

Without a plan, each request uses ONE exposure (the median of the pairs'
ideal exposures) and SensorKit chooses a single camera for it, so pairs only
accumulate as cameras happen to be chosen and completion covers only pairs
seen so far.

Implements the TaskSource protocol defined in burr.task_source.protocol.
"""

from __future__ import annotations

import time
import uuid as uuidlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List  # retained for generate_twilight_periods signature

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from loguru import logger
from pydantic import BaseModel
from zoneinfo import ZoneInfo

from burr.models.observation import (
    AltAzTarget,
    ExposureSpec,
    ObservationRequest,
)
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.models.tracking import TrackingMode
from burr.task_source.flats.plan import FlatPairPlan, FlatPlanConfig
from burr.task_source.flats.products import (
    DEFAULT_CAMERA,
    SEQUENCE_CARD,
    SOURCE_HASH_CARD,
    SOURCE_PATH_CARD,
    UUID_CARD,
    FlatSequenceKey,
    UnsafeIdentityError,
    find_case_collision,
    log_restore_report,
    product_dir,
    product_name,
    publish_product,
    reconcile_history,
    scan_products,
    source_digest,
    validate_key,
)

__all__ = [
    "DEFAULT_CAMERA",
    "FlatFieldManager",
    "FlatSequenceKey",
    "load_image_hdu",
    "validate_sk_flat_header",
]


@dataclass
class ExposureRecord:
    """Record of a single flat field exposure."""

    timestamp: datetime
    exposure_time: float
    median_counts: float
    saturation_percentage: float
    is_good: bool


@dataclass
class FlatSequenceState:
    """Collection state for one camera/filter sequence.

    ``good_frames`` counts verified accepted products. ``next_sequence`` is
    the next free filename number; it is kept separately so that gaps in the
    numbering (e.g. a removed product) never imply accepted frames.
    """

    good_frames: int = 0
    next_sequence: int = 1
    next_exposure: float | None = None
    exposures: List[ExposureRecord] = field(default_factory=list)


@dataclass(frozen=True)
class _Frame:
    """Identity and provenance of a frame being processed."""

    path: Path
    key: FlatSequenceKey
    uuid: str | None
    digest: str | None


class FlatRejected(ValueError):
    """A frame cannot be accepted for a reason other than image quality."""


@dataclass
class FrameStats:
    """Count statistics for one flat frame."""

    average_counts: float
    median_counts: float
    mean_counts: float
    std_counts: float
    min_counts: float
    max_counts: float
    saturation_percentage: float
    saturation_threshold: int


class Twilight(BaseModel):
    start_time: datetime
    end_time: datetime


def _header_text(header: fits.Header, key: str) -> str:
    """Stripped string value of ``key``; a missing or valueless card is ``""``."""
    value = header.get(key)
    return "" if value is None else str(value).strip()


def validate_sk_flat_header(header: fits.Header) -> tuple[str, str]:
    """Return the camera and filter after validating a SensorKit flat."""
    image_type = _header_text(header, "IMAGETYP")
    camera = _header_text(header, "UNITID")
    filter_name = _header_text(header, "FILTER")

    if image_type.upper() != "FLAT":
        raise ValueError(f"expected IMAGETYP=FLAT, got {image_type!r}")
    if not camera:
        raise ValueError("Sensorkit flat is missing UNITID")
    if not filter_name:
        raise ValueError("Sensorkit flat is missing FILTER")
    return camera, filter_name


def load_image_hdu(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Return ``(data, header)`` of the first HDU in ``path`` holding an image.

    SK's ``compress_fits`` writes an empty primary HDU followed by a
    ``CompImageHDU``, so the pixels are not in HDU 0. The returned header
    is the image HDU's own (where SK puts EXPTIME, UNITID, IMAGETYP, ...).
    """
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return np.asarray(hdu.data), hdu.header.copy()
    raise ValueError(f"no image data in any HDU of {path}")


def _parse_date_obs(date_obs) -> datetime | None:
    """Parse a FITS DATE-OBS string as UTC, or None if absent/invalid."""
    if not (date_obs and isinstance(date_obs, str) and date_obs.strip()):
        return None
    try:
        parsed = datetime.fromisoformat(date_obs.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=ZoneInfo("UTC"))


def _is_flat_frame(path: Path) -> bool:
    """True if any HDU header in ``path`` has ``IMAGETYP = flat``.

    Reads headers only, so a compressed image is not decompressed just to
    classify it.
    """
    try:
        with fits.open(path) as hdul:
            return any(str(hdu.header.get("IMAGETYP", "")).strip().lower() == "flat" for hdu in hdul)
    except Exception as e:
        logger.debug(f"Skipping unreadable FITS {path}: {e}")
        return False


def _image_header_cards(path: Path) -> fits.Header | None:
    """Header of the first image HDU in ``path``, without reading pixels; None if unreadable.

    Matches ``load_image_hdu``'s choice of HDU (SK's compressed image, not
    the empty primary) using header shape only.
    """
    try:
        with fits.open(path) as hdul:
            for hdu in hdul:
                if (
                    isinstance(hdu, fits.CompImageHDU)
                    or hdu.header.get("NAXIS", 0) >= 2
                ):
                    return hdu.header.copy()
            return None
    except (OSError, ValueError, fits.VerifyError) as e:
        logger.debug(f"Skipping unreadable FITS {path}: {e}")
        return None


def _central_cutout(data: np.ndarray, size: int | None) -> np.ndarray:
    """Central ``size``x``size`` box of ``data`` (whole frame if None/too big)."""
    if size is None:
        return data
    ny, nx = data.shape[-2:]
    if size >= ny and size >= nx:
        return data
    y0, x0 = max((ny - size) // 2, 0), max((nx - size) // 2, 0)
    return data[..., y0 : y0 + size, x0 : x0 + size]


class FlatFieldManager:
    """
    Manages the collection of flat field calibration frames during twilight.

    This class handles:
    - Determining optimal exposure times based on sky brightness
    - Tracking the quality of collected frames, per camera
    - Adjusting exposure times based on measured counts
    """

    def __init__(self, run: BurrRun, sk_data_root: Path | None = None):
        """
        Initialize the flat field manager.

        Parameters
        ----------
        run : BurrRun
            The run configuration and state.
        sk_data_root : Path, optional
            Directory where the SK camera data graph writes FITS frames.
            When set, ``on_request_completed`` finds every flat frame the
            cameras wrote for the request (by ``IMAGETYP`` and write time,
            not filename) and feeds each into ``process_exposure`` so the
            exposure-adjustment loop closes. Direct mode leaves this
            ``None`` because its executor calls ``process_exposure``
            directly with the temp FITS path before invoking
            ``on_request_completed``.
        """
        self.run = run
        self.config = run.config.schedule.twilight_flats
        self.target_flats = self.config.n_frames
        self.min_counts = self.config.min_counts
        self.max_counts = self.config.max_counts
        self.target_counts = (self.min_counts + self.max_counts) / 2
        self.minimum_exposure = self.config.minimum_exposure_seconds
        self.initial_exposure = self.config.initial_exposure_seconds
        self.maximum_exposure = self.config.maximum_exposure_seconds
        self.sky_doubling_seconds = self.config.sky_doubling_seconds
        self.cutout_size = self.config.cutout_size
        self.frame_settle_seconds = self.config.frame_settle_seconds
        self.max_saturation_percentage = self.config.max_saturation_percentage
        self.saturation_threshold = self.config.saturation_threshold
        self.sk_data_root = Path(sk_data_root) if sk_data_root else None
        self.plan: FlatPlanConfig | None = self.config.plan
        if self.plan is not None and self.sk_data_root is None:
            raise ValueError(
                "twilight_flats.plan needs SensorKit mode (a raw_data_root to read frames from)"
            )
        self._planned: dict[FlatSequenceKey, FlatPairPlan] = (
            self.plan.by_key() if self.plan else {}
        )
        # SensorKit task ID and per-pair request IDs of each dispatched request.
        self._dispatched: dict[str, str] = {}

        # Track the last in-flight request's start time so on_request_completed
        # only picks up frames written for this request.
        self._request_started_at: datetime | None = None
        # Cameras that wrote a frame for the last completed request; used to
        # decide how many frames to wait for on the next one.
        self._expected_cameras: int = 0

        # Restore every camera/filter sequence from verified accepted products,
        # reconciled with run history (see _restore_sequences).
        self.cameras: dict[FlatSequenceKey, FlatSequenceState] = {}
        self._accepted_sources: set[str] = set()
        self._accepted_digests: set[str] = set()
        self._restore_sequences()
        for key in self._planned:
            self.cameras.setdefault(key, FlatSequenceState())
        for key in sorted(set(self.cameras) - set(self._planned)) if self.plan else ():
            logger.warning(
                f"Accepted flats exist for {key.label()}, which is not in the flat plan; "
                f"left in place and not counted toward completion"
            )
        self.current_exposure = self._clip_exposure(self.initial_exposure)

        logger.info(f"Initialized FlatFieldManager (target: {self.target_flats} flats per camera/filter)")
        logger.info(f"├── Count range: {self.min_counts:,} - {self.max_counts:,} ADU (target {self.target_counts:,.0f})")
        logger.info(f"├── Exposure range: {self.minimum_exposure}s - {self.maximum_exposure}s")
        logger.info(f"├── Sky doubling time: {self.sky_doubling_seconds}s")
        logger.info(f"├── Stats cutout: {self.cutout_size or 'full frame'}")
        logger.info(f"├── Max saturation: {self.max_saturation_percentage}%")
        logger.info(f"├── Saturation threshold: {self.saturation_threshold or 'auto-detect'}")
        logger.info(f"└── Resuming from {self.good_frames} existing good frames ({len(self.cameras)} sequence(s))")
        if self.plan is not None:
            logger.info(
                f"Flat plan: {len(self._planned)} pair(s): {self.progress_summary()}"
            )

    # ------------------------------------------------------------------
    # TaskSource protocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for this task source."""
        return "twilight_flats"

    @property
    def suitable_conditions(self) -> list[LightingCondition]:
        """Lighting conditions under which this source can generate tasks.

        Reads from the ``windows`` list in the twilight_flats config section,
        e.g. ``["nautical_twilight", "astronomical_twilight"]``.
        """
        return [
            LightingCondition(window.lower())
            for window in self.config.windows
        ]

    @property
    def good_frames(self) -> int:
        """Good frames of the sequence furthest behind (done when it is).

        With a plan, only planned pairs count.
        """
        states = (
            [self.cameras[key] for key in self._planned]
            if self.plan
            else list(self.cameras.values())
        )
        if not states:
            return 0
        return min(state.good_frames for state in states)

    def target_for(self, key: FlatSequenceKey) -> int:
        """Accepted frames required for ``key``: its plan entry's, else ``n_frames``."""
        pair = self._planned.get(key)
        return self.plan.target(pair) if pair is not None else self.target_flats

    def remaining_pairs(self) -> list[FlatSequenceKey]:
        """Planned pairs still short of their target, in plan order."""
        return [
            key
            for key in self._planned
            if self.cameras[key].good_frames < self.target_for(key)
        ]

    def _ordered_remaining_pairs(self, now: datetime) -> list[FlatSequenceKey]:
        """Return incomplete planned pairs in the correct twilight filter order.

        Without a throughput table, preserve existing plan order.

        With a table, reorder filters independently for each camera:
        - dusk: lowest throughput first
        - dawn: highest throughput first

        Camera positions in the overall plan are preserved. Python's stable
        sort preserves plan order for filters with equal throughput.
        """
        pairs = self.remaining_pairs()

        if self.plan is None or self.plan.filter_throughput is None:
            return pairs

        throughput = self.plan.filter_throughput
        dawn = self._is_dawn(now)

        def pair_throughput(key: FlatSequenceKey) -> float:
            # Planned SensorKit pairs must always identify a filter.
            if key.filter is None:
                raise ValueError(f"planned flat pair {key.label()} has no filter")
            return throughput[key.filter]

        ordered = list(pairs)

        camera_order = list(dict.fromkeys(key.camera for key in pairs))
        for camera in camera_order:
            positions = [
                index for index, key in enumerate(pairs) if key.camera == camera
            ]
            camera_pairs = [pairs[index] for index in positions]
            camera_pairs = sorted(
                camera_pairs,
                key=pair_throughput,
                reverse=dawn,
            )

            for position, key in zip(positions, camera_pairs, strict=True):
                ordered[position] = key

        return ordered

    def progress_summary(self) -> str:
        """``C0/Lum=3/10, C1/Lum=0/10`` for planned pairs, or seen sequences without a plan."""
        keys = list(self._planned) if self.plan else sorted(self.cameras)
        return ", ".join(
            f"{key.label()}={self.cameras[key].good_frames}/{self.target_for(key)}"
            for key in keys
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _restore_sequences(self) -> None:
        """Rebuild each camera/filter sequence from accepted products and history.

        Counts come only from verified products on disk (plus attributable
        legacy products); history is reconciled against them and anything
        that does not match is reported, never counted. See
        ``products.reconcile_history`` and ``docs/flats.md``.
        """
        output_directory = Path(self.run.output_directory)
        restored = scan_products(
            output_directory, sensorkit=self.sk_data_root is not None
        )
        reconcile_history(restored, output_directory, self.run.executed_commands)
        log_restore_report(restored)
        for key in sorted({*restored.products, *restored.legacy_counts}):
            self.cameras[key] = FlatSequenceState(
                good_frames=restored.accepted_count(key),
                next_sequence=restored.next_sequence.get(key, 1),
            )
        self._accepted_sources = set(restored.source_paths)
        self._accepted_digests = set(restored.source_hashes)
        self.restore_report = restored

    def _is_dawn(self, when: datetime) -> bool:
        """True if ``when`` is after local midnight, i.e. the sky is brightening."""
        midnight = self.run.lighting_schedule.local_midnight
        if midnight.tzinfo is None:
            midnight = midnight.replace(tzinfo=ZoneInfo("UTC"))
        return when >= midnight

    def _next_exposure_for(
        self,
        last_exposure: float,
        average_counts: float,
        taken_at: datetime,
        *,
        now: datetime | None = None,
    ) -> float:
        """Exposure that should hit ``target_counts`` at ``now``.

        Scale the previous exposure to the target counts, then compensate for
        sky brightening or darkening since the frame was taken. If ``now`` is
        omitted, use the current UTC time.
        """
        if now is None:
            now = datetime.now(ZoneInfo("UTC"))
        dawn = self._is_dawn(now)

        if dawn != self._is_dawn(taken_at):
            return self.minimum_exposure

        exposure = last_exposure * (self.target_counts / max(average_counts, 1.0))
        elapsed = max((now - taken_at).total_seconds(), 0.0)
        sky_factor = 2.0 ** min(elapsed / self.sky_doubling_seconds, 64.0)
        exposure = exposure / sky_factor if dawn else exposure * sky_factor

        return float(np.clip(exposure, self.minimum_exposure, self.maximum_exposure))

    def _shared_exposure(self) -> float:
        """One exposure for the next request, serving cameras still collecting.

        Takes the median of the per-sequence ideal exposures (re-projected to
        now for sky change) over camera/filter sequences not yet complete.
        """
        candidates = []
        for key, state in self.cameras.items():
            if state.good_frames >= self.target_for(key) or not state.exposures:
                continue
            last = state.exposures[-1]
            candidates.append(self._next_exposure_for(last.exposure_time, last.median_counts, last.timestamp))
        if not candidates:
            return self.current_exposure
        return float(np.median(candidates))

    def _frame_stats(self, data: np.ndarray) -> FrameStats:
        """Count statistics over the central cutout, plus full-frame saturation."""
        cutout = _central_cutout(data, self.cutout_size).astype(np.float32)
        average, median, std = sigma_clipped_stats(cutout)
        saturation_percentage, threshold = self._calculate_saturation_percentage(data)
        return FrameStats(
            average_counts=float(average),
            median_counts=float(median),
            mean_counts=float(np.mean(cutout)),
            std_counts=float(std),
            min_counts=float(np.min(cutout)),
            max_counts=float(np.max(cutout)),
            saturation_percentage=saturation_percentage,
            saturation_threshold=threshold,
        )

    def process_exposure(self, frame_path: str, device_control=None) -> bool:
        """
        Process a collected flat field frame.

        Parameters
        ----------
        frame_path : str
            Path to the FITS file produced by the camera.
        device_control : DirectDeviceControl, optional
            Direct-mode device control. When supplied (direct path),
            good frames are re-written via ``device_control.save_fits``
            using the camera's most recent image and full live metadata.
            When ``None`` (SK path), the FITS at ``frame_path`` is
            already the canonical SK output, so we copy it into the
            run's flats directory and stamp the flat-specific headers
            into the copy.

        Returns
        -------
        bool
            True if the frame meets quality criteria
        """
        try:
            frame_path = Path(frame_path).resolve()
            logger.info(f"Processing flat field frame: {frame_path}")
            data, header = load_image_hdu(frame_path)
            frame = self._identify(frame_path, header, sensorkit=device_control is None)
        except (FlatRejected, UnsafeIdentityError) as e:
            logger.warning(f"Flat frame not accepted: {e}")
            return False
        except Exception as e:
            logger.error(f"Error processing flat field frame: {str(e)}")
            logger.exception(e)  # Log full traceback
            return False
        return self._measure_and_keep(frame, data, header, device_control)

    def _identify(
        self, frame_path: Path, header: fits.Header, *, sensorkit: bool
    ) -> _Frame:
        """Resolve a frame's camera/filter sequence and provenance.

        SensorKit frames must carry a valid IMAGETYP/UNITID/FILTER identity
        and a task UUID, and are rejected if this exact source frame was
        already accepted (by path or content, including before a restart).
        Direct mode keeps its fallback: missing UNITID is the default camera
        and a missing FILTER is allowed.

        Raises:
            FlatRejected: Not a usable flat, duplicate source, or ambiguous identity.
            UnsafeIdentityError: Camera or filter cannot be used as a path component.
        """
        if not sensorkit:
            camera = _header_text(header, "UNITID") or DEFAULT_CAMERA
            key = FlatSequenceKey(camera, _header_text(header, "FILTER") or None)
            return _Frame(frame_path, validate_key(key), None, None)

        try:
            key = validate_key(FlatSequenceKey(*validate_sk_flat_header(header)))
        except UnsafeIdentityError:
            raise
        except ValueError as e:
            raise FlatRejected(str(e)) from e
        if self.plan is not None and key not in self._planned:
            raise FlatRejected(f"{key.label()} is not in the flat plan; not counted")
        uuid = _header_text(header, "SKTASKID") or _header_text(header, "TASKID")
        if not uuid:
            raise FlatRejected("SensorKit flat has no SKTASKID or TASKID")
        collision = find_case_collision(key, self.cameras)
        if collision is not None:
            raise FlatRejected(
                f"{key.label()} would share output with {collision.label()} on a case-insensitive filesystem"
            )
        digest = source_digest(frame_path)
        if (
            str(frame_path) in self._accepted_sources
            or digest in self._accepted_digests
        ):
            raise FlatRejected(f"source frame {frame_path} was already accepted")
        return _Frame(frame_path, key, uuid, digest)

    def _measure_and_keep(
        self, frame: _Frame, data: np.ndarray, header: fits.Header, device_control
    ) -> bool:
        """Measure a frame, update its sequence's exposure model, and keep it if good."""
        key = frame.key
        state = self.cameras.get(key, FlatSequenceState())
        exposure_time = float(header.get("EXPTIME", 0) or self.current_exposure)
        timestamp = _parse_date_obs(header.get("DATE-OBS"))
        if timestamp is None:
            logger.debug("No valid DATE-OBS in header, using current time")
            timestamp = datetime.now(ZoneInfo("UTC"))

        stats = self._frame_stats(data)
        counts = stats.average_counts

        logger.info(f"Frame statistics [{key.label()}] ({exposure_time:.3f}s):")
        logger.info(
            f"├── Clipped mean: {counts:,.0f} ADU (median {stats.median_counts:,.0f}, std {stats.std_counts:,.0f})"
        )
        logger.info(
            f"├── Cutout min/max: {stats.min_counts:,.0f} / {stats.max_counts:,.0f} ADU"
        )
        logger.info(
            f"└── Saturation: {stats.saturation_percentage:.2f}% (threshold: {stats.saturation_threshold})"
        )

        counts_good = self.min_counts <= counts <= self.max_counts
        saturation_good = stats.saturation_percentage <= self.max_saturation_percentage
        # A sequence that is already complete still updates its exposure
        # model, but its extra frames are not kept.
        target = self.target_for(key)
        is_good = counts_good and saturation_good and state.good_frames < target

        self.cameras[key] = state
        state.exposures.append(
            ExposureRecord(
                timestamp=timestamp,
                exposure_time=exposure_time,
                median_counts=counts,
                saturation_percentage=stats.saturation_percentage,
                is_good=is_good,
            )
        )
        state.next_exposure = self._next_exposure_for(exposure_time, counts, timestamp)
        logger.info(f"[{key.label()}] ideal next exposure: {state.next_exposure:.3f}s")

        if is_good:
            return self._save_good_frame(
                frame, state, stats, exposure_time, timestamp, device_control
            )
        if state.good_frames >= target:
            logger.info(
                f"[{key.label()}] already has {state.good_frames} flats — frame not kept"
            )
        else:
            self._log_rejection(key, stats, exposure_time, counts_good, saturation_good)
        return False

    def _save_good_frame(
        self,
        frame: _Frame,
        state: FlatSequenceState,
        stats: FrameStats,
        exposure_time: float,
        acquired: datetime,
        device_control,
    ) -> bool:
        """Publish a good flat, then count it and record it in history.

        Nothing is counted or recorded unless publication succeeded, so a
        failed write leaves state, history, existing products and the source
        as they were. Returns True if the product was published.
        """
        key = frame.key
        sequence = state.next_sequence
        output_dir = product_dir(Path(self.run.output_directory), key)
        if device_control is None:
            output_path = output_dir / product_name(key, acquired, sequence, frame.uuid)
        else:
            output_path = output_dir / f"flat_{sequence:03d}.fits"

        try:
            self._publish(
                frame,
                output_path,
                self._product_cards(frame, sequence, stats),
                device_control,
            )
        except FileExistsError:
            logger.error(
                f"[{key.label()}] refusing to overwrite existing flat {output_path}; frame not kept"
            )
            state.next_sequence = sequence + 1  # that number is taken; never reuse it
            return False
        except (OSError, ValueError, fits.VerifyError) as e:
            # Write, link or verification failure: nothing was published.
            logger.error(
                f"[{key.label()}] could not publish flat {output_path}: {e}; frame not kept"
            )
            return False

        state.next_sequence = sequence + 1
        state.good_frames += 1
        if frame.digest is not None:
            self._accepted_sources.add(str(frame.path))
            self._accepted_digests.add(frame.digest)

        logger.success(
            f"Good flat frame [{key.label()}] ({state.good_frames}/{self.target_for(key)}):"
        )
        logger.info(f"├── Clipped mean counts: {stats.average_counts:,.0f}")
        logger.info(f"├── Saturation: {stats.saturation_percentage:.2f}%")
        logger.info(f"├── Exposure time: {exposure_time:.3f}s")
        logger.info(f"└── Saved to: {output_path}")

        self.run.log_command(
            "flat_field_saved",
            f"{state.good_frames}/{self.target_for(key)} flats collected",
            {
                "camera": key.camera,
                "filter": key.filter,
                "frame_number": state.good_frames,
                "sequence": sequence,
                "median_counts": stats.average_counts,
                "saturation_percentage": stats.saturation_percentage,
                "exposure_time": exposure_time,
                "source_path": str(frame.path),
                "source_sha256": frame.digest,
                "task_uuid": frame.uuid,
                "output_path": str(output_path.relative_to(self.run.output_directory)),
            },
        )
        return True

    def _product_cards(
        self, frame: _Frame, sequence: int, stats: FrameStats
    ) -> dict[str, tuple]:
        """Header cards stamped on an accepted product."""
        key = frame.key
        cards = {
            "IMAGETYP": ("FLAT", "Type of image"),
            "MEDIAN": (stats.median_counts, "Clipped median counts in cutout"),
            "MEAN": (stats.average_counts, "Clipped mean counts in cutout"),
            "STDDEV": (stats.std_counts, "Clipped std dev of counts in cutout"),
            "MINCNTS": (stats.min_counts, "Minimum counts in cutout"),
            "MAXCNTS": (stats.max_counts, "Maximum counts in cutout"),
            "FLATSNUM": (sequence, "Flat field sequence number"),
            SEQUENCE_CARD: (sequence, "Camera/filter sequence number"),
            "SATPCT": (stats.saturation_percentage, "Percentage of saturated pixels"),
            "SATTHRES": (stats.saturation_threshold, "Saturation threshold used"),
        }
        if key.camera != DEFAULT_CAMERA:
            cards["UNITID"] = (key.camera, "Camera unit identifier")
        if key.filter is not None:
            cards["FILTER"] = (key.filter, "Filter reported by the camera")
        if frame.digest is not None:
            # FITS string cards hold 68 characters; long paths keep the hash only.
            if len(str(frame.path)) <= 68:
                cards[SOURCE_PATH_CARD] = (
                    str(frame.path),
                    "Resolved source frame path",
                )
            cards[SOURCE_HASH_CARD] = (frame.digest, "SHA-256 of the source frame")
            cards[UUID_CARD] = (frame.uuid, "SensorKit/burr task UUID")
        return cards

    def _publish(
        self, frame: _Frame, output_path: Path, cards: dict[str, tuple], device_control
    ) -> None:
        """Publish the product without ever overwriting ``output_path``.

        SK mode copies the source frame and stamps ``cards``. Direct mode has
        ``device_control.save_fits`` write the camera's last image (with
        ``cards``) into the temporary file instead. Both then go through the
        same verify-and-link publication.

        Raises:
            FileExistsError: ``output_path`` already exists.
            Exception: The write or verification failed.
        """
        if device_control is None:
            publish_product(
                frame.path,
                output_path,
                cards,
                verify=lambda path: self._verify_product(path, frame),
            )
            return

        def save(temp: Path) -> None:
            device_control.save_fits(
                output_path=temp,
                tracking_mode=TrackingMode.OFF,
                rates=None,
                extra_headers=cards,
            )

        publish_product(None, output_path, cards, verify=lambda path: None, write=save)

    def _verify_product(self, path: Path, frame: _Frame) -> None:
        """Check a stamped temporary product before it is published.

        Raises:
            ValueError: The product's identity or provenance does not match the frame.
        """
        _, header = load_image_hdu(path)
        if FlatSequenceKey(*validate_sk_flat_header(header)) != frame.key:
            raise ValueError(
                f"stamped product does not carry identity {frame.key.label()}"
            )
        if _header_text(header, SOURCE_HASH_CARD) != frame.digest:
            raise ValueError("stamped product does not carry the source hash")

    def _log_rejection(
        self,
        key: FlatSequenceKey,
        stats: FrameStats,
        exposure_time: float,
        counts_good: bool,
        saturation_good: bool,
    ) -> None:
        """Log why a flat frame was rejected, to the log and run history."""
        rejection_reasons = []
        if not counts_good:
            rejection_reasons.append(
                f"counts ({stats.average_counts:,.0f}) outside range [{self.min_counts:,.0f}, {self.max_counts:,.0f}]"
            )
        if not saturation_good:
            rejection_reasons.append(
                f"saturation ({stats.saturation_percentage:.2f}%) exceeds limit ({self.max_saturation_percentage}%)"
            )

        reason = " and ".join(rejection_reasons)
        logger.warning(f"Frame rejected [{key.label()}] - {reason}")

        self.run.log_command(
            "flat_field_rejected",
            f"Frame rejected: {reason}",
            {
                "camera": key.camera,
                "filter": key.filter,
                "median_counts": stats.average_counts,
                "saturation_percentage": stats.saturation_percentage,
                "exposure_time": exposure_time,
                "min_counts": self.min_counts,
                "max_counts": self.max_counts,
                "max_saturation_percentage": self.max_saturation_percentage,
                "rejection_reason": reason,
            },
        )

    def generate_request(
        self, task_id: str, binning: int = 1
    ) -> ObservationRequest | None:
        """Generate the next observation request.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task.
        binning : int
            Current camera binning factor (unused for flats but required by protocol).

        Returns
        -------
        ObservationRequest | None
            A request if one is available, None otherwise.
        """
        # Check if we're done
        if self.is_collection_complete():
            logger.success(
                f"Flat collection complete! Collected {self.good_frames} good frames per camera/filter"
            )
            return None

        if self.plan is not None:
            return self._planned_request(task_id)

        # Re-project every camera's ideal exposure to now (the sky has moved
        # on since the last frames were analysed) and share one exposure.
        self.current_exposure = self._shared_exposure()

        # Stamp the request's start time so on_request_completed only
        # picks up frames written for this request.
        self._request_started_at = datetime.now(ZoneInfo("UTC"))

        # Generate request with current exposure time
        request = ObservationRequest(
            task_id=task_id,
            source_name="twilight_flats",
            target=AltAzTarget(altitude_deg=75.0, azimuth_deg=0.0),
            exposure_seconds=[self.current_exposure],
            exposure_map=[ExposureSpec(tracking_mode=TrackingMode.OFF)],
            rates=None,
            metadata={"good_frames": self.good_frames, "target_flats": self.target_flats},
        )

        logger.info(f"Generated flat field request ({self.good_frames + 1}/{self.target_flats}):")
        logger.info(f"├── Task ID: {task_id}")
        logger.info(f"├── Exposure: {self.current_exposure:.3f}s")
        logger.info(f"├── Pointing: Alt={75.0}, Az={0.0}")
        logger.info("└── Tracking: OFF")

        return request


    def _clip_exposure(self, exposure: float) -> float:
        """Clip an exposure to the configured hardware limits."""
        return float(
            np.clip(
                exposure,
                self.minimum_exposure,
                self.maximum_exposure,
            )
        )

    def _current_twilight_record(
        self,
        state: FlatSequenceState,
        now: datetime,
    ) -> ExposureRecord | None:
        """Return the newest record from the same dusk/dawn as ``now``."""
        dawn = self._is_dawn(now)
        candidates = [
            record
            for record in state.exposures
            if self._is_dawn(record.timestamp) == dawn
        ]

        return max(
            candidates,
            key=lambda record: record.timestamp,
            default=None,
        )

    def _pair_exposure(
        self, key: FlatSequenceKey, *, now: datetime | None = None
    ) -> float:
        """Return the best exposure estimate for a planned pair"""
        if now is None:
            now = datetime.now(ZoneInfo("UTC"))
        state = self.cameras[key]
        own_record = self._current_twilight_record(state, now)

        # once the pair has its own current-twilight measurement
        # normal count feedback takes precedence

        if own_record is not None:
            return self._next_exposure_for(
                own_record.exposure_time,
                own_record.median_counts,
                own_record.timestamp,
                now=now,
            )
        throughput = (self.plan.filter_throughput
                      if self.plan is not None
                      else None)
        
        # without relative filter data use reference start
        if throughput is None:
            return self._clip_exposure(self.initial_exposure)

        if key.filter is None:
            raise ValueError(f"planned flat pair {key.label()} has no filter")

        seeds = []

        for source_key in self._planned:
            if source_key == key:
                continue
            if source_key.camera != key.camera:
                continue
            if source_key.filter is None:
                continue

            record = self._current_twilight_record(self.cameras[source_key], now)
            if record is not None:
                seeds.append((record.timestamp, source_key, record))

        if seeds:
            _, source_key, record = max(seeds, key=lambda item: item[0])
            estimate = (record.exposure_time * throughput[source_key.filter] / throughput[key.filter])

        else:
            estimate = (self.initial_exposure / throughput[key.filter])

        return self._clip_exposure(estimate)

    def _planned_request(self, task_id: str) -> ObservationRequest:
        """One request with its own pinned exposure for every incomplete planned pair.

        ``metadata["flat_plan"]`` lists, per pair, the camera, the SensorKit
        instrument to pin, the filter, its exposure and a ``request_id``. The
        SensorKit burr program turns each entry into its own
        ``CameraParameterSet``. ``exposure_seconds`` holds the shortest, only
        to keep the request well formed.
        """
        now = datetime.now(ZoneInfo("UTC"))
        ordered_pairs = self._ordered_remaining_pairs(now)

        entries = [
            {
                "camera": key.camera,
                "instrument": self._planned[key].instrument,
                "filter": key.filter,
                "exposure_seconds": self._pair_exposure(key, now=now),
                "request_id": str(uuidlib.uuid4()),
            }
            for key in ordered_pairs
        ]
        self._request_started_at = now
        self.current_exposure = min(
            entry["exposure_seconds"] for entry in entries
        )
        request = ObservationRequest(
            task_id=task_id,
            source_name="twilight_flats",
            target=AltAzTarget(altitude_deg=75.0, azimuth_deg=0.0),
            exposure_seconds=[self.current_exposure],
            exposure_map=[ExposureSpec(tracking_mode=TrackingMode.OFF)],
            rates=None,
            metadata={"flat_plan": entries},
        )
        twilight = "dawn" if self._is_dawn(now) else "dusk"
        logger.info(
            f"Generated planned flat request {task_id} for "
            f"{len(entries)} pair(s), ordered for {twilight}:"
        )
        for entry in entries:
            key = FlatSequenceKey(entry["camera"], entry["filter"])
            logger.info(
                f"├── {key.label()} on {entry['instrument']}: "
                f"{entry['exposure_seconds']:.3f}s "
                f"({self.cameras[key].good_frames}/{self.target_for(key)})"
            )
        logger.info("└── Pointing: Alt=75.0, Az=0.0, tracking OFF")
        return request

    def is_collection_complete(self) -> bool:
        """Check whether collection is done.

        With a plan: every planned pair has reached its target, including
        pairs that have not yet returned any frame. Without one: every
        sequence seen so far has ``n_frames``; only sequences that produced a
        frame are known, so this cannot tell whether every camera/filter was
        collected.
        """
        if self.plan is not None:
            return not self.remaining_pairs()
        return bool(self.cameras) and self.good_frames >= self.target_flats

    def on_request_completed(
        self, request: ObservationRequest, duration_seconds: float
    ) -> None:
        """Called when an observation request completes successfully.

        In direct mode, ``process_exposure`` was already invoked by the
        executor before this callback fires, so this just logs.

        In SK mode, each camera's data graph writes its FITS to disk
        asynchronously, after the collect task returns, and burr never
        sees the paths. We close the loop here by waiting for the flat
        frames written since the request started to land under
        ``sk_data_root`` and feeding each into ``process_exposure`` so
        every camera's exposure model and good-frame count advance.

        Parameters
        ----------
        request : ObservationRequest
            The completed request.
        duration_seconds : float
            Wall-clock time the request took.
        """
        logger.info(
            f"Flat request {request.task_id} completed in {duration_seconds:.1f}s "
            f"({self.good_frames}/{self.target_flats} good frames so far)"
        )

        if self.sk_data_root is None:
            return
        if self.plan is not None:
            self._complete_planned(request)
            return

        frame_paths = self._wait_for_sk_flats(request)
        if not frame_paths:
            logger.warning(
                f"No flat FITS found under {self.sk_data_root} for request "
                f"{request.task_id} — exposure adjustment skipped"
            )
            return

        logger.info(f"Reading back {len(frame_paths)} SK-written flat(s) for analysis")
        for frame_path in frame_paths:
            self.process_exposure(str(frame_path), device_control=None)
        self._expected_cameras = max(self._expected_cameras, len(frame_paths))

        logger.info(
            "Flats per camera/filter: "
            + ", ".join(
                f"{key.label()}={state.good_frames}"
                for key, state in sorted(self.cameras.items())
            )
        )

    def on_task_dispatched(self, request: ObservationRequest, sk_task_id: str) -> None:
        """Record the SensorKit task ID minted for ``request``.

        The SensorKit burr program calls this once the controller accepts the
        task. Planned frames are matched to the request by this ID, which the
        camera data graph writes into every frame as ``SKTASKID``.
        """
        self._dispatched[request.task_id] = str(sk_task_id)
        logger.debug(
            f"Flat request {request.task_id} dispatched as SensorKit task {sk_task_id}"
        )

    def _complete_planned(self, request: ObservationRequest) -> None:
        """Match a planned request's frames exactly, then measure each pair's frame."""
        sk_task_id = self._dispatched.pop(request.task_id, None)
        expected = {
            FlatSequenceKey(e["camera"], e["filter"]): e
            for e in request.metadata.get("flat_plan", [])
        }
        if sk_task_id is None:
            logger.error(
                f"Flat request {request.task_id} has no recorded SensorKit task ID; its frames cannot be "
                f"matched exactly and are not counted ({', '.join(k.label() for k in expected)})"
            )
            return
        matched, unexpected = self._wait_for_task_frames(sk_task_id, expected)
        for path, reason in unexpected:
            logger.warning(
                f"Frame {path} from task {sk_task_id} is not counted: {reason}"
            )
        for key in expected:
            path = matched.get(key)
            if path is None:
                logger.warning(
                    f"[{key.label()}] returned no frame for request {request.task_id} (task {sk_task_id})"
                )
                continue
            self.process_exposure(str(path), device_control=None)
        logger.info(f"Flats per planned pair: {self.progress_summary()}")

    def _wait_for_task_frames(
        self, sk_task_id: str, expected: dict[FlatSequenceKey, dict]
    ) -> tuple[dict[FlatSequenceKey, Path], list[tuple[Path, str]]]:
        """Wait for one frame per expected pair carrying ``sk_task_id``.

        Returns the frame matched to each pair, and frames from this task that
        match no expected pair (an unplanned camera or filter, or a second
        frame for one pair). Polls until every pair has a frame or
        ``frame_settle_seconds`` elapses.
        """
        deadline = time.monotonic() + self.frame_settle_seconds
        while True:
            matched, unexpected = self._task_frames(sk_task_id, expected)
            if len(matched) == len(expected) or time.monotonic() >= deadline:
                return matched, unexpected
            time.sleep(0.5)

    def _task_frames(
        self, sk_task_id: str, expected: dict[FlatSequenceKey, dict]
    ) -> tuple[dict[FlatSequenceKey, Path], list[tuple[Path, str]]]:
        """Frames under ``sk_data_root`` whose ``SKTASKID`` is ``sk_task_id``, sorted to pairs."""
        matched: dict[FlatSequenceKey, Path] = {}
        unexpected: list[tuple[Path, str]] = []
        output_root = Path(self.run.output_directory).resolve()
        for path in sorted(self.sk_data_root.rglob("*.fits")):
            if path.resolve().is_relative_to(output_root):
                continue
            header = _image_header_cards(path)
            if header is None or _header_text(header, "SKTASKID") != sk_task_id:
                continue
            key = FlatSequenceKey(
                _header_text(header, "UNITID"), _header_text(header, "FILTER")
            )
            if key not in expected:
                unexpected.append(
                    (path, f"{key.label()} was not requested by this task")
                )
            elif key in matched:
                unexpected.append((path, f"second frame for {key.label()}"))
            else:
                matched[key] = path
        return matched, unexpected

    def _wait_for_sk_flats(self, request: ObservationRequest) -> list[Path]:
        """Wait for the cameras' flat frames for ``request`` to land, then return them.

        Polls until as many frames as the previous request produced have
        appeared and the set has been stable for a couple of seconds, or
        ``frame_settle_seconds`` elapses.
        """
        deadline = time.monotonic() + self.frame_settle_seconds
        found: list[Path] = []
        stable_since = time.monotonic()
        while True:
            current = self._find_sk_flats(request)
            if len(current) != len(found):
                found, stable_since = current, time.monotonic()
            settled = time.monotonic() - stable_since >= 2.0
            if found and settled and len(found) >= self._expected_cameras:
                return found
            if time.monotonic() >= deadline:
                return found
            time.sleep(0.5)

    def _find_sk_flats(self, request: ObservationRequest) -> list[Path]:
        """Locate the flat FITS the SK data graph wrote for this request.

        Matches on content rather than filename (the controller's
        ``FileNameTemplate`` may override burr's): a ``*.fits`` file under
        ``sk_data_root`` whose mtime is at or after the request start and
        whose image header has ``IMAGETYP = flat``. Excludes burr's own
        processed copies and in-progress ``.tmp`` writes.
        """
        if self.sk_data_root is None or not self.sk_data_root.exists():
            return []

        cutoff = self._request_started_at
        output_root = Path(self.run.output_directory).resolve()
        frames = []
        for path in self.sk_data_root.rglob("*.fits"):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=ZoneInfo("UTC"))
            except OSError:
                continue
            if cutoff is not None and mtime < cutoff:
                continue
            if path.resolve().is_relative_to(output_root):
                continue
            if _is_flat_frame(path):
                frames.append(path)

        return sorted(frames)

    def on_request_failed(
        self, request: ObservationRequest, error: str
    ) -> None:
        """Called when an observation request fails.

        Parameters
        ----------
        request : ObservationRequest
            The failed request.
        error : str
            Description of the failure.
        """
        self._dispatched.pop(request.task_id, None)
        planned = request.metadata.get("flat_plan") or []
        if planned:
            pairs = ", ".join(
                f"{e['camera']}/{e['filter']} on {e['instrument']}" for e in planned
            )
            logger.error(
                f"Planned flat request {request.task_id} failed; none of its pairs were collected "
                f"({pairs}). SensorKit reported: {error}"
            )
            logger.info(f"Flats per planned pair: {self.progress_summary()}")
            return
        logger.warning(
            f"Flat request {request.task_id} failed: {error} "
            f"({self.good_frames}/{self.target_flats} good frames so far)"
        )

    # ------------------------------------------------------------------
    # Twilight period helpers
    # ------------------------------------------------------------------

    def generate_twilight_periods(
        self,
        useful_conditions: List[LightingCondition] = [
            LightingCondition.ASTRONOMICAL_TWILIGHT,
            LightingCondition.NAUTICAL_TWILIGHT,
        ],
    ) -> List[Twilight]:
        """
        Generate a list of twilight periods where lighting conditions are suitable for flat collection.

        Parameters
        ----------
        useful_conditions : List[LightingCondition]
            List of lighting conditions that are suitable for flat collection.
            Defaults to astronomical and nautical twilight.

        Returns
        -------
        List[Twilight]
            List of Twilight objects containing start and end times for suitable periods
        """
        # Get current time in UTC
        now = datetime.now(ZoneInfo("UTC"))
        logger.info(f"Generating twilight periods from {now} UTC")
        logger.info(f"Useful conditions: {[c.name for c in useful_conditions]}")

        # Use the pre-calculated lighting schedule from the run
        schedule = self.run.lighting_schedule
        logger.info(f"Schedule bounds: {schedule.schedule_start_utc} to {schedule.schedule_end_utc}")

        # Get all time boundaries for each useful condition
        all_boundaries = []
        for condition in useful_conditions:
            boundaries = schedule.get_condition_boundaries(condition)
            logger.info(f"Found {len(boundaries)} periods for {condition.name}:")
            for start, end in boundaries:
                # Ensure boundary times are timezone-aware
                if start.tzinfo is None:
                    start = start.replace(tzinfo=ZoneInfo("UTC"))
                if end.tzinfo is None:
                    end = end.replace(tzinfo=ZoneInfo("UTC"))
                logger.info(f"  ├── {start} UTC")
                logger.info(f"  └── {end} UTC")
                all_boundaries.append((start, end))

        if not all_boundaries:
            logger.warning("No boundaries found for any conditions")
            return []

        # Sort boundaries by start time
        all_boundaries.sort(key=lambda x: x[0])

        # Merge overlapping or adjacent periods
        merged_periods = []
        current_start, current_end = all_boundaries[0]

        for start, end in all_boundaries[1:]:
            # If this period starts after current period ends (with a small gap tolerance)
            if start > current_end + timedelta(minutes=1):
                # Save the current period and start a new one
                merged_periods.append(Twilight(start_time=current_start, end_time=current_end))
                current_start = start
                current_end = end
            else:
                # Extend current period
                current_end = max(current_end, end)

        # Add the last period
        merged_periods.append(Twilight(start_time=current_start, end_time=current_end))

        # Filter out periods that are completely in the past
        future_periods = [period for period in merged_periods if period.end_time > now]

        # For periods that span current time, update start time to now
        for period in future_periods:
            if period.start_time < now:
                period.start_time = now

        logger.info(f"Generated {len(future_periods)} usable twilight periods:")
        for i, period in enumerate(future_periods, 1):
            logger.info(f"Period {i}:")
            logger.info(f"├── Start: {period.start_time} UTC")
            logger.info(f"└── End: {period.end_time} UTC")

        return future_periods

    def _calculate_saturation_percentage(self, data: np.ndarray) -> tuple[float, int]:
        """Calculate the percentage of saturated pixels in the frame.

        Args:
            data: The image data array

        Returns:
            Tuple of (saturation_percentage, saturation_threshold_used)
        """
        # Auto-detect saturation threshold if not configured
        if self.saturation_threshold is None:
            if data.dtype == np.uint8:
                threshold = 255
            elif np.issubdtype(data.dtype, np.integer):
                # SK writes int32 even for 12/14/16-bit sensors (the ASI183's
                # 12-bit ADC is scaled to 16 bits, saturating at 65520), so
                # take the full scale of the smallest bit depth that holds the
                # frame's maximum and allow 0.5% headroom below it.
                bits = max(int(np.ceil(np.log2(float(np.max(data)) + 1.0))), 8)
                threshold = int((2**bits - 1) * 0.995)
            else:
                # For other data types, use 99.9% of the maximum value in the frame
                threshold = int(np.max(data) * 0.999)
        else:
            threshold = self.saturation_threshold

        # Count saturated pixels
        saturated_pixels = np.sum(data >= threshold)
        total_pixels = data.size
        saturation_percentage = (saturated_pixels / total_pixels) * 100.0

        return saturation_percentage, threshold
