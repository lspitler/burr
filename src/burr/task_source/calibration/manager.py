"""Task source for targetless DARK and BIAS calibration collection."""
import math
import time
from pathlib import Path

from astropy.io import fits

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from burr.models.observation import ObservationRequest
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.task_source.calibration.plan import (
    CalibrationPlanConfig,
    CalibrationSequencePlan,
)

SequenceIdentity = tuple[str, str, str, float]
SequenceGroup = tuple[str, str, float]

_EXPOSURE_REL_TOLERANCE = 1e-6
_EXPOSURE_ABS_TOLERANCE_SECONDS = 1e-6
_FITS_READ_ERRORS = (OSError, ValueError, fits.VerifyError)

def _header_text(header: fits.Header, key: str) -> str:
    """Return a stripped fits header value or an empty string"""
    value = header.get(key)
    return "" if value is None else str(value).strip()

def _image_header_cards(path: Path) -> fits.Header | None:
    """Read the first image-HDU header without decompressing its pixels"""
    try:
        with fits.open(path) as hdul:
            for hdu in hdul:
                if (
                    isinstance(hdu, fits.CompImageHDU)
                    or hdu.header.get("NAXIS", 0) >=2
                ):
                    return hdu.header.copy()
    except _FITS_READ_ERRORS as error:
        logger.debug(
            f"skipping unreadable calibration FITS {path}: {error}"
        )
    return None


@dataclass
class CalibrationSequenceState:
    """Runtime state for one configured calibration sequence"""

    completed_frames: int = 0
    consecutive_failures: int = 0


class CalibrationFrameManager:
    """Generate and track targetless DARK and BIAS requests.

    Sequences with the same filter, frame type and exposure are grouped into
    one request so sensorkit can collect that calibration condition from several cameras together.

    Calibration progress advances only after SensorKit FITS files have been associated with the dispatched and validated against the
    configured camer, filter, frame type and exposure.
    """

    def __init__(self, run: BurrRun, sk_data_root: Path | None):
        self.run = run
        self.config = run.config.schedule.calibration_frames
        self.plan = self.config.plan

        if self.plan is None:
            raise ValueError(
                "CalibrationFrameManager requires calibration_frames_plan"
            )
        if sk_data_root is None:
            raise ValueError("CalibrationFrameManager requires a SensorKit data root")

        self.sk_data_root = Path(sk_data_root)
        self.frame_settle_seconds = self.config.frame_settle_seconds
        self.max_request_retries = self.config.max_request_retries
        self._dispatched: dict[str, str] = {}

        self.configured: dict[ SequenceIdentity, CalibrationSequencePlan] = {
            sequence.identity(): sequence
            for sequence in self.plan.sequences
        }
        self._states: dict[
            SequenceIdentity,
            CalibrationSequenceState,
        ] = {
            identity: CalibrationSequenceState()
            for identity in self._configured
        }
        self._completed_logged = False
        self._exhausted_logged = False

    @property
    def name(self) -> str:
        """TaskSource name used for scheduler"""
        return "calibration_frames"

    @property
    def suitable_conditions(self) -> list[LightingCondition]:
        """Lighting conditions where calibration can run"""
        return [
            LightingCondition(window.lower())
            for window in self.config.windows
        ]

    def offer_windows(
        self,
        *,
        withdraw_finished: bool = True,
    ) -> list[tuple]:
        """Return only the explicitly configured calibration windows.

        Once collection is complete, withdraw the windows so SensorKit can
        hand the controller to another program. A window remains advertised
        while a request is in flight.
        """
        if withdraw_finished and self.is_collection_complete():
            return []
        return self.run.lighting_schedule.merged_windows_for(
            self.suitable_conditions
        )

    @property
    def all_targets_met(self) -> bool:
        """whether every configured sequence reached frame target"""
        return all(
            self._remaining(sequence) == 0
            for sequence in self.plan.sequences
        )

    def is_collection_complete(self) -> bool:
        """Whether collection has reached a terminal state.

        - meaning every sequence is either complete or exhausted.
        `all_targets_met` distingushes fully successful completion
        from termination due to repeat faulures
        """
        return all(
            self._remaining(sequence) == 0
            or self._is_exhausted(sequence)
            for sequence in self.plan.sequences
        )

    def progress_totals(self) -> tuple[int, int]:
        """Return completed and required frame totals"""
        completed = sum(
            min(
                self._state(sequence).completed_frames,
                self.plan.target(sequence),
            )
            for sequence in self.plan.sequences
        )
        required = sum(
            self.plan.target(sequence)
            for sequence in self.plan.sequences
        )
        return completed, required

    def generate_request(
        self,
        task_id: str,
        binning: int = 1,
    ) -> ObservationRequest | None:
        """Generate the next camera only calibration request"""
        del binning

        if self.is_collection_complete():
            self._log_terminal_state_once()
            return None

        available = [
            sequence
            for sequence in self.plan.sequences
            if self._remaining(sequence) > 0
            and not self._is_exhausted(sequence)
        ]

        if not available:
            self._log_terminal_state_once()
            return None

        first_group = self._group_key(available[0])
        selected = [
            sequence
            for sequence in available
            if self._group_key(sequence) == first_group
        ]

        request_sequences = [
            sequence.model_copy(
                update={
                    "frames": self._remaining(sequence),
                }
            )
            for sequence in selected
        ]
        request_plan = CalibrationPlanConfig(
            frames_per_sequence=self.plan.frames_per_sequence,
            sequences=request_sequences,
        )

        completed, required = self.progress_totals()
        filter_name, frame_type, exposure_seconds = first_group

        logger.info(
            f"Generated calibration request {task_id}: "
            f"{frame_type} {exposure_seconds:g}s through {filter_name}, "
            f"{len(request_sequences)} camera(s), "
            f"progress {completed}/{required}"
        )

        return ObservationRequest(
            task_id=task_id,
            source_name="calibration_frames",
            target=None,
            exposure_seconds=[],
            exposure_map=[],
            rates=None,
            metadata={
                "target_name": "calibration",
                "calibration_group": {
                    "filter": filter_name,
                    "frame_type": frame_type,
                    "exposure_seconds": exposure_seconds,
                },
                "calibration_plan": request_plan.model_dump(
                    mode="json"
                ),
            },
        )

    def on_task_dispatched(
            self,
            request: ObservationRequest,
            sk_task_id: str,
    ) -> None:
        """Asspcoate a Burr request with its SensorKit execution"""
        self._dispatched(request.task_id) = str(sk_task_id)
        logger.debug(
            f"Calibration request {request.task_id} dispatched as"
            f"SensorKit task {sk_task_id}"
        )

    def on_request_completed(
        self,
        request: ObservationRequest,
        duration_seconds: float,
    ) -> None:
        """Accept only frames validated against the completed task"""
        request_plan = self._request_plan(request)
        updates = []

        # validate the complete request before reading or changing state
        for requested in request_plan.sequences:
            configured = self._configured_sequence(requested)
            state = self._state(configured)
            requested_frames = request_plan.target(requested)
            remaining = self._remaining(configured)

            if requested_frames > remaining:
                raise ValueError(
                    f"request for {configured.label()} contains "
                    f"{requested_frames} frame(s), but only {remaining} "
                    "remain"
                )

            updates.append(
                (requested, configured, state, requested_frames)
            )

        accepted = self._readback_request(
            request,
            request_plan,
        )

        for(requested, configured, state, requested_frames
            ) in updates:
                paths = accepted.get(configured.identity(), [])
                accepted_frames = len(paths)

                if accepted_frames > requested_frames:
                    raise RuntimeError(
                        f"readback returned {accepted_frames} for"
                        f"{configured.label()}, but the request expected only"
                        f"{requested_frames}"
                    )

                if accepted_frames:
                    state.completed_frames += accepted_frames
                    state.consecutive_failures = 0

                    target = self.plan.targer(configured)
                    complete = state.completed_frames >= target
                    command = (
                        "calibration_sequence_completed"
                        if complete
                        else "calibration_sequence_progress"
                    )

                    logger.info(
                        f"Calibration frames accepted"
                        f"{configured.label()}"
                        f"{state.completed_frames}/{target}"
                    )
                    self.run.log_command(
                        command,
                        (f"{state.completed_frames}/{target}"
                        "validated franes"),
                    {
                        **self._sequence_metadata(configured),
                        "request_id": request.task_id,
                        "request_frames": requested_frames,
                        "accepted_frames": accepted_frames,
                        "completed_frames": state.completed_frames,
                        "target_frames": target,
                        "duration_seconds": duration_seconds,
                        "source_paths": [str(path) for path in paths],
                    }
                )
                missing = requested_frames - accepted_frames
                if missing:
                    self._record_sequence_failure(
                        configured,
                        request,
                        (
                            f"readback accepted {accepted_frames}/"
                            f"{requested_frames} expected frames(s)"
                            f"{missing} missing or invalid"
                        )
                    )

        if self.all_targets_met:
            completed, required = self.progress_totals()
            logger.success(
                f"Calibration frame collection complete:"
                f"{completed}/{required}"
            )


    def on_request_failed(
        self,
        request: ObservationRequest,
        error: str,
    ) -> None:
        """Record a failed request and exhaust repeatedly failing sequences."""
        try:
            request_plan = self._request_plan(request)
        except (TypeError, ValueError) as validation_error:
            logger.error(
                f"Could not account for failed calibration request "
                f"{request.task_id}: {validation_error}"
            )
            return

        for requested in request_plan.sequences:
            try:
                configured = self._configured_sequence(requested)
            except ValueError as validation_error:
                logger.error(
                    f"Could not match failed calibration request "
                    f"{request.task_id}: {validation_error}"
                )
                continue

            state = self._state(configured)
            state.consecutive_failures += 1
            exhausted = self._is_exhausted(configured)

            logger.warning(
                f"Calibration sequence failed: {configured.label()} "
                f"(failure {state.consecutive_failures}, "
                f"retry limit {self.max_request_retries}): {error}"
            )
            self.run.log_command(
                "calibration_sequence_failed",
                error,
                {
                    **self._sequence_metadata(configured),
                    "request_id": request.task_id,
                    "consecutive_failures": state.consecutive_failures,
                    "max_request_retries": self.max_request_retries,
                    "exhausted": exhausted,
                },
                error=error,
            )

            if exhausted:
                logger.error(
                    f"Calibration sequence exhausted after repeated "
                    f"failures: {configured.label()}"
                )

    def _readback_request(
            self,
            request: ObservationRequest,
            request_plan: CalibrationPlanConfig,
    ) -> dict[SequenceIdentity, list[Path]]:
        """Read and validate FITS files produced by one SensorKit task."""
        sk_task_id = self._dispatched.pop(
            request.task_id,
            None,
        )
        if sk_task_id is None:
            raise ValueError(
                f"calibration request {request.task_id} has no recorded SensorKit task ID"
            )
        if not self.sk_data_root.exists():
            raise FileNotFoundError(
                f"SensorKit data root does not exist:"
                f"{self.sk_data_root}"
            )

        expected: dict[
            SequenceIdentity,
            tuple[CalibrationSequencePlan, int],
        ] = {}

        for requested in request_plan.sequences:
            configured = self._configured_sequence(requested)
            expected[configured.identity()] = (
                configured,
                request_plan.target(requested)
            )

        accepted, rejected = self._wait_for_task_frames(
            sk_task_id,
            expected
        )

        for path, reason in rejected:
            logger.warning(
                f"Calibration frame {path} from task {sk_task_id} was rejected {reason}"
            )
            self.run.log_command(
                "calibration_frame_rejected",
                reason,
                {
                    "request_id": request.task_idm
                    "sk_task_id": sk_task_id,
                    "source_path": str(path),
                    "reason": reason,
                },
            )
        return accepted

    def _wait_for_task_frames(
            self, sk_task_id: str, expected: dict[SequenceIdentity, tuple[CalibrationSequencePlan, int]],
    ) -> tuple[dict[SequenceIdentity, list[Path]],
        list[tuple[Path, str]]
    ]:
        """Wait for all expected files carryng SensorKit task ID."""
        required = sum(frame_count for _, frame_count in expected.values())
        deadline = time.monotonic() + self.frame_settle_seconds

        while True:
            accepted, rejected = self._task_frames(
                sk_task_id,
                expected,
            )
            found = sum(
                len(paths)
                for paths in accepted.values()
            )

            if (
                found >=required
                or time.monotonic() >= deadline
            ):
                return accepted, rejected

            time.sleep(0.5)

    def _task_frames(
            self,
            sk_task_id: str,
            expected: dict[SequenceIdentity, tuple[CalibrationSequencePlan, int]]
    ) -> tuple [
        dict[SequenceIdentity, list[Path]],
        list[tuple[Path, str]]
    ]:
        

    def _request_plan(
        self,
        request: ObservationRequest,
    ) -> CalibrationPlanConfig:
        """Validate and return the plan carried by one Burr request"""
        if request.source_name != self.name:
            raise ValueError(
                f"expected source_name={self.name!r}"
                f"got {request.source_name!r}"
            )
        if request.target is not None:
            raise ValueError(
                "calibration requests must have target=None"
            )

        raw_plan = request.metadata.get("calibration_plan")
        if raw_plan is None:
            raise ValueError(
                "calibration request has no calibration_plan"
            )

        return CalibrationPlanConfig.model_validate(raw_plan)

    def _configured_sequence(
        self,
        requested: CalibrationSequencePlan,
    ) -> CalibrationSequencePlan:
        """Return the configured sequence for a request entry"""
        configured = self._configured.get(requested.identity())

        if configured is None:
            raise ValueError(
                f"{requested.label()} is not in the configured plan"
            )

        if (
            requested.camera != configured.camera
            or requested.instrument != configured.instrument
            or requested.filter != configured.filter
            or requested.frame_type != configured.frame_type
            or requested.exposure_seconds
            != configured.exposure_seconds
        ):
            raise ValueError(
                f"{requested.label()} does not exactly match its configured sequence"
            )
        return configured

    def _state(
        self,
        sequence: CalibrationSequencePlan,
    ) -> CalibrationSequenceState:
        return self._states[sequence.identity()]

    def _remaining(
        self,
        sequence: CalibrationSequencePlan,
    ) -> int:
        state = self._state(sequence)
        return max(
            self.plan.target(sequence) - state.completed_frames,
            0,
        )

    def _is_exhausted(
        self,
        sequence: CalibrationSequencePlan,
    ) -> bool:
        state = self._state(sequence)
        return (
            state.consecutive_failures
            > self.max_request_retries
        )

    @staticmethod
    def _group_key(
        sequence: CalibrationSequencePlan,
    ) -> SequenceGroup:
        return (
            sequence.filter,
            sequence.frame_type,
            sequence.exposure_seconds,
        )

    @staticmethod
    def _sequence_metadata(
        sequence: CalibrationSequencePlan,
    ) -> dict:
        return {
            "camera": sequence.camera,
            "instrument": sequence.instrument,
            "filter": sequence.filter,
            "frame_type": sequence.frame_type,
            "exposure_seconds": sequence.exposure_seconds,
        }

    def _log_terminal_state_once(self) -> None:
        """Log successful or exhausted terminal state only once."""
        completed, required = self.progress_totals()

        if self.all_targets_met:
            if self._completed_logged:
                return
            self._completed_logged = True
            logger.success(
                f"Calibration-frame collection complete: "
                f"{completed}/{required} frames"
            )
            return

        if self._exhausted_logged:
            return
        self._exhausted_logged = True
        exhausted = [
            sequence.label()
            for sequence in self.plan.sequences
            if self._remaining(sequence) > 0
            and self._is_exhausted(sequence)
        ]
        logger.error(
            "Calibration-frame collection stopped with exhausted "
            "sequence(s): " + ", ".join(exhausted)
        )
