"""Task source for targetless DARK and BIAS calibration collection."""

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


@dataclass
class CalibrationSequenceState:
    """Runtime state for one configured calibration sequence"""

    completed_frames: int = 0
    consecutive_failures: int = 0


class CalibrationFrameManager:
    """Generate and track targetless DARK and BIAS requests.

    Sequences with the same filter, frame type and exposure are grouped into
    one request so sensorkit can collect that calibration condition from several cameras together.

    This source is disabled by default, until calibration FITS readback is implemented.
    """

    def __init__(self, run: BurrRun):
        self.run = run
        self.config = run.config.schedule.calibration_frames
        self.plan = self.config.plan

        if self.plan is None:
            raise ValueError("CalibrationFrameManager requires calibration_frames.plan")

        self.max_request_retries = self.config.max_request_retries
        self._configured: dict[SequenceIdentity, CalibrationSequencePlan] = {
            sequence.identity(): sequence
            for sequence in self.plan.sequences
        }
        self._states: dict[SequenceIdentity, CalibrationSequenceState] = {
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

    def on_request_completed(
        self,
        request: ObservationRequest,
        duration_seconds: float,
    ) -> None:
        """Count frames from a successfully completed SensorKit request."""
        request_plan = self._request_plan(request)
        updates = []

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
                (configured, state, requested_frames)
            )

        for configured, state, requested_frames in updates:
            state.completed_frames += requested_frames
            state.consecutive_failures = 0

            target = self.plan.target(configured)
            logger.info(
                f"Calibration sequence completed: "
                f"{configured.label()} "
                f"{state.completed_frames}/{target}"
            )
            self.run.log_command(
                "calibration_sequence_completed",
                (
                    f"{state.completed_frames}/{target} "
                    "frames completed"
                ),
                {
                    **self._sequence_metadata(configured),
                    "request_id": request.task_id,
                    "request_frames": requested_frames,
                    "completed_frames": state.completed_frames,
                    "target_frames": target,
                    "duration_seconds": duration_seconds,
                },
            )

        if self.all_targets_met:
            completed, required = self.progress_totals()
            logger.success(
                f"Calibration-frame collection complete:"
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
