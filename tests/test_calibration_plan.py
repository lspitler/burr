"""Validation tests for per-camera DARK and BIAS plans."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from burr.core.config import (
    CalibrationFramesConfig,
    ScheduleConfig,
)
from burr.task_source.calibration.plan import CalibrationPlanConfig


def valid_plan_data():
    return {
        "frames_per_sequence": 20,
        "sequences": [
            {
                "camera": "C0",
                "instrument": "C0-ASI183MM",
                "filter": "Blank",
                "frame_type": "bias",
                "exposure_seconds": 0.0,
            },
            {
                "camera": "C0",
                "instrument": "C0-ASI183MM",
                "filter": "Blank",
                "frame_type": "dark",
                "exposure_seconds": 10.0,
            },
            {
                "camera": "C1",
                "instrument": "C1-ASI183MM",
                "filter": "Blank",
                "frame_type": "dark",
                "exposure_seconds": 30.0,
                "frames": 10,
            },
        ],
    }


def test_plan_resolves_default_and_sequence_frame_counts():
    plan = CalibrationPlanConfig.model_validate(valid_plan_data())

    assert plan.target(plan.sequences[0]) == 20
    assert plan.target(plan.sequences[1]) == 20
    assert plan.target(plan.sequences[2]) == 10
    assert plan.sequences[0].frame_type == "bias"
    assert plan.sequences[0].exposure_seconds == 0.0


def test_zero_second_dark_is_rejected():
    data = valid_plan_data()
    data["sequences"][1]["exposure_seconds"] = 0.0

    with pytest.raises(
        ValidationError,
        match="greater than zero",
    ):
        CalibrationPlanConfig.model_validate(data)


@pytest.mark.parametrize("exposure", [float("inf"), float("nan")])
def test_nonfinite_exposure_is_rejected(exposure):
    data = valid_plan_data()
    data["sequences"][1]["exposure_seconds"] = exposure

    with pytest.raises(ValidationError):
        CalibrationPlanConfig.model_validate(data)


def test_duplicate_sequence_is_rejected():
    data = valid_plan_data()
    data["sequences"].append(
        deepcopy(data["sequences"][1])
    )

    with pytest.raises(
        ValidationError,
        match="duplicate calibration sequence",
    ):
        CalibrationPlanConfig.model_validate(data)


def test_camera_cannot_map_to_two_instruments():
    data = valid_plan_data()
    data["sequences"][1]["instrument"] = "other-camera"

    with pytest.raises(
        ValidationError,
        match="conflicts with existing camera",
    ):
        CalibrationPlanConfig.model_validate(data)


def test_instrument_cannot_map_to_two_cameras():
    data = valid_plan_data()
    data["sequences"][2]["instrument"] = "C0-ASI183MM"

    with pytest.raises(
        ValidationError,
        match="conflicts with existing instrument",
    ):
        CalibrationPlanConfig.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera", "../C0"),
        ("filter", "Blank/Unsafe"),
        ("instrument", "camera with spaces"),
    ],
)
def test_unsafe_identity_is_rejected(field, value):
    data = valid_plan_data()
    data["sequences"][0][field] = value

    with pytest.raises(ValidationError):
        CalibrationPlanConfig.model_validate(data)


def test_unknown_sequence_field_is_rejected():
    data = valid_plan_data()
    data["sequences"][0]["unexpected"] = True

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        CalibrationPlanConfig.model_validate(data)


def test_enabled_collection_requires_windows_and_plan():
    plan = CalibrationPlanConfig.model_validate(valid_plan_data())

    with pytest.raises(
        ValidationError,
        match="windows",
    ):
        CalibrationFramesConfig(
            collect=True,
            plan=plan,
        )

    with pytest.raises(
        ValidationError,
        match="plan",
    ):
        CalibrationFramesConfig(
            collect=True,
            windows=["day"],
        )


def test_schedule_defaults_calibration_collection_off():
    schedule = ScheduleConfig()

    assert schedule.calibration_frames.collect is False
    assert schedule.calibration_frames.plan is None


def test_schedule_parses_enabled_calibration_plan():
    schedule = ScheduleConfig.model_validate(
        {
            "calibration_frames": {
                "collect": True,
                "windows": ["day"],
                "frame_settle_seconds": 60,
                "plan": valid_plan_data(),
            }
        }
    )

    assert schedule.calibration_frames.collect is True
    assert len(schedule.calibration_frames.plan.sequences) == 3