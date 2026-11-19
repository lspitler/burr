"""Task-source tests for DARK and BIAS calibration collection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from burr.core.config import CalibrationFramesConfig
from burr.models.observation import RADecTarget
from burr.task_source.calibration.manager import (
    CalibrationFrameManager,
)
from burr.task_source.calibration.plan import CalibrationPlanConfig


def plan_data():
    return {
        "frames_per_sequence": 2,
        "sequences": [
            {
                "camera": "C0",
                "instrument": "east",
                "filter": "Blank",
                "frame_type": "bias",
                "exposure_seconds": 0.0,
            },
            {
                "camera": "C1",
                "instrument": "west",
                "filter": "Blank",
                "frame_type": "bias",
                "exposure_seconds": 0.0,
                "frames": 3,
            },
            {
                "camera": "C0",
                "instrument": "east",
                "filter": "Blank",
                "frame_type": "dark",
                "exposure_seconds": 30.0,
            },
            {
                "camera": "C1",
                "instrument": "west",
                "filter": "Blank",
                "frame_type": "dark",
                "exposure_seconds": 30.0,
            },
        ],
    }


def make_manager(
    *,
    data=None,
    max_request_retries=2,
):
    plan = CalibrationPlanConfig.model_validate(
        data or plan_data()
    )
    config = CalibrationFramesConfig(
        collect=True,
        windows=["day"],
        max_request_retries=max_request_retries,
        plan=plan,
    )
    run = SimpleNamespace(
        config=SimpleNamespace(
            schedule=SimpleNamespace(
                calibration_frames=config,
            )
        ),
        log_command=Mock(),
    )
    return CalibrationFrameManager(run), run


def request_plan(request):
    return CalibrationPlanConfig.model_validate(
        request.metadata["calibration_plan"]
    )


def test_manager_properties_and_initial_progress():
    manager, _ = make_manager()

    assert manager.name == "calibration_frames"
    assert [
        condition.value
        for condition in manager.suitable_conditions
    ] == ["day"]
    assert manager.progress_totals() == (0, 9)
    assert manager.all_targets_met is False
    assert manager.is_collection_complete() is False


def test_first_request_groups_biases_across_cameras():
    manager, _ = make_manager()

    request = manager.generate_request("bias-request")
    plan = request_plan(request)

    assert request.target is None
    assert request.exposure_seconds == []
    assert request.exposure_map == []
    assert [
        (
            sequence.camera,
            sequence.frame_type,
            sequence.exposure_seconds,
            plan.target(sequence),
        )
        for sequence in plan.sequences
    ] == [
        ("C0", "bias", 0.0, 2),
        ("C1", "bias", 0.0, 3),
    ]


def test_success_advances_to_the_next_group():
    manager, run = make_manager()

    bias_request = manager.generate_request("bias-request")
    manager.on_request_completed(
        bias_request,
        duration_seconds=12.5,
    )

    assert manager.progress_totals() == (5, 9)
    assert manager.is_collection_complete() is False

    dark_request = manager.generate_request("dark-request")
    dark_plan = request_plan(dark_request)

    assert {
        (
            sequence.camera,
            sequence.frame_type,
            sequence.exposure_seconds,
            dark_plan.target(sequence),
        )
        for sequence in dark_plan.sequences
    } == {
        ("C0", "dark", 30.0, 2),
        ("C1", "dark", 30.0, 2),
    }

    assert [
        call.args[0]
        for call in run.log_command.call_args_list
    ] == [
        "calibration_sequence_completed",
        "calibration_sequence_completed",
    ]


def test_all_successful_groups_complete_collection():
    manager, _ = make_manager()

    first = manager.generate_request("bias-request")
    manager.on_request_completed(first, 10.0)

    second = manager.generate_request("dark-request")
    manager.on_request_completed(second, 20.0)

    assert manager.progress_totals() == (9, 9)
    assert manager.all_targets_met is True
    assert manager.is_collection_complete() is True
    assert manager.generate_request("unused") is None


def test_failed_group_is_retried_before_later_groups():
    manager, _ = make_manager()

    first = manager.generate_request("bias-1")
    manager.on_request_failed(first, "camera timeout")

    retry = manager.generate_request("bias-2")
    retry_plan = request_plan(retry)

    assert {
        sequence.frame_type
        for sequence in retry_plan.sequences
    } == {"bias"}

    manager.on_request_completed(retry, 10.0)
    next_request = manager.generate_request("dark-1")

    assert {
        sequence.frame_type
        for sequence in request_plan(next_request).sequences
    } == {"dark"}


def test_sequence_group_exhausts_after_retry_limit():
    data = {
        "frames_per_sequence": 2,
        "sequences": [
            {
                "camera": "C0",
                "instrument": "east",
                "filter": "Blank",
                "frame_type": "dark",
                "exposure_seconds": 30.0,
            }
        ],
    }
    manager, run = make_manager(
        data=data,
        max_request_retries=2,
    )

    for attempt in range(3):
        request = manager.generate_request(
            f"failed-{attempt}"
        )
        assert request is not None
        manager.on_request_failed(
            request,
            "camera timeout",
        )

    assert manager.all_targets_met is False
    assert manager.is_collection_complete() is True
    assert manager.generate_request("unused") is None
    assert run.log_command.call_count == 3
    assert (
        run.log_command.call_args_list[-1]
        .args[2]["exhausted"]
        is True
    )


def test_success_rejects_more_frames_than_remain():
    manager, _ = make_manager()

    request = manager.generate_request("bias-request")
    request.metadata["calibration_plan"]["sequences"][0][
        "frames"
    ] = 100

    with pytest.raises(
        ValueError,
        match="only 2 remain",
    ):
        manager.on_request_completed(request, 10.0)

    assert manager.progress_totals() == (0, 9)


def test_success_rejects_foreign_sequence():
    manager, _ = make_manager()

    request = manager.generate_request("bias-request")
    request.metadata["calibration_plan"]["sequences"][0][
        "instrument"
    ] = "another-camera"

    with pytest.raises(
        ValueError,
        match="does not exactly match",
    ):
        manager.on_request_completed(request, 10.0)

    assert manager.progress_totals() == (0, 9)


def test_success_requires_targetless_calibration_request():
    manager, _ = make_manager()

    request = manager.generate_request("bias-request")
    request.target = RADecTarget(
        right_ascension_deg=10,
        declination_deg=20,
    )

    with pytest.raises(
        ValueError,
        match="target=None",
    ):
        manager.on_request_completed(request, 10.0)


def test_manager_requires_a_plan():
    config = CalibrationFramesConfig(
        collect=False,
    )
    run = SimpleNamespace(
        config=SimpleNamespace(
            schedule=SimpleNamespace(
                calibration_frames=config,
            )
        ),
        log_command=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="requires calibration_frames.plan",
    ):
        CalibrationFrameManager(run)