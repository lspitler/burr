"""SensorKit FITS readback tests for DARK and BIAS collection."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from astropy.io import fits

from burr.core.config import CalibrationFramesConfig
from burr.task_source.calibration.manager import CalibrationFrameManager
from burr.task_source.calibration.plan import CalibrationPlanConfig


def _write_frame(
    path,
    *,
    task_id="sk-task-1",
    camera="C0",
    filter_name="Blank",
    image_type="DARK",
    exposure=30.0,
    date_obs="2026-11-20T01:02:03.123456",
    readout_mode=0,
    bin_x=1,
    bin_y=1,
    omit=(),
):
    header = fits.Header()
    cards = {
        "SKTASKID": task_id,
        "UNITID": camera,
        "FILTER": filter_name,
        "IMAGETYP": image_type,
        "EXPTIME": exposure,
        "DATE-OBS": date_obs,
        "READOUTM": readout_mode,
        "XBINNING": bin_x,
        "YBINNING": bin_y,
    }
    for key, value in cards.items():
        if key not in omit:
            header[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((20, 20), dtype=np.int32)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.CompImageHDU(data=data, header=header),
        ]
    ).writeto(path)
    return path


def _manager(
    tmp_path,
    *,
    frame_type="dark",
    exposure=30.0,
    frames=2,
):
    plan = CalibrationPlanConfig.model_validate(
        {
            "frames_per_sequence": frames,
            "sequences": [
                {
                    "camera": "C0",
                    "instrument": "east",
                    "filter": "Blank",
                    "frame_type": frame_type,
                    "exposure_seconds": exposure,
                }
            ],
        }
    )
    config = CalibrationFramesConfig(
        collect=True,
        windows=["day"],
        frame_settle_seconds=0.01,
        plan=plan,
    )
    run = SimpleNamespace(
        output_directory=tmp_path / "processed",
        config=SimpleNamespace(
            schedule=SimpleNamespace(
                calibration_frames=config,
            )
        ),
        log_command=Mock(),
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    return CalibrationFrameManager(run, sk_data_root=raw), run, raw


def _request(manager, request_id="calibration-request"):
    request = manager.generate_request(request_id)
    assert request is not None
    manager.on_task_dispatched(request, "sk-task-1")
    return request


def _commands(run):
    return [call.args[0] for call in run.log_command.call_args_list]


@pytest.mark.parametrize(
    ("frame_type", "exposure"),
    [("dark", 30.0), ("bias", 0.0)],
)
def test_valid_calibration_frames_are_accepted(
    tmp_path,
    frame_type,
    exposure,
):
    manager, run, raw = _manager(
        tmp_path,
        frame_type=frame_type,
        exposure=exposure,
    )
    request = _request(manager)
    for index in range(2):
        _write_frame(
            raw / f"frame-{index}.fits",
            image_type=frame_type.upper(),
            exposure=exposure,
            date_obs=f"2026-11-20T01:02:03.12345{index}",
            readout_mode=0,
            bin_x=1,
            bin_y=1,
        )

    manager.on_request_completed(request, 2.5)

    assert manager.progress_totals() == (2, 2)
    assert manager.all_targets_met is True
    assert _commands(run) == [
        "calibration_frame_saved",
        "calibration_frame_saved",
        "calibration_sequence_completed",
    ]
    metadata = run.log_command.call_args.args[2]
    assert metadata["accepted_frames"] == 2
    assert len(metadata["source_paths"]) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera", "C9"),
        ("filter_name", "Lum"),
        ("image_type", "LIGHT"),
        ("exposure", 31.0),
        ("readout_mode", "unsafe/name"),
    ],
)
def test_incorrect_metadata_is_rejected(
    tmp_path,
    field,
    value,
):
    manager, run, raw = _manager(tmp_path, frames=1)
    request = _request(manager)
    values = {
        "camera": "C0",
        "filter_name": "Blank",
        "image_type": "DARK",
        "exposure": 30.0,
    }
    values[field] = value
    _write_frame(raw / "wrong.fits", **values)

    manager.on_request_completed(request, 1.0)

    assert manager.progress_totals() == (0, 1)
    assert "calibration_frame_rejected" in _commands(run)
    assert "calibration_sequence_failed" in _commands(run)


@pytest.mark.parametrize(
    "missing_card",
    [
        "UNITID",
        "FILTER",
        "IMAGETYP",
        "EXPTIME",
        "DATE-OBS",
        "READOUTM",
        "XBINNING",
        "YBINNING",
    ],
)
def test_missing_required_metadata_is_rejected(
    tmp_path,
    missing_card,
):
    manager, run, raw = _manager(tmp_path, frames=1)
    request = _request(manager)
    _write_frame(
        raw / "missing.fits",
        omit={missing_card},
    )

    manager.on_request_completed(request, 1.0)

    assert manager.progress_totals() == (0, 1)
    rejected = [
        call
        for call in run.log_command.call_args_list
        if call.args[0] == "calibration_frame_rejected"
    ]
    assert len(rejected) == 1
    assert missing_card in rejected[0].args[1]


def test_wrong_sensor_task_is_ignored(tmp_path):
    manager, run, raw = _manager(tmp_path, frames=1)
    request = _request(manager)
    _write_frame(
        raw / "other-task.fits",
        task_id="another-task",
    )

    manager.on_request_completed(request, 1.0)

    assert manager.progress_totals() == (0, 1)
    assert "calibration_frame_rejected" not in _commands(run)
    assert "calibration_sequence_failed" in _commands(run)


def test_partial_readback_counts_only_valid_frames(tmp_path):
    manager, run, raw = _manager(tmp_path, frames=2)
    request = _request(manager)
    _write_frame(raw / "valid.fits")
    _write_frame(
        raw / "light.fits",
        image_type="LIGHT",
    )

    manager.on_request_completed(request, 1.0)

    assert manager.progress_totals() == (1, 2)
    assert "calibration_sequence_progress" in _commands(run)
    assert "calibration_frame_rejected" in _commands(run)
    retry = manager.generate_request("retry")
    retry_sequence = retry.metadata["calibration_plan"]["sequences"][0]
    assert retry_sequence["frames"] == 1


def test_extra_frame_is_rejected_without_overcounting(tmp_path):
    manager, run, raw = _manager(tmp_path, frames=2)
    request = _request(manager)
    for index in range(3):
        _write_frame(
            raw / f"frame-{index}.fits",
            date_obs=f"2026-11-20T01:02:03.12345{index}",
        )

    manager.on_request_completed(request, 1.0)

    assert manager.progress_totals() == (2, 2)
    rejected = [
        call
        for call in run.log_command.call_args_list
        if call.args[0] == "calibration_frame_rejected"
    ]
    assert len(rejected) == 1
    assert "extra frame" in rejected[0].args[1]


def test_completion_requires_recorded_sensor_task(tmp_path):
    manager, _, _ = _manager(tmp_path, frames=1)
    request = manager.generate_request("not-dispatched")

    with pytest.raises(
        ValueError,
        match="no recorded SensorKit task ID",
    ):
        manager.on_request_completed(request, 1.0)

    assert manager.progress_totals() == (0, 1)


def test_published_products_restore_progress(tmp_path):
    manager, run, raw = _manager(tmp_path, frames=1)
    request = _request(manager)
    _write_frame(raw / "frame.fits")

    manager.on_request_completed(request, 1.0)

    restored = CalibrationFrameManager(
        run,
        sk_data_root=raw,
    )
    assert restored.progress_totals() == (1, 1)
    assert restored.generate_request("unused") is None
