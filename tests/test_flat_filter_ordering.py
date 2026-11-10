"""Filter-throughput ordering for planned twilight flats."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import burr.task_source.flats.manager as manager_module
from burr.core.config import TwilightFlatsConfig
from burr.task_source.flats.manager import ExposureRecord, FlatFieldManager
from burr.task_source.flats.products import FlatSequenceKey


NOW = datetime(2026, 11, 10, 18, 0, tzinfo=UTC)
THROUGHPUT = {"Lum": 1.0, "g": 0.69, "r": 0.21}


def _pair(camera, filter_name):
    return {
        "camera": camera,
        "instrument": f"{camera}-ASI183MM",
        "filter": filter_name,
    }


def _interleaved_pairs():
    return [
        _pair("C0", "Lum"),
        _pair("C1", "Lum"),
        _pair("C0", "g"),
        _pair("C1", "g"),
        _pair("C0", "r"),
        _pair("C1", "r"),
    ]


def _manager(
    tmp_path,
    *,
    dawn=False,
    pairs=None,
    throughput=THROUGHPUT,
):
    plan = {"pairs": pairs or _interleaved_pairs()}
    if throughput is not None:
        plan["filter_throughput"] = throughput

    config = TwilightFlatsConfig(plan=plan)
    midnight = NOW - timedelta(hours=4) if dawn else NOW + timedelta(hours=4)
    run = SimpleNamespace(
        config=SimpleNamespace(schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=[],
        output_directory=tmp_path / "processed",
        lighting_schedule=SimpleNamespace(local_midnight=midnight),
        log_command=lambda *args, **kwargs: None,
    )
    return FlatFieldManager(run, sk_data_root=tmp_path / "raw")


def _labels(keys):
    return [(key.camera, key.filter) for key in keys]


def test_dusk_orders_each_camera_from_low_to_high_throughput(tmp_path):
    manager = _manager(tmp_path)

    assert _labels(manager._ordered_remaining_pairs(NOW)) == [
        ("C0", "r"),
        ("C1", "r"),
        ("C0", "g"),
        ("C1", "g"),
        ("C0", "Lum"),
        ("C1", "Lum"),
    ]


def test_dawn_orders_each_camera_from_high_to_low_throughput(tmp_path):
    manager = _manager(tmp_path, dawn=True)

    assert _labels(manager._ordered_remaining_pairs(NOW)) == [
        ("C0", "Lum"),
        ("C1", "Lum"),
        ("C0", "g"),
        ("C1", "g"),
        ("C0", "r"),
        ("C1", "r"),
    ]


def test_no_throughput_table_preserves_exact_plan_order(tmp_path):
    manager = _manager(tmp_path, throughput=None)

    assert manager._ordered_remaining_pairs(NOW) == manager.remaining_pairs()


@pytest.mark.parametrize("dawn", [False, True], ids=["dusk", "dawn"])
def test_equal_throughputs_preserve_relative_plan_order(tmp_path, dawn):
    manager = _manager(
        tmp_path,
        dawn=dawn,
        throughput={"Lum": 1.0, "g": 0.5, "r": 0.5},
    )

    filters = [
        key.filter
        for key in manager._ordered_remaining_pairs(NOW)
        if key.camera == "C0"
    ]
    assert filters.index("g") < filters.index("r")


def test_completed_pair_is_omitted_without_affecting_other_cameras(tmp_path):
    manager = _manager(tmp_path)
    completed = FlatSequenceKey("C0", "r")
    manager.cameras[completed].good_frames = manager.target_for(completed)

    assert _labels(manager._ordered_remaining_pairs(NOW)) == [
        ("C0", "g"),
        ("C1", "r"),
        ("C0", "Lum"),
        ("C1", "g"),
        ("C1", "Lum"),
    ]


def test_planned_request_uses_order_and_one_request_time(tmp_path, monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is None else NOW.astimezone(tz)

    monkeypatch.setattr(manager_module, "datetime", FixedDateTime)
    manager = _manager(tmp_path)

    request = manager.generate_request(task_id="ordered")

    assert [
        (entry["camera"], entry["filter"])
        for entry in request.metadata["flat_plan"]
    ] == [
        ("C0", "r"),
        ("C1", "r"),
        ("C0", "g"),
        ("C1", "g"),
        ("C0", "Lum"),
        ("C1", "Lum"),
    ]
    assert manager._request_started_at == NOW
    exposures = {
        (entry["camera"], entry["filter"]): entry["exposure_seconds"]
        for entry in request.metadata["flat_plan"]
    }
    for camera in ("C0", "C1"):
        assert exposures[(camera, "Lum")] == pytest.approx(1.0)
        assert exposures[(camera, "g")] == pytest.approx(1.0 / 0.69)
        assert exposures[(camera, "r")] == pytest.approx(1.0 / 0.21)


def test_pair_exposure_uses_the_request_time(tmp_path):
    manager = _manager(tmp_path)
    key = FlatSequenceKey("C0", "Lum")
    manager.cameras[key].exposures.append(
        ExposureRecord(
            timestamp=NOW - timedelta(seconds=manager.sky_doubling_seconds),
            exposure_time=1.0,
            median_counts=manager.target_counts,
            saturation_percentage=0.0,
            is_good=False,
        )
    )

    assert manager._pair_exposure(key, now=NOW) == pytest.approx(2.0)
