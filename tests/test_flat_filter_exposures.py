"""Starting exposures and same-camera seeding for planned twilight flats."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

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


def _manager(
    tmp_path,
    *,
    pairs=None,
    throughput=THROUGHPUT,
    initial=1.0,
    minimum=0.1,
    maximum=60.0,
    dawn=False,
):
    pairs = pairs or [
        _pair("C0", "Lum"),
        _pair("C0", "g"),
        _pair("C0", "r"),
    ]
    plan = {"pairs": pairs}
    if throughput is not None:
        plan["filter_throughput"] = throughput

    config = TwilightFlatsConfig(
        initial_exposure_seconds=initial,
        minimum_exposure_seconds=minimum,
        maximum_exposure_seconds=maximum,
        plan=plan,
    )
    midnight = NOW - timedelta(hours=4) if dawn else NOW + timedelta(hours=4)
    run = SimpleNamespace(
        config=SimpleNamespace(schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=[],
        output_directory=tmp_path / "processed",
        lighting_schedule=SimpleNamespace(local_midnight=midnight),
        log_command=lambda *args, **kwargs: None,
    )
    return FlatFieldManager(run, sk_data_root=tmp_path / "raw")


def _add_record(
    manager,
    key,
    *,
    timestamp=NOW,
    exposure=1.0,
    counts=None,
):
    manager.cameras[key].exposures.append(
        ExposureRecord(
            timestamp=timestamp,
            exposure_time=exposure,
            median_counts=manager.target_counts if counts is None else counts,
            saturation_percentage=0.0,
            is_good=False,
        )
    )


def test_initial_exposures_scale_by_filter_throughput(tmp_path):
    manager = _manager(tmp_path)

    assert manager._pair_exposure(
        FlatSequenceKey("C0", "Lum"), now=NOW
    ) == pytest.approx(1.0)
    assert manager._pair_exposure(
        FlatSequenceKey("C0", "g"), now=NOW
    ) == pytest.approx(1.0 / 0.69)
    assert manager._pair_exposure(
        FlatSequenceKey("C0", "r"), now=NOW
    ) == pytest.approx(1.0 / 0.21)


def test_no_throughput_table_uses_reference_initial_exposure(tmp_path):
    manager = _manager(
        tmp_path,
        pairs=[_pair("C0", "Lum")],
        throughput=None,
        initial=2.0,
    )

    assert manager._pair_exposure(
        FlatSequenceKey("C0", "Lum"), now=NOW
    ) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("filter_name", "throughput", "expected"),
    [
        pytest.param("Clear", 20.0, 0.1, id="minimum"),
        pytest.param("Halpha", 0.001, 60.0, id="maximum"),
    ],
)
def test_initial_exposure_is_clipped_to_hardware_limits(
    tmp_path, filter_name, throughput, expected
):
    manager = _manager(
        tmp_path,
        pairs=[_pair("C0", filter_name)],
        throughput={filter_name: throughput},
    )

    assert manager._pair_exposure(
        FlatSequenceKey("C0", filter_name), now=NOW
    ) == pytest.approx(expected)


def test_measured_filter_seeds_unmeasured_filter_on_same_camera(tmp_path):
    manager = _manager(
        tmp_path,
        pairs=[_pair("C0", "Lum"), _pair("C0", "r")],
        throughput={"Lum": 1.0, "r": 0.2},
    )
    _add_record(manager, FlatSequenceKey("C0", "Lum"), exposure=2.0)

    assert manager._pair_exposure(
        FlatSequenceKey("C0", "r"), now=NOW
    ) == pytest.approx(10.0)


def test_measurement_does_not_seed_another_camera(tmp_path):
    manager = _manager(
        tmp_path,
        pairs=[_pair("C0", "r"), _pair("C1", "Lum")],
        throughput={"Lum": 1.0, "r": 0.2},
    )
    _add_record(manager, FlatSequenceKey("C1", "Lum"), exposure=2.0)

    assert manager._pair_exposure(
        FlatSequenceKey("C0", "r"), now=NOW
    ) == pytest.approx(5.0)


def test_dusk_measurement_does_not_seed_at_dawn(tmp_path):
    manager = _manager(
        tmp_path,
        pairs=[_pair("C0", "Lum"), _pair("C0", "r")],
        throughput={"Lum": 1.0, "r": 0.2},
        dawn=True,
    )
    _add_record(
        manager,
        FlatSequenceKey("C0", "Lum"),
        timestamp=NOW - timedelta(hours=8),
        exposure=2.0,
    )

    assert manager._pair_exposure(
        FlatSequenceKey("C0", "r"), now=NOW
    ) == pytest.approx(5.0)


def test_pairs_own_measurement_takes_precedence_over_seed(tmp_path):
    manager = _manager(
        tmp_path,
        pairs=[_pair("C0", "Lum"), _pair("C0", "r")],
        throughput={"Lum": 1.0, "r": 0.2},
    )
    _add_record(manager, FlatSequenceKey("C0", "Lum"), exposure=2.0)
    _add_record(manager, FlatSequenceKey("C0", "r"), exposure=3.0)

    assert manager._pair_exposure(
        FlatSequenceKey("C0", "r"), now=NOW
    ) == pytest.approx(3.0)


def test_newest_same_camera_measurement_is_used_as_seed(tmp_path):
    manager = _manager(tmp_path)
    _add_record(
        manager,
        FlatSequenceKey("C0", "Lum"),
        timestamp=NOW - timedelta(minutes=2),
        exposure=1.0,
    )
    _add_record(
        manager,
        FlatSequenceKey("C0", "g"),
        timestamp=NOW - timedelta(minutes=1),
        exposure=4.0,
    )

    assert manager._pair_exposure(
        FlatSequenceKey("C0", "r"), now=NOW
    ) == pytest.approx(4.0 * 0.69 / 0.21)


def test_planless_first_request_uses_clipped_initial_exposure(tmp_path):
    config = TwilightFlatsConfig(
        initial_exposure_seconds=2.0,
        minimum_exposure_seconds=0.1,
        maximum_exposure_seconds=60.0,
    )
    run = SimpleNamespace(
        config=SimpleNamespace(schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=[],
        output_directory=tmp_path / "processed",
        lighting_schedule=SimpleNamespace(local_midnight=NOW + timedelta(hours=4)),
        log_command=lambda *args, **kwargs: None,
    )
    manager = FlatFieldManager(run, sk_data_root=tmp_path / "raw")

    request = manager.generate_request(task_id="planless")

    assert request.exposure_seconds == [2.0]
