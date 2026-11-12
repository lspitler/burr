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


def test_previous_nights_dusk_record_does_not_drive_tonights_exposure(tmp_path):
    """A record from the same side of midnight but a different night is stale.

    Exposure records live in memory, so if burr stays enabled across nights a
    dusk frame from yesterday is still "a dusk frame" today; it must not set
    tonight's exposure or seed tonight's other filters.
    """
    manager = _manager(
        tmp_path,
        pairs=[_pair("C0", "Lum"), _pair("C0", "r")],
        throughput={"Lum": 1.0, "r": 0.2},
    )
    _add_record(
        manager,
        FlatSequenceKey("C0", "Lum"),
        timestamp=NOW - timedelta(hours=24),
        exposure=60.0,
    )

    # Neither the pair's own path nor seeding uses yesterday's frame: both cold-start.
    assert manager._pair_exposure(FlatSequenceKey("C0", "Lum"), now=NOW) == pytest.approx(1.0)
    assert manager._pair_exposure(FlatSequenceKey("C0", "r"), now=NOW) == pytest.approx(5.0)


def test_recent_record_in_the_same_twilight_is_still_used(tmp_path):
    """The staleness limit must not discard frames from this twilight."""
    manager = _manager(
        tmp_path,
        pairs=[_pair("C0", "Lum"), _pair("C0", "r")],
        throughput={"Lum": 1.0, "r": 0.2},
    )
    _add_record(
        manager,
        FlatSequenceKey("C0", "Lum"),
        timestamp=NOW - timedelta(minutes=20),
        exposure=2.0,
    )

    assert manager._pair_exposure(FlatSequenceKey("C0", "r"), now=NOW) == pytest.approx(10.0, rel=1e-6)


def _twilight_manager(tmp_path, *, dawn, dusk_initial=1.0, dawn_initial=60.0, **kw):
    pairs = [_pair("C0", "g"), _pair("C0", "r")]
    config = TwilightFlatsConfig(
        initial_exposure_seconds=1.0,
        initial_exposure_dusk_seconds=dusk_initial,
        initial_exposure_dawn_seconds=dawn_initial,
        minimum_exposure_seconds=0.1,
        maximum_exposure_seconds=kw.pop("maximum", 60.0),
        plan={"pairs": pairs, "filter_throughput": kw.pop("throughput", {"g": 1.0, "r": 0.34})},
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


def test_dusk_uses_the_dusk_reference_exposure(tmp_path):
    manager = _twilight_manager(tmp_path, dawn=False, dusk_initial=1.0, dawn_initial=60.0)
    assert manager._pair_exposure(FlatSequenceKey("C0", "g"), now=NOW) == pytest.approx(1.0)
    assert manager._pair_exposure(FlatSequenceKey("C0", "r"), now=NOW) == pytest.approx(1.0 / 0.34)


def test_dawn_uses_the_dawn_reference_exposure_then_throughput_scaling(tmp_path):
    manager = _twilight_manager(tmp_path, dawn=True, dusk_initial=1.0, dawn_initial=20.0)
    assert manager._pair_exposure(FlatSequenceKey("C0", "g"), now=NOW) == pytest.approx(20.0)
    # Same throughput scaling as at dusk, from the dawn reference.
    assert manager._pair_exposure(FlatSequenceKey("C0", "r"), now=NOW) == pytest.approx(20.0 / 0.34)


def test_dawn_reference_at_the_cap_clips_fainter_filters_to_the_cap(tmp_path):
    manager = _twilight_manager(tmp_path, dawn=True, dawn_initial=60.0, maximum=60.0)
    assert manager._pair_exposure(FlatSequenceKey("C0", "g"), now=NOW) == pytest.approx(60.0)
    assert manager._pair_exposure(FlatSequenceKey("C0", "r"), now=NOW) == pytest.approx(60.0)  # 176 s -> cap


def test_per_twilight_references_default_to_initial_exposure_seconds(tmp_path):
    config = TwilightFlatsConfig(initial_exposure_seconds=2.5)
    assert config.initial_exposure_dusk_seconds is None and config.initial_exposure_dawn_seconds is None
    for dawn in (False, True):
        manager = _manager(tmp_path, pairs=[_pair("C0", "g")], throughput={"g": 1.0}, initial=2.5, dawn=dawn)
        assert manager._pair_exposure(FlatSequenceKey("C0", "g"), now=NOW) == pytest.approx(2.5)


def test_dawn_reference_without_a_throughput_table(tmp_path):
    manager = _twilight_manager(tmp_path, dawn=True, dawn_initial=30.0, throughput=None)
    assert manager._pair_exposure(FlatSequenceKey("C0", "g"), now=NOW) == pytest.approx(30.0)


def test_bias_level_frame_does_not_seed_another_filter(tmp_path):
    """11-11 dusk: C3 u at 60 s read ~bias, and z was seeded from it to 60 x 0.004/0.05 = 4.8 s."""
    manager = _manager(
        tmp_path,
        pairs=[_pair("C3", "u"), _pair("C3", "z")],
        throughput={"u": 0.004, "z": 0.05},
    )
    _add_record(manager, FlatSequenceKey("C3", "u"), exposure=60.0, counts=144.0)

    # Falls back to the reference / throughput estimate: 1.0 / 0.05 = 20 s.
    assert manager._pair_exposure(FlatSequenceKey("C3", "z"), now=NOW) == pytest.approx(20.0)


def test_bias_level_frame_is_skipped_but_a_good_older_seed_is_used(tmp_path):
    manager = _manager(
        tmp_path,
        pairs=[_pair("C0", "Lum"), _pair("C0", "g"), _pair("C0", "r")],
    )
    _add_record(manager, FlatSequenceKey("C0", "Lum"), timestamp=NOW - timedelta(minutes=2), exposure=2.0)
    _add_record(manager, FlatSequenceKey("C0", "g"), timestamp=NOW - timedelta(minutes=1), exposure=60.0, counts=150.0)

    assert manager._pair_exposure(FlatSequenceKey("C0", "r"), now=NOW) == pytest.approx(2.0 * 1.0 / 0.21)


def test_seed_floor_is_configurable(tmp_path):
    pairs = [_pair("C0", "Lum"), _pair("C0", "r")]
    config = TwilightFlatsConfig(
        initial_exposure_seconds=1.0,
        seed_min_counts=0.0,
        plan={"pairs": pairs, "filter_throughput": {"Lum": 1.0, "r": 0.2}},
    )
    run = SimpleNamespace(
        config=SimpleNamespace(schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=[],
        output_directory=tmp_path / "processed",
        lighting_schedule=SimpleNamespace(local_midnight=NOW + timedelta(hours=4)),
        log_command=lambda *args, **kwargs: None,
    )
    manager = FlatFieldManager(run, sk_data_root=tmp_path / "raw")
    _add_record(manager, FlatSequenceKey("C0", "Lum"), exposure=2.0, counts=150.0)

    # With the floor off, the old behaviour: seeded from Lum regardless of its level.
    assert manager._pair_exposure(FlatSequenceKey("C0", "r"), now=NOW) == pytest.approx(10.0)


def test_default_seed_floor_is_far_below_min_counts():
    config = TwilightFlatsConfig()
    assert 160 < config.seed_min_counts <= config.min_counts / 10
