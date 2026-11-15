"""Dawn-start / dusk-end Sun-altitude limits for twilight flats.

At dawn the first ~15 min of nautical twilight is too faint even at the
maximum exposure (11-16: 81 of 83 rejections before Sun ~-9 deg), so the
flats window can open at a configured Sun altitude instead, and SensorKit
gives the minutes before it to other programs.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from burr.core.config import TwilightFlatsConfig
from burr.models.site import LightingSchedule, SiteMetadata
from burr.scheduler.scheduler import Scheduler
from burr.scheduler.strategies import OneShotStrategy
from burr.task_source.flats.manager import (
    FlatFieldManager,
    _sun_altitudes,
    clip_windows_by_sun_altitude,
)

SITE = SiteMetadata(name="Test", latitude=-31.0, longitude=149.0, altitude_km=1.1)
NIGHT = datetime(2026, 11, 16, 20, 0, tzinfo=UTC) - timedelta(hours=11)  # ~20:00 local, AEDT


@pytest.fixture(scope="module")
def schedule():
    return LightingSchedule.calculate_twilights(SITE, NIGHT)


def _altitude(when):
    _, alts = _sun_altitudes(SITE, when, when)
    return float(alts[0])


def _manager(tmp_path, schedule, **limits):
    config = TwilightFlatsConfig(windows=["nautical_twilight", "civil_twilight"], **limits)
    run = SimpleNamespace(
        config=SimpleNamespace(site=SITE, schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=[],
        output_directory=tmp_path / "processed",
        lighting_schedule=schedule,
        log_command=lambda *a, **k: None,
    )
    return FlatFieldManager(run)


def test_no_limits_offers_the_configured_windows(tmp_path, schedule):
    m = _manager(tmp_path, schedule)
    expected = schedule.merged_windows_for(m.suitable_conditions)
    assert m.offer_windows() == expected and len(expected) == 2


def test_dawn_start_opens_at_the_sun_altitude_and_leaves_dusk_alone(tmp_path, schedule):
    m = _manager(tmp_path, schedule, dawn_start_sun_altitude_deg=-9.0)
    full_dusk, full_dawn = schedule.merged_windows_for(m.suitable_conditions)
    dusk, dawn = m.offer_windows()

    assert dusk == full_dusk
    assert dawn[1] == full_dawn[1]
    assert full_dawn[0] < dawn[0] < full_dawn[1]
    assert _altitude(dawn[0]) == pytest.approx(-9.0, abs=0.05)
    # ~3 deg at ~0.2-0.3 deg/min after the -12 deg nautical start.
    assert timedelta(minutes=10) < dawn[0] - full_dawn[0] < timedelta(minutes=20)


def test_dusk_end_closes_at_the_sun_altitude(tmp_path, schedule):
    m = _manager(tmp_path, schedule, dusk_end_sun_altitude_deg=-10.0)
    full_dusk, full_dawn = schedule.merged_windows_for(m.suitable_conditions)
    dusk, dawn = m.offer_windows()

    assert dawn == full_dawn
    assert dusk[0] == full_dusk[0] and dusk[1] < full_dusk[1]
    assert _altitude(dusk[1]) == pytest.approx(-10.0, abs=0.05)


def test_a_window_wholly_below_the_limit_is_dropped(schedule):
    dawn = (schedule.morning_nautical_start, schedule.morning_nautical_start + timedelta(minutes=5))
    kept = clip_windows_by_sun_altitude(
        [dawn], schedule.local_midnight, SITE, dawn_start=-9.0, dusk_end=None
    )
    assert kept == []


def test_a_window_wholly_above_the_limit_is_kept(schedule):
    dawn = (schedule.morning_civil_start, schedule.morning_civil_end)  # Sun -6..0 deg
    kept = clip_windows_by_sun_altitude(
        [dawn], schedule.local_midnight, SITE, dawn_start=-9.0, dusk_end=None
    )
    assert kept == [dawn]


def test_generate_request_returns_none_before_the_dawn_start(tmp_path, schedule, monkeypatch):
    import burr.task_source.flats.manager as mod

    m = _manager(tmp_path, schedule, dawn_start_sun_altitude_deg=-9.0)
    _, dawn = m.offer_windows()
    for when, expect_request in ((dawn[0] - timedelta(minutes=5), False), (dawn[0] + timedelta(minutes=1), True)):

        class Fixed(datetime):
            @classmethod
            def now(cls, tz=None, _when=when):
                return _when if tz is None else _when.astimezone(tz)

        monkeypatch.setattr(mod, "datetime", Fixed)
        assert (m.generate_request(task_id="t") is not None) is expect_request


def test_scheduler_offers_the_clipped_window(tmp_path, schedule):
    m = _manager(tmp_path, schedule, dawn_start_sun_altitude_deg=-9.0)
    other = SimpleNamespace(name="other", suitable_conditions=m.suitable_conditions[:1])  # nautical only, no own windows
    scheduler = Scheduler()
    scheduler.register(m, priority=1, strategy=OneShotStrategy())
    assert scheduler.offer_windows(schedule) == m.offer_windows()

    scheduler.register(other, priority=2, strategy=OneShotStrategy())
    merged = scheduler.offer_windows(schedule)
    # The unclipped nautical source re-extends dawn back to nautical start.
    assert merged[-1][0] == schedule.morning_nautical_start


@pytest.mark.parametrize("value", [-19.0, 1.0])
def test_sun_limits_are_range_checked(value):
    with pytest.raises(ValidationError):
        TwilightFlatsConfig(dawn_start_sun_altitude_deg=value)


def test_limits_default_to_none():
    config = TwilightFlatsConfig()
    assert config.dawn_start_sun_altitude_deg is None and config.dusk_end_sun_altitude_deg is None
