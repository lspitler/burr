"""Withdraw the rest of a twilight's offer once burr has nothing left to collect in it.

11-17 dusk: the last planned request finished at 20:28; burr then returned no
request every 5 s but kept the controller until its window ended at 20:40, so
Otto lost 12 minutes. Now the twilight's window ends at "now" once every
remaining pair is exhausted (or all are complete); the next twilight is
offered as normal.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from burr.core.config import TwilightFlatsConfig
from burr.models.site import LightingSchedule, SiteMetadata
from burr.scheduler.scheduler import Scheduler
from burr.scheduler.strategies import OneShotStrategy
from burr.task_source.flats.manager import ExposureRecord, FlatFieldManager
from burr.task_source.flats.products import FlatSequenceKey

SITE = SiteMetadata(name="Test", latitude=-31.0, longitude=149.0, altitude_km=1.1)
NIGHT = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)  # ~20:00 local
PAIRS = [("C0", "g"), ("C0", "r"), ("C3", "z")]


@pytest.fixture(scope="module")
def schedule():
    return LightingSchedule.calculate_twilights(SITE, NIGHT)


def _manager(tmp_path, schedule, *, sequencing="per_filter"):
    config = TwilightFlatsConfig(
        windows=["nautical_twilight", "civil_twilight"],
        minimum_exposure_seconds=0.1,
        maximum_exposure_seconds=60.0,
        min_counts=25000,
        max_counts=50000,
        plan={
            "frames_per_pair": 3,
            "pairs": [{"camera": c, "instrument": f"{c}-ASI183MM", "filter": f} for c, f in PAIRS],
            "filter_sequencing": sequencing,
            "filter_phase_order": "throughput",
            "filter_throughput": {"g": 1.0, "r": 0.3, "z": 0.02},
        },
    )
    run = SimpleNamespace(
        config=SimpleNamespace(site=SITE, schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=[],
        output_directory=tmp_path / "processed",
        lighting_schedule=schedule,
        log_command=lambda *a, **k: None,
    )
    return FlatFieldManager(run, sk_data_root=tmp_path / "raw")


def _dusk_now(m):
    dusk, _ = m._sun_clipped_windows()
    return dusk[0] + (dusk[1] - dusk[0]) * 0.8


def _exhaust(m, keys, when):
    """A dusk frame at the 60 s cap, below min_counts: the pair can't recover this twilight."""
    for camera, filter_name in keys:
        m.cameras[FlatSequenceKey(camera, filter_name)].exposures.append(
            ExposureRecord(timestamp=when, exposure_time=60.0, median_counts=500.0,
                           saturation_percentage=0.0, is_good=False)
        )


def test_window_kept_while_a_pair_can_still_collect(tmp_path, schedule):
    """(b) A pair that is merely behind (not exhausted) keeps the window."""
    m = _manager(tmp_path, schedule)
    now = _dusk_now(m)
    _exhaust(m, [("C0", "g"), ("C0", "r")], now)  # C3/z still collectable
    assert not m.twilight_done(now)
    assert m.offer_windows(now) == m._sun_clipped_windows()


def test_exhausted_dusk_window_ends_now_and_dawn_is_still_offered(tmp_path, schedule):
    """(a) The bug: every remaining pair exhausted at dusk."""
    m = _manager(tmp_path, schedule)
    now = _dusk_now(m)
    _exhaust(m, PAIRS, now)
    full_dusk, full_dawn = m._sun_clipped_windows()

    assert m.twilight_done(now)
    dusk, dawn = m.offer_windows(now)
    assert dusk == (full_dusk[0], now)
    assert dawn == full_dawn


def test_dawn_starts_fresh_after_an_exhausted_dusk(tmp_path, schedule):
    """Dusk records don't make dawn exhausted: the dawn window is offered whole."""
    m = _manager(tmp_path, schedule)
    _exhaust(m, PAIRS, _dusk_now(m))
    _, dawn = m._sun_clipped_windows()
    dawn_now = dawn[0] + timedelta(minutes=5)

    assert not m.twilight_done(dawn_now)
    assert m.offer_windows(dawn_now)[-1] == dawn


def test_all_pairs_complete_also_withdraws(tmp_path, schedule):
    m = _manager(tmp_path, schedule, sequencing="batched")
    for camera, filter_name in PAIRS:
        m.cameras[FlatSequenceKey(camera, filter_name)].good_frames = 3
    now = _dusk_now(m)
    assert m.twilight_done(now)
    assert m.offer_windows(now)[0][1] == now


def test_batched_plan_never_withdraws_on_exhaustion(tmp_path, schedule):
    """Batched sequencing has no exhaustion rule; it keeps today's behaviour."""
    m = _manager(tmp_path, schedule, sequencing="batched")
    now = _dusk_now(m)
    _exhaust(m, PAIRS, now)
    assert not m.twilight_done(now)


def test_request_in_flight_keeps_the_window(tmp_path, schedule):
    """(c) The SK program passes withdraw_finished=False while a request is busy."""
    m = _manager(tmp_path, schedule)
    now = _dusk_now(m)
    _exhaust(m, PAIRS, now)
    assert m.offer_windows(now, withdraw_finished=False) == m._sun_clipped_windows()

    scheduler = Scheduler()
    scheduler.register(m, priority=1, strategy=OneShotStrategy())
    kept = scheduler.offer_windows(schedule, withdraw_finished=False)
    assert kept == m._sun_clipped_windows()


def test_new_night_offers_a_whole_window_again(tmp_path, schedule):
    """(d) Rollover builds a new manager for the new night, with fresh windows."""
    m = _manager(tmp_path, schedule)
    _exhaust(m, PAIRS, _dusk_now(m))
    tomorrow = LightingSchedule.calculate_twilights(SITE, NIGHT + timedelta(days=1))
    fresh = _manager(tmp_path / "next", tomorrow)
    dusk, _ = fresh._sun_clipped_windows()
    now = dusk[0] + timedelta(minutes=5)
    assert not fresh.twilight_done(now)
    assert fresh.offer_windows(now) == fresh._sun_clipped_windows()


def test_exhausted_message_is_logged_once_per_twilight(tmp_path, schedule, monkeypatch):
    import burr.task_source.flats.manager as mod

    m = _manager(tmp_path, schedule)
    now = _dusk_now(m)
    _exhaust(m, PAIRS, now)

    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is None else now.astimezone(tz)

    monkeypatch.setattr(mod, "datetime", Fixed)
    lines = []
    sink = mod.logger.add(lines.append, format="{message}", filter=lambda r: "exhausted" in r["message"])
    try:
        for _ in range(5):
            assert m._planned_request("t") is None
    finally:
        mod.logger.remove(sink)
    assert len(lines) == 1
