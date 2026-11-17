"""Observing-night rollover checks for Burr's SensorKit program."""

# ruff: noqa: E402

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("sensorkit")

from burr.hardware.sk import burr_program as program
from burr.models.site import LightingCondition


def _schedule(start: datetime, end: datetime):
    return SimpleNamespace(
        schedule_start_utc=start,
        schedule_end_utc=end,
        get_lighting_condition=Mock(return_value=LightingCondition.NIGHT),
    )


def _context(start: datetime, end: datetime):
    scheduler = Mock()
    return program._ProgramContext(
        run=SimpleNamespace(
            run_id="old-night",
            lighting_schedule=_schedule(start, end),
        ),
        config=SimpleNamespace(site=SimpleNamespace(name="Huntsman")),
        controller_entity="Huntsman",
        scheduler=scheduler,
    )


def test_run_covers_only_its_noon_to_noon_window():
    start = datetime(2026, 11, 16, 1, 0, tzinfo=UTC)
    end = start + timedelta(days=1)
    ctx = _context(start, end)

    assert program._run_covers(ctx, start)
    assert program._run_covers(ctx, end - timedelta(microseconds=1))
    assert not program._run_covers(ctx, start - timedelta(microseconds=1))
    assert not program._run_covers(ctx, end)


def test_stale_run_is_replaced_when_idle(monkeypatch):
    now = datetime(2026, 11, 17, 2, 0, tzinfo=UTC)
    ctx = _context(now - timedelta(days=2), now - timedelta(days=1))
    new_scheduler = Mock()
    new_run = SimpleNamespace(
        run_id="new-night",
        lighting_schedule=_schedule(
            now - timedelta(hours=1),
            now + timedelta(hours=23),
        ),
    )
    build = Mock(return_value=(new_run, new_scheduler))
    monkeypatch.setattr(program, "_build_run_and_scheduler", build)

    changed = asyncio.run(program._rollover_if_needed(ctx, now))

    assert changed
    assert ctx.run is new_run
    assert ctx.scheduler is new_scheduler
    build.assert_called_once_with(ctx.config, now)


@pytest.mark.parametrize("busy", ["building", "active", "pending"])
def test_stale_run_waits_for_current_request(monkeypatch, busy):
    now = datetime(2026, 11, 17, 2, 0, tzinfo=UTC)
    ctx = _context(now - timedelta(days=2), now - timedelta(days=1))
    if busy == "building":
        ctx.building_request = True
    elif busy == "active":
        ctx.active = SimpleNamespace()
    else:
        ctx.pending.append(SimpleNamespace())

    build = Mock()
    monkeypatch.setattr(program, "_build_run_and_scheduler", build)

    changed = asyncio.run(program._rollover_if_needed(ctx, now))

    assert not changed
    build.assert_not_called()
    assert ctx.run.run_id == "old-night"


def test_current_run_is_not_rebuilt(monkeypatch):
    now = datetime(2026, 11, 17, 2, 0, tzinfo=UTC)
    ctx = _context(now - timedelta(hours=1), now + timedelta(hours=23))
    build = Mock()
    monkeypatch.setattr(program, "_build_run_and_scheduler", build)

    changed = asyncio.run(program._rollover_if_needed(ctx, now))

    assert not changed
    build.assert_not_called()


def test_refresh_clears_stale_offers_before_rebuilding(monkeypatch):
    now = datetime.now(UTC)
    ctx = _context(now - timedelta(days=2), now - timedelta(days=1))
    events: list[str] = []
    window = (now + timedelta(minutes=1), now + timedelta(minutes=10))
    new_scheduler = Mock()
    new_scheduler.offer_windows.side_effect = lambda schedule: (
        events.append("offer_windows") or [window]
    )
    new_run = SimpleNamespace(
        run_id="new-night",
        lighting_schedule=_schedule(
            now - timedelta(hours=1),
            now + timedelta(hours=23),
        ),
    )

    def build(config, observation_date):
        events.append("build")
        return new_run, new_scheduler

    async def publish_offers():
        events.append("publish")

    api = SimpleNamespace(
        clear_offers=lambda: events.append("clear"),
        add_offer=lambda start, end: events.append("add"),
        publish_offers=publish_offers,
    )
    monkeypatch.setattr(program, "_build_run_and_scheduler", build)
    monkeypatch.setattr(program.sk, "program", lambda: api)

    asyncio.run(program._refresh_offers(ctx))

    assert events[:3] == ["clear", "publish", "build"]
    assert events[-4:] == ["offer_windows", "clear", "add", "publish"]
    assert ctx.run is new_run
    new_scheduler.log_status.assert_called_once()


def test_enable_only_wakes_background_refresh(monkeypatch):
    now = datetime.now(UTC)
    ctx = _context(now - timedelta(hours=1), now + timedelta(hours=23))
    start = Mock()
    monkeypatch.setattr(program, "_ctx", ctx)
    monkeypatch.setattr(program, "_start_offer_loop", start)

    asyncio.run(program._on_enable())

    start.assert_called_once_with(ctx)


def test_task_factory_does_not_build_while_run_is_being_replaced(monkeypatch):
    now = datetime.now(UTC)
    ctx = _context(now - timedelta(hours=1), now + timedelta(hours=23))
    ctx.rolling_over = True
    fill = AsyncMock()
    monkeypatch.setattr(program, "_ctx", ctx)
    monkeypatch.setattr(program, "_fill_queue", fill)

    async def run():
        factory = program._next_task()
        with pytest.raises(StopAsyncIteration):
            await anext(factory)

    asyncio.run(run())

    fill.assert_not_awaited()

