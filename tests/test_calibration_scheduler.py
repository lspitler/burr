"""Scheduler integration tests for DARK and BIAS collection."""

from types import SimpleNamespace

from burr.models.site import LightingCondition
from burr.scheduler.factory import build_scheduler
from burr.scheduler.strategies import OneShotStrategy


def test_scheduler_registers_calibration_as_one_shot():
    source = SimpleNamespace(
        name="calibration_frames",
        suitable_conditions=[LightingCondition.DAY],
        is_collection_complete=lambda: False,
    )
    run = SimpleNamespace(executed_commands=[])
    config = SimpleNamespace(
        schedule=SimpleNamespace(),
    )

    scheduler = build_scheduler(
        config,
        run,
        {"calibration_frames": source},
    )

    assert len(scheduler.slots) == 1

    slot = scheduler.slots[0]
    assert slot.source is source
    assert slot.priority == 0
    assert isinstance(slot.strategy, OneShotStrategy)