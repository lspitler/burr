"""Execution and lifecycle checks for Burr's native SensorKit program."""

# ruff: noqa: E402

import asyncio
import contextlib
import threading
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("sensorkit")

import sensorkit.api as sk
from sensorkit.astro.coords import Horizontal
from sensorkit.astro.target import AltAzTarget
from sensorkit.core.controller import TaskExecutionResult
from sensorkit.std.collect import CameraParameterSet, StandardCollectTask
from sensorkit.std.instrument import FrameType

from burr.hardware.sk import burr_program as program
from burr.models.site import LightingCondition


@pytest.fixture
def setup(monkeypatch):
    source = SimpleNamespace(
        on_task_dispatched=Mock(),
        on_request_completed=Mock(),
        on_request_failed=Mock(),
        generate_request=Mock(),
    )
    slot = SimpleNamespace(
        name="twilight_flats",
        source=source,
        record_success=Mock(),
        record_failure=Mock(),
    )
    request = SimpleNamespace(task_id="burr-request")
    state = program._RequestState(
        slot=slot,
        request=request,
    )
    task = StandardCollectTask(
        target=AltAzTarget(coords=Horizontal(az=0, alt=75)),
        camera_params=CameraParameterSet(
            instrument="east",
            frame_type=FrameType.FLAT,
            integration_time_seconds=1.0,
            frame_count=1,
        ),
    )
    pending = program._PendingTask(
        task.submit(),
        state,
        True,
    )
    schedule = SimpleNamespace(
        get_lighting_condition=Mock(return_value=LightingCondition.NIGHT)
    )
    config = SimpleNamespace(
        schedule=SimpleNamespace(
            twilight_flats=SimpleNamespace(
                collect=True,
                plan=SimpleNamespace(pairs=[SimpleNamespace(instrument="east")]),
            )
        )
    )
    ctx = program._ProgramContext(
        run=SimpleNamespace(lighting_schedule=schedule),
        config=config,
        controller_entity="Huntsman",
        scheduler=Mock(),
    )
    ctx.pending.append(pending)
    monkeypatch.setattr(program, "_ctx", ctx)
    return ctx, pending, source, slot


@pytest.mark.parametrize(
    "outcome",
    ["success", "failure", "cancel", "readback"],
)
@pytest.mark.parametrize("with_hook", [True, False])
def test_factory_waits_and_finalizes_once(setup, monkeypatch, outcome, with_hook):
    ctx, pending, source, slot = setup
    monkeypatch.setattr(
        program,
        "_validate_instruments",
        AsyncMock(),
    )

    if not with_hook:
        del source.on_task_dispatched
    if outcome == "readback":
        source.on_request_completed.side_effect = RuntimeError("FITS read failed")

    async def run():
        awaiting = asyncio.Event()

        class ObservedExecution(sk.TaskExecution):
            def __await__(self):
                awaiting.set()
                return super().__await__()

        execution = ObservedExecution(
            task=pending.submission.task,
            task_id=uuid.uuid4(),
            controller_id="Huntsman",
        )
        result = asyncio.get_running_loop().create_future()
        execution.bind_result(result)

        if outcome in {"failure", "cancel"}:
            pending.is_last = False
            ctx.pending.append(
                program._PendingTask(
                    pending.submission,
                    pending.state,
                    True,
                )
            )

        factory = program._next_task()
        assert await anext(factory) is pending.submission
        finishing = asyncio.create_task(factory.asend(execution))

        try:
            await asyncio.wait_for(
                awaiting.wait(),
                timeout=2,
            )
            if with_hook:
                source.on_task_dispatched.assert_called_once_with(
                    pending.state.request,
                    str(execution.task_id),
                )

            source.on_request_completed.assert_not_called()
            source.on_request_failed.assert_not_called()
            assert not finishing.done()

            if outcome == "failure":
                result.set_exception(RuntimeError("camera failed"))
            elif outcome == "cancel":
                result.cancel()
            else:
                now = datetime.now(UTC)
                result.set_result(
                    TaskExecutionResult(
                        task_id=execution.task_id,
                        start_time=now,
                        end_time=now,
                    )
                )

            expected = {
                "success": StopAsyncIteration,
                "failure": RuntimeError,
                "cancel": asyncio.CancelledError,
                "readback": RuntimeError,
            }[outcome]

            with pytest.raises(expected):
                await asyncio.wait_for(
                    finishing,
                    timeout=2,
                )
        finally:
            if not finishing.done():
                finishing.cancel()

            with contextlib.suppress(
                asyncio.CancelledError,
                StopAsyncIteration,
                RuntimeError,
            ):
                await finishing

            await factory.aclose()

    asyncio.run(run())
    assert pending.state.finished

    if outcome == "success":
        source.on_request_completed.assert_called_once()
        source.on_request_failed.assert_not_called()
        slot.record_success.assert_called_once()
        slot.record_failure.assert_not_called()
    else:
        source.on_request_failed.assert_called_once()
        slot.record_failure.assert_called_once()
        slot.record_success.assert_not_called()
        assert not ctx.pending


def test_dispatch_rejection_drops_remaining_tasks(setup, monkeypatch):
    ctx, pending, source, slot = setup
    ctx.pending.append(
        program._PendingTask(
            pending.submission,
            pending.state,
            True,
        )
    )
    monkeypatch.setattr(
        program,
        "_validate_instruments",
        AsyncMock(),
    )

    async def run():
        factory = program._next_task()
        await anext(factory)

        with pytest.raises(RuntimeError, match="rejected"):
            await factory.athrow(RuntimeError("controller rejected"))

        await factory.aclose()

    asyncio.run(run())

    source.on_task_dispatched.assert_not_called()
    source.on_request_failed.assert_called_once()
    slot.record_failure.assert_called_once()
    assert not ctx.pending


def test_disable_accounts_for_a_queued_request_once(setup):
    ctx, pending, source, slot = setup
    ctx.pending.append(
        program._PendingTask(
            pending.submission,
            pending.state,
            True,
        )
    )

    asyncio.run(program._on_disable())
    asyncio.run(program._on_disable())

    source.on_request_failed.assert_called_once()
    source.on_request_completed.assert_not_called()
    slot.record_failure.assert_called_once()
    assert not ctx.pending


def test_offers_use_clipped_windows_and_remove_stale_offers(setup, monkeypatch):
    ctx, _, _, _ = setup
    start = datetime.now(UTC)
    windows = [(start, start + timedelta(minutes=20))]
    ctx.scheduler.offer_windows.return_value = windows
    advertised = [("stale-start", "stale-end")]
    api = SimpleNamespace(
        clear_offers=advertised.clear,
        add_offer=lambda begin, end: advertised.append((begin, end)),
        publish_offers=AsyncMock(),
    )
    monkeypatch.setattr(
        program.sk,
        "program",
        lambda: api,
    )

    asyncio.run(program._publish_offers())

    assert advertised == windows
    ctx.scheduler.offer_windows.assert_called_once_with(
        ctx.run.lighting_schedule, withdraw_finished=False  # setup leaves a task pending
    )


@pytest.mark.parametrize(
    ("assigned", "available", "error"),
    [
        ("Huntsman", ["east"], None),
        ("Huntsman", ["other"], "east"),
        (
            "OtherController",
            ["east"],
            "assigned controller",
        ),
    ],
)
def test_validates_plan_instruments_without_global_selection(
    setup, monkeypatch, assigned, available, error
):
    ctx, _, _, _ = setup
    controller = SimpleNamespace(
        kv_get_model=AsyncMock(
            return_value=sk.EntityInfo(
                entity_type="controller",
                details=sk.ControllerDetails(
                    supported_tasks=["standard_collect"],
                    controlled_devices=[],
                    instruments=tuple(
                        sk.InstrumentInfo(name=name) for name in available
                    ),
                ),
            )
        )
    )
    api = SimpleNamespace(
        kv_get_model=AsyncMock(
            return_value=SimpleNamespace(
                enable_state=SimpleNamespace(
                    enabled=True,
                    controller=assigned,
                )
            )
        ),
        sensorkit=lambda: SimpleNamespace(controller=lambda name: controller),
    )
    monkeypatch.setattr(
        program.sk,
        "program",
        lambda: api,
    )

    if error:
        with pytest.raises(ValueError, match=error):
            asyncio.run(program._validate_instruments(ctx))
    else:
        asyncio.run(program._validate_instruments(ctx))


def test_cancelled_generation_accounts_for_its_request(setup, monkeypatch):
    ctx, pending, source, slot = setup
    ctx.pending.clear()
    started = threading.Event()
    release = threading.Event()

    def build(_ctx):
        started.set()
        assert release.wait(timeout=5)
        return [pending]

    monkeypatch.setattr(
        program,
        "_build_next_request",
        build,
    )

    async def run():
        filling = asyncio.create_task(program._fill_queue(ctx))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            filling.cancel()
            release.set()

            with pytest.raises(asyncio.CancelledError):
                await filling
        finally:
            release.set()

            if not filling.done():
                filling.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await filling

    asyncio.run(run())

    source.on_request_failed.assert_called_once()
    slot.record_failure.assert_called_once()
    assert not ctx.pending


def test_success_is_recorded_only_after_the_last_task(setup, monkeypatch):
    ctx, first, source, slot = setup
    first.is_last = False
    ctx.pending.append(
        program._PendingTask(
            first.submission,
            first.state,
            True,
        )
    )
    monkeypatch.setattr(
        program,
        "_validate_instruments",
        AsyncMock(),
    )

    async def run():
        for index in range(2):
            factory = program._next_task()
            submission = await anext(factory)
            execution = sk.TaskExecution(
                task=submission.task,
                task_id=uuid.uuid4(),
                controller_id="Huntsman",
            )
            future = asyncio.get_running_loop().create_future()
            execution.bind_result(future)
            now = datetime.now(UTC)
            future.set_result(
                TaskExecutionResult(
                    task_id=execution.task_id,
                    start_time=now,
                    end_time=now,
                )
            )

            with pytest.raises(StopAsyncIteration):
                await factory.asend(execution)

            if index == 0:
                source.on_request_completed.assert_not_called()
                slot.record_success.assert_not_called()

            await factory.aclose()

    asyncio.run(run())

    source.on_request_completed.assert_called_once()
    slot.record_success.assert_called_once()
    slot.record_failure.assert_not_called()


def test_task_build_failure_is_reported_to_the_source(setup, monkeypatch):
    ctx, pending, source, slot = setup
    ctx.pending.clear()
    source.generate_request.return_value = pending.state.request
    ctx.scheduler.ranked_suitable.return_value = [slot]
    monkeypatch.setattr(
        program,
        "build_requested_tasks",
        Mock(side_effect=ValueError("bad flat plan")),
    )

    assert program._build_next_request(ctx) == []

    source.on_request_failed.assert_called_once_with(
        pending.state.request,
        "bad flat plan",
    )


@pytest.mark.parametrize(
    ("busy", "withdraw"),
    [("pending", False), ("active", False), ("building", False), (None, True)],
)
def test_offers_never_withdraw_a_finished_twilight_mid_request(setup, monkeypatch, busy, withdraw):
    """A finished twilight's window is withdrawn only once no Burr request is in progress."""
    ctx, pending, _source, _slot = setup
    ctx.pending.clear()
    if busy == "pending":
        ctx.pending.append(pending)
    elif busy == "active":
        ctx.active = pending.state
    elif busy == "building":
        ctx.building_request = True
    ctx.scheduler.offer_windows = Mock(return_value=[])
    ctx.scheduler.log_status = Mock()
    published = SimpleNamespace(clear_offers=Mock(), add_offer=Mock(), publish_offers=AsyncMock())
    monkeypatch.setattr(program.sk, "program", lambda: published)

    asyncio.run(program._publish_offers(ctx))

    assert ctx.scheduler.offer_windows.call_args.kwargs == {"withdraw_finished": withdraw}
