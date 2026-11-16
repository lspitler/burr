"""Burr-owned SensorKit scheduling program.

Uses SensorKit's functional declaration API: sk.declare_program plus
explicit lifecycle and task-factory callbacks. SensorKit's configured
sensor controller executes the submitted tasks.

Launch modes:

  # Run directly through SensorKit:
  sensorkit service run Burr burr.hardware.sk.burr_program:main

  # Or declare a standalone service (BURR_CONFIG mode):
  services:
    - id: Burr
      python_module: burr.hardware.sk.burr_program

  # Or use Burr's CLI, which launches the same service entry point:
  burr-sk --site deploy/example/site.yaml --preset deploy/presets/sda.yaml --name Burr

  # A single Burr configuration file is also supported:
  burr-sk --config burr.yaml --name Burr

The program declaration is nameless, so it inherits the service name.
Keep that name consistent with the program referenced by SensorKit's
automation configuration. For Huntsman this should remain "Burr".

A unified SensorKit burr: section launches this service automatically when
sensorkit.imports contains burr.hardware.sk.burr_program. Do not also add
an explicit services: entry for the same program. Reload configuration
before starting so the BurrServiceConfig model is uploaded.

BURR_CONFIG remains an explicit standalone override selecting Burr's
configuration file or composed site/preset files. The examples below
apply to that standalone mode.

Linux:
  export BURR_CONFIG="deploy/example/site.yaml;deploy/presets/sda.yaml"

PowerShell:
  $env:BURR_CONFIG = "deploy/example/site.yaml;deploy/presets/sda.yaml"

Semicolon-separated paths work on both platforms. Legacy colon-separated
paths are also accepted on POSIX. Burr's CLI sets BURR_CONFIG automatically
from its --config or --site/--preset arguments.

The program publishes the scheduler's observing windows, submits native
SensorKit tasks, receives controller-assigned execution IDs, and reports
request completion or failure to Burr's task sources.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import sensorkit.api as sk
from loguru import logger

from burr.bootstrap import initialize_engine
from burr.core.config import (
    AppConfig,
    get_or_initialize_config,
    initialize_config,
    set_config,
)
from burr.core.logging import setup_loguru
from burr.hardware.sk.service_config import BurrServiceConfig, service_config_for
from burr.hardware.sk.settings import CollectConfig
from burr.hardware.sk.tasks import build_requested_tasks
from burr.models.observation import ObservationRequest
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.run.utils import get_or_create_run
from burr.scheduler.factory import build_scheduler
from burr.scheduler.scheduler import Scheduler
from burr.scheduler.slot import TaskSlot
from burr.task_source.protocol import TaskSource


@dataclass
class _RequestState:
    slot: TaskSlot
    request: ObservationRequest
    started_at: float | None = None
    finished: bool = False


@dataclass
class _PendingTask:
    submission: sk.TaskSubmission
    state: _RequestState
    is_last: bool


@dataclass
class _ProgramContext:
    run: BurrRun
    config: AppConfig
    controller_entity: str
    scheduler: Scheduler
    collect: CollectConfig = field(default_factory=CollectConfig)
    pending: deque[_PendingTask] = field(default_factory=deque)


_ctx: _ProgramContext | None = None


def _context() -> _ProgramContext:
    if _ctx is None:
        raise RuntimeError("Burr program has not been initialised")
    return _ctx


def _current_condition(ctx: _ProgramContext) -> LightingCondition:
    return ctx.run.lighting_schedule.get_lighting_condition(datetime.now(UTC))


async def _publish_offers() -> None:
    ctx = _context()
    intervals = await asyncio.to_thread(
        ctx.scheduler.offer_windows,
        ctx.run.lighting_schedule,
    )
    program = sk.program()
    program.clear_offers()

    for start, end in intervals:
        program.add_offer(start, end)

    await program.publish_offers()
    ctx.scheduler.log_status(
        _current_condition(ctx),
        offer_intervals=intervals,
    )


async def _on_attach() -> None:
    await _publish_offers()


async def _on_enable() -> None:
    await _publish_offers()


async def _on_disable() -> None:
    ctx = _context()
    batch = list(ctx.pending)
    ctx.pending.clear()
    await _abandon_batch(
        ctx,
        batch,
        "Burr disabled before the request completed",
    )
    # Keep bounded offers advertised so the agent can enable Burr again.


async def _on_detach() -> None:
    ctx = _context()
    batch = list(ctx.pending)
    ctx.pending.clear()
    await _abandon_batch(ctx, batch, "Burr service stopped")


async def _validate_instruments(ctx: _ProgramContext) -> None:
    program = sk.program()
    state = await program.kv_get_model(sk.ProgramState)
    assigned = state.enable_state.controller

    if not state.enable_state.enabled or assigned != ctx.controller_entity:
        raise ValueError(
            f"Burr requires enabled controller "
            f"{ctx.controller_entity!r}; "
            f"assigned controller is {assigned!r}"
        )

    names = {name for name, _ in ctx.collect.instrument_settings() if name is not None}
    flats = ctx.config.schedule.twilight_flats
    plan = getattr(flats, "plan", None)

    if flats.collect and plan is not None:
        names.update(pair.instrument for pair in plan.pairs)

    if not names:
        return

    controller = program.sensorkit().controller(ctx.controller_entity)
    info = await controller.kv_get_model(sk.EntityInfo)

    if not isinstance(info.details, sk.ControllerDetails):
        raise ValueError(
            f"Controller {ctx.controller_entity!r} " "has no controller details"
        )

    available = {instrument.name for instrument in info.details.instruments}
    if missing := sorted(names - available):
        raise ValueError(
            f"Controller {ctx.controller_entity!r} "
            "has no instrument(s): " + ", ".join(repr(name) for name in missing)
        )


def _record_failure(state: _RequestState, error: str) -> None:
    state.slot.record_failure()
    state.slot.source.on_request_failed(state.request, error)


def _finish_request(
    ctx: _ProgramContext,
    state: _RequestState,
    error: str | None = None,
) -> None:
    if state.finished:
        return

    state.finished = True

    if error is not None:
        _record_failure(state, error)
        return

    duration = (
        time.monotonic() - state.started_at if state.started_at is not None else 0.0
    )
    try:
        state.slot.source.on_request_completed(
            state.request,
            duration,
        )
    except Exception as exc:
        _record_failure(
            state,
            f"request readback failed: {exc}",
        )
        raise

    state.slot.record_success(
        duration,
        _current_condition(ctx),
    )


def _build_next_request(
    ctx: _ProgramContext,
) -> list[_PendingTask]:
    candidates = ctx.scheduler.ranked_suitable(_current_condition(ctx))

    for slot in candidates:
        try:
            request = slot.source.generate_request(task_id=str(uuid4()))
        except Exception:
            logger.exception(f"{slot.name}: could not generate request")
            slot.record_failure()
            continue

        if request is None:
            slot.record_failure()
            continue

        state = _RequestState(slot=slot, request=request)

        try:
            submissions = build_requested_tasks(
                ctx.run,
                ctx.config,
                request,
                collect=ctx.collect,
            )
            if not submissions:
                raise ValueError("request produced no SensorKit tasks")
        except Exception as exc:
            logger.exception(f"{slot.name}: could not build request")
            _finish_request(ctx, state, str(exc))
            continue

        logger.info(
            f"{slot.name}: queued {len(submissions)} "
            f"SensorKit task(s) for Burr request {request.task_id}"
        )
        return [
            _PendingTask(
                submission,
                state,
                index == len(submissions) - 1,
            )
            for index, submission in enumerate(submissions)
        ]

    return []


async def _abandon_batch(
    ctx: _ProgramContext,
    batch: list[_PendingTask],
    reason: str,
) -> None:
    states = {id(pending.state): pending.state for pending in batch}
    for state in states.values():
        await asyncio.to_thread(
            _finish_request,
            ctx,
            state,
            reason,
        )


async def _fill_queue(ctx: _ProgramContext) -> None:
    # Cancellation cannot stop the worker thread. Wait for its result
    # and account for any request it built before propagating cancellation.
    worker = asyncio.create_task(asyncio.to_thread(_build_next_request, ctx))
    try:
        batch = await asyncio.shield(worker)
    except asyncio.CancelledError:
        batch = await asyncio.shield(worker)
        await _abandon_batch(
            ctx,
            batch,
            "Burr stopped before request dispatch",
        )
        raise

    ctx.pending.extend(batch)


def _drop_remaining(
    ctx: _ProgramContext,
    state: _RequestState,
) -> None:
    ctx.pending = deque(item for item in ctx.pending if item.state is not state)


def _report_dispatched(
    state: _RequestState,
    execution: sk.TaskExecution,
) -> None:
    callback = getattr(
        state.slot.source,
        "on_task_dispatched",
        None,
    )
    if callback is not None:
        callback(state.request, str(execution.task_id))


async def _next_task():
    """Yield once, receive an execution, and await actual completion."""
    ctx = _context()

    if not ctx.pending:
        await _fill_queue(ctx)
    if not ctx.pending:
        return

    pending = ctx.pending.popleft()
    state = pending.state

    try:
        await _validate_instruments(ctx)

        if state.started_at is None:
            state.started_at = time.monotonic()

        execution = yield pending.submission

        if execution.controller_id != ctx.controller_entity:
            raise ValueError(
                "SensorKit accepted the task on " "an unexpected controller"
            )

        _report_dispatched(state, execution)
        logger.info(
            f"Burr request {state.request.task_id} accepted as "
            f"SensorKit task {execution.task_id}"
        )

        await execution

        if pending.is_last:
            await asyncio.to_thread(
                _finish_request,
                ctx,
                state,
            )

    except (asyncio.CancelledError, GeneratorExit):
        _drop_remaining(ctx, state)
        await asyncio.to_thread(
            _finish_request,
            ctx,
            state,
            "SensorKit task cancelled",
        )
        raise

    except Exception as exc:
        _drop_remaining(ctx, state)
        await asyncio.to_thread(
            _finish_request,
            ctx,
            state,
            str(exc),
        )
        raise


def build_task_sources(
    run: BurrRun,
    config: AppConfig,
) -> dict[str, TaskSource]:
    from burr.task_source.calsats.manager import (
        CalibrationSatelliteManager,
    )
    from burr.task_source.coverage.manager import SkyCoverageManager
    from burr.task_source.flats.manager import FlatFieldManager
    from burr.task_source.lunar.manager import LunarBackgroundManager
    from burr.task_source.photometry.manager import (
        PhotometricStandardsManager,
    )

    schedule = config.schedule
    sources: dict[str, TaskSource] = {}

    if schedule.twilight_flats.collect:
        sources["twilight_flats"] = FlatFieldManager(
            run,
            sk_data_root=config.sk.data_root,
        )
    if schedule.lunar_background.collect:
        sources["lunar_background"] = LunarBackgroundManager(run=run)
    if schedule.photometric_standards.collect:
        sources["photometric_standards"] = PhotometricStandardsManager(run=run)
    if schedule.coverage.collect:
        sources["coverage"] = SkyCoverageManager(run=run)
    if schedule.calsats.collect:
        sources["calsats"] = CalibrationSatelliteManager(run=run)

    return sources


def _resolve_config_path() -> Path | list[Path] | None:
    value = os.getenv("BURR_CONFIG")
    if not value:
        return None

    separator = ";" if os.name == "nt" or ";" in value else ":"
    paths = [Path(path) for path in value.split(separator) if path]
    return paths[0] if len(paths) == 1 else paths


def _build_context(
    config: AppConfig,
    collect: CollectConfig | None = None,
) -> _ProgramContext:
    run = get_or_create_run(
        site=config.site,
        verbose=True,
        new_config=config,
    )
    sources = build_task_sources(run, config)

    return _ProgramContext(
        run=run,
        config=config,
        controller_entity=config.sk.burr_controller_entity,
        scheduler=build_scheduler(config, run, sources),
        collect=collect or CollectConfig(),
    )


async def _load_settings(
    service: sk.Service,
) -> tuple[AppConfig, CollectConfig]:
    paths = _resolve_config_path()
    if paths is not None:
        config = await asyncio.to_thread(initialize_config, paths)
        return config, CollectConfig()

    try:
        resolved = await sk.load_config(fail_policy="error")
    except FileNotFoundError as exc:
        default_file = Path("sensorkit.yaml").resolve()
        missing = Path(exc.filename).resolve() if exc.filename else None
        if "SENSORKIT_CONFIG" in os.environ or missing != default_file:
            raise
        config = await asyncio.to_thread(get_or_initialize_config)
        return config, CollectConfig()

    authored = service_config_for(resolved, service.name)
    if authored is None:
        config = await asyncio.to_thread(get_or_initialize_config)
        return config, CollectConfig()

    # Reuse this client when Service.register() registers the service later.
    # Do not register the service before configuration/initialisation succeeds.
    if service.client is None:
        service.client = await sk.connect()

    entity = service.client.entity(service.name)
    try:
        settings = await entity.kv_get_model(BurrServiceConfig)
    except sk.KVError as exc:
        raise RuntimeError(
            f"No BurrServiceConfig is loaded for {service.name!r}. "
            "Load the migrated SensorKit configuration before starting Burr."
        ) from exc
    if settings.id != service.name:
        raise ValueError("Uploaded Burr configuration has the wrong program id")

    config = await settings.to_app_config(service.client)
    set_config(config)
    return config, settings.collect


@sk.service_entrypoint(version=sk.VERSION)
async def main(service: sk.Service) -> None:
    global _ctx

    # Finish synchronous setup before registering program callbacks.
    await asyncio.to_thread(initialize_engine)
    config, collect = await _load_settings(service)
    setup_loguru(level=config.logging.level)
    _ctx = await asyncio.to_thread(_build_context, config, collect)

    declaration = sk.declare_program()
    sk.on_attach(declaration)(_on_attach)
    sk.on_detach(declaration)(_on_detach)
    sk.on_enable(declaration)(_on_enable)
    sk.on_disable(declaration)(_on_disable)
    sk.task_factory(declaration)(_next_task)
    service.add(declaration)

    try:
        await service.run()
    finally:
        _ctx = None
