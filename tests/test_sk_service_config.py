"""Configuration wiring for Burr's own SensorKit service."""
# ruff: noqa: E402

import asyncio
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("sensorkit")

import yaml

import sensorkit.api as sk
from sensorkit.astro.common import SitePosition
from sensorkit.config.parser import PARSER_VERSION, parse_config

import burr.core.config as core_config
from burr.hardware.sk import burr_program as program
from burr.hardware.sk import service_config
from burr.hardware.sk.service_config import (
    SERVICE_MODULE,
    BurrServiceConfig,
    service_config_for,
)


@pytest.fixture
def section():
    return {
        "id": "Burr",
        "controller": "Huntsman",
        "raw_data_root": "/data/raw/burr",
        "runtime": {"output_dir": "/data/processed/burr"},
        "minimum_altitude_degrees": 25,
        "collect": {
            "gain": 100, "binning": 2,
            "instruments": {"C0-ASI183MM": {}, "C1-ASI183MM": {"binning": None}},
        },
        "schedule": {
            "twilight_flats": {
                "collect": True,
                "windows": ["nautical_twilight", "civil_twilight"],
                "initial_exposure_dawn_seconds": 60,
                "plan": {
                    "frames_per_pair": 10,
                    "pairs": [
                        {"camera": "C0", "instrument": "C0-ASI183MM", "filter": "g"},
                        {"camera": "C1", "instrument": "C1-ASI183MM", "filter": "r"},
                    ],
                },
            },
        },
    }


def resolve(section, services=None):
    return parse_config({
        "version": PARSER_VERSION,
        "burr": [section],
        "services": services or [],
    }).resolve_dynamic_sections()


def test_burr_section_launches_one_owned_service(section):
    resolved = resolve(section)
    assert [(s.id, s.python_module) for s in resolved.services] == [
        ("Burr", SERVICE_MODULE)
    ]
    model = service_config_for(resolved, "Burr")
    assert isinstance(model, BurrServiceConfig)
    assert model.controller == "Huntsman"
    assert model.schedule.twilight_flats.plan.frames_per_pair == 10


def test_multiple_burr_entries_are_rejected(section):
    other = {**deepcopy(section), "id": "OtherBurr"}
    with pytest.raises(ValueError):
        parse_config({
            "version": PARSER_VERSION, "burr": [section, other]
        }).resolve_dynamic_sections()


@pytest.mark.parametrize(("extra", "message"), [
    ({"id": "Burr", "python_module": SERVICE_MODULE}, "only one"),
    ({"id": "OtherBurr", "python_module": SERVICE_MODULE}, "only one"),
    (
        {"id": "OldBurr", "python_module": "sensorkit.burr.program"},
        "sensorkit.burr.program",
    ),
    ({"id": "Burr", "python_module": "some.other.service"}, "duplicate"),
])
def test_conflicting_service_definitions_fail(section, extra, message):
    resolved = resolve(section, [extra])
    with pytest.raises(ValueError, match=message):
        service_config_for(resolved, "Burr")


def test_program_name_is_preserved_exactly(section):
    with pytest.raises(ValueError, match="id 'burr'"):
        service_config_for(resolve(section), "burr")


def test_registration_rejects_an_existing_provider(monkeypatch):
    monkeypatch.setattr(
        service_config, "get_config_section",
        lambda name: SimpleNamespace(service_path="sensorkit.burr.program"),
    )
    with pytest.raises(RuntimeError, match="already registered"):
        service_config._register_section()


def test_planned_flats_require_the_raw_data_root(section):
    section.pop("raw_data_root")
    with pytest.raises(ValueError, match="raw_data_root"):
        BurrServiceConfig.model_validate(section)


def test_controller_paths_schedule_and_camera_settings_survive_round_trip(section):
    model = BurrServiceConfig.model_validate_json(
        BurrServiceConfig.model_validate(section).model_dump_json()
    )
    client = SimpleNamespace(kv_get_model=AsyncMock(return_value=SitePosition(
        latitude_degrees=-31.27, longitude_degrees=149.06, altitude_km=1.165
    )))
    kit = SimpleNamespace(controller=Mock(return_value=client))
    app = asyncio.run(model.to_app_config(kit))
    kit.controller.assert_called_once_with("Huntsman")
    client.kv_get_model.assert_awaited_once_with(SitePosition)
    assert app.site.name == "Huntsman"
    assert app.site.minimum_altitude_degrees == 25
    assert app.sk.burr_controller_entity == "Huntsman"
    assert app.sk.data_root == Path(section["raw_data_root"])
    assert app.runtime.output_dir == section["runtime"]["output_dir"]
    assert app.schedule == model.schedule
    camera = dict(model.collect.instrument_settings())
    assert camera["C0-ASI183MM"].binning == 2
    assert camera["C1-ASI183MM"].binning is None
    assert camera["C1-ASI183MM"].gain == 100


def test_loader_reads_uploaded_settings_and_reuses_the_client(section, monkeypatch):
    monkeypatch.delenv("BURR_CONFIG", raising=False)
    monkeypatch.setattr(core_config, "_config_instance", None)
    resolved = resolve(section)
    uploaded = BurrServiceConfig.model_validate(section)
    entity = SimpleNamespace(kv_get_model=AsyncMock(return_value=uploaded))
    controller = SimpleNamespace(kv_get_model=AsyncMock(return_value=SitePosition(
        latitude_degrees=-31.27, longitude_degrees=149.06, altitude_km=1.165
    )))
    kit = SimpleNamespace(
        entity=Mock(return_value=entity),
        controller=Mock(return_value=controller),
    )
    service = SimpleNamespace(name="Burr", client=None)
    connect = AsyncMock(return_value=kit)
    monkeypatch.setattr(program.sk, "load_config", AsyncMock(return_value=resolved))
    monkeypatch.setattr(program.sk, "connect", connect)

    app, collect = asyncio.run(program._load_settings(service))

    entity.kv_get_model.assert_awaited_once_with(BurrServiceConfig)
    kit.entity.assert_called_once_with("Burr")
    connect.assert_awaited_once()
    assert service.client is kit
    assert core_config.get_config() is app
    assert collect is uploaded.collect
    assert app.sk.data_root == uploaded.raw_data_root


def test_explicit_burr_config_retains_standalone_mode(monkeypatch):
    monkeypatch.setenv("BURR_CONFIG", "site.yaml;preset.yaml")
    marker = object()
    initialise = Mock(return_value=marker)
    load = AsyncMock()
    monkeypatch.setattr(program, "initialize_config", initialise)
    monkeypatch.setattr(program.sk, "load_config", load)
    service = SimpleNamespace(name="Burr", client=None)
    app, _ = asyncio.run(program._load_settings(service))
    assert app is marker
    initialise.assert_called_once_with([Path("site.yaml"), Path("preset.yaml")])
    load.assert_not_awaited()


def test_bad_sensorkit_config_does_not_fall_back(monkeypatch):
    monkeypatch.delenv("BURR_CONFIG", raising=False)
    fallback = Mock()
    monkeypatch.setattr(program, "get_or_initialize_config", fallback)
    monkeypatch.setattr(
        program.sk,
        "load_config",
        AsyncMock(side_effect=ValueError("invalid configuration")),
    )
    with pytest.raises(ValueError, match="invalid configuration"):
        service = SimpleNamespace(name="Burr", client=None)
        asyncio.run(program._load_settings(service))
    fallback.assert_not_called()


def test_missing_import_does_not_fall_back(monkeypatch):
    monkeypatch.delenv("BURR_CONFIG", raising=False)
    monkeypatch.delenv("SENSORKIT_CONFIG", raising=False)
    fallback = Mock()
    monkeypatch.setattr(program, "get_or_initialize_config", fallback)
    monkeypatch.setattr(program.sk, "load_config", AsyncMock(side_effect=FileNotFoundError(
        2, "missing import", "missing_plugin.py"
    )))
    with pytest.raises(FileNotFoundError):
        service = SimpleNamespace(name="Burr", client=None)
        asyncio.run(program._load_settings(service))
    fallback.assert_not_called()


def test_cli_does_not_force_a_standalone_default(monkeypatch):
    import sensorkit.api.entrypoint as entrypoint
    from burr.cli import sk as cli

    monkeypatch.delenv("BURR_CONFIG", raising=False)
    monkeypatch.setattr("sys.argv", ["burr-sk"])
    run_services = AsyncMock()
    monkeypatch.setattr(entrypoint, "run_services", run_services)
    assert cli.main() == 0
    assert "BURR_CONFIG" not in os.environ
    assert set(run_services.call_args.args[0]) == {"Burr"}


def test_huntsman_example_selects_the_owned_service():
    path = Path(__file__).parents[1] / "deploy/example/huntsman-flats-lum.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["sensorkit"]["imports"] == [SERVICE_MODULE]
    resolved = parse_config({
        "version": PARSER_VERSION,
        **document,
    }).resolve_dynamic_sections()
    settings = service_config_for(resolved, "Burr")
    assert settings.controller == "Huntsman"
    assert settings.raw_data_root == Path("/opt/huntsman/data/raw/burr")
    assert settings.runtime.output_dir == "/opt/huntsman/data/processed/burr/"
    assert len(settings.schedule.twilight_flats.plan.pairs) == 9


def test_entry_point_declares_one_named_program(section, monkeypatch):
    settings = BurrServiceConfig.model_validate(section)
    context = SimpleNamespace(collect=settings.collect)
    monkeypatch.setattr(program, "initialize_engine", Mock())
    monkeypatch.setattr(program, "_load_settings", AsyncMock(return_value=(
        SimpleNamespace(logging=SimpleNamespace(level="INFO")), settings.collect,
    )))
    build = Mock(return_value=context)
    monkeypatch.setattr(program, "_build_context", build)

    async def run_without_backend(service):
        service.running.set_result(True)
        await service.shutdown

    monkeypatch.setattr(sk.Service, "run", run_without_backend)

    async def run():
        service, task = await program.main.run("Burr")
        try:
            assert len(service.declarations) == 1
            assert {decl.name for decl in service.declarations} == {"Burr"}
            assert program._ctx is context
            assert build.call_args.args[1] is settings.collect
        finally:
            service.shutdown.set_result(True)
            await task
        assert program._ctx is None

    asyncio.run(run())


def test_service_start_keeps_the_host_logging_sinks(section, monkeypatch):
    """11-17: main() called setup_loguru(), whose remove() deleted SensorKit's
    --log-file sink (loguru is process-global), so log.txt went silent."""
    from loguru import logger

    settings = BurrServiceConfig.model_validate(section)
    host_lines: list[str] = []
    host_sink = logger.add(host_lines.append, level="DEBUG", format="{message}")
    monkeypatch.setattr(program, "initialize_engine", Mock())
    monkeypatch.setattr(program, "_load_settings", AsyncMock(return_value=(
        SimpleNamespace(logging=SimpleNamespace(level="INFO")), settings.collect,
    )))
    monkeypatch.setattr(program, "_build_context", Mock(return_value=SimpleNamespace(collect=settings.collect)))

    async def run_without_backend(service):
        service.running.set_result(True)
        await service.shutdown

    monkeypatch.setattr(sk.Service, "run", run_without_backend)

    async def run():
        service, task = await program.main.run("Burr")
        try:
            logger.debug("host debug line after burr start-up")
        finally:
            service.shutdown.set_result(True)
            await task

    try:
        asyncio.run(run())
        assert any("host debug line after burr start-up" in line for line in host_lines)
    finally:
        logger.remove(host_sink)


def test_burr_sk_cli_reads_its_log_level_from_the_standalone_config(tmp_path):
    from burr.cli.sk import _standalone_log_level

    site = tmp_path / "site.yaml"
    site.write_text("app:\n  logging:\n    level: DEBUG\n")
    preset = tmp_path / "preset.yaml"
    preset.write_text("app:\n  schedule: {}\n")
    assert _standalone_log_level([site, preset]) == "DEBUG"
    assert _standalone_log_level([preset]) == "INFO"
    assert _standalone_log_level([]) == "INFO"
