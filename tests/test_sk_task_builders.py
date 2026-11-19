"""Burr task construction against SensorKit's native collection APIs."""
# ruff: noqa: E402

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("sensorkit")

from burr.core.config import AppConfig, PlottingConfig
from burr.hardware.sk.settings import CollectConfig
from burr.hardware.sk.tasks import BurrContext, build_requested_tasks
from burr.models.observation import (
    AltAzTarget,
    ExposureSpec,
    ObservationRequest,
    RADecTarget,
)
from burr.models.site import SiteMetadata
from burr.models.tracking import Rates, TrackingMode

import sensorkit.api as sk
from sensorkit.astro.target import ICRSTarget, RateTarget
from sensorkit.core.entity import DeviceDetails, InstrumentInfo
from sensorkit.core.task import TaskInfo
from sensorkit.data.context import Context
from sensorkit.data.filesys import FileNameTemplate
from sensorkit.sensor.binding import BoundSensor
from sensorkit.sensor.collect import SettingUnsatisfiable, pack
from sensorkit.sensor.standard_task import translate
from sensorkit.sensor.topology import Structure, Topology
from sensorkit.std.collect import CollectInfo
from sensorkit.std.instrument import FrameType
from sensorkit.std.optics import Filter, Filters

SIDEREAL = TrackingMode.SIDEREAL
RATE = TrackingMode.RATE
OFF = TrackingMode.OFF


@pytest.fixture
def inputs():
    run = SimpleNamespace(
        lighting_schedule=SimpleNamespace(
            schedule_end_utc=datetime.now(UTC) + timedelta(hours=12)
        )
    )
    config = AppConfig(
        stage="test",
        plotting=PlottingConfig(debug=False, review=False),
        site=SiteMetadata(
            name="test", latitude=0, longitude=0, altitude_km=0
        ),
    )
    return run, config


def observation(
    modes, *, source="coverage", metadata=None, target=None
):
    return ObservationRequest(
        task_id="request",
        source_name=source,
        target=target or RADecTarget(
            right_ascension_deg=10,
            declination_deg=20,
        ),
        exposure_seconds=[0.5, 2.0],
        exposure_map=[
            ExposureSpec(tracking_mode=mode) for mode in modes
        ],
        rates=Rates(
            right_ascension_rate_arcsec_s=36,
            declination_rate_arcsec_s=-18,
        ),
        metadata=metadata or {},
    )


def flat_request():
    return observation(
        [OFF],
        source="twilight_flats",
        target=AltAzTarget(altitude_deg=75, azimuth_deg=0),
        metadata={
            "flat_plan": [
                {
                    "camera": "C1",
                    "instrument": "west",
                    "filter": "Lum",
                    "exposure_seconds": 2.0,
                },
                {
                    "camera": "C0",
                    "instrument": "east",
                    "filter": "g",
                    "exposure_seconds": 0.5,
                },
            ]
        },
    )


def calibration_request():
    return ObservationRequest(
        task_id="calibration-request",
        source_name="calibration_frames",
        target=None,
        exposure_seconds=[],
        exposure_map=[],
        rates=None,
        metadata={
            "target_name": "calibration",
            "calibration_plan": [
                {
                    "camera": "C1",
                    "instrument": "west",
                    "filter": "Blank",
                    "frame_type": "dark",
                    "exposure_seconds": 30.0,
                    "frame_count": 2,
                },
                {
                    "camera": "C0",
                    "instrument": "east",
                    "filter": "Blank",
                    "frame_type": "bias",
                    "exposure_seconds": 0.0,
                    "frame_count": 3,
                },
            ],
        },
    )


@pytest.fixture
def sensor():
    topology = Topology(
        Structure.model_validate(
            {
                "components": [
                    {"device": "mount"},
                    *[
                        {
                            "unit": name,
                            "components": [
                                {"device": f"wheel-{name}"},
                                {
                                    "device": f"cam-{name}",
                                    "instrument": True,
                                },
                            ],
                        }
                        for name in ("east", "west")
                    ],
                ]
            }
        )
    )
    details = {
        "mount": DeviceDetails(
            supported_commands={"Connect", "FollowTarget", "Stop"}
        )
    }
    for name in ("east", "west"):
        details[f"wheel-{name}"] = DeviceDetails(
            supported_commands={"SetFilter"}
        )
        details[f"cam-{name}"] = DeviceDetails(
            supported_commands={"Connect", "ConfigureCameraSensor"}
        )

    bound = BoundSensor.bind(topology, details)
    named = {
        placement.device.removeprefix("cam-"): placement
        for placement in topology.instruments()
    }
    return bound, named


def test_calibration_task_is_targetless_and_pinned(
    inputs, sensor
):
    run, config = inputs
    settings = CollectConfig(
        gain=100,
        binning=1,
        readout_mode=2,
        instruments={
            "west": {"gain": 200},
            "east": {},
        },
    )

    (submission,) = build_requested_tasks(
        run, config, calibration_request(), settings
    )

    assert submission.task.target is None
    assert submission.task.target_id == "calibration"
    assert submission.task.sidereal_frames == []
    assert [
        (
            params.instrument,
            params.filter_name,
            params.frame_type,
            params.integration_time_seconds,
            params.frame_count,
            params.gain,
            params.binning_x,
            params.binning_y,
            params.readout_mode,
        )
        for params in submission.task.exposures
    ] == [
        (
            "west",
            "Blank",
            FrameType.DARK,
            30.0,
            2,
            200,
            1,
            1,
            2,
        ),
        (
            "east",
            "Blank",
            FrameType.BIAS,
            0.0,
            3,
            100,
            1,
            1,
            2,
        ),
    ]

    bound, named = sensor
    packed = pack(
        translate(submission.task, instruments=named),
        bound,
    )

    # This is the key behaviour supplied by SensorKit 146396a.
    assert packed.prepare == ()
    assert packed.cleanup == ()
    assert all(
        setting.target.device != "mount"
        for epoch in packed.epochs
        for setting in epoch.settings
    )

    captures = [
        (
            unit.target.device,
            unit.command.frame_type,
            unit.command.integration_time,
        )
        for epoch in packed.epochs
        for unit in epoch.units
    ]
    assert captures.count(
        ("cam-west", FrameType.DARK, 30.0)
    ) == 2
    assert captures.count(
        ("cam-east", FrameType.BIAS, 0.0)
    ) == 3


@pytest.mark.parametrize(
    ("modes", "expected"),
    [
        ([SIDEREAL] * 3, [(ICRSTarget, 3, [])]),
        ([OFF] * 2, [(ICRSTarget, 2, [])]),
        ([RATE] * 2, [(RateTarget, 2, [])]),
        ([RATE, SIDEREAL, SIDEREAL], [(RateTarget, 3, [1, 2])]),
        (
            [SIDEREAL, RATE, RATE],
            [(ICRSTarget, 1, []), (RateTarget, 2, [])],
        ),
    ],
)
def test_ordinary_tracking_patterns(inputs, modes, expected):
    run, config = inputs
    submissions = build_requested_tasks(
        run, config, observation(modes)
    )
    actual = [
        (
            type(submission.task.target),
            submission.task.exposures[0].frame_count,
            submission.task.sidereal_frames,
        )
        for submission in submissions
    ]

    assert actual == expected * 2
    assert all(
        submission.expiry_time
        == run.lighting_schedule.schedule_end_utc
        for submission in submissions
    )
    assert all(
        "task_id" not in type(submission.task).model_fields
        for submission in submissions
    )


def test_split_tasks_share_sequence_only_within_their_exposure(inputs):
    run, config = inputs
    submissions = build_requested_tasks(
        run, config, observation([SIDEREAL, RATE])
    )
    sequences = [
        submission.context[BurrContext].seq
        for submission in submissions
    ]

    assert sequences[0] == sequences[1]
    assert sequences[2] == sequences[3]
    assert sequences[0] != sequences[2]


def test_rate_units_and_planless_flat_type(inputs):
    run, config = inputs
    first, _ = build_requested_tasks(
        run, config, observation([RATE])
    )

    assert first.task.target.rates.ra == pytest.approx(0.01)
    assert first.task.target.rates.dec == pytest.approx(-0.005)

    plain_flats = build_requested_tasks(
        run, config, observation([OFF], source="twilight_flats")
    )
    assert all(
        submission.task.exposures[0].frame_type == FrameType.FLAT
        for submission in plain_flats
    )


def test_planned_pairs_preserve_settings_and_capture_as_flats(
    inputs, sensor
):
    run, config = inputs
    settings = CollectConfig(
        filter_name="wrong",
        gain=100,
        binning=2,
        readout_mode=1,
        instruments={
            "west": {
                "filter_name": "also-wrong",
                "gain": 200,
                "binning": None,
            }
        },
    )
    (submission,) = build_requested_tasks(
        run, config, flat_request(), settings
    )

    assert [
        (
            params.instrument,
            params.filter_name,
            params.integration_time_seconds,
            params.gain,
            params.binning_x,
            params.binning_y,
            params.readout_mode,
        )
        for params in submission.task.exposures
    ] == [
        ("west", "Lum", 2.0, 200, None, None, 1),
        ("east", "g", 0.5, 100, 2, 2, 1),
    ]

    bound, named = sensor
    packed = pack(
        translate(submission.task, instruments=named),
        bound,
    )

    assert len(packed.epochs) == 1
    assert {
        (
            unit.target.device,
            unit.command.integration_time,
            unit.command.frame_type,
        )
        for epoch in packed.epochs
        for unit in epoch.units
    } == {
        ("cam-west", 2.0, FrameType.FLAT),
        ("cam-east", 0.5, FrameType.FLAT),
    }


def test_missing_filter_cannot_reassign_the_planned_camera(
    inputs, sensor
):
    run, config = inputs
    (submission,) = build_requested_tasks(
        run, config, flat_request()
    )
    bound, named = sensor
    held = {
        "wheel-west": sk.KeywordDict(
            Filters(filters=[Filter(name="g")])
        ),
        "wheel-east": sk.KeywordDict(
            Filters(filters=[Filter(name="Lum"), Filter(name="g")])
        ),
    }

    with pytest.raises(SettingUnsatisfiable):
        pack(
            translate(submission.task, instruments=named),
            bound,
            device_keywords=held,
        )


def test_multiple_filters_keep_the_plan_order(inputs):
    run, config = inputs
    request = flat_request()
    request.metadata["flat_plan"].append(
        {
            "camera": "C1",
            "instrument": "west",
            "filter": "g",
            "exposure_seconds": 1.0,
        }
    )
    (submission,) = build_requested_tasks(run, config, request)

    assert [
        (params.instrument, params.filter_name)
        for params in submission.task.exposures
    ] == [
        ("west", "Lum"),
        ("east", "g"),
        ("west", "g"),
    ]


def test_filename_uses_execution_identity_and_distinguishes_instruments(
    inputs,
):
    run, config = inputs
    (submission,) = build_requested_tasks(
        run, config, flat_request()
    )
    minted = uuid.uuid4()
    names = []

    for params in submission.task.exposures:
        context = Context(
            submission.context,
            TaskInfo(
                task=submission.task,
                id=minted,
                controller_id="ctrl",
            ),
            InstrumentInfo(name=params.instrument),
            CollectInfo(
                target=submission.task.target,
                params=params,
                frame_number=0,
            ),
        )
        names.append(
            context.eval(
                submission.context[FileNameTemplate].template
            )
        )

    assert len(set(names)) == 2
    assert all(str(minted) in name for name in names)
    assert all("/" not in name and "\\" not in name for name in names)


def test_settings_round_trip_preserves_omitted_and_null_overrides():
    original = CollectConfig(
        gain=100,
        binning=2,
        instruments={
            "east": {},
            "west": {"binning": None},
        },
    )
    restored = CollectConfig.model_validate_json(
        original.model_dump_json()
    )
    resolved = dict(restored.instrument_settings())

    assert resolved["east"].binning == 2
    assert resolved["west"].binning is None
    assert resolved["east"].gain == resolved["west"].gain == 100


def test_invalid_requests_fail_instead_of_using_a_fallback(inputs):
    run, config = inputs

    with pytest.raises(ValueError, match="at least one"):
        build_requested_tasks(
            run,
            config,
            observation(
                [OFF],
                source="twilight_flats",
                metadata={"flat_plan": []},
            ),
        )

    with pytest.raises(ValueError, match="untracked"):
        request = flat_request()
        request.exposure_map = [
            ExposureSpec(tracking_mode=SIDEREAL)
        ]
        build_requested_tasks(run, config, request)

    with pytest.raises(ValueError, match="interleaving"):
        build_requested_tasks(
            run,
            config,
            observation([SIDEREAL, RATE, SIDEREAL]),
        )

    with pytest.raises(ValueError, match="require tracking rates"):
        request = observation([RATE])
        request.rates = None
        build_requested_tasks(run, config, request)


def test_calibration_request_requires_no_target(inputs):
    run, config = inputs
    request = calibration_request()
    request.target = RADecTarget(
        right_ascension_deg=10,
        declination_deg=20,
    )

    with pytest.raises(ValueError, match="must have target=None"):
        build_requested_tasks(run, config, request)


def test_calibration_request_rejects_empty_plan(inputs):
    run, config = inputs
    request = calibration_request()
    request.metadata["calibration_plan"] = []

    with pytest.raises(ValueError, match="at least one sequence"):
        build_requested_tasks(run, config, request)


def test_calibration_request_rejects_light_frames(inputs):
    run, config = inputs
    request = calibration_request()
    request.metadata["calibration_plan"][0]["frame_type"] = "light"

    with pytest.raises(ValueError, match="DARK or BIAS"):
        build_requested_tasks(run, config, request)


def test_calibration_request_rejects_tracking_frames(inputs):
    run, config = inputs
    request = calibration_request()
    request.exposure_map = [
        ExposureSpec(tracking_mode=TrackingMode.OFF)
    ]

    with pytest.raises(
        ValueError,
        match="only through metadata.calibration_plan",
    ):
        build_requested_tasks(run, config, request)


def test_dark_requires_positive_exposure(inputs):
    run, config = inputs
    request = calibration_request()
    request.metadata["calibration_plan"][0][
        "exposure_seconds"
    ] = 0.0

    with pytest.raises(ValueError, match="greater than zero"):
        build_requested_tasks(run, config, request)
