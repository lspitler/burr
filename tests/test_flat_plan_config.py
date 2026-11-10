"""Validation of the twilight-flat collection plan (``twilight_flats.plan``).

The plan lists every camera/filter pair that needs flats. These tests check
it is rejected when empty, duplicated, ambiguous or unsafe, and that the
manager tracks planned pairs from the start. End-to-end routing through
sensorkit is tested in sensorkit's ``modules/burr/tests/test_flat_plan.py``.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from burr.core.config import TwilightFlatsConfig
from burr.task_source.flats.manager import FlatFieldManager, FlatSequenceKey
from burr.task_source.flats.plan import FlatPlanConfig


def _pair(camera="C0", filter_name="Lum", instrument=None, **kw):
    return {
        "camera": camera,
        "instrument": instrument or f"{camera}-ASI183MM",
        "filter": filter_name,
        **kw,
    }


def test_a_valid_plan_with_defaults_and_overrides():
    plan = FlatPlanConfig(
        frames_per_pair=10, pairs=[_pair("C0"), _pair("C1", frames=4)]
    )
    assert [(p.key.label(), plan.target(p)) for p in plan.pairs] == [
        ("C0/Lum", 10),
        ("C1/Lum", 4),
    ]
    assert plan.filter_throughput is None


def test_filter_throughput_covers_unique_planned_filters():
    plan = FlatPlanConfig(
        pairs=[
            _pair("C0", "Lum"),
            _pair("C1", "Lum"),
            _pair("C0", "g"),
            _pair("C0", "r"),
        ],
        filter_throughput={"Lum": 1.0, "g": 0.69, "r": 0.21},
    )

    assert plan.filter_throughput == {"Lum": 1.0, "g": 0.69, "r": 0.21}


def test_filter_throughput_may_be_above_one():
    plan = FlatPlanConfig(
        pairs=[_pair("C0", "Clear")],
        filter_throughput={"Clear": 1.4},
    )

    assert plan.filter_throughput == {"Clear": 1.4}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param(0.0, "greater than zero", id="zero"),
        pytest.param(-0.1, "greater than zero", id="negative"),
        pytest.param(float("nan"), "finite", id="nan"),
        pytest.param(float("inf"), "finite", id="positive-infinity"),
        pytest.param(float("-inf"), "finite", id="negative-infinity"),
    ],
)
def test_filter_throughput_must_be_positive_and_finite(value, message):
    with pytest.raises(ValidationError, match=message):
        FlatPlanConfig(
            pairs=[_pair("C0", "Lum")],
            filter_throughput={"Lum": value},
        )


def test_filter_throughput_rejects_missing_planned_filter():
    with pytest.raises(ValidationError, match=r"missing planned filter\(s\): r"):
        FlatPlanConfig(
            pairs=[_pair("C0", "Lum"), _pair("C0", "r")],
            filter_throughput={"Lum": 1.0},
        )


def test_filter_throughput_rejects_unplanned_filter():
    with pytest.raises(ValidationError, match=r"unplanned filter\(s\): r"):
        FlatPlanConfig(
            pairs=[_pair("C0", "Lum")],
            filter_throughput={"Lum": 1.0, "r": 0.21},
        )


def test_filter_throughput_names_are_case_sensitive():
    with pytest.raises(ValidationError, match=r"missing planned filter\(s\): Lum"):
        FlatPlanConfig(
            pairs=[_pair("C0", "Lum")],
            filter_throughput={"lum": 1.0},
        )


def test_filter_throughput_rejects_empty_table():
    with pytest.raises(ValidationError, match="cannot be empty"):
        FlatPlanConfig(
            pairs=[_pair("C0", "Lum")],
            filter_throughput={},
        )


@pytest.mark.parametrize("name", [" Lum", "r/g", "g_r"])
def test_filter_throughput_rejects_unsafe_names(name):
    with pytest.raises(ValidationError):
        FlatPlanConfig(
            pairs=[_pair("C0", "Lum")],
            filter_throughput={"Lum": 1.0, name: 0.5},
        )


def test_twilight_config_accepts_filter_throughput():
    config = TwilightFlatsConfig(
        plan={
            "frames_per_pair": 10,
            "filter_throughput": {"Lum": 1.0, "g": 0.69, "r": 0.21},
            "pairs": [
                _pair("C0", "Lum"),
                _pair("C0", "g"),
                _pair("C0", "r"),
            ],
        }
    )

    assert config.plan.filter_throughput["r"] == 0.21


@pytest.mark.parametrize(
    "pairs, message",
    [
        pytest.param([], "at least one", id="empty"),
        pytest.param(
            [_pair("C0"), _pair("C0")], "duplicate plan entry C0/Lum", id="duplicate"
        ),
        pytest.param(
            [_pair("C0"), _pair("c0", "lum", instrument="C0-ASI183MM")],
            "duplicate",
            id="case-duplicate",
        ),
        pytest.param([_pair("../C0")], "unsafe", id="camera-traversal"),
        pytest.param([_pair("C0", "a/b")], "unsafe", id="filter-separator"),
        pytest.param([_pair("C0", "g_r")], "unsafe", id="filter-underscore"),
        pytest.param([_pair("C0", " Lum")], "whitespace", id="filter-whitespace"),
        pytest.param([_pair("C0", "")], "not a usable", id="filter-empty"),
        pytest.param(
            [_pair("C0", instrument="C0 ASI")],
            "not a valid device key",
            id="instrument-space",
        ),
        pytest.param(
            [_pair("C0", instrument="X"), _pair("C1", instrument="X")],
            "mapped to both",
            id="shared-instrument",
        ),
        pytest.param(
            [_pair("C0", "Lum", instrument="A"), _pair("C0", "g", instrument="B")],
            "mapped to both",
            id="two-instruments",
        ),
        pytest.param(
            [_pair("C0", frames=0)], "greater than or equal to 1", id="zero-frames"
        ),
        pytest.param([_pair("C0", typo=1)], "Extra inputs", id="unknown-key"),
    ],
)
def test_invalid_plans_are_rejected(pairs, message):
    with pytest.raises(ValidationError, match=message):
        FlatPlanConfig(pairs=pairs)


def test_frames_per_pair_must_be_positive():
    with pytest.raises(ValidationError):
        FlatPlanConfig(frames_per_pair=0, pairs=[_pair()])


def test_the_lum_example_in_the_docs_validates():
    """The example plan documented in docs/flats.md (Lum on every Huntsman camera)."""
    cameras = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C10"]
    config = TwilightFlatsConfig(
        windows=["nautical_twilight"],
        plan={"frames_per_pair": 10, "pairs": [_pair(c, "Lum") for c in cameras]},
    )
    assert len(config.plan.pairs) == 9
    assert all(config.plan.target(p) == 10 for p in config.plan.pairs)


def _run(tmp_path, config):
    return SimpleNamespace(
        config=SimpleNamespace(schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=[],
        output_directory=tmp_path / "processed",
        lighting_schedule=SimpleNamespace(
            local_midnight=datetime.now(UTC) + timedelta(hours=4)
        ),
        log_command=lambda *a, **k: None,
    )


def test_planned_pairs_are_tracked_before_any_frame(tmp_path):
    config = TwilightFlatsConfig(
        plan={"frames_per_pair": 3, "pairs": [_pair("C0"), _pair("C1", "g")]}
    )
    manager = FlatFieldManager(_run(tmp_path, config), sk_data_root=tmp_path / "raw")
    assert set(manager.cameras) == {
        FlatSequenceKey("C0", "Lum"),
        FlatSequenceKey("C1", "g"),
    }
    assert manager.progress_summary() == "C0/Lum=0/3, C1/g=0/3"
    assert not manager.is_collection_complete()
    assert manager.good_frames == 0


def test_a_plan_requires_sensorkit_mode(tmp_path):
    config = TwilightFlatsConfig(plan={"pairs": [_pair()]})
    with pytest.raises(ValueError, match="SensorKit mode"):
        FlatFieldManager(_run(tmp_path, config), sk_data_root=None)


def test_without_a_plan_requests_and_completion_are_unchanged(tmp_path):
    config = TwilightFlatsConfig(windows=["nautical_twilight"], n_frames=2)
    manager = FlatFieldManager(_run(tmp_path, config), sk_data_root=tmp_path / "raw")
    assert manager.plan is None and manager.cameras == {}
    request = manager.generate_request(task_id="t")
    assert "flat_plan" not in request.metadata
    assert request.exposure_seconds == [manager.minimum_exposure]
    assert not manager.is_collection_complete()
