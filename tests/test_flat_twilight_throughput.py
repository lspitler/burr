"""Separate dusk and dawn filter throughput tables (the dusk sky is bluer than the dawn sky)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from burr.core.config import TwilightFlatsConfig
from burr.task_source.flats.manager import ExposureRecord, FlatFieldManager
from burr.task_source.flats.plan import FlatPlanConfig
from burr.task_source.flats.products import FlatSequenceKey

NOW = datetime(2026, 11, 10, 18, 0, tzinfo=UTC)
DUSK = {"g": 1.0, "r": 0.27, "i": 0.14, "z": 0.015, "u": 0.007}
DAWN = {"g": 1.0, "r": 0.45, "i": 0.34, "z": 0.05, "u": 0.004}
FILTERS = ("g", "r", "i", "z", "u")


def _pairs(filters=FILTERS):
    return [{"camera": "C3", "instrument": "C3-ASI183MM", "filter": f} for f in filters]


def _manager(tmp_path, *, dawn, sequencing="batched", **tables):
    plan = {"pairs": _pairs(), "filter_sequencing": sequencing, **tables}
    if sequencing == "per_filter":
        plan["filter_phase_order"] = "throughput"
    config = TwilightFlatsConfig(
        initial_exposure_seconds=1.0,
        minimum_exposure_seconds=0.1,
        maximum_exposure_seconds=60.0,
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


def _first(manager, f):
    return manager._pair_exposure(FlatSequenceKey("C3", f), now=NOW)


@pytest.mark.parametrize(("dawn", "table"), [(False, DUSK), (True, DAWN)])
def test_each_twilight_scales_from_its_own_table(tmp_path, dawn, table):
    m = _manager(tmp_path, dawn=dawn, filter_throughput_dusk=DUSK, filter_throughput_dawn=DAWN)
    assert _first(m, "r") == pytest.approx(1.0 / table["r"])
    assert _first(m, "i") == pytest.approx(1.0 / table["i"])


@pytest.mark.parametrize("dawn", [False, True])
def test_missing_twilight_table_falls_back_to_filter_throughput(tmp_path, dawn):
    m = _manager(tmp_path, dawn=dawn, filter_throughput=DAWN, filter_throughput_dusk=DUSK)
    expected = DAWN if dawn else DUSK
    assert _first(m, "r") == pytest.approx(1.0 / expected["r"])


def test_only_filter_throughput_behaves_as_before(tmp_path):
    for dawn in (False, True):
        m = _manager(tmp_path, dawn=dawn, filter_throughput=DAWN)
        assert _first(m, "r") == pytest.approx(1.0 / DAWN["r"])


def test_seeding_uses_this_twilights_ratios(tmp_path):
    m = _manager(tmp_path, dawn=False, filter_throughput_dusk=DUSK, filter_throughput_dawn=DAWN)
    m.cameras[FlatSequenceKey("C3", "g")].exposures.append(
        ExposureRecord(timestamp=NOW, exposure_time=2.0, median_counts=37500.0, saturation_percentage=0.0, is_good=True)
    )
    assert _first(m, "r") == pytest.approx(2.0 * DUSK["g"] / DUSK["r"])


def test_batched_ordering_uses_this_twilights_table(tmp_path):
    # i and r swap between tables here, so the order shows which table was used.
    dusk = {**DUSK, "r": 0.10, "i": 0.20}
    m = _manager(tmp_path, dawn=False, filter_throughput_dusk=dusk, filter_throughput_dawn=DAWN)
    assert [k.filter for k in m._ordered_remaining_pairs(NOW)] == ["u", "z", "r", "i", "g"]


@pytest.mark.parametrize(("dawn", "order"), [(False, "u"), (True, "g")])
def test_per_filter_phase_uses_this_twilights_table(tmp_path, dawn, order):
    m = _manager(
        tmp_path, dawn=dawn, sequencing="per_filter", filter_throughput_dusk=DUSK, filter_throughput_dawn=DAWN
    )
    assert [k.filter for k in m._filter_phase_pairs(NOW)] == [order]


def test_per_filter_throughput_order_needs_a_table_for_both_twilights():
    with pytest.raises(ValidationError, match="both filter_throughput_dusk and filter_throughput_dawn"):
        FlatPlanConfig(
            pairs=_pairs(),
            filter_sequencing="per_filter",
            filter_phase_order="throughput",
            filter_throughput_dusk=DUSK,
        )


def test_per_filter_with_both_twilight_tables_and_no_shared_table_is_valid():
    plan = FlatPlanConfig(
        pairs=_pairs(),
        filter_sequencing="per_filter",
        filter_phase_order="throughput",
        filter_throughput_dusk=DUSK,
        filter_throughput_dawn=DAWN,
    )
    assert plan.throughput_for(dawn=False) == DUSK and plan.throughput_for(dawn=True) == DAWN


@pytest.mark.parametrize("name", ["filter_throughput_dusk", "filter_throughput_dawn"])
def test_twilight_tables_are_validated_like_filter_throughput(name):
    with pytest.raises(ValidationError, match="must list every planned filter"):
        FlatPlanConfig(pairs=_pairs(), **{name: {"g": 1.0, "r": 0.3}})
    with pytest.raises(ValidationError, match="finite"):
        FlatPlanConfig(pairs=_pairs(), **{name: {**DUSK, "u": 0.0}})
    with pytest.raises(ValidationError, match="cannot be empty"):
        FlatPlanConfig(pairs=_pairs(), **{name: {}})
