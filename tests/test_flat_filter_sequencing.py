"""per_filter sequencing: each camera works one filter at a time, all cameras concurrently.

Dusk goes red to blue, dawn blue to red (by ``filter_wavelength_nm``). A camera
moves to its next filter once the current one is complete or exhausted for
this twilight (at the maximum exposure but too faint at dusk; at the minimum
exposure but too bright at dawn).
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from burr.core.config import TwilightFlatsConfig
from burr.task_source.flats.manager import ExposureRecord, FlatFieldManager
from burr.task_source.flats.products import FlatSequenceKey

NOW = datetime(2026, 11, 10, 9, 10, tzinfo=UTC)
WAVELENGTH = {"u": 355.0, "g": 475.0, "r": 620.0, "i": 760.0, "z": 900.0}
THROUGHPUT = {"u": 0.01, "g": 1.0, "r": 0.34, "i": 0.15, "z": 0.02}


def _pair(camera, f, frames=None):
    p = {"camera": camera, "instrument": f"{camera}-ASI183MM", "filter": f}
    if frames:
        p["frames"] = frames
    return p


def _manager(tmp_path, pairs, *, dawn=False, sequencing="per_filter", frames_per_pair=3):
    filters = {p["filter"] for p in pairs}
    config = TwilightFlatsConfig(
        minimum_exposure_seconds=0.1,
        maximum_exposure_seconds=60.0,
        min_counts=25000,
        max_counts=50000,
        plan={
            "frames_per_pair": frames_per_pair,
            "pairs": pairs,
            "filter_sequencing": sequencing,
            "filter_wavelength_nm": {f: WAVELENGTH[f] for f in filters},
            "filter_throughput": {f: THROUGHPUT[f] for f in filters},
        },
    )
    midnight = NOW - timedelta(hours=4) if dawn else NOW + timedelta(hours=4)
    run = SimpleNamespace(
        config=SimpleNamespace(schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=[],
        output_directory=tmp_path / "processed",
        lighting_schedule=SimpleNamespace(local_midnight=midnight),
        log_command=lambda *a, **k: None,
    )
    return FlatFieldManager(run, sk_data_root=tmp_path / "raw")


def _record(manager, key, *, exposure, counts, when=NOW):
    manager.cameras[key].exposures.append(
        ExposureRecord(timestamp=when, exposure_time=exposure, median_counts=counts,
                       saturation_percentage=0.0, is_good=False)
    )


def _phase(manager):
    return {(k.camera, k.filter) for k in manager._filter_phase_pairs(NOW)}


HUNTSMAN = [_pair("C0", "g"), _pair("C0", "r")] + [_pair("C3", f) for f in ("g", "r", "i", "z", "u")]


def test_one_pair_per_camera_reddest_first_at_dusk(tmp_path):
    m = _manager(tmp_path, HUNTSMAN)
    assert _phase(m) == {("C0", "r"), ("C3", "z")}  # each camera starts on its reddest filter


def test_bluest_first_at_dawn(tmp_path):
    m = _manager(tmp_path, HUNTSMAN, dawn=True)
    assert _phase(m) == {("C0", "g"), ("C3", "u")}


def test_a_camera_moves_on_when_its_filter_is_complete(tmp_path):
    m = _manager(tmp_path, HUNTSMAN)
    m.cameras[FlatSequenceKey("C0", "r")].good_frames = 3  # target reached
    assert ("C0", "g") in _phase(m)


def test_dusk_filter_at_the_cap_but_too_faint_is_exhausted(tmp_path):
    m = _manager(tmp_path, HUNTSMAN)
    _record(m, FlatSequenceKey("C3", "z"), exposure=60.0, counts=1300)  # what happened on 11-10
    assert ("C3", "i") in _phase(m)  # C3 moves on to its next-reddest filter


def test_dusk_filter_short_of_counts_below_the_cap_is_not_exhausted(tmp_path):
    m = _manager(tmp_path, HUNTSMAN)
    _record(m, FlatSequenceKey("C3", "z"), exposure=10.0, counts=1300)  # can still lengthen
    assert ("C3", "z") in _phase(m)


def test_dawn_filter_at_the_minimum_but_too_bright_is_exhausted(tmp_path):
    m = _manager(tmp_path, HUNTSMAN, dawn=True)
    _record(m, FlatSequenceKey("C0", "g"), exposure=0.1, counts=60000)
    assert ("C0", "r") in _phase(m)


def test_one_cameras_hard_filter_never_holds_up_the_others(tmp_path):
    m = _manager(tmp_path, HUNTSMAN)
    # C3 is still trying z (not yet exhausted); C0 is unaffected and on its own phase.
    _record(m, FlatSequenceKey("C3", "z"), exposure=20.0, counts=500)
    assert _phase(m) == {("C0", "r"), ("C3", "z")}


def test_exhaustion_is_judged_on_this_twilight_only(tmp_path):
    m = _manager(tmp_path, HUNTSMAN)
    _record(m, FlatSequenceKey("C3", "z"), exposure=60.0, counts=1300, when=NOW - timedelta(hours=24))
    assert ("C3", "z") in _phase(m)  # yesterday's failure doesn't skip z tonight


def test_no_request_when_every_remaining_pair_is_exhausted(tmp_path):
    m = _manager(tmp_path, [_pair("C3", "z"), _pair("C3", "u")])
    for f in ("z", "u"):
        _record(m, FlatSequenceKey("C3", f), exposure=60.0, counts=500)
    assert m._filter_phase_pairs(NOW) == []
    assert m._planned_request("t") is None


def test_request_holds_exactly_the_phase(tmp_path, monkeypatch):
    m = _manager(tmp_path, HUNTSMAN)
    import burr.task_source.flats.manager as mod

    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is None else NOW.astimezone(tz)

    monkeypatch.setattr(mod, "datetime", Fixed)
    req = m._planned_request("t")
    assert sorted((e["camera"], e["filter"]) for e in req.metadata["flat_plan"]) == [("C0", "r"), ("C3", "z")]


def test_batched_default_is_unchanged(tmp_path):
    m = _manager(tmp_path, HUNTSMAN, sequencing="batched")
    assert len(m._ordered_remaining_pairs(NOW)) == len(HUNTSMAN)


@pytest.mark.parametrize("bad", [
    {"filter_sequencing": "per_filter", "filter_wavelength_nm": None},
    {"filter_wavelength_nm": {"g": 475.0}},  # missing r
])
def test_per_filter_config_is_validated(bad):
    pairs = [_pair("C0", "g"), _pair("C0", "r")]
    plan = {"pairs": pairs, "filter_sequencing": "per_filter", "filter_wavelength_nm": {"g": 475.0, "r": 620.0}, **bad}
    with pytest.raises(ValueError):
        TwilightFlatsConfig(plan=plan)
