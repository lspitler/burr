"""Twilight flat readback and exposure control (SK mode).

SK's data graph writes each camera's frame as an empty primary HDU plus a
CompImageHDU, named by the controller's FileNameTemplate rather than burr's,
and only after the collect task has returned. These tests build frames in
that shape for several cameras and check the manager finds, measures and
counts them per camera/filter, that SK flat identity headers (IMAGETYP,
UNITID, FILTER) are validated and normalised, and that accepted products are
named, published and restored safely across restarts.
"""

import uuid as uuidlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from burr.core.config import TwilightFlatsConfig
from burr.task_source.flats import products
from burr.task_source.flats.manager import (
    FlatFieldManager,
    FlatSequenceKey,
    load_image_hdu,
)


def _write_sk_frame(
    path,
    level,
    *,
    unit="C0",
    filter_name="Lum",
    exptime=1.0,
    imagetyp="flat",
    date_obs=None,
    task_id=None,
    seed=0,
    omit=(),
):
    """Write a frame the way SK's compress_fits + write_file do.

    ``unit``, ``filter_name`` and ``imagetyp`` set the UNITID, FILTER and
    IMAGETYP cards; passing ``None`` writes the card with an undefined value.
    Card names in ``omit`` are left out of the header entirely. Each frame
    gets a fresh ``SKTASKID`` unless ``task_id`` is given; ``seed`` varies the
    pixels so distinct frames have distinct content.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(level, 50, size=(400, 600)).clip(0, 65520).astype(np.int32)
    cards = {
        "IMAGETYP": imagetyp,
        "EXPTIME": exptime,
        "UNITID": unit,
        "FILTER": filter_name,
        "SKTASKID": task_id or str(uuidlib.uuid4()),
        "DATE-OBS": (date_obs or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%S.%f"),
    }
    header = fits.Header()
    for key, value in cards.items():
        if key not in omit:
            header[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    hdus = [fits.PrimaryHDU(), fits.CompImageHDU(data=data, header=header)]
    fits.HDUList(hdus).writeto(path)


def _manager(
    tmp_path, *, local_midnight, commands=None, sk_data_root="raw", **overrides
):
    settings = {
        "windows": ["nautical_twilight"],
        "n_frames": 2,
        "min_counts": 40000,
        "max_counts": 50000,
        "frame_settle_seconds": 3.0,
    }
    config = TwilightFlatsConfig(**{**settings, **overrides})
    commands = [] if commands is None else commands
    run = SimpleNamespace(
        config=SimpleNamespace(schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=commands,
        output_directory=tmp_path / "processed",
        lighting_schedule=SimpleNamespace(local_midnight=local_midnight),
        log_command=lambda command, result=None, metadata=None, error=None: commands.append(
            {"command": command, "result": result, "metadata": metadata}
        ),
    )
    return FlatFieldManager(
        run, sk_data_root=tmp_path / sk_data_root if sk_data_root else None
    )


def test_load_image_hdu_reads_compressed_hdu(tmp_path):
    path = tmp_path / "frame.fits"
    _write_sk_frame(path, 45000, unit="C0")
    data, header = load_image_hdu(path)
    assert data.shape == (400, 600)
    assert header["UNITID"] == "C0"


def test_readback_counts_each_camera_and_ignores_non_flats(tmp_path):
    dusk_midnight = datetime.now(UTC) + timedelta(hours=4)
    manager = _manager(tmp_path, local_midnight=dusk_midnight)
    request = manager.generate_request(task_id="t1")
    raw = tmp_path / "raw" / "Huntsman"

    # Names follow the controller template, not burr's *twilight_flats* one.
    _write_sk_frame(raw / "C0" / "uuid-a_0.fits", 45000, unit="C0")
    _write_sk_frame(raw / "C1" / "uuid-a_0.fits", 20000, unit="C1")  # too faint
    _write_sk_frame(raw / "C2" / "uuid-x_0.fits", 45000, unit="C2", imagetyp="light")  # otto frame
    manager.on_request_completed(request, duration_seconds=1.0)

    c0, c1 = FlatSequenceKey("C0", "Lum"), FlatSequenceKey("C1", "Lum")
    assert set(manager.cameras) == {c0, c1}
    assert manager.cameras[c0].good_frames == 1
    assert manager.cameras[c1].good_frames == 0
    (product,) = (tmp_path / "processed" / "data" / "flats" / "C0" / "Lum").glob(
        "flat_C0_Lum_*_0001_*.fits"
    )
    assert product.exists()
    assert manager.good_frames == 0  # collection waits for the sequence furthest behind
    assert not manager.is_collection_complete()

    # C1 at 20000 ADU wants ~2.25x; C0 is on target -> shared exposure in between, > 1s at dusk.
    assert manager.cameras[c1].next_exposure > 2.0
    assert 1.0 < manager._shared_exposure() < manager.cameras[c1].next_exposure


def test_sky_factor_shortens_at_dawn_and_lengthens_at_dusk(tmp_path):
    taken = datetime.now(UTC) - timedelta(seconds=180)  # one doubling time ago
    dusk = _manager(tmp_path, local_midnight=taken + timedelta(hours=3))
    dawn = _manager(tmp_path, local_midnight=taken - timedelta(hours=3))
    on_target = dusk.target_counts
    assert np.isclose(dusk._next_exposure_for(1.0, on_target, taken), 2.0, rtol=0.02)
    assert np.isclose(dawn._next_exposure_for(1.0, on_target, taken), 0.5, rtol=0.02)

    # Long gaps within one twilight clip to the limits instead of overflowing.
    long_ago = datetime.now(UTC) - timedelta(hours=2)
    late_dusk = _manager(tmp_path, local_midnight=datetime.now(UTC) + timedelta(hours=1))
    early_dawn = _manager(tmp_path, local_midnight=long_ago - timedelta(hours=1))
    assert late_dusk._next_exposure_for(1.0, on_target, long_ago) == late_dusk.maximum_exposure
    assert early_dawn._next_exposure_for(1.0, on_target, long_ago) == early_dawn.minimum_exposure

    # Last evening's frame seen at dawn restarts the sequence at the minimum.
    evening = datetime.now(UTC) - timedelta(hours=9)
    crossed = _manager(tmp_path, local_midnight=evening + timedelta(hours=4))
    assert crossed._next_exposure_for(20.0, 1000.0, evening) == crossed.minimum_exposure


def test_resume_restores_per_camera_counts(tmp_path):
    midnight = datetime.now(UTC) + timedelta(hours=4)
    first = _manager(tmp_path, local_midnight=midnight)
    request = first.generate_request(task_id="t1")
    _write_sk_frame(tmp_path / "raw" / "C0" / "a_0.fits", 45000, unit="C0")
    first.on_request_completed(request, duration_seconds=1.0)

    resumed = _manager(
        tmp_path, local_midnight=midnight, commands=list(first.run.executed_commands)
    )
    assert resumed.cameras[FlatSequenceKey("C0", "Lum")].good_frames == 1


def test_saturation_autodetect_for_12bit_in_int32(tmp_path):
    manager = _manager(tmp_path, local_midnight=datetime.now(UTC))
    data = np.full((100, 100), 45000, dtype=np.int32)
    data[0, :10] = 65520  # 0.1% saturated at the ASI183's scaled full well
    pct, threshold = manager._calculate_saturation_percentage(data)
    assert threshold < 65520
    assert np.isclose(pct, 0.1)


def _image_header(path):
    """Header of the image HDU (HDU 1 in an SK-format file)."""
    return load_image_hdu(path)[1]


def _saved(manager):
    """flat_field_saved history entries."""
    commands = manager.run.executed_commands
    return [cmd for cmd in commands if cmd["command"] == "flat_field_saved"]


def _processed_fits(tmp_path):
    flats = tmp_path / "processed" / "data" / "flats"
    return sorted(flats.rglob("*.fits")) if flats.exists() else []


def test_sk_flat_metadata_is_normalised_and_preserved(tmp_path):
    manager = _manager(tmp_path, local_midnight=datetime.now(UTC) + timedelta(hours=4))
    source = tmp_path / "raw" / "Huntsman" / "C1" / "uuid-a_0.fits"
    _write_sk_frame(
        source, 45000, unit=" C1 ", filter_name=" Lum ", imagetyp=" flat ", exptime=1.5
    )
    source_bytes = source.read_bytes()

    assert manager.process_exposure(str(source), device_control=None) is True

    (output,) = _processed_fits(tmp_path)
    assert output.parent == tmp_path / "processed" / "data" / "flats" / "C1" / "Lum"
    assert output.name.startswith("flat_C1_Lum_") and "_0001_" in output.name
    header = _image_header(output)
    assert header["IMAGETYP"] == "FLAT"
    assert header["UNITID"] == "C1"
    assert header["FILTER"] == "Lum"
    assert header["EXPTIME"] == 1.5

    assert source.read_bytes() == source_bytes

    (entry,) = _saved(manager)
    assert entry["metadata"]["camera"] == "C1"
    assert entry["metadata"]["filter"] == "Lum"
    assert set(manager.cameras) == {FlatSequenceKey("C1", "Lum")}


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param({"omit": {"UNITID"}}, id="missing-unitid"),
        pytest.param({"unit": None}, id="unitid-none"),
        pytest.param({"omit": {"FILTER"}}, id="missing-filter"),
        pytest.param({"filter_name": None}, id="filter-none"),
        pytest.param({"imagetyp": "light"}, id="imagetyp-light"),
        pytest.param({"omit": {"IMAGETYP"}}, id="missing-imagetyp"),
    ],
)
def test_sk_flat_with_invalid_identity_headers_is_rejected(tmp_path, frame):
    manager = _manager(tmp_path, local_midnight=datetime.now(UTC) + timedelta(hours=4))
    source = tmp_path / "raw" / "Huntsman" / "C1" / "uuid-a_0.fits"
    _write_sk_frame(source, 45000, **frame)

    assert manager.process_exposure(str(source), device_control=None) is False
    assert manager.cameras == {}
    assert _processed_fits(tmp_path) == []
    assert _saved(manager) == []


# ----------------------------------------------------------------------------
# Camera/filter sequences, safe publication and restart
# ----------------------------------------------------------------------------

DUSK = timedelta(hours=4)  # local midnight this far ahead: we are at dusk


def _dusk_manager(tmp_path, **kw):
    return _manager(tmp_path, local_midnight=datetime.now(UTC) + DUSK, **kw)


def _source(tmp_path, camera, filter_name, name, **kw):
    """Write a good SK flat for ``camera``/``filter_name`` and return its path."""
    path = tmp_path / "raw" / "Huntsman" / camera / name
    _write_sk_frame(path, 45000, unit=camera, filter_name=filter_name, **kw)
    return path


def _products(tmp_path, camera, filter_name):
    directory = tmp_path / "processed" / "data" / "flats" / camera / filter_name
    return sorted(directory.glob("flat_*.fits")) if directory.exists() else []


def _sequence(path):
    return int(path.name.rsplit("_", 2)[-2])


PAIRS = [("C0", "Lum"), ("C0", "g"), ("C1", "Lum"), ("C1", "g")]


def test_camera_filter_pairs_count_and_number_independently(tmp_path):
    manager = _dusk_manager(tmp_path, n_frames=5)
    acquired = datetime(2026, 11, 4, 8, 15, 30, 123456, tzinfo=UTC)
    sources = {}
    # Interleave: C0/Lum twice, every other pair once.
    for i, (camera, filt) in enumerate([*PAIRS, ("C0", "Lum")]):
        task = f"{i:08x}-0000-4000-8000-000000000000"
        path = _source(
            tmp_path,
            camera,
            filt,
            f"f{i}.fits",
            seed=i,
            task_id=task,
            date_obs=acquired,
        )
        sources[path] = path.read_bytes()
        assert manager.process_exposure(str(path)) is True

    counts = {key.label(): state.good_frames for key, state in manager.cameras.items()}
    assert counts == {"C0/Lum": 2, "C0/g": 1, "C1/Lum": 1, "C1/g": 1}
    assert [_sequence(p) for p in _products(tmp_path, "C0", "Lum")] == [1, 2]
    for camera, filt in PAIRS[1:]:
        (only,) = _products(tmp_path, camera, filt)
        assert _sequence(only) == 1

    # Name: camera, filter, DATE-OBS to the millisecond, sequence, 8-char task UUID.
    first = _products(tmp_path, "C0", "Lum")[0]
    assert first.name == "flat_C0_Lum_20261104T081530123_0001_00000000.fits"
    header = _image_header(first)
    assert (
        header["IMAGETYP"],
        header["UNITID"],
        header["FILTER"],
        header["EXPTIME"],
    ) == ("FLAT", "C0", "Lum", 1.0)
    assert header["FLATSEQ"] == 1
    assert header["SRCSHA"] == products.source_digest(next(iter(sources)))
    assert all(
        path.read_bytes() == data for path, data in sources.items()
    )  # raw sources untouched

    saved = {
        (e["metadata"]["camera"], e["metadata"]["filter"], e["metadata"]["sequence"])
        for e in _saved(manager)
    }
    assert saved == {
        ("C0", "Lum", 1),
        ("C0", "Lum", 2),
        ("C0", "g", 1),
        ("C1", "Lum", 1),
        ("C1", "g", 1),
    }


def test_restart_restores_interleaved_pairs_and_continues_numbering(tmp_path):
    first = _dusk_manager(tmp_path, n_frames=5)
    for i, (camera, filt) in enumerate(
        [("C0", "Lum"), ("C1", "g"), ("C0", "Lum"), ("C0", "g")]
    ):
        assert first.process_exposure(
            str(_source(tmp_path, camera, filt, f"f{i}.fits", seed=i))
        )

    resumed = _dusk_manager(
        tmp_path, n_frames=5, commands=list(first.run.executed_commands)
    )
    counts = {key.label(): state.good_frames for key, state in resumed.cameras.items()}
    assert counts == {"C0/Lum": 2, "C0/g": 1, "C1/g": 1}
    assert resumed.restore_report.unrecorded_products == []
    assert resumed.restore_report.missing_products == []

    assert resumed.process_exposure(
        str(_source(tmp_path, "C0", "Lum", "f9.fits", seed=9))
    )
    assert [_sequence(p) for p in _products(tmp_path, "C0", "Lum")] == [1, 2, 3]
    assert resumed.cameras[FlatSequenceKey("C0", "Lum")].good_frames == 3


def test_numbering_gap_does_not_create_accepted_frames(tmp_path):
    first = _dusk_manager(tmp_path, n_frames=5)
    for i in range(3):
        assert first.process_exposure(
            str(_source(tmp_path, "C0", "Lum", f"f{i}.fits", seed=i))
        )
    _products(tmp_path, "C0", "Lum")[1].unlink()  # remove #2: a gap at 2

    resumed = _dusk_manager(
        tmp_path, n_frames=5, commands=list(first.run.executed_commands)
    )
    state = resumed.cameras[FlatSequenceKey("C0", "Lum")]
    assert state.good_frames == 2  # verified products, not the highest number
    assert state.next_sequence == 4  # never reuse a number
    (missing,) = resumed.restore_report.missing_products
    assert missing["sequence"] == 2


def test_accepted_product_absent_from_history_is_recovered(tmp_path):
    first = _dusk_manager(tmp_path, n_frames=5)
    for i in range(2):
        assert first.process_exposure(
            str(_source(tmp_path, "C1", "g", f"f{i}.fits", seed=i))
        )

    # Simulate a restart whose history was lost entirely.
    resumed = _dusk_manager(tmp_path, n_frames=5, commands=[])
    assert resumed.cameras[FlatSequenceKey("C1", "g")].good_frames == 2
    assert len(resumed.restore_report.unrecorded_products) == 2


def test_history_entry_with_missing_or_invalid_product_is_reported_not_counted(
    tmp_path,
):
    first = _dusk_manager(tmp_path, n_frames=5)
    for i in range(3):
        assert first.process_exposure(
            str(_source(tmp_path, "C0", "g", f"f{i}.fits", seed=i))
        )
    gone, damaged, _ = _products(tmp_path, "C0", "g")
    gone.unlink()
    damaged.write_bytes(b"not a FITS file")

    resumed = _dusk_manager(
        tmp_path, n_frames=5, commands=list(first.run.executed_commands)
    )
    assert resumed.cameras[FlatSequenceKey("C0", "g")].good_frames == 1
    assert {m["sequence"] for m in resumed.restore_report.missing_products} == {1, 2}
    assert [p for p, _ in resumed.restore_report.invalid_products] == [damaged]
    assert (
        damaged.read_bytes() == b"not a FITS file"
    )  # reported, not repaired or removed


def test_recovery_after_publication_succeeded_but_history_write_was_interrupted(
    tmp_path, monkeypatch
):
    commands = []
    manager = _dusk_manager(tmp_path, n_frames=5, commands=commands)
    source = _source(tmp_path, "C1", "Lum", "f0.fits")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("power lost after publish")

    monkeypatch.setattr(manager.run, "log_command", interrupted)
    with pytest.raises(KeyboardInterrupt):
        manager.process_exposure(str(source))
    (product,) = _products(tmp_path, "C1", "Lum")
    assert commands == []  # history never got the entry

    resumed = _dusk_manager(tmp_path, n_frames=5, commands=commands)
    assert resumed.cameras[FlatSequenceKey("C1", "Lum")].good_frames == 1
    assert [r.path for r in resumed.restore_report.unrecorded_products] == [product]
    # The provenance in the product blocks re-accepting the same source.
    assert resumed.process_exposure(str(source)) is False
    assert _products(tmp_path, "C1", "Lum") == [product]


def test_duplicate_source_is_rejected_before_and_after_restart(tmp_path):
    manager = _dusk_manager(tmp_path, n_frames=5)
    source = _source(tmp_path, "C0", "Lum", "f0.fits")
    assert manager.process_exposure(str(source)) is True
    assert manager.process_exposure(str(source)) is False  # same resolved path
    copy = tmp_path / "raw" / "elsewhere" / "copy.fits"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(source.read_bytes())
    assert manager.process_exposure(str(copy)) is False  # same content, different path

    resumed = _dusk_manager(
        tmp_path, n_frames=5, commands=list(manager.run.executed_commands)
    )
    assert resumed.process_exposure(str(source)) is False
    assert resumed.cameras[FlatSequenceKey("C0", "Lum")].good_frames == 1
    assert len(_products(tmp_path, "C0", "Lum")) == 1
    assert len(_saved(resumed)) == 1


def test_existing_destination_is_never_overwritten(tmp_path):
    manager = _dusk_manager(tmp_path, n_frames=5)
    acquired = datetime(2026, 11, 4, 8, 0, tzinfo=UTC)
    task = "abcdef01-0000-4000-8000-000000000000"
    source = _source(tmp_path, "C0", "Lum", "f0.fits", task_id=task, date_obs=acquired)
    blocker = (
        tmp_path
        / "processed"
        / "data"
        / "flats"
        / "C0"
        / "Lum"
        / "flat_C0_Lum_20260104T080000000_0001_abcdef01.fits"
    )
    blocker = blocker.with_name(
        products.product_name(FlatSequenceKey("C0", "Lum"), acquired, 1, task)
    )
    blocker.parent.mkdir(parents=True)
    blocker.write_bytes(b"someone else's file")

    assert manager.process_exposure(str(source)) is False
    assert blocker.read_bytes() == b"someone else's file"
    assert manager.cameras[FlatSequenceKey("C0", "Lum")].good_frames == 0
    assert _saved(manager) == []
    assert list(blocker.parent.glob(".publishing-*")) == []

    # The taken number is skipped, so the next attempt publishes as #2.
    assert manager.process_exposure(str(source)) is True
    assert sorted(_sequence(p) for p in blocker.parent.glob("flat_*.fits")) == [1, 2]


def test_failed_publication_records_nothing_and_cleans_up(tmp_path, monkeypatch):
    manager = _dusk_manager(tmp_path, n_frames=5)
    keep = _source(tmp_path, "C0", "Lum", "f0.fits", seed=0)
    assert manager.process_exposure(str(keep)) is True
    (existing,) = _products(tmp_path, "C0", "Lum")
    existing_bytes = existing.read_bytes()

    source = _source(tmp_path, "C0", "Lum", "f1.fits", seed=1)
    source_bytes = source.read_bytes()

    def broken_link(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(products.os, "link", broken_link)
    assert manager.process_exposure(str(source)) is False

    state = manager.cameras[FlatSequenceKey("C0", "Lum")]
    assert state.good_frames == 1
    assert len(_saved(manager)) == 1
    assert _products(tmp_path, "C0", "Lum") == [existing]
    assert existing.read_bytes() == existing_bytes
    assert source.read_bytes() == source_bytes
    assert list(existing.parent.glob(".publishing-*")) == []

    # Not marked accepted, so it can be retried once the disk recovers.
    monkeypatch.undo()
    assert manager.process_exposure(str(source)) is True


def test_failed_verification_publishes_nothing(tmp_path, monkeypatch):
    manager = _dusk_manager(tmp_path, n_frames=5)
    source = _source(tmp_path, "C1", "g", "f0.fits")

    def bad_verify(self, path, frame):
        raise ValueError("stamp did not stick")

    monkeypatch.setattr(FlatFieldManager, "_verify_product", bad_verify)
    assert manager.process_exposure(str(source)) is False
    directory = tmp_path / "processed" / "data" / "flats" / "C1" / "g"
    assert list(directory.iterdir()) == []
    assert _saved(manager) == []


@pytest.mark.parametrize(
    "camera, filter_name",
    [
        pytest.param("..", "Lum", id="camera-dotdot"),
        pytest.param(".", "Lum", id="camera-dot"),
        pytest.param("C0/../C1", "Lum", id="camera-separator"),
        pytest.param("C0\\x", "Lum", id="camera-backslash"),
        pytest.param("C0", "a/b", id="filter-separator"),
        pytest.param("C0", "r:", id="filter-colon"),
        pytest.param("C0", "Lum.", id="filter-trailing-dot"),
        pytest.param("CON", "Lum", id="camera-windows-reserved"),
        pytest.param("C0", "g_1", id="filter-underscore"),
    ],
)
def test_unsafe_identity_is_rejected_without_sanitising(tmp_path, camera, filter_name):
    manager = _dusk_manager(tmp_path, n_frames=5)
    path = tmp_path / "raw" / "frame.fits"
    _write_sk_frame(path, 45000, unit=camera, filter_name=filter_name)

    assert manager.process_exposure(str(path)) is False
    assert manager.cameras == {}
    assert _processed_fits(tmp_path) == []
    assert _saved(manager) == []


def test_case_colliding_identities_are_not_combined(tmp_path):
    manager = _dusk_manager(tmp_path, n_frames=5)
    assert (
        manager.process_exposure(str(_source(tmp_path, "C0", "Lum", "f0.fits", seed=0)))
        is True
    )
    # Same directory on a case-insensitive filesystem: refuse rather than merge.
    assert (
        manager.process_exposure(str(_source(tmp_path, "C0", "lum", "f1.fits", seed=1)))
        is False
    )
    assert (
        manager.process_exposure(str(_source(tmp_path, "c0", "Lum", "f2.fits", seed=2)))
        is False
    )
    assert set(manager.cameras) == {FlatSequenceKey("C0", "Lum")}
    assert len(_saved(manager)) == 1


def _legacy_product(tmp_path, number, *, filter_name):
    """A product in the old camera-only layout, data/flats/<camera>/flat_NNN.fits."""
    path = tmp_path / "processed" / "data" / "flats" / "C0" / f"flat_{number:03d}.fits"
    _write_sk_frame(
        path,
        45000,
        unit="C0",
        filter_name=filter_name,
        imagetyp="Flat Field",
        seed=number,
    )
    return path


def _legacy_entry(number, **extra):
    metadata = {
        "camera": "C0",
        "frame_number": number,
        "output_path": f"data/flats/C0/flat_{number:03d}.fits",
    }
    return {
        "command": "flat_field_saved",
        "result": f"{number}/50 flats collected",
        "metadata": {**metadata, **extra},
    }


def test_legacy_camera_only_history_uses_trustworthy_filter_evidence(tmp_path):
    with_header = _legacy_product(tmp_path, 1, filter_name="g")
    recorded = _legacy_product(tmp_path, 2, filter_name=None)
    commands = [_legacy_entry(1), _legacy_entry(2, filter="g")]
    legacy_bytes = {p: p.read_bytes() for p in (with_header, recorded)}

    manager = _dusk_manager(tmp_path, n_frames=5, commands=commands)

    # #1 attributed from its validated FILTER header, #2 from the filter in its history entry.
    assert manager.cameras[FlatSequenceKey("C0", "g")].good_frames == 2
    assert manager.restore_report.ambiguous_legacy == []
    assert {
        p: p.read_bytes() for p in legacy_bytes
    } == legacy_bytes  # not migrated or rewritten
    # New products for the pair go to the new layout and start at #1.
    assert (
        manager.process_exposure(str(_source(tmp_path, "C0", "g", "new.fits", seed=9)))
        is True
    )
    (new,) = _products(tmp_path, "C0", "g")
    assert _sequence(new) == 1
    assert manager.cameras[FlatSequenceKey("C0", "g")].good_frames == 3


def test_legacy_history_without_trustworthy_filter_is_reported_not_assigned(tmp_path):
    no_filter = _legacy_product(tmp_path, 1, filter_name=None)
    _legacy_product(tmp_path, 2, filter_name="g")
    commands = [
        _legacy_entry(1),  # its product has no FILTER header: ambiguous
        _legacy_entry(2, filter="r"),  # history says r, the product says g: conflicting
        {
            "command": "flat_field_saved",
            "result": "3/50 flats collected",
            "metadata": {"camera": "C0"},
        },
    ]
    manager = _dusk_manager(tmp_path, n_frames=5, commands=commands)

    assert manager.cameras == {}  # nothing assigned to an arbitrary filter
    assert len(manager.restore_report.ambiguous_legacy) == 3
    assert no_filter.exists()


def test_direct_mode_keeps_its_fallback_layout_and_no_synthetic_unitid(tmp_path):
    manager = _dusk_manager(tmp_path, n_frames=5, sk_data_root=None)
    saved = []
    frame = tmp_path / "direct" / "frame.tmp.fits"
    _write_sk_frame(frame, 45000, omit={"UNITID", "FILTER", "SKTASKID"})

    class FakeDevice:
        def save_fits(self, output_path, tracking_mode, rates, extra_headers):
            saved.append(extra_headers)
            output_path.write_bytes(frame.read_bytes())

    assert manager.process_exposure(str(frame), device_control=FakeDevice()) is True
    key = FlatSequenceKey("default", None)
    assert set(manager.cameras) == {key}
    assert (tmp_path / "processed" / "data" / "flats" / "flat_001.fits").exists()
    (headers,) = saved
    assert "UNITID" not in headers and "FILTER" not in headers
    (entry,) = _saved(manager)
    assert (entry["metadata"]["camera"], entry["metadata"]["filter"]) == (
        "default",
        None,
    )

    # Direct mode reports camera/filter when the camera does, under its own pair.
    named = tmp_path / "direct" / "named.tmp.fits"
    _write_sk_frame(named, 45000, unit="C3", filter_name="r", omit={"SKTASKID"})
    assert manager.process_exposure(str(named), device_control=FakeDevice()) is True
    assert saved[-1]["UNITID"] == ("C3", "Camera unit identifier")
    assert (
        tmp_path / "processed" / "data" / "flats" / "C3" / "r" / "flat_001.fits"
    ).exists()


def test_sk_frame_without_task_uuid_is_rejected(tmp_path):
    manager = _dusk_manager(tmp_path, n_frames=5)
    path = tmp_path / "raw" / "frame.fits"
    _write_sk_frame(path, 45000, omit={"SKTASKID"})
    assert manager.process_exposure(str(path)) is False
    assert manager.cameras == {}
    assert _processed_fits(tmp_path) == []

    # burr's own TASKID is accepted as the fallback UUID.
    path2 = tmp_path / "raw" / "frame2.fits"
    _write_sk_frame(path2, 45000, omit={"SKTASKID"}, seed=1)
    with fits.open(path2, mode="update") as hdul:
        hdul[1].header["TASKID"] = "fedcba98-0000-4000-8000-000000000000"
    assert manager.process_exposure(str(path2)) is True
    (product,) = _products(tmp_path, "C0", "Lum")
    assert product.name.endswith("_fedcba98.fits")


def test_direct_mode_publication_cannot_overwrite_an_existing_product(tmp_path):
    manager = _dusk_manager(tmp_path, n_frames=5, sk_data_root=None)
    frame = tmp_path / "direct" / "frame.tmp.fits"
    _write_sk_frame(frame, 45000, omit={"UNITID", "FILTER", "SKTASKID"})
    existing = tmp_path / "processed" / "data" / "flats" / "flat_001.fits"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"earlier product")

    class OverwritingDevice:
        """Like DirectDeviceControl.save_fits, which writes with overwrite=True."""

        def save_fits(self, output_path, tracking_mode, rates, extra_headers):
            output_path.write_bytes(frame.read_bytes())

    assert (
        manager.process_exposure(str(frame), device_control=OverwritingDevice())
        is False
    )
    assert existing.read_bytes() == b"earlier product"
    assert _saved(manager) == []
    assert list(existing.parent.glob(".publishing-*")) == []


def test_direct_mode_restores_its_products_after_restart(tmp_path):
    manager = _dusk_manager(tmp_path, n_frames=5, sk_data_root=None)
    frame = tmp_path / "direct" / "frame.tmp.fits"
    _write_sk_frame(frame, 45000, omit={"UNITID", "FILTER", "SKTASKID"})

    class FakeDevice:
        def save_fits(self, output_path, tracking_mode, rates, extra_headers):
            output_path.write_bytes(frame.read_bytes())
            with fits.open(output_path, mode="update") as hdul:
                for card, value in extra_headers.items():
                    hdul[1].header[card] = value

    for _ in range(2):
        assert manager.process_exposure(str(frame), device_control=FakeDevice()) is True

    resumed = _dusk_manager(
        tmp_path,
        n_frames=5,
        sk_data_root=None,
        commands=list(manager.run.executed_commands),
    )
    state = resumed.cameras[FlatSequenceKey("default", None)]
    assert (state.good_frames, state.next_sequence) == (2, 3)
    assert resumed.restore_report.legacy_products == []
    assert resumed.restore_report.missing_products == []
    assert resumed.process_exposure(str(frame), device_control=FakeDevice()) is True
    assert (tmp_path / "processed" / "data" / "flats" / "flat_003.fits").exists()


def test_products_with_inconsistent_identity_or_provenance_are_not_counted(tmp_path):
    first = _dusk_manager(tmp_path, n_frames=5)
    for i in range(4):
        assert first.process_exposure(
            str(_source(tmp_path, "C0", "Lum", f"f{i}.fits", seed=i))
        )
    no_hash, moved, wrong_seq, good = _products(tmp_path, "C0", "Lum")

    with fits.open(no_hash, mode="update") as hdul:
        del hdul[1].header["SRCSHA"]  # no provenance: cannot rule out a duplicate
    elsewhere = tmp_path / "processed" / "data" / "flats" / "C1" / "Lum"
    elsewhere.mkdir(parents=True)
    moved = moved.rename(
        elsewhere / moved.name
    )  # headers say C0/Lum, directory says C1/Lum
    with fits.open(wrong_seq, mode="update") as hdul:
        hdul[1].header["FLATSEQ"] = 99  # contradicts the filename

    resumed = _dusk_manager(
        tmp_path, n_frames=5, commands=list(first.run.executed_commands)
    )
    assert set(resumed.cameras) == {
        FlatSequenceKey("C0", "Lum")
    }  # the moved one adds no C1/Lum
    assert resumed.cameras[FlatSequenceKey("C0", "Lum")].good_frames == 1  # only `good`
    reasons = {path: reason for path, reason in resumed.restore_report.invalid_products}
    assert set(reasons) == {no_hash, moved, wrong_seq}
    assert "SRCSHA" in reasons[no_hash]
    assert "is in" in reasons[moved]
    assert "FLATSEQ" in reasons[wrong_seq]
    assert good.exists() and all(p.exists() for p in reasons)  # nothing removed
