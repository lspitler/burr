"""Accepted twilight-flat products: identity, naming, publication and restore.

A flat sequence is keyed by ``FlatSequenceKey(camera, filter)``; every pair
counts, numbers and restores independently. Accepted SensorKit products are
published as::

    data/flats/<camera>/<filter>/
        flat_<camera>_<filter>_<YYYYMMDDTHHMMSSfff>_<NNNN>_<uuid8>.fits

where the timestamp is the frame's DATE-OBS (UTC), ``NNNN`` is that pair's
next free sequence number and ``uuid8`` is the first eight characters of the
frame's SensorKit task ID (``SKTASKID``), or burr's ``TASKID`` if absent.
Uniqueness comes from the sequence number plus no-overwrite publication; the
UUID prefix is informative only (UUIDv7 prefixes are timestamps and repeat).

Each product carries its own provenance (``SRCPATH``, ``SRCSHA``,
``FLATSEQ``, ``FLATUUID``) so a restart can rebuild state from the products
themselves and recover a product whose ``flat_field_saved`` history entry was
never written. See ``docs/flats.md`` for the restart and legacy rules.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from astropy.io import fits
from loguru import logger

# Camera key used when a frame carries no unit identity (direct mode drives a
# single camera). Never written into a product's UNITID.
DEFAULT_CAMERA = "default"

FLATS_SUBDIR = Path("data") / "flats"

# Header cards burr stamps on an accepted product to record its provenance.
SOURCE_PATH_CARD = "SRCPATH"
SOURCE_HASH_CARD = "SRCSHA"
SEQUENCE_CARD = "FLATSEQ"
UUID_CARD = "FLATUUID"

_DIRECT_NAME = re.compile(r"^flat_(?P<seq>\d+)\.fits$")
_PRODUCT_NAME = re.compile(
    r"^flat_(?P<camera>.+)_(?P<filter>.+)_(?P<stamp>\d{8}T\d{9})_(?P<seq>\d{4,})_(?P<uuid>[0-9A-Za-z-]{1,8})\.fits$"
)
# Characters that are invalid in a path component on Linux, macOS or Windows,
# plus ASCII control characters. "_" is excluded too: it is the filename's
# field separator, so a camera or filter containing it would be ambiguous.
_INVALID_CHARS = frozenset('<>:"/\\|?*_') | {chr(c) for c in range(32)}
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{n}" for n in range(1, 10)),
        *(f"LPT{n}" for n in range(1, 10)),
    }
)


# What astropy raises for an unreadable or damaged FITS file.
_FITS_READ_ERRORS = (OSError, ValueError, fits.VerifyError)


class UnsafeIdentityError(ValueError):
    """A camera or filter name cannot be used verbatim as a path component."""


class ProductInvalidError(ValueError):
    """An accepted-product file is unreadable or inconsistent with its location."""


@dataclass(frozen=True, order=True)
class FlatSequenceKey:
    """Independent flat sequence: one camera through one filter.

    ``filter`` is ``None`` only in direct mode, where the camera may not
    report a filter; SensorKit frames always carry one.
    """

    camera: str
    filter: str | None

    def label(self) -> str:
        """``camera/filter`` for logs, e.g. ``C1/Lum`` (``C1/-`` with no filter)."""
        return f"{self.camera}/{self.filter or '-'}"

    def folded(self) -> tuple[str, str | None]:
        """Case-folded identity, for detecting case-insensitive collisions."""
        return self.camera.casefold(), self.filter.casefold() if self.filter else None


def validate_component(kind: str, value: str) -> str:
    """Return ``value`` unchanged if it is safe as a single path component.

    Names are never sanitised: a name that would need changing is rejected,
    so two different identities can never be silently written to one place.

    Raises:
        UnsafeIdentityError: ``value`` is empty, ``.``/``..``, contains a path
            separator, ``_`` or a platform-invalid character, ends in a space
            or dot, or is a Windows reserved device name.
    """
    if not value or value in {".", ".."}:
        raise UnsafeIdentityError(f"{kind} {value!r} is not a usable path component")
    bad = sorted({ch for ch in value if ch in _INVALID_CHARS})
    if bad:
        raise UnsafeIdentityError(
            f"{kind} {value!r} contains unsafe character(s) {bad!r}"
        )
    if value[-1] in " .":
        raise UnsafeIdentityError(f"{kind} {value!r} ends with a space or dot")
    if value.split(".")[0].upper() in _WINDOWS_RESERVED:
        raise UnsafeIdentityError(f"{kind} {value!r} is a reserved device name")
    return value


def validate_key(key: FlatSequenceKey) -> FlatSequenceKey:
    """Validate both components of ``key`` for use in product paths."""
    validate_component("camera", key.camera)
    if key.filter is not None:
        validate_component("filter", key.filter)
    return key


def product_dir(output_directory: Path, key: FlatSequenceKey) -> Path:
    """Directory for a sequence's accepted products.

    Direct mode keeps its historical layout: no camera directory for the
    default camera, and no filter directory when no filter is known.
    """
    directory = Path(output_directory) / FLATS_SUBDIR
    if key.camera != DEFAULT_CAMERA:
        directory = directory / key.camera
    if key.filter is not None:
        directory = directory / key.filter
    return directory


def product_name(
    key: FlatSequenceKey, acquired: datetime, sequence: int, uuid: str
) -> str:
    """Filename for an accepted product (see the module docstring)."""
    stamp = acquired.astimezone(UTC)
    millis = f"{stamp.microsecond // 1000:03d}"
    return (
        f"flat_{key.camera}_{key.filter or 'none'}_{stamp:%Y%m%dT%H%M%S}{millis}_"
        f"{sequence:04d}_{uuid_prefix(uuid)}.fits"
    )


def uuid_prefix(uuid: str) -> str:
    """First eight characters of a task UUID, as used in product names."""
    return uuid.strip()[:8]


def source_digest(path: Path) -> str:
    """SHA-256 of a source frame's bytes; identifies it independently of its path."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp_image_header(path: Path, cards: Mapping[str, tuple]) -> None:
    """Write ``cards`` into the image HDU of the FITS at ``path``, in place."""
    with fits.open(path, mode="update") as hdul:
        image_hdus = [hdu for hdu in hdul if hdu.data is not None]
        header = (image_hdus[0] if image_hdus else hdul[0]).header
        for key, value in cards.items():
            header[key] = value


def publish_product(
    source: Path | None,
    destination: Path,
    cards: Mapping[str, tuple],
    verify,
    *,
    write=None,
) -> None:
    """Write a product to a temporary file, check it, and publish it at ``destination``.

    The temporary file is created in the destination directory and filled
    either by copying ``source`` and stamping ``cards`` into it, or, when
    ``write`` is given, by ``write(temp_path)`` (direct mode's ``save_fits``,
    which stamps its own cards). It is checked by ``verify(temp_path)`` and
    then published with a hard link. ``os.link`` fails atomically if
    ``destination`` already exists, so an existing product is never
    overwritten (an exists-check followed by a write or rename could race).
    The temporary file is always removed; ``source`` is only ever read.

    Raises:
        FileExistsError: ``destination`` already exists.
        OSError: The copy, stamp or link failed.
        Exception: Whatever ``verify`` raises for an invalid product.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=".publishing-", suffix=".fits.tmp", dir=destination.parent
    )
    temp = Path(temp_name)
    try:
        if write is None:
            with os.fdopen(handle, "wb") as out, Path(source).open("rb") as src:
                for block in iter(lambda: src.read(1 << 20), b""):
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            _stamp_image_header(temp, cards)
        else:
            os.close(handle)
            write(temp)
        verify(temp)
        os.link(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


@dataclass(frozen=True)
class ProductRecord:
    """A verified accepted product found on disk."""

    path: Path
    key: FlatSequenceKey
    sequence: int
    source_path: str | None
    source_hash: str | None


@dataclass
class RestoredFlats:
    """Result of reconciling accepted products with run history."""

    products: dict[FlatSequenceKey, list[ProductRecord]] = field(default_factory=dict)
    next_sequence: dict[FlatSequenceKey, int] = field(default_factory=dict)
    source_paths: set[str] = field(default_factory=set)
    source_hashes: set[str] = field(default_factory=set)
    # Reported, never counted:
    invalid_products: list[tuple[Path, str]] = field(default_factory=list)
    missing_products: list[dict] = field(default_factory=list)
    unrecorded_products: list[ProductRecord] = field(default_factory=list)
    ambiguous_legacy: list[dict] = field(default_factory=list)
    legacy_products: list[Path] = field(default_factory=list)
    # Legacy products (pre-camera/filter layout) whose own validated headers
    # name a camera and filter. Counted, but never renamed or rewritten.
    legacy_counts: dict[FlatSequenceKey, int] = field(default_factory=dict)

    def accepted_count(self, key: FlatSequenceKey) -> int:
        """Verified accepted products for ``key``: new-layout plus attributable legacy."""
        return len(self.products.get(key, ())) + self.legacy_counts.get(key, 0)


def _read_header(path: Path) -> dict:
    """Identity and provenance cards from a product's image header.

    Raises:
        ProductInvalidError: The file cannot be read as FITS.
    """
    try:
        with fits.open(path, memmap=False) as hdul:
            headers = [hdu.header for hdu in hdul if hdu.data is not None]
            header = headers[0] if headers else hdul[0].header
            return {
                k: header.get(k)
                for k in (
                    "IMAGETYP",
                    "UNITID",
                    "FILTER",
                    SEQUENCE_CARD,
                    SOURCE_PATH_CARD,
                    SOURCE_HASH_CARD,
                )
            }
    except _FITS_READ_ERRORS as e:
        raise ProductInvalidError(f"unreadable FITS: {e}") from e


def _card_text(cards: Mapping, key: str) -> str:
    """Stripped text of a header card; a missing or valueless card is ``""``."""
    value = cards.get(key)
    return "" if value is None else str(value).strip()


def _located_key(
    path: Path, flats_root: Path, cards: Mapping, image_types: set[str]
) -> FlatSequenceKey:
    """Sequence key a product's headers name, checked against its directory.

    Raises:
        ProductInvalidError: IMAGETYP is not one of ``image_types``, or the
            headers name a different camera/filter than the file's directory.
    """
    if _card_text(cards, "IMAGETYP").upper() not in image_types:
        raise ProductInvalidError(f"IMAGETYP is {cards.get('IMAGETYP')!r}, not FLAT")
    key = FlatSequenceKey(
        _card_text(cards, "UNITID") or DEFAULT_CAMERA,
        _card_text(cards, "FILTER") or None,
    )
    if path.parent != product_dir(flats_root.parent.parent, key):
        raise ProductInvalidError(
            f"headers say {key.label()} but the file is in {path.parent}"
        )
    return key


def _read_direct_product(path: Path, flats_root: Path) -> ProductRecord:
    """Validate a direct-mode product (``flat_NNN.fits``) and return its record.

    Direct mode keeps its historical names and carries no source provenance.

    Raises:
        ProductInvalidError: Unreadable, not a flat, or in the wrong directory.
    """
    key = _located_key(path, flats_root, _read_header(path), {"FLAT", "FLAT FIELD"})
    return ProductRecord(
        path, key, int(_DIRECT_NAME.match(path.name)["seq"]), None, None
    )


def _read_product(path: Path, flats_root: Path, *, sensorkit: bool) -> ProductRecord:
    """Validate an accepted product and return its record.

    Raises:
        ProductInvalidError: Unreadable, not a flat, missing identity or
            provenance, or its headers disagree with its directory or name.
    """
    if not sensorkit and _DIRECT_NAME.match(path.name):
        return _read_direct_product(path, flats_root)
    match = _PRODUCT_NAME.match(path.name)
    if match is None:
        raise ProductInvalidError(
            "filename does not match the accepted-product pattern"
        )
    cards = _read_header(path)
    key = _located_key(path, flats_root, cards, {"FLAT"})
    if sensorkit and (key.camera == DEFAULT_CAMERA or key.filter is None):
        raise ProductInvalidError("product is missing UNITID or FILTER")
    if (match["camera"], match["filter"]) != (key.camera, key.filter or "none"):
        raise ProductInvalidError(
            f"headers say {key.label()} but the filename does not"
        )
    sequence = int(match["seq"])
    if cards[SEQUENCE_CARD] is not None and int(cards[SEQUENCE_CARD]) != sequence:
        raise ProductInvalidError(
            f"{SEQUENCE_CARD}={cards[SEQUENCE_CARD]} disagrees with filename sequence {sequence}"
        )
    source_hash = _card_text(cards, SOURCE_HASH_CARD)
    if not source_hash:
        raise ProductInvalidError(f"missing {SOURCE_HASH_CARD} provenance")
    return ProductRecord(
        path, key, sequence, _card_text(cards, SOURCE_PATH_CARD) or None, source_hash
    )


def _legacy_key(path: Path) -> FlatSequenceKey | None:
    """Camera/filter a legacy product's own headers support, or None.

    A legacy file is attributable only if it is a readable FLAT whose image
    header names both UNITID and FILTER; anything less is ambiguous.
    """
    try:
        cards = _read_header(path)
    except ProductInvalidError as e:
        logger.debug(f"Legacy flat {path} is unreadable: {e}")
        return None
    camera, filter_name = _card_text(cards, "UNITID"), _card_text(cards, "FILTER")
    if (
        _card_text(cards, "IMAGETYP").upper() not in {"FLAT", "FLAT FIELD"}
        or not camera
        or not filter_name
    ):
        return None
    return FlatSequenceKey(camera, filter_name)


def scan_products(output_directory: Path, *, sensorkit: bool) -> RestoredFlats:
    """Find and verify accepted products under ``data/flats``.

    In SensorKit mode, files named like the pre-camera/filter layout
    (``flat_NNN.fits``) are legacy: listed and left untouched, and counted
    only through ``reconcile_history``'s legacy rules. In direct mode that
    name is still the current one, so those files are verified products.
    Temporary ``.publishing-*`` files from an interrupted publication are
    ignored (they were never published).
    """
    restored = RestoredFlats()
    flats_root = Path(output_directory) / FLATS_SUBDIR
    if not flats_root.exists():
        return restored
    for path in sorted(flats_root.rglob("*.fits")):
        if path.name.startswith(".publishing-"):
            continue
        if sensorkit and _DIRECT_NAME.match(path.name):
            restored.legacy_products.append(path)
            continue
        try:
            record = _read_product(path, flats_root, sensorkit=sensorkit)
        except ProductInvalidError as e:
            restored.invalid_products.append((path, str(e)))
            continue
        restored.products.setdefault(record.key, []).append(record)
        if record.source_path:
            restored.source_paths.add(record.source_path)
        if record.source_hash:
            restored.source_hashes.add(record.source_hash)
    for key, records in restored.products.items():
        records.sort(key=lambda r: r.sequence)
        restored.next_sequence[key] = records[-1].sequence + 1
    return restored


def _history_key(metadata: Mapping) -> FlatSequenceKey | None:
    """Sequence key recorded in a history entry, or None if it has no filter."""
    camera = str(metadata.get("camera") or DEFAULT_CAMERA).strip()
    filter_name = metadata.get("filter")
    filter_name = str(filter_name).strip() if filter_name is not None else ""
    return FlatSequenceKey(camera, filter_name) if filter_name else None


def reconcile_history(
    restored: RestoredFlats, output_directory: Path, history: Iterable[Mapping]
) -> RestoredFlats:
    """Compare ``flat_field_saved`` history with the verified products.

    Products are the source of truth for counts; history never adds to them.

    - An entry whose new-layout product verified is matched to it.
    - A verified product with no matching entry (published, then interrupted
      before history was written) is listed in ``unrecorded_products``. It is
      a valid accepted product, so it counts.
    - An entry whose product is missing or failed verification is listed in
      ``missing_products`` and does not count.
    - A legacy camera-only entry (no ``filter``) is attributed only from a
      trustworthy source: a filter recorded with it, or the validated
      UNITID/FILTER headers of the legacy product it names. Its product is
      counted in ``legacy_counts`` and left in place. Otherwise the entry is
      listed in ``ambiguous_legacy`` and not assigned to any filter.
    """
    reconciler = _Reconciler(restored, Path(output_directory))
    for cmd in history:
        if cmd.get("command") == "flat_field_saved":
            reconciler.entry(cmd.get("metadata") or {})
    restored.unrecorded_products = sorted(
        (
            r
            for path, r in reconciler.by_path.items()
            if path not in reconciler.recorded
        ),
        key=lambda r: (r.key, r.sequence),
    )
    return restored


class _Reconciler:
    """Classifies ``flat_field_saved`` entries against verified products."""

    def __init__(self, restored: RestoredFlats, output_directory: Path):
        self.restored = restored
        self.output_directory = output_directory
        self.by_path = {
            r.path.resolve(): r
            for records in restored.products.values()
            for r in records
        }
        self.legacy = {p.resolve() for p in restored.legacy_products}
        self.recorded: set[Path] = set()
        self.attributed: set[Path] = set()

    def entry(self, metadata: Mapping) -> None:
        """Match, attribute or report one history entry (see ``reconcile_history``)."""
        output = metadata.get("output_path")
        product = (self.output_directory / output).resolve() if output else None
        if product in self.by_path:
            self.recorded.add(product)
        elif product in self.legacy:
            if product not in self.attributed:
                self.attributed.add(product)
                _attribute_legacy(self.restored, product, metadata)
        elif product is None and _history_key(metadata) is None:
            self.restored.ambiguous_legacy.append(dict(metadata))
        else:
            self.restored.missing_products.append(dict(metadata))


def _attribute_legacy(
    restored: RestoredFlats, product: Path, metadata: Mapping
) -> None:
    """Count a legacy product under the camera/filter its evidence supports."""
    recorded = _history_key(metadata)
    from_header = _legacy_key(product)
    key = recorded or from_header
    if key is None or (recorded and from_header and recorded != from_header):
        restored.ambiguous_legacy.append(dict(metadata))
        return
    restored.legacy_counts[key] = restored.legacy_counts.get(key, 0) + 1


def find_case_collision(
    key: FlatSequenceKey, known: Iterable[FlatSequenceKey]
) -> FlatSequenceKey | None:
    """A different known key that ``key`` would share a directory with on a case-insensitive filesystem."""
    folded = key.folded()
    return next(
        (other for other in known if other != key and other.folded() == folded), None
    )


def log_restore_report(restored: RestoredFlats) -> None:
    """Log every product or history entry that restore did not count."""
    for path, reason in restored.invalid_products:
        logger.warning(f"Ignoring invalid accepted flat {path}: {reason}")
    for metadata in restored.missing_products:
        logger.warning(
            f"History records accepted flat {metadata.get('output_path')!r} "
            f"({metadata.get('camera')}/{metadata.get('filter')}) but it is missing or invalid; not counted"
        )
    for record in restored.unrecorded_products:
        logger.warning(
            f"Recovered accepted flat with no history entry (interrupted before history was written): {record.path}"
        )
    for metadata in restored.ambiguous_legacy:
        logger.warning(
            f"Legacy camera-only flat history for camera {metadata.get('camera')!r} "
            f"({metadata.get('output_path')!r}) has no trustworthy filter; not assigned to any filter"
        )
    for path in restored.legacy_products:
        logger.info(
            f"Leaving legacy flat product in place (not counted, not migrated): {path}"
        )
