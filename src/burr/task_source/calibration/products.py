"""Calibration product publication and restoration."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from astropy.io import fits
from loguru import logger

from burr.task_source.calibration.plan import CalibrationSequencePlan
from burr.task_source.flats.products import (
    SOURCE_HASH_CARD,
    SOURCE_PATH_CARD,
    UnsafeIdentityError,
    publish_product,
    source_digest,
    validate_component,
)

CALIBRATION_SUBDIR = Path("data") / "calibration"
SEQUENCE_CARD = "CALSEQ"
UUID_CARD = "CALUUID"

_PRODUCT_NAME = re.compile(
    r"^(?P<type>dark|bias)_"
    r"(?P<camera>.+)_"
    r"(?P<stamp>\d{8}T\d{9})_"
    r"(?P<sequence>\d{4,})_"
    r"(?P<uuid>[0-9A-Za-z-]{1,8})\.fits$"
)

_FITS_READ_ERRORS = (
    OSError,
    ValueError,
    fits.VerifyError,
)


class CalibrationProductError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class CalibrationProductKey:
    frame_type: str
    camera: str
    filter: str
    readout_mode: str
    bin_x: int
    bin_y: int
    exposure_seconds: float

    def matches(
        self,
        sequence: CalibrationSequencePlan,
    ) -> bool:
        return (
            self.frame_type == sequence.frame_type
            and self.camera == sequence.camera
            and self.filter == sequence.filter
            and math.isclose(
                self.exposure_seconds,
                sequence.exposure_seconds,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        )


@dataclass(frozen=True)
class SourceCalibrationFrame:
    path: Path
    key: CalibrationProductKey
    acquired: datetime
    task_uuid: str
    source_hash: str


@dataclass(frozen=True)
class CalibrationProductRecord:
    path: Path
    key: CalibrationProductKey
    sequence: int
    source_path: str | None
    source_hash: str
    task_uuid: str


@dataclass
class RestoredCalibration:
    products: dict[
        CalibrationProductKey,
        list[CalibrationProductRecord],
    ] = field(default_factory=dict)
    next_sequence: dict[
        CalibrationProductKey,
        int,
    ] = field(default_factory=dict)
    source_paths: set[str] = field(default_factory=set)
    source_hashes: set[str] = field(default_factory=set)
    invalid_products: list[
        tuple[Path, str]
    ] = field(default_factory=list)

    def records_for(
        self,
        sequence: CalibrationSequencePlan,
    ) -> list[CalibrationProductRecord]:
        return [
            record
            for key, records in self.products.items()
            if key.matches(sequence)
            for record in records
        ]


def _text(header: fits.Header, key: str) -> str:
    value = header.get(key)
    return "" if value is None else str(value).strip()


def _image_header(path: Path) -> fits.Header:
    try:
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    return hdu.header.copy()
    except _FITS_READ_ERRORS as error:
        raise CalibrationProductError(
            f"unreadable FITS: {error}"
        ) from error

    raise CalibrationProductError(
        "FITS has no image HDU"
    )


def _positive_integer(
    header: fits.Header,
    card: str,
) -> int:
    try:
        value = int(header[card])
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationProductError(
            f"missing or invalid {card}"
        ) from error

    if value < 1:
        raise CalibrationProductError(
            f"{card} must be positive"
        )

    return value


def _date_obs(header: fits.Header) -> datetime:
    value = _text(header, "DATE-OBS")
    if not value:
        raise CalibrationProductError(
            "missing DATE-OBS"
        )

    try:
        acquired = datetime.fromisoformat(value)
    except ValueError as error:
        raise CalibrationProductError(
            f"invalid DATE-OBS={value!r}"
        ) from error

    if acquired.tzinfo is None:
        acquired = acquired.replace(tzinfo=UTC)

    return acquired.astimezone(UTC)


def key_from_header(
    header: fits.Header,
) -> CalibrationProductKey:
    frame_type = _text(
        header,
        "IMAGETYP",
    ).lower()
    camera = _text(header, "UNITID")
    filter_name = _text(header, "FILTER")
    readout_mode = _text(header, "READOUTM")

    if frame_type not in {"dark", "bias"}:
        raise CalibrationProductError(
            f"invalid IMAGETYP={frame_type!r}"
        )

    try:
        validate_component("camera", camera)
        validate_component("filter", filter_name)
        validate_component(
            "readout mode",
            readout_mode,
        )
    except UnsafeIdentityError as error:
        raise CalibrationProductError(
            str(error)
        ) from error

    try:
        exposure = float(header["EXPTIME"])
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationProductError(
            "missing or invalid EXPTIME"
        ) from error

    if not math.isfinite(exposure):
        raise CalibrationProductError(
            f"invalid EXPTIME={exposure!r}"
        )
    if frame_type == "dark" and exposure <= 0:
        raise CalibrationProductError(
            "DARK exposure must be positive"
        )
    if frame_type == "bias" and exposure < 0:
        raise CalibrationProductError(
            "BIAS exposure cannot be negative"
        )

    return CalibrationProductKey(
        frame_type=frame_type,
        camera=camera,
        filter=filter_name,
        readout_mode=readout_mode,
        bin_x=_positive_integer(
            header,
            "XBINNING",
        ),
        bin_y=_positive_integer(
            header,
            "YBINNING",
        ),
        exposure_seconds=exposure,
    )


def inspect_source(
    path: Path,
) -> SourceCalibrationFrame:
    path = Path(path).resolve()
    header = _image_header(path)
    task_uuid = _text(header, "SKTASKID")

    if not task_uuid:
        raise CalibrationProductError(
            "source frame has no SKTASKID"
        )

    return SourceCalibrationFrame(
        path=path,
        key=key_from_header(header),
        acquired=_date_obs(header),
        task_uuid=task_uuid,
        source_hash=source_digest(path),
    )

def _exposure_component(seconds: float) -> str:
    value = f"{seconds:.9f}".rstrip("0").rstrip(".")
    return "exposure_" + value.replace(".", "p") + "s"

def product_dir(
    output_directory: Path,
    key: CalibrationProductKey,
) -> Path:
    directory = (
        Path(output_directory)
        / CALIBRATION_SUBDIR
        / key.frame_type
        / key.camera
        / key.filter
        / f"readout_{key.readout_mode}"
        / f"bin_{key.bin_x}x{key.bin_y}"
    )

    if key.frame_type == "dark":
        directory /= _exposure_component(
            key.exposure_seconds
        )

    return directory

def product_name(
    source: SourceCalibrationFrame,
    sequence: int,
) -> str:
    acquired = source.acquired
    milliseconds = (
        f"{acquired.microsecond // 1000:03d}"
    )

    return (
        f"{source.key.frame_type}_"
        f"{source.key.camera}_"
        f"{acquired:%Y%m%dT%H%M%S}"
        f"{milliseconds}_"
        f"{sequence:04d}_"
        f"{source.task_uuid[:8]}.fits"
    )

def _cards(
        source: SourceCalibrationFrame,
        sequence: int,
) -> dict[str, tuple]:
    cards = {
        SOURCE_HASH_CARD:(
            source.source_hash,
            "SHA-256 of source frame"
        ),
        SEQUENCE_CARD: (
            sequence,
            "Calibration sequence number"
        ),
        UUID_CARD: (
            source.task_uuid,
            "SensorKIt task UUID"
        ),
    }

    if len(str(source.path)) <= 68:
        cards[SOURCE_PATH_CARD] = (
            str(source.path),
            "Resolved source frame path"
        )
    return cards

def _verify_stamped(
        path: Path,
        source: SourceCalibrationFrame,
        sequence: int,
) -> None:
    header = _image_header(path)

    if key_from_header(header) != source.key:
        raise CalibrationProductError(
            "product identity differs from source"
        )
    if _text(
        header,
        SOURCE_HASH_CARD,
    ) != source.source_hash:
        raise CalibrationProductError(
            f"missing or incorrect {SOURCE_HASH_CARD}"
        )
    if int(
        header.get(SEQUENCE_CARD, -1)
    ) != sequence:
        raise CalibrationProductError(
            f"missing or incorrect {SEQUENCE_CARD}"
        )
    if _text(
        header,
        UUID_CARD,
    ) != source.task_uuid:
        raise CalibrationProductError(
            f"missing or incorrect {UUID_CARD}"
        )

def publish(
        source: SourceCalibrationFrame,
        output_directory: Path,
        sequence: int,
) -> CalibrationProductRecord:
    destination = (
        product_dir(
            output_directory,
            source.key
        )
        / product_name(source, sequence)
    )

    publish_product(
        source.path,
        destination,
        _cards(source,sequence),
        verify=lambda path: _verify_stamped(
            path,
            source,
            sequence
        )
    )

    return CalibrationProductRecord(
        path=destination,
        key=source.key,
        sequence=sequence,
        source_path=str(source.path),
        source_hash=source.source_hash,
        task_uuid=source.task_uuid
    )

def _read_product(
        path: Path,
        output_directory: Path
) -> CalibrationProductRecord:
    match = _PRODUCT_NAME.match(path.name)
    if match is None:
        raise CalibrationProductError(
            "filename does not match product pattern"
        )

    header = _image_header(path)
    key = key_from_header(header)
    sequence = int(match["sequence"])
    source_hash = _text(header, SOURCE_HASH_CARD)
    task_uuid = _text(header, UUID_CARD)

    if not source_hash:
        raise CalibrationProductError(
            f"missing {SOURCE_HASH_CARD}"
        )
    if not task_uuid:
        raise CalibrationProductError(
            f"missing {UUID_CARD}"
        )
    if int(
        header.get(SEQUENCE_CARD, -1)
    ) != sequence:
        raise CalibrationProductError(
            f"{SEQUENCE_CARD} disagrees with filename"
        )
    if path.parent != product_dir(
        output_directory,
        key
    ):
        raise CalibrationProductError(
            "header identity disagrees with directory"
        )
    if (
        match ["type"] != key.frame_type
        or match["camera"] != key.camera
    ):
        raise CalibrationProductError(
            "header identity disagrees with filename"
        )
    expected_name = product_name(
        SourceCalibrationFrame(
            path=path,
            key=key,
            acquired=_date_obs(header),
            task_uuid=task_uuid,
            source_hash=source_hash
        ),
        sequence,
    )
    if path.name != expected_name:
        raise CalibrationProductError(
            "filename timestamp or UUID is invalid"
        )

    return CalibrationProductRecord(
        path = path,
        key=key,
        sequence=sequence,
        source_path=(
            _text(header, SOURCE_PATH_CARD)
            or None
        ),
        source_hash=source_hash,
        task_uuid=task_uuid
    )

def scan_products(
        output_directory: Path,
) -> RestoredCalibration:
    restored = RestoredCalibration()
    root = (Path(output_directory)/ CALIBRATION_SUBDIR)

    if not root.exists():
        return restored

    for path in sorted(root.rglob("*.fits")):
        if path.name.startswith(".publishing-"):
            continue
        try:
            record = _read_product(
                path, Path(output_directory),
            )
        except CalibrationProductError as error:
            restored.invalid_products.append(
                (path, str(error))
            )
            continue
        restored.products.setdefault(
            record.key,
            []
        ).append(record)
        restored.source_hashes.add(
            record.source_hash
        )

        if record.source_path: restored.source_paths.add(
            record.source_path
        )

    for key, records in restored.products.items():
        records.sort(
            key=lambda record: record.sequence
        )
        restored.next_sequence[key] = (
            records[-1].sequence + 1
        )
    return restored

def log_restore_report(
        restored: RestoredCalibration
) -> None:
    for path, reason in restored.invalid_products:
        logger.warning(
            f"Ignoring invalid calibration product"
            f"{path}: {reason}"
        )
