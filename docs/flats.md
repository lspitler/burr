# Twilight flats: products, restart and recovery

How `FlatFieldManager` (`src/burr/task_source/flats/`) names accepted flats,
publishes them safely, and rebuilds its state when a run restarts. The code
is in `manager.py` (measuring and accepting frames) and `products.py`
(identity, naming, publication and restore).

## Sequences

State is kept per **camera/filter pair**, `FlatSequenceKey(camera, filter)`,
taken from the frame's `UNITID` and `FILTER` headers. Each pair has its own
accepted count, exposure history, ideal next exposure and filename numbering.
`C0/Lum`, `C0/g`, `C1/Lum` and `C1/g` never share a count or a number.

`n_frames` is the target for **each** pair. Collection is complete when every
pair seen so far has `n_frames` accepted products (see Limitations).

## Accepted SensorKit products

```
data/flats/<camera>/<filter>/flat_<camera>_<filter>_<acquired>_<NNNN>_<uuid8>.fits
```

| Field      | Source                                                                  |
| ---------- | ----------------------------------------------------------------------- |
| `camera`   | `UNITID`, whitespace-trimmed                                            |
| `filter`   | `FILTER`, whitespace-trimmed                                            |
| `acquired` | `DATE-OBS` in UTC, `YYYYMMDDTHHMMSSmmm` (milliseconds)                  |
| `NNNN`     | that pair's next free sequence number, from `0001`                      |
| `uuid8`    | first 8 characters of `SKTASKID` (SensorKit task), else burr's `TASKID` |

Example: `data/flats/C1/Lum/flat_C1_Lum_20261104T081530123_0001_01a1a7cb.fits`.

The sequence number together with no-overwrite publication is what makes a
name unique. The UUID prefix is for traceability only: SensorKit task IDs
are UUIDv7, whose first 8 characters are a timestamp, so they repeat for
tasks started close together.

A SensorKit flat is rejected (not counted, nothing written) if `IMAGETYP` is
not `flat`, `UNITID` or `FILTER` is missing or has no value, or it has
neither `SKTASKID` nor `TASKID`.

### Headers

Products keep every card from the raw SensorKit frame, including the actual
`EXPTIME`. On top of that, burr stamps:

- the normalised identity: `IMAGETYP = 'FLAT'`, `UNITID`, `FILTER`;
- statistics: `MEDIAN`, `MEAN`, `STDDEV`, `MINCNTS`, `MAXCNTS`, `SATPCT`, `SATTHRES`;
- the pair's sequence number: `FLATSEQ` (and `FLATSNUM`);
- provenance, used by restart and duplicate checks:
  - `SRCSHA`: SHA-256 of the raw source file;
  - `SRCPATH`: its resolved path, omitted if longer than a FITS card allows;
  - `FLATUUID`: the full task UUID.

Raw source frames are only ever read, never modified.

### Identity safety

Camera and filter names are used verbatim as directory and filename
components and are **never sanitised**. A name that would need changing is
rejected instead, so two different identities can never end up sharing an
output. Rejected:

- empty, `.` or `..`;
- path separators (`/`, `\`) and `<>:"|?*`;
- control characters, or a trailing space or dot;
- Windows reserved names (`CON`, `NUL`, `COM1`, …);
- `_`, the filename's field separator.

Two identities that differ only in case (`C0/Lum` and `C0/lum`) would share
a directory on a case-insensitive filesystem. Once one is known, frames for
the other are rejected rather than merged.

## Safe publication

1. The product is written to a temporary `.publishing-*.fits.tmp` file in its
   destination directory: the source is copied and burr's cards stamped in
   (SensorKit mode), or `save_fits` writes it (direct mode).
2. It is re-read and checked: identity and source hash (SensorKit mode).
3. It is published with `os.link`. This fails atomically if the destination
   already exists, so an existing file is never overwritten. A check for
   existence followed by a write or rename could race, so none is used.
4. The temporary file is always removed.

Only after a successful step 3 does burr increase the pair's count, mark the
source as accepted and write `flat_field_saved` to history. If any step
fails, the frame is not kept: the count, history, existing products and the
raw source are all left exactly as they were. A frame that failed to publish
can be retried.

If the destination name is already taken, that sequence number is skipped,
so the next accepted frame gets the following number.

### Duplicate sources

A source frame is accepted at most once. It is recognised by its resolved
path **and** by its content hash, so a copy of an accepted frame under
another name is also refused. Both are rebuilt from the products' own
provenance cards on restart, so a frame accepted before a restart is refused
after it. This protects against reprocessing a source file. Matching frames
to specific SensorKit task IDs is a later milestone.

## Restart

On construction the manager rebuilds every pair from **verified accepted
products on disk**, then reconciles them with run history. Products are the
source of truth for counts; history never adds to a count.

A product counts if it is a readable FITS with `IMAGETYP = FLAT`, its
`UNITID`/`FILTER` headers match its directory and filename, its `FLATSEQ`
matches its filename, and it carries `SRCSHA`. The pair's next sequence
number is one past the highest verified number. It is kept separately from
the count, so a gap (say #2 was removed) never implies an accepted frame,
and numbers are never reused.

Reconciliation reports, in `manager.restore_report` and in the log:

| Situation                                                              | Report                | Counted?               |
| ---------------------------------------------------------------------- | --------------------- | ---------------------- |
| History entry whose product verified                                   | (matched)             | yes                    |
| Verified product with no history entry                                 | `unrecorded_products` | **yes**                |
| History entry whose product is missing                                 | `missing_products`    | no                     |
| Product that fails verification                                        | `invalid_products`    | no; the file is kept   |
| Leftover `.publishing-*` temporary file                                | ignored               | no                     |

### Interrupted publication

If the process stops after step 3 but before `flat_field_saved` is written,
the product is on disk with no history entry. On restart it verifies, is
listed in `unrecorded_products` and **counts**: it is a valid accepted flat.
Its `SRCSHA`/`SRCPATH` cards are restored into the duplicate checks, so the
same source frame is not accepted a second time.

## Legacy camera-only history

Earlier versions wrote SensorKit flats as `data/flats/<camera>/flat_NNN.fits`
and recorded `flat_field_saved` with a camera but **no filter**. In SensorKit
mode those files are treated as legacy:

- They are **left in place**: never moved, renamed, rewritten or deleted.
  They are not part of the new numbering.
- A legacy history entry is attributed to a camera/filter pair only on
  trustworthy evidence:
  - a `filter` recorded in the entry itself; or
  - the validated `UNITID`/`FILTER` headers of the legacy product the entry
    names.

  If both are present and disagree, it is not attributed.
- An attributed legacy product counts toward its pair's total. A new product
  for that pair still starts at sequence `0001` in the new layout.
- An entry with no trustworthy filter, conflicting evidence, or no product
  path is reported in `ambiguous_legacy` and **not assigned to any filter**,
  not even the current one.

## Direct mode

Direct (ASCOM) mode keeps its existing behaviour:

- A frame without `UNITID` uses the `default` camera, and `FILTER` may be
  absent. A synthetic `UNITID = default` is never written into a product.
- Products keep the historical names: `data/flats/flat_NNN.fits` for the
  default camera, or `data/flats/<camera>/<filter>/flat_NNN.fits` when the
  camera reports its identity.
- `save_fits` writes the product into the temporary file. Publication is the
  same no-overwrite link as SensorKit mode, so an existing product is never
  overwritten, even though `save_fits` itself writes with `overwrite=True`.
- On restart, verified `flat_NNN.fits` products are counted (they are
  current, not legacy, in direct mode).
- Direct mode has no source-file duplicate check: the executor reuses one
  temporary path per exposure, and its frames carry no task UUID.

## Limitations

These are known and left for later milestones:

- **One exposure per request.** Each request uses a single exposure: the
  median of the pairs' ideal exposures. Pairs whose filter or optics need a
  very different exposure are rejected more often.
- **One camera per request in SensorKit.** Current SensorKit gives a request
  with one `CameraParameterSet` to a single camera, so pairs accumulate only
  as cameras happen to be chosen. All-camera collection is **not** solved
  here.
- **Completion covers only pairs seen so far.** There is no configured
  expected camera/filter matrix, so collection can report complete while
  pairs that never produced a frame have none.
- **No requested filters.** burr does not choose filters; it records the
  filter each camera reports.
- **Source-file duplicate protection only.** Frames are not yet matched to
  the exact SensorKit task that produced them; discovery still selects
  candidates by modification time and `IMAGETYP`.

Recommended next milestone: explicit requested filters and a configured
expected camera/filter matrix. Then emit one `CameraParameterSet` per pair
with its own exposure, and associate each frame with its SensorKit task ID.
That closes the shared-exposure, completion and discovery gaps together.
