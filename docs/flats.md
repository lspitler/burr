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

## Planned collection (SensorKit)

`twilight_flats.plan` lists every camera/filter pair that needs flats, the
SensorKit instrument that takes it, and how many accepted frames it needs:

```yaml
twilight_flats:
  windows: [nautical_twilight]
  plan:
    frames_per_pair: 10            # target for each pair unless overridden
    pairs:
      - {camera: C0, instrument: C0-ASI183MM, filter: Lum}
      - {camera: C1, instrument: C1-ASI183MM, filter: Lum, frames: 5}
```

`deploy/example/huntsman-flats-lum.yaml` is a complete example covering all
nine Huntsman cameras with Lum. It is an example only; nothing loads it.

| Field        | Meaning                                                                 |
| ------------ | ----------------------------------------------------------------------- |
| `camera`     | the `UNITID` the camera's data graph writes into each frame             |
| `instrument` | the SensorKit camera device key that takes this pair                    |
| `filter`     | the filter name **exactly** as that camera's wheel reports it (case-sensitive) |
| `frames`     | accepted frames this pair needs; defaults to `frames_per_pair`          |

### Validation

The plan is rejected at config load if:

- it has no pairs;
- a camera/filter pair is listed twice, including twice with only a
  difference in case (`C0/Lum` and `c0/lum`);
- a camera or filter name is unsafe as a path component (the rules under
  Identity safety), or has leading or trailing whitespace;
- an instrument is not a single device-key token;
- one camera maps to two instruments, or one instrument to two cameras;
- `frames` or `frames_per_pair` is below 1;
- an entry has an unknown key (typos are errors, not ignored).

A plan needs SensorKit mode. A manager given a plan without a
`raw_data_root` refuses to start.

### Requests and routing

Every planned pair is tracked from the start, at `0/N`, including pairs
whose camera never returns an image.

Each request carries one exposure for every pair still short of its
target:

- that pair's own ideal exposure: its last frame re-projected to now, or
  the minimum for a pair with no frames yet;
- its filter;
- its instrument.

The SensorKit burr program turns these into one `StandardCollectTask` with
one `CameraParameterSet` per pair. Each set sets `instrument`, which
SensorKit's planner honours by assigning the exposure **only** to that
camera, and `filter_name`, which goes to that camera's wheel. The cameras
expose together.

If a pinned camera cannot take its filter (its wheel reports a filter list
without it), SensorKit fails the whole collect rather than moving the
exposure to another camera. burr then logs the request as failed, naming
every pair in it and SensorKit's reason. Nothing is counted.

A global `collect.filter_name` does not override a pair's filter. Global
gain, binning and readout mode still apply.

### Matching frames to requests

When SensorKit accepts the task, burr records the task ID the controller
mints for it; cameras write that ID into every frame as `SKTASKID`. After
the collect, burr reads back only frames whose `SKTASKID` is exactly that
ID, and assigns each to the planned pair named by its `UNITID` and
`FILTER`. Modification time and file name play no part.

| Frame                                                   | Result                            |
| ------------------------------------------------------- | --------------------------------- |
| this task, a requested pair, first frame for it         | measured, kept if good            |
| this task, a camera or filter not requested             | reported, not counted             |
| this task, a second frame for one pair                  | reported, not counted             |
| another task                                            | ignored                           |
| a requested pair with no frame within `frame_settle_seconds` | that pair is reported as returning no frame |

If no task ID was recorded for a request, its frames cannot be matched
exactly, and none are counted.

### Completion and unexpected identities

Collection is complete only when **every planned pair** has reached its
target. A frame for a camera or filter outside the plan is never accepted,
never creates state, and never counts. Accepted products found on disk for
pairs outside the plan (from an earlier, different plan) are reported at
start-up, left in place, and not counted.

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

### Dawn start and dusk end by Sun altitude

`dawn_start_sun_altitude_deg` (e.g. `-9.0`) opens the dawn flats window only
once the Sun has risen to that altitude; `dusk_end_sun_altitude_deg` closes
the dusk window when it sets to it. Both are optional (default: the full
configured `windows`). On 11-16 dawn, 81 of 83 rejected frames came before
Sun ~-9 deg: the first frames start at the 60 s cap and are still too faint.

The limit clips the source's `offer_windows()`, which `Scheduler.offer_windows`
merges for SensorKit, so SensorKit hands those minutes to other programs (a
burr that merely declines to generate keeps the controller and idles).
`generate_request` also returns None outside the clipped window as a safety
net. Burr's own SensorKit program publishes `scheduler.offer_windows()`
directly, so no SensorKit Burr-module patch is needed for clipping.

## Pilot checklist

Planned flats have worked on Huntsman through the previous adapter. The
Burr-owned entry point needs its own supervised pilot after migration.
See [the migration guide](sensorkit-migration.md) for deployment and validation.

1. **Filter names.** Confirm each wheel reports the exact planned names.
   A mismatch fails the collect at planning time.
2. **Program and SensorKit build.** Use native instrument selection and
   frame-type propagation with `burr.hardware.sk.burr_program`. Run one
   Burr program. The stock SensorKit Burr module does not supply this
   planned-flat adapter. No additional `request_id` field is required.
3. **Task ID in headers.** Check the image HDU of a pilot frame. Its
   `IMAGETYP` must be flat, and `SKTASKID` must
   equal the task ID in burr's "dispatched as SensorKit task" log line.
4. **Concurrent exposures.** Confirm all planned cameras exposed in one
   collect, each with the exposure burr logged for it.
5. **Settle time.** Confirm every camera's frame lands within
   `frame_settle_seconds`. A slow camera is reported as returning no frame;
   raise the setting if so.
6. **Counts.** Confirm the per-pair counts, and that the exposure each pair
   converges to keeps its median in the count window.
7. **Pointing and vignetting.** Flats still point at a fixed Alt 75°, Az 0°.
   Check for slit vignetting on the outer cameras.

## Limitations

These are known and left for later milestones:

- **Without a plan**, the requested exposure is shared across the selected
  instruments and completion covers only pairs seen so far. A plan supplies
  independent exposures and explicit completion targets.
- **Filter sequencing** supports batched or per-filter collection and
  configured ordering. Throughput tables still need tuning to the site
  and twilight; automatic learning is not implemented.
- **No preflight filter check.** burr does not read the wheels' filter lists
  before requesting; an unavailable filter is discovered when SensorKit
  plans the collect, which fails the whole request.
- **Unchanged pointing.** Flats point at a fixed Alt 75°, Az 0°.
- **Direct mode** is unchanged and has no plan support.
- **Plan-less SensorKit discovery** still selects candidate frames by
  modification time and `IMAGETYP`; only planned collection matches by
  task ID.

The next operational improvement is a preflight check of each planned
wheel's reported filters, with an explicit policy for unavailable pairs.

## Filter-aware ordering and starting exposures: design notes

**Status:** ordering, throughput scaling and separate dusk/dawn starting
exposures are implemented. Automatic throughput learning remains future
work. The following records the original motivation and design.

### Original problem

burr has no knowledge of a filter's bandwidth or wavelength:

- **Order** comes only from the plan's list. On a camera with several filters,
  SensorKit takes them in that order, the same at dusk and at dawn.
- **Starting exposure** is `minimum_exposure_seconds` for every pair, Lum or
  narrowband alike. A pair's exposure only adapts after it has taken a frame.

Twilight flats want the opposite. Filters differ hugely in throughput, and the
sky changes by about 2x every 3 minutes, so each filter should be taken while
the sky suits it:

- **Dusk** (sky darkening): lowest throughput first, while there is still
  light; highest last. For example narrowband, then r, g, Lum.
- **Dawn** (sky brightening): highest throughput first, in the faint early
  sky; lowest last. For example Lum, then g, r, narrowband.

Measured on 11-09 evening (Huntsman, 60 s, same request):

| Camera | Lum | g | r |
|---|---|---|---|
| C0 | 4,070 | 2,804 | 854 |
| C1 | 4,158 | 2,872 | 887 |
| C3 | 4,265 | 2,958 | 884 |

So Lum ≈ 1.4× g ≈ 4.6× r. With plan order Lum, g, r at dusk, r was taken
last, in the darkest sky, the worst case for the faintest filter. (r is
fainter than g here although sunset light is red; the post-sunset sky is
blue. Don't build a colour assumption in; measure.)

### Design

1. **Per-filter throughput in the plan.** A new optional mapping on
   `FlatPlanConfig`, relative to a reference filter:

   ```yaml
   plan:
     frames_per_pair: 10
     filter_throughput:   # counts per second relative to Lum; higher = brighter
       Lum: 1.0
       g: 0.69
       r: 0.21
       Halpha: 0.02
     pairs: [...]
   ```

   Validation: values > 0; every filter in `pairs` must be listed, or none
   (then ordering falls back to plan order, as today). Names follow the same
   rules as `filter`.

2. **Order each camera's pairs by throughput, flipping at dawn.** In
   `_planned_request`, sort `remaining_pairs()` per camera by throughput,
   ascending at dusk and descending at dawn (`_is_dawn(now)`). SensorKit keeps
   the order of a task's `CameraParameterSet` list per instrument (verified
   on 11-09: the wheel went Lum → g → r in that order), so no SensorKit
   change is needed.

3. **Throughput-scaled starting exposure.** A pair with no frames yet starts
   at `initial_exposure_seconds / throughput` for its filter (clipped to the
   min/max), instead of `minimum_exposure_seconds`. Once any pair on a camera
   has a measured frame, derive the others from it:
   `t_new = t_measured × (throughput_measured / throughput_new)`. That saves
   the first rejected frames per filter, which is most of the cost in a
   ~30 min window.
   **Dusk and dawn tables.** The twilight sky's colour differs (dawn is much
   redder: r/g 0.27 at dusk, 0.45 at dawn), so the plan takes optional
   `filter_throughput_dusk` and `filter_throughput_dawn`. Each falls back to
   `filter_throughput`; the table for the current twilight drives ordering,
   the starting exposure and seeding. With `per_filter` +
   `filter_phase_order: throughput`, each twilight must resolve to a table.
   A frame below `seed_min_counts` (default 1000 ADU raw, bias included) is
   never used as a seed: at bias level it says nothing about the sky. On
   11-11 dusk, C3's u at 60 s read ~bias and seeded z at
   60 × 0.004 / 0.05 = 4.8 s; now z falls back to the reference/throughput
   estimate.

4. **`initial_exposure_seconds`** (new; default 1.0) replaces
   `minimum_exposure_seconds` as the first exposure. On 11-09 the 0.1 s
   start gave bias-level frames (~162 ADU), giving the scaling nothing to
   work from.

5. **Optional: learn throughput.** Log each pair's measured
   counts-per-second, so the `filter_throughput` table can be refreshed from
   data rather than estimated.

### Validation cases

- Dusk orders lowest-throughput first; dawn reverses it; no table keeps plan order.
- First exposure per filter scales by throughput, within the min/max.
- A measured pair seeds an unmeasured pair on the same camera.
- Validation: a non-positive throughput, or a planned filter missing from a partial table, is rejected.
- Task translation (`tests/test_sk_task_builders.py`): planned parameters
  preserve each instrument's filter order and exposure. Verify execution
  order through SensorKit during the pilot.

### Non-goals

- Time-splitting one camera's filters into separate windows within a twilight.
- Per-filter count targets (all pairs share `min_counts`/`max_counts`).
