# Migrating to the Burr-owned SensorKit program

Burr owns request translation, scheduling, flat readback and completion.
SensorKit provides native task execution, instrument selection and camera
control. This migration needs no changes to SensorKit source.

The development target is SensorKit commit `701bbe4`. Run compatibility
tests and a supervised pilot of this entry point before unattended use.
Earlier on-sky success used the previous adapter.

## Configuration

Replace `sensorkit.burr.program` in the existing imports with:

```yaml
sensorkit:
  imports:
    - burr.hardware.sk.burr_program
```

Keep the other imports. Preserve the existing `burr:` entry, including:

- `id: Burr` and `controller: Huntsman`;
- the schedule, twilight limits and camera/filter plan;
- `collect` defaults and instrument overrides;
- `raw_data_root` and `runtime.output_dir`.

The `burr:` section creates the service. Remove a duplicate explicit
`services:` entry for Burr and leave `BURR_CONFIG` unset in unified mode.
Keep the existing automation priorities and interrupt policy.

Reload the migrated configuration before starting the service. This program
uses the Burr-owned `BurrServiceConfig` model; an older uploaded
SensorKit Burr configuration does not supply that model.

Standalone configuration remains supported:

```sh
burr-sk --site deploy/example/site.yaml --preset deploy/presets/sda.yaml --name Burr
burr-sk --config burr.yaml --name Burr
```

Use one launch mode at a time.

## Data graph compatibility

Review deployment expressions and explicit FITS-header include lists for
SensorKit's current keyword names: `Task`, `Burr`, `Otto`, `Exposure`,
`Image`, `Array` and `File`. The `Collect` keyword retains that name.

Use `Instrument.name` for the selected instrument and `Exposure.detector`
for the detector. Preserve the short camera `UNITID` values used by the
plan, such as `C0`; an instrument name such as `C0-ASI183MM` is different.

Keep the raw writer's actual directory consistent with `raw_data_root`.
Check the case of `Program.program_id` when it appears in a directory
expression: `Burr` can produce a different path from the previous `burr`.
Preserve the configured storage location rather than copying a default
deployment path.

## Validation and rollout

1. Record both repositories' revisions and local diffs. Save the deployment
   YAML, dependency pins and lock file. Preserve unrelated SensorKit edits.
2. In an isolated environment with the candidate Burr and SensorKit
   installed, run:

   ```sh
   uv run --no-sync pytest tests/test_sk_task_builders.py tests/test_sk_program.py tests/test_sk_service_config.py -q
   uv run --no-sync pytest -q
   ```

   The SensorKit tests must execute. Skipping them because SensorKit is
   unavailable does not establish compatibility.
3. Validate the migrated deployment without writing configuration:

   ```sh
   sensorkit config load -n huntsman-inwork.yaml
   ```

4. In the agreed maintenance window, stop the previous Burr program,
   install the tested versions and load the migrated configuration:

   ```sh
   sensorkit config load huntsman-inwork.yaml
   ```

   Restart through the existing service manager. Confirm exactly one
   program named `Burr`, using controller `Huntsman`.
5. Follow the [flat pilot checklist](flats.md#pilot-checklist). Verify
   image type, task IDs, independent exposures, products and progress.
   Also confirm the clipped dawn window releases time to other programs.
   Record the tested revisions and results.

To roll back, restore the matching package pins, lock file and deployment
configuration together, reload that configuration and restart the previous
entry point. Preserve unrelated local edits.

The cross-night run rollover bug, filter preflight, throughput learning,
pointing and calibration choices remain separate work.
