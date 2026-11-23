# Dark and bias calibration

Burr can collect targetless dark and bias frames through sensorkit using a per-camera plan

## requirements

- Each camera needs a opaque filter, normally 'blank'
- SensorKit must support StandardCollectTask(target=None)
- Calibration tasks must not initialize, unpark, slew or otherwise command the mount
- raw_data_root must point to the SensorKit FITS output directory

## config

```yaml
calibration_frames:
    collect: true
    windows: [day, cloudy?]
    frame_settle_seconds: 60
    plan:
        frames_per_sequence: 20
        sequences:
        - camera: C0
            instrument: C0-ASI183MM
            filter: Blank
            frame_type: bias
            exposure_seconds: 0.0

        - camera: C0
            instrument: C0-ASI183MM
            filter: Blank
            frame_type: dark
            exposure_seconds: 30.0
```

## Produces

Saves frames under 
data/calibration/bias/<camera>/<filter>/<readout>/<binning>/
data/calibration/dark/<camera>/<filter>/<readout>/<binning>/<exposure>/

they are published without overwriting existing files, burr restores sequence progress from validated products after a restart.

## Verification
Begin with one camera and one frame per sequence, confirm

- the wheel selects the correct filter
- the mount does not move or recieve commands
- FITS headers contain the expected camera, filter, image type and exposure
- products appear in the correct directory
- restarting Burr restores completed progress.