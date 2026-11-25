# Per-segment rotor forces (thrust, torque, loading)

Reads the OpenFOAM case directly and splits the force on a rotor patch into equal
segments from tip to tip, following the rotating blade. No ParaView or CSV export.

## Needed before running

| Item | How |
|---|---|
| Reconstructed case | `reconstructPar` (decomposed `processor*` is not read) |
| `p` in each time dir | Written by the solver |
| Mesh files | `constant/polyMesh/{points,faces,owner,boundary}` (or a time-dir `polyMesh` for a moving mesh) |
| Viscous force (optional) | `postProcess -func "grad(U)" -time 0.1:0.2` or `postProcess -func wallShearStress -time 0.1:0.2` |

- With **no** viscous field, the force is pressure only.
- With turbulence, prefer `wallShearStress` (it includes nuEff). If both fields exist,
  `auto` picks `grad(U)`, so set `VISCOUS_SOURCE = "wallShearStress"`.
- Geometry assumed: rotation axis = Y, rotor disc in the XZ plane, hub at `hub_x, hub_z`.

## Python setup

- Put the script in the case folder.
- `pip install numpy pandas matplotlib imageio imageio-ffmpeg pillow`
- LaTeX is optional; the script falls back to plain text if it is missing.

## Main settings (top of the script)

| Setting | Meaning |
|---|---|
| `PATCH_NAMES` | Rotor patch(es), wildcards allowed |
| `LAST_N_STEPS`, `TIME_MIN/MAX`, `TIME_STRIDE` | Which times to use (36 steps at 5° = one 180° period) |
| `N_SEG` | Number of tip-to-tip segments |
| `rho`, `nu`, `PRESSURE_IS_KINEMATIC`, `P_REF_PA` | Fluid and pressure conventions |
| `NORMAL_SIGN`, `THRUST_DIR` | Sign fixes; the first-step checks tell you if they are wrong |
| `WEIGHT_BY_ANGLE`, `ROTOR_PERIOD_DEG` | Angle-weighted time average |
| `THRUST_TARGET` | Reference thrust to compare the mean against |

## What you get

In `postProcessing/topRotor_force_plots/`:

- `segment_forces.csv`: per timestep and segment: thrust, dThrust/ds, tangential and radial force, Fx/Fy/Fz, torque about Y
- `thrust_timestep_<n>.png`: sectional thrust per timestep
- `thrust_all_timesteps.png` / `.eps`: all timesteps with the mean of each segment
- `summary.png`, `cumulative_thrust.png`, `face_map.png`
- `thrust_timesteps.mp4`

The console also prints total thrust, mean thrust (angle-weighted) against the target,
min/max, torque, and sanity checks (normal direction, patch closure, pressure level).
