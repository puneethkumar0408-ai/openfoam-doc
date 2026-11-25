# Vorticity plots from a sampled plane

Plots the x-component of vorticity (ω_x) on a sampled plane (x = const) for every
saved time, then joins the frames into an MP4. Reads the OpenFOAM `.vtk`/`.vtp`
output directly (no ParaView).

## Run this in OpenFOAM first

1. **Compute vorticity** so a `vorticity` field exists in the time directories:
   ```bash
   postProcess -func vorticity -time 0.1:0.2        # adjust time range
   ```
2. **Sample a plane** that writes the `vorticity` field as VTK:
   ```bash
   postProcess -func samplePlane -time 0.1:0.2
   ```
   `samplePlane` must be a surface-sampling function object with a plane surface
   named to match `SURFACE` in the script (default `planeX003`), `fields (vorticity);`
   and `surfaceFormat vtk;`.

Expected layout: `postProcessing/samplePlane/<time>/<surface>.vtk`

## Python setup

- Put the script in the case folder (it uses its own folder as `CASE_DIR`).
- `pip install numpy scipy matplotlib pyvista imageio imageio-ffmpeg`
- LaTeX text (`USE_LATEX = True`) needs MiKTeX, `dvipng` and Ghostscript. Set
  `MIKTEX_BIN` / `GS_BIN` at the top to your install paths, or set `USE_LATEX = False`.

## Main settings (bottom of the script)

| Setting | Meaning |
|---|---|
| `SURFACE`, `SCALAR` | Surface name and field name (`vorticity`) |
| `T_MIN`, `T_MAX` | Time range (`None` = all) |
| `SHADING`, `BLEND` | `flat`, `gouraud` or `blend` look |
| `SMOOTH_PASSES`, `SMOOTH_STRENGTH` | Noise reduction (0 = off) |
| `R`, `XLIM`, `YLIM`, `VMIN`, `VMAX` | Length scale, plot window, colour range (±500) |
| `MAKE_VIDEO`, `FPS` | Video on/off and speed |

## What you get

In `postProcessing/samplePlane/Vorticity_Plots/`:

- `vorticity_t<time>.png`, one image per time
- `vorticity.mp4`, all frames (t = 0 excluded by default)

Axes are z/R (horizontal) and y/R (vertical) of the OpenFOAM coordinates. The labels
are swapped on purpose to match the earlier script.
