# ============================================================
# VORTICITY PLOTS FROM OpenFOAM SAMPLED PLANE (postProcess output)
# Reads postProcessing/samplePlane/<time>/<surface>.vtk directly:
# no ParaView, no .vtr conversion.
#
# Shading modes:
#   "flat"    : one colour per triangle (sharp)
#   "gouraud" : smooth interpolation between vertices
#   "blend"   : flat layer + partially transparent Gouraud layer
#               (1 - BLEND) * flat + BLEND * gouraud
#
# Noise reduction (applied to the data before plotting):
#   SMOOTH_PASSES   : number of neighbour-averaging passes (0 = off)
#   SMOOTH_STRENGTH : weight of the neighbour mean per pass (0..1)
#       new = (1 - s) * old + s * mean(neighbours)
#   Effective smoothing scale ~ sqrt(passes * strength) cells.
# ============================================================

import gc
import os
import shutil
from pathlib import Path

# ------------------------------------------------------------
# Make MiKTeX / Ghostscript visible to this Python process (usetex)
# ------------------------------------------------------------
MIKTEX_BIN = r"C:\Users\punee\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
if os.path.isdir(MIKTEX_BIN):
    os.environ["PATH"] = MIKTEX_BIN + os.pathsep + os.environ["PATH"]

GS_BIN = r"C:\Program Files\gs\gs10.08.0\bin"
if os.path.isdir(GS_BIN):
    os.environ["PATH"] = GS_BIN + os.pathsep + os.environ["PATH"]


def check_latex():
    """Fail early with a clear message if the LaTeX toolchain is incomplete."""
    required = {
        "latex": ("latex",),
        "dvipng": ("dvipng",),
        "ghostscript": ("gswin64c", "gs"),
    }
    missing = [
        name for name, exes in required.items()
        if not any(shutil.which(e) for e in exes)
    ]
    if missing:
        raise RuntimeError(
            "LaTeX rendering requested but these tools were not found on PATH: "
            + ", ".join(missing)
            + ". Install them or set USE_LATEX = False."
        )


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
import pyvista as pv
from scipy import sparse


# ------------------------------------------------------------
# Custom colormap (matches the RedWhiteBlue_500 ParaView preset)
# ------------------------------------------------------------
def make_redwhiteblue_cmap():
    stops = [
        (-500, (0.0, 0.0, 1.0)),
        (-375, (0.298, 0.298, 1.0)),
        (-250, (0.6, 0.6, 1.0)),
        (-125, (0.8, 0.8, 1.0)),
        (0,    (1.0, 1.0, 1.0)),
        (125,  (1.0, 0.8, 0.8)),
        (250,  (1.0, 0.6, 0.6)),
        (375,  (1.0, 0.298, 0.298)),
        (500,  (1.0, 0.0, 0.0)),
    ]
    vmin, vmax = stops[0][0], stops[-1][0]
    positions = [(v - vmin) / (vmax - vmin) for v, _ in stops]
    colors = [c for _, c in stops]
    return LinearSegmentedColormap.from_list(
        "RedWhiteBlue_500", list(zip(positions, colors))
    )


CMAP = make_redwhiteblue_cmap()
VMIN, VMAX = -500, 500
R = 0.11  # length scale used to normalise coordinates (y/R, z/R)

XLIM = (-1.3, 1.3)
YLIM = (-2.0, 0.5)


# ------------------------------------------------------------
# Path helper: lets a Windows path (E:\...) work when Python runs in WSL/Linux
# ------------------------------------------------------------
def to_local_path(p):
    p = str(p)
    if os.name != "nt" and len(p) > 2 and p[1] == ":" and p[0].isalpha():
        p = f"/mnt/{p[0].lower()}/" + p[2:].replace("\\", "/").lstrip("/")
    return Path(p)


# ------------------------------------------------------------
# Noise reduction: neighbour averaging on the triangulated surface
# ------------------------------------------------------------
_SMOOTH_CACHE = {}


def _neighbour_operator(faces, n_points, kind):
    """Row-normalised neighbour-mean operator (cached; geometry is static).

    kind = "point": points sharing a triangle are neighbours.
    kind = "cell" : triangles sharing at least one vertex are neighbours.
    Returns (W, has_neighbours_mask).
    """
    key = (kind, n_points, faces.shape[0], hash(faces.tobytes()))
    if key in _SMOOTH_CACHE:
        return _SMOOTH_CACHE[key]

    n_cells = faces.shape[0]
    rows = np.repeat(np.arange(n_cells), 3)
    cols = faces.ravel()
    C = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(n_cells, n_points))
    A = (C.T @ C) if kind == "point" else (C @ C.T)
    A = A.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    W = sparse.diags(inv) @ A
    _SMOOTH_CACHE.clear()          # keep only the current geometry in memory
    _SMOOTH_CACHE[key] = (W.tocsr(), deg > 0)
    return _SMOOTH_CACHE[key]


def smooth_field(values, faces, n_points, kind, passes, strength):
    """Neighbour averaging: new = (1 - s) * old + s * mean(neighbours)."""
    if passes <= 0 or strength <= 0:
        return values
    s = float(np.clip(strength, 0.0, 1.0))
    W, has_nb = _neighbour_operator(faces, n_points, kind)
    out = np.asarray(values, dtype=float).copy()
    for _ in range(int(passes)):
        out = np.where(has_nb, (1.0 - s) * out + s * (W @ out), out)
    return out


# ------------------------------------------------------------
# Find the sampled files (one per time directory)
# ------------------------------------------------------------
def find_sampled_files(sample_dir, surface_name=None, t_min=None, t_max=None):
    """Return [(time_value, time_string, file_path), ...] sorted by time."""
    sample_dir = Path(sample_dir)
    found = []
    for d in sample_dir.iterdir():
        if not d.is_dir():
            continue
        try:
            t = float(d.name)
        except ValueError:
            continue                      # not a time directory
        if (t_min is not None and t < t_min) or (t_max is not None and t > t_max):
            continue

        candidates = []
        if surface_name:
            candidates = sorted(d.glob(f"{surface_name}*.vtk")) \
                         + sorted(d.glob(f"{surface_name}*.vtp"))
        if not candidates:
            candidates = sorted(d.glob("*.vtk")) + sorted(d.glob("*.vtp"))
        if candidates:
            found.append((t, d.name, candidates[0]))

    return sorted(found, key=lambda item: item[0])


# ------------------------------------------------------------
# Plot one sampled plane
# ------------------------------------------------------------
def plot_sampled_plane(filename, scalar_name="vorticity", save_path=None,
                       show=True, use_latex=True, dpi=300, time_label=None,
                       shading="gouraud", blend=0.3,
                       smooth_passes=0, smooth_strength=0.5):
    """Plot the x-component of vorticity on a sampled plane (x = const).

    Horizontal axis: OpenFOAM z / R   (labelled y/R, as in the earlier script)
    Vertical axis  : OpenFOAM y / R   (labelled z/R)

    shading:
      "flat"    - one colour per triangle (cell data is always drawn this way
                  unless shading == "blend")
      "gouraud" - smooth interpolation (point data only)
      "blend"   - opaque flat layer + Gouraud layer drawn with alpha = blend.
                  blend = 0 -> fully flat, blend = 1 -> fully Gouraud.
                  Works for both point data and cell data (cell data is
                  averaged to the points for the Gouraud layer).

    smooth_passes / smooth_strength:
      Neighbour-averaging noise reduction applied to the data before plotting.
      new = (1 - strength) * old + strength * mean(neighbours), repeated
      `smooth_passes` times (0 = off). Same setting should be used for every
      frame of a video.
    """
    filename = Path(filename)
    if save_path is None:
        save_path = filename.with_suffix(".png")
    save_path = Path(save_path)
    blend = float(np.clip(blend, 0.0, 1.0))

    mesh = pv.read(str(filename))

    # Find the field (case-insensitive, e.g. "vorticity" / "Vorticity")
    def find_array(data):
        for name in data.keys():
            if name.lower() == scalar_name.lower():
                return name
        return None

    # Prefer cell data (true cell values, sharpest); otherwise point data
    cell_name = find_array(mesh.cell_data)
    point_name = find_array(mesh.point_data)
    if cell_name is None and point_name is None:
        raise KeyError(
            f"'{scalar_name}' not found in {filename.name}. "
            f"Point arrays: {list(mesh.point_data.keys())}, "
            f"cell arrays: {list(mesh.cell_data.keys())}"
        )
    use_cell = cell_name is not None

    def x_component(arr):
        arr = np.asarray(arr)
        return arr[:, 0] if arr.ndim > 1 else arr

    # Triangulated surface (cutting planes are usually triangles already)
    tri = mesh if mesh.is_all_triangles else mesh.triangulate()
    pts = np.asarray(tri.points)
    faces = np.asarray(tri.faces).reshape(-1, 4)[:, 1:]

    cell_values = None    # one value per triangle
    point_values = None   # one value per point
    if use_cell:
        cell_values = x_component(tri.cell_data[cell_name])
        if shading == "blend":
            # Average cell values onto the points for the Gouraud layer
            tri_pd = tri.cell_data_to_point_data()
            point_values = x_component(tri_pd.point_data[cell_name])
    else:
        point_values = x_component(tri.point_data[point_name])

    # Noise reduction on the full surface (before cropping to the window)
    if smooth_passes > 0 and smooth_strength > 0:
        if cell_values is not None:
            cell_values = smooth_field(cell_values, faces, len(pts), "cell",
                                       smooth_passes, smooth_strength)
        if point_values is not None:
            point_values = smooth_field(point_values, faces, len(pts), "point",
                                        smooth_passes, smooth_strength)

    h = pts[:, 2] / R          # horizontal: z / R
    v = pts[:, 1] / R          # vertical  : y / R

    # Keep only triangles that touch the plotted window (+ margin);
    # nothing visible is removed.
    pad_x = 0.05 * (XLIM[1] - XLIM[0])
    pad_y = 0.05 * (YLIM[1] - YLIM[0])
    inside = (
        (h >= XLIM[0] - pad_x) & (h <= XLIM[1] + pad_x)
        & (v >= YLIM[0] - pad_y) & (v <= YLIM[1] + pad_y)
    )
    keep = inside[faces].any(axis=1)
    faces = faces[keep]
    if cell_values is not None:
        cell_values = cell_values[keep]

    # Per-triangle values for the flat layer
    if cell_values is not None:
        flat_values = cell_values
    else:
        flat_values = point_values[faces].mean(axis=1)

    if shading == "blend":
        mode = f"blend {blend:.2f} ({'cell' if use_cell else 'point'} data)"
    elif use_cell:
        mode = "cell data, flat"
    else:
        mode = "point data, " + shading
    if smooth_passes > 0 and smooth_strength > 0:
        mode += f", smoothed {smooth_passes}x @ {smooth_strength:.2f}"
    print(f"  Points: {len(pts)}  Triangles plotted: {len(faces)}  ({mode})")

    triang = Triangulation(h, v, faces)

    plt.rcParams["text.usetex"] = use_latex
    if use_latex:
        check_latex()
        plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
        cbar_label = r"\textbf{Vorticity $\boldsymbol{\omega_x}$}"
        title = r"\textbf{Vorticity Field}"
        if time_label is not None:
            title = r"\textbf{Vorticity Field, $t = " + time_label + r"$ s}"
    else:
        cbar_label = r"Vorticity $\omega_x$"
        title = "Vorticity Field"
        if time_label is not None:
            title = f"Vorticity Field, t = {time_label} s"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    flat_kwargs = dict(cmap=CMAP, vmin=VMIN, vmax=VMAX, edgecolors="face",
                       linewidth=0.0, antialiased=False)

    if shading == "blend":
        # Opaque flat layer underneath
        pcm = ax.tripcolor(triang, facecolors=flat_values, **flat_kwargs)
        # Partially transparent Gouraud layer on top
        if blend > 0:
            ax.tripcolor(triang, point_values, shading="gouraud",
                         cmap=CMAP, vmin=VMIN, vmax=VMAX, alpha=blend)
    elif use_cell or shading == "flat":
        pcm = ax.tripcolor(triang, facecolors=flat_values, **flat_kwargs)
    else:
        pcm = ax.tripcolor(triang, point_values, shading="gouraud",
                           cmap=CMAP, vmin=VMIN, vmax=VMAX)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=15)
    cbar.ax.tick_params(labelsize=15)

    # NOTE: horizontal axis is OpenFOAM z/R, vertical is OpenFOAM y/R;
    # labels are intentionally swapped to match the earlier script.
    ax.set_xlabel("y/R", fontsize=15)
    ax.set_ylabel("z/R", fontsize=15)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)
    del mesh, tri, pts, faces, triang, flat_values, point_values, cell_values
    gc.collect()


# ------------------------------------------------------------
# Video from saved frames
# ------------------------------------------------------------
def make_video(frames, output_path, fps=15):
    """Combine PNG frames (in the order given) into an MP4 at `fps` fps.

    Needs:  pip install imageio imageio-ffmpeg
    """
    import imageio.v2 as imageio

    frames = [Path(f) for f in frames]
    if not frames:
        raise FileNotFoundError("No frames to write")
    output_path = Path(output_path)

    first = imageio.imread(frames[0])
    height, width = first.shape[:2]
    print(f"Writing {len(frames)} frames ({width}x{height}) at {fps} fps "
          f"-> {output_path}")

    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=1,
    )
    try:
        for i, frame_file in enumerate(frames, start=1):
            frame = imageio.imread(frame_file)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"{frame_file.name} is {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}; all frames must match."
                )
            writer.append_data(frame)
            if i % 50 == 0 or i == len(frames):
                print(f"  {i}/{len(frames)} frames")
    finally:
        writer.close()

    print(f"Video saved: {output_path}")
    return output_path


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # USER INPUTS
    # --------------------------------------------------------
    # Case folder. Default: the folder this script sits in.
    # Windows-style paths (E:\...) are converted automatically when running in WSL/Linux.
    CASE_DIR = Path(__file__).resolve().parent
    # CASE_DIR = to_local_path(r"E:\LaptopRescue\Openfoam\sRotorFin\mesh2")

    # postProcessing/<function object name>/<time>/<surface>.vtk
    SAMPLE_DIR = CASE_DIR / "postProcessing" / "samplePlane"
    SURFACE = "planeX003"          # surface name in the sampling dictionary

    SCALAR = "vorticity"           # field name written by OpenFOAM

    # Time range in seconds (None = all available times)
    T_MIN = None
    T_MAX = None

    # "flat"    : sharp, per-triangle colours
    # "gouraud" : fully smooth (point data only)
    # "blend"   : flat + partially transparent Gouraud layer
    SHADING = "blend"
    BLEND = 0.3                    # 0 = fully flat, 1 = fully Gouraud (blend mode only)

    # Noise reduction (neighbour averaging on the data, before plotting)
    SMOOTH_PASSES = 2              # 0 = off; more passes = larger smoothing scale
    SMOOTH_STRENGTH = 0.5          # 0..1 weight of neighbour mean per pass

    USE_LATEX = True
    SHOW_TIME_IN_TITLE = False     # True -> "Vorticity Field, t = 0.5 s"

    OUTPUT_DIR = SAMPLE_DIR / "Vorticity_Plots"

    # Video of all frames
    MAKE_VIDEO = True
    EXCLUDE_T0_FROM_VIDEO = True   # t = 0 is still plotted as a PNG, just not in the video
    FPS = 15
    VIDEO_FILE = OUTPUT_DIR / "vorticity.mp4"

    # --------------------------------------------------------
    # FIND FILES
    # --------------------------------------------------------
    items = find_sampled_files(SAMPLE_DIR, SURFACE, T_MIN, T_MAX)

    print(f"Number of time steps found: {len(items)}")
    if not items:
        raise FileNotFoundError(
            f"No sampled .vtk files found in {SAMPLE_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # PROCESS ALL TIME STEPS
    # --------------------------------------------------------
    saved_frames = []
    for index, (t, t_name, vtk_file) in enumerate(items, start=1):
        print(f"\nProcessing {index}/{len(items)}: t = {t_name}")

        output_file = OUTPUT_DIR / f"vorticity_t{t_name}.png"

        plot_sampled_plane(
            filename=vtk_file,
            scalar_name=SCALAR,
            save_path=output_file,
            show=False,
            use_latex=USE_LATEX,
            time_label=t_name if SHOW_TIME_IN_TITLE else None,
            shading=SHADING,
            blend=BLEND,
            smooth_passes=SMOOTH_PASSES,
            smooth_strength=SMOOTH_STRENGTH,
        )
        saved_frames.append((t, output_file))

    if MAKE_VIDEO:
        video_frames = [f for t, f in saved_frames
                        if not (EXCLUDE_T0_FROM_VIDEO and t == 0)]
        make_video(video_frames, VIDEO_FILE, fps=FPS)

    print("\nPost-processing completed!")
    print(f"Plots saved in: {OUTPUT_DIR}")