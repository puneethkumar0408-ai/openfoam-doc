"""
Per-segment rotor forces straight from an OpenFOAM case (no ParaView / CSV export).

What is read, per selected time directory
-----------------------------------------
  polyMesh/{points,faces,owner,boundary}   geometry of the patch (face centres, area vectors, areas)
                                           (time-dir polyMesh is used if present, e.g. moving mesh,
                                            else constant/polyMesh, same rule OpenFOAM uses)
  p                                        patch values (zeroGradient patches that write no 'value'
                                           fall back to the adjacent cell value)
  grad(U)   (preferred)                    patch values, for the viscous force mu*(gradU + gradU^T).n dA
  wallShearStress (alternative)            used for the viscous force if grad(U) is missing, and it
                                           includes the turbulent nuEff

Create the optional fields once with, for example:
    postProcess -func "grad(U)" -time 0.1:0.2            (adjust the time range)
    postProcess -func wallShearStress -time 0.1:0.2

The case must be reconstructed (reconstructPar), not decomposed.
ASCII and binary files are supported, as are .gz compressed files.
"""
import fnmatch
import gzip
import itertools
import re
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # no GUI needed, files only
import matplotlib.pyplot as plt

# ---------------- Setup ----------------
try:
    current_dir_path = Path(__file__).resolve().parent
except NameError:                          # notebook / interactive session
    current_dir_path = Path.cwd()

CASE_DIR = current_dir_path               # OpenFOAM case directory (contains constant/, 0.1/, ...)
PATCH_NAMES = ["wallsTop"]                # patch name(s); shell-style wildcards allowed, e.g. ["rotor*"]
OUTPUT_NAME = "topRotor_force_plots"      # folder created inside <case>/postProcessing/
output_dir = Path(CASE_DIR) / "postProcessing" / OUTPUT_NAME
output_dir.mkdir(parents=True, exist_ok=True)

# ---- Which time directories to use (a step's index = its position in the selected list) ----
TIME_MIN = None                   # only times >= TIME_MIN (None = no limit)
TIME_MAX = None                   # only times <= TIME_MAX
TIME_STRIDE = 1                   # use every n-th selected time
LAST_N_STEPS = 36                 # keep only the last N selected times (None = all), e.g. one 180 deg period
INCLUDE_ZERO = False              # include the "0" directory

VIDEO_FPS = 5
MAX_VIDEO_WIDTH = 1920
N_SEG = 80                        # number of equal-width segments from one tip to the other

FACE_SIZE_EXPECTED = 0.75e-3      # m, surface mesh sizing (used only for a sanity check)
DIAG_STEP = 0                     # step index used for the per-bin diagnostic table and face map

# ---- Appearance of the sectional-thrust plots (frames + combined figure) ----
USE_LATEX = True                  # LaTeX text; falls back to plain mode if LaTeX is not available
CMAP_NAME = "jet"                 # one gradient shared by all timestep curves
LINEWIDTH = 2
SHOW_FIGURE = False               # True: plt.show() the combined figure, False: close it
COMBINED_NAME = "thrust_all_timesteps"   # combined figure, saved as .png and .eps in output_dir
FRAME_DPI = 200                   # video frames; saved WITHOUT bbox_inches="tight" so that all frames
                                  # have identical pixel size (needed for the video)

SHOW_BARS = True                  # light-blue bar graph behind the lines
BAR_COLOR = "#bdd7e7"             # light blue bars
BAR_EDGE = "#2171b5"              # darker blue outline, clearly visible around each bar
BAR_EDGE_WIDTH = 1.3              # bar outline thickness
MEAN_LINE_WIDTH = 1.2             # thin dotted line for the mean of each segment

# Fluid properties
rho = 1.22        # kg/m^3
nu = 1.5e-5       # m^2/s
mu = rho * nu     # Pa.s, used with grad(U). With turbulence prefer VISCOUS_SOURCE = "wallShearStress".

PRESSURE_IS_KINEMATIC = True      # OpenFOAM-style p/rho
p_scale = rho if PRESSURE_IS_KINEMATIC else 1.0

# Viscous force source: "auto" (grad(U) if present, else wallShearStress, else none),
# "gradU", "wallShearStress" or "none".
VISCOUS_SOURCE = "auto"
GRADU_NAMES = ["grad(U)", "gradU"]            # candidate field file names
WSS_NAMES = ["wallShearStress"]
WSS_IS_KINEMATIC = True           # incompressible solvers write wallShearStress in m^2/s^2
WSS_SIGN = -1.0                   # OpenFOAM's wallShearStress = -(Sf/|Sf|).Reff, so the force on the
                                  # blade is -wss*|Sf|. Flip only if the viscous Fy looks wrong.

# Reference (ambient / far-field) pressure in Pa, subtracted AFTER the rho scaling.
# A constant offset cancels in the total (closed patch) but NOT per segment, where it
# adds large noise. Set to 0.0 if the p files are already gauge.
P_REF_PA = 0

# OpenFOAM boundary face vectors point OUT of the fluid domain, i.e. INTO the blade, so -1
# makes them point out of the blade. The first-step check below confirms this.
NORMAL_SIGN = -1.0
THRUST_DIR = 1.0                  # +1: thrust is +Y, -1: thrust is -Y

# Rotor: hub on the origin axis, rotation axis = Y, rotor disc in the XZ plane.
hub_x, hub_z = 0.0, 0.0

# ---- Segmentation (along the rotor's own tip-to-tip line, which rotates) ----
S_MIN = None                      # one tip (m). If None, taken from the data (all timesteps).
S_MAX = None                      # other tip (m). If None, taken from the data (all timesteps).
SPAN_NAME = "s"
MAX_ANGLE_JUMP_DEG = 60.0         # warn if the detected axis turns more than this between steps
MIN_AXIS_QUALITY = 4.0            # warn if the rotor is not clearly elongated (major/minor moment)

# Time averaging: weight each output step by the rotation angle it represents.
WEIGHT_BY_ANGLE = True
ROTOR_PERIOD_DEG = 180.0

THRUST_TARGET = 2.829            # N, cycle-averaged reference (compare with the MEAN)


# =====================================================================================
#                          Minimal OpenFOAM file reader
# =====================================================================================
_WS = re.compile(rb"(?:\s+|//[^\n]*|/\*.*?\*/)*", re.S)
_INT = re.compile(rb"[+-]?\d+")
_WORD = re.compile(rb"[^\s;{}()]+")
_LISTTYPE = re.compile(rb"List<(\w+)>")
_LIST_NCOMP = {b"scalar": 1, b"label": 1, b"vector": 3, b"symmTensor": 6,
               b"tensor": 9, b"sphericalTensor": 1}


def _exists(path):
    return path.exists() or path.with_name(path.name + ".gz").exists()


def _read_bytes(path):
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        with gzip.open(gz, "rb") as fh:
            return fh.read()
    raise FileNotFoundError(path)


def _mtime(path):
    path = Path(path)
    p = path if path.exists() else path.with_name(path.name + ".gz")
    return p.stat().st_mtime


class FoamFile:
    """One OpenFOAM file held in memory, with readers for lists and entries."""

    def __init__(self, path):
        self.path = Path(path)
        self.buf = _read_bytes(self.path)
        m = re.search(rb"FoamFile\s*\{(.*?)\}", self.buf[:8192], re.S)
        if not m:
            raise ValueError(f"{self.path}: no FoamFile header")
        hdr = m.group(1)
        fmt = re.search(rb"format\s+(\w+)", hdr)
        cls = re.search(rb"class\s+(\w+)", hdr)
        arch = re.search(rb'arch\s+"([^"]*)"', hdr)
        arch = arch.group(1) if arch else b""
        self.binary = bool(fmt) and fmt.group(1) == b"binary"
        self.cls = cls.group(1).decode() if cls else ""
        self.label_bytes = 8 if b"label=64" in arch else 4
        self.scalar_bytes = 4 if b"scalar=32" in arch else 8
        self.pos0 = m.end()

    def skip(self, pos):
        return _WS.match(self.buf, pos).end()

    # ---- numeric lists: "N ( ... )" -------------------------------------------------
    def read_list(self, pos, ncomp, is_label=False):
        buf = self.buf
        pos = self.skip(pos)
        m = _INT.match(buf, pos)
        if not m:
            raise ValueError(f"{self.path.name}: expected a list size at byte {pos}")
        count = int(m.group())
        pos = self.skip(m.end())
        dt_int = np.int64
        if buf[pos:pos + 1] == b"{":                      # "N{value}" uniform list (ASCII)
            end = buf.index(b"}", pos)
            item = np.array(buf[pos + 1:end].translate(None, b"()").split(),
                            dtype=dt_int if is_label else float)
            return np.tile(item, (count, 1)), end + 1
        if buf[pos:pos + 1] != b"(":
            raise ValueError(f"{self.path.name}: expected '(' after list size")
        start = pos + 1
        if count == 0:
            return np.zeros((0, ncomp), dtype=dt_int if is_label else float), \
                buf.index(b")", start) + 1
        if self.binary:
            isz = self.label_bytes if is_label else self.scalar_bytes
            code = ("<i" if is_label else "<f") + str(isz)
            arr = np.frombuffer(buf, dtype=code, count=count * ncomp, offset=start)
            end = start + count * ncomp * isz
            if buf[end:end + 1] != b")":
                raise ValueError(f"{self.path.name}: binary list not terminated "
                                 f"(wrong label/scalar size in the arch string?)")
            return arr.astype(dt_int if is_label else float).reshape(count, ncomp), end + 1
        close = self._ascii_close(start, count, ncomp)
        chunk = buf[start:close].translate(None, b"()")
        arr = np.fromstring(chunk, dtype=dt_int if is_label else float, sep=" ")
        if arr.size != count * ncomp:
            raise ValueError(f"{self.path.name}: expected {count * ncomp} numbers, read {arr.size}")
        return arr.reshape(count, ncomp), close + 1

    def _ascii_close(self, start, count, ncomp):
        """Position of the ')' that closes an ASCII list (items of ncomp>1 are '(...)')."""
        arr = np.frombuffer(self.buf, dtype=np.uint8)
        need = count if ncomp > 1 else 0               # item-closing parens before the list's own
        win = max(1 << 16, count * ncomp * 12)
        pos, seen = start, 0
        while pos < len(arr):
            idx = np.flatnonzero(arr[pos:pos + win] == 41)
            if seen + len(idx) > need:
                return pos + int(idx[need - seen])
            seen += len(idx)
            pos += win
        raise ValueError(f"{self.path.name}: unterminated list")

    # ---- dictionary entries -----------------------------------------------------------
    def _skip_dict(self, pos):
        depth = 0
        for m in re.finditer(rb"[{}]", self.buf, pos):
            depth += 1 if m.group() == b"{" else -1
            if depth == 0:
                return m.end()
        raise ValueError(f"{self.path.name}: unbalanced braces")

    def read_value(self, pos, default_ncomp):
        """Value part of 'keyword value;'. Returns (kind, data, pos_after)."""
        buf = self.buf
        pos = self.skip(pos)
        m = _WORD.match(buf, pos)
        word = m.group() if m else b""
        if word == b"uniform":
            end = buf.index(b";", m.end())
            try:
                vals = np.array(buf[m.end():end].translate(None, b"()").split(), dtype=float)
                return "uniform", vals, end + 1
            except ValueError:
                return "other", None, end + 1
        if word == b"nonuniform":
            p = self.skip(m.end())
            nc = default_ncomp
            lt = _LISTTYPE.match(buf, p)
            if lt:
                nc = _LIST_NCOMP.get(lt.group(1), default_ncomp)
                p = lt.end()
            arr, p = self.read_list(p, nc)
            p = self.skip(p)
            if buf[p:p + 1] == b";":
                p += 1
            return "nonuniform", arr, p
        if buf[pos:pos + 1] == b"{":
            return "other", None, self._skip_dict(pos)
        return "other", None, buf.index(b";", pos) + 1

    def patch_value(self, patch, default_ncomp):
        """'value' entry of boundaryField/<patch>. kind: uniform | nonuniform | none | missing."""
        buf = self.buf
        bf = buf.rfind(b"boundaryField")
        if bf < 0:
            raise ValueError(f"{self.path.name}: no boundaryField")
        m = re.compile(rb"(?<![\w.])" + re.escape(patch.encode()) + rb"\s*\{").search(buf, bf)
        if not m:
            return "missing", None
        pos = m.end()
        while True:
            pos = self.skip(pos)
            if buf[pos:pos + 1] == b"}":
                return "none", None
            kw = _WORD.match(buf, pos)
            if not kw:
                raise ValueError(f"{self.path.name}: cannot parse patch '{patch}'")
            kind, data, pos = self.read_value(kw.end(), default_ncomp)
            if kw.group() == b"value":
                return kind, data

    def internal(self, default_ncomp):
        m = re.compile(rb"\binternalField\s").search(self.buf, self.pos0)
        if not m:
            raise ValueError(f"{self.path.name}: no internalField")
        kind, data, _ = self.read_value(m.end(), default_ncomp)
        return kind, data


# ---- polyMesh pieces (cached: a static mesh is only parsed once) -----------------------
@lru_cache(maxsize=2)
def _points(path_str, mtime):
    ff = FoamFile(path_str)
    return ff.read_list(ff.pos0, 3)[0]


@lru_cache(maxsize=1)
def _owner(path_str, mtime):
    ff = FoamFile(path_str)
    return ff.read_list(ff.pos0, 1, is_label=True)[0].ravel()


@lru_cache(maxsize=8)
def _boundary(path_str, mtime):
    ff = FoamFile(path_str)
    text = ff.buf[ff.pos0:].decode("ascii", "ignore")
    out = {}
    for m in re.finditer(r"([^\s{}()]+)\s*\{([^{}]*)\}", text):
        nf = re.search(r"\bnFaces\s+(\d+)", m.group(2))
        sf = re.search(r"\bstartFace\s+(\d+)", m.group(2))
        if nf and sf:
            out[m.group(1)] = (int(sf.group(1)), int(nf.group(1)))
    return out


@lru_cache(maxsize=8)
def _patch_faces(path_str, mtime, start, n):
    """(vertex counts, flat vertex labels) of faces start .. start+n-1."""
    ff = FoamFile(path_str)
    buf = ff.buf
    if ff.cls == "faceCompactList":
        offsets, pos = ff.read_list(ff.pos0, 1, is_label=True)
        labels, _ = ff.read_list(pos, 1, is_label=True)
        offsets, labels = offsets.ravel(), labels.ravel()
        o = offsets[start:start + n + 1]
        return np.diff(o), labels[o[0]:o[-1]]
    if ff.binary:
        raise NotImplementedError(f"{ff.path}: binary 'faceList' not supported "
                                  f"(expected faceCompactList)")
    pos = ff.skip(ff.pos0)
    m = _INT.match(buf, pos)
    pos = ff.skip(m.end())
    it = re.finditer(rb"(\d+)\(([^)]*)\)", buf[pos + 1:])
    sel = list(itertools.islice(it, start, start + n))
    counts = np.array([int(x.group(1)) for x in sel], dtype=np.int64)
    flat = np.array(b" ".join(x.group(2) for x in sel).split(), dtype=np.int64)
    return counts, flat


def face_geometry(points, counts, flat):
    """Face area vectors Sf (out of the fluid domain) and face centres, like OpenFOAM."""
    nf = len(counts)
    offs = np.concatenate([[0], np.cumsum(counts)])
    Sf = np.zeros((nf, 3))
    Cf = np.zeros((nf, 3))
    for k in np.unique(counts):
        sel = np.flatnonzero(counts == k)
        P = points[flat[offs[sel][:, None] + np.arange(k)[None, :]]]      # (m, k, 3)
        if k == 3:
            Sf[sel] = 0.5 * np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            Cf[sel] = P.mean(axis=1)
            continue
        c = P.mean(axis=1)
        P2 = np.roll(P, -1, axis=1)
        n_i = 0.5 * np.cross(P - c[:, None], P2 - c[:, None])
        tc = (P + P2 + c[:, None]) / 3.0
        sumN = n_i.sum(axis=1)
        hat = sumN / np.maximum(np.linalg.norm(sumN, axis=1), 1e-300)[:, None]
        a_i = np.einsum("mkj,mj->mk", n_i, hat)
        sumA = a_i.sum(axis=1)
        good = sumA > 1e-300
        Cf[sel] = c
        Cf[sel[good]] = (a_i[good][..., None] * tc[good]).sum(axis=1) / sumA[good][:, None]
        Sf[sel] = sumN
    return Sf, Cf


@lru_cache(maxsize=64)
def _geometry(fpath_s, fmt, ppath_s, pmt, patches):
    """Face area, outward-from-fluid unit normals and face centres of the selected patches.

    Cached on the faces/points files, so a static mesh is computed ONCE for all timesteps
    (and Pass 0 / Pass 1 share the result for a moving mesh).
    """
    pts = _points(ppath_s, pmt)
    parts = [_patch_faces(fpath_s, fmt, s, n) for _, s, n in patches]
    counts = np.concatenate([c for c, _ in parts])
    flat = np.concatenate([f for _, f in parts])
    Sf, Cf = face_geometry(pts, counts, flat)
    area = np.linalg.norm(Sf, axis=1)
    normals = Sf / np.maximum(area, 1e-300)[:, None]
    for a in (area, normals, Cf):
        a.flags.writeable = False              # shared between calls: never modify in place
    return area, normals, Cf


class FoamCase:
    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / "constant").is_dir():
            raise SystemExit(f"'{self.root}' does not look like an OpenFOAM case (no constant/).")
        self.all_times = []
        for d in self.root.iterdir():
            if d.is_dir():
                try:
                    self.all_times.append((float(d.name), d.name))
                except ValueError:
                    pass
        self.all_times.sort()

    def mesh_file(self, tname, name):
        """polyMesh file valid at time tname: latest time dir <= tname holding it, else constant."""
        tv = float(tname)
        for v, n in reversed(self.all_times):
            if v <= tv + 1e-12:
                p = self.root / n / "polyMesh" / name
                if _exists(p):
                    return p
        return self.root / "constant" / "polyMesh" / name

    def field_file(self, tname, name):
        p = self.root / tname / name
        return p if _exists(p) else None


def read_patch_values(path, patches, ncomp, owner_getter, note=None):
    """Patch face values of a volField, concatenated over `patches` [(name, startFace, nFaces)]."""
    ff = FoamFile(path)
    internal = None
    parts = []
    for name, start, n in patches:
        kind, data = ff.patch_value(name, ncomp)
        if kind == "missing":
            raise KeyError(f"patch '{name}' not found in {path}")
        if kind == "nonuniform":
            if data.shape != (n, ncomp):
                raise ValueError(f"{path}: patch '{name}' has {data.shape}, expected ({n}, {ncomp})")
            parts.append(data)
        elif kind == "uniform":
            if data.size != ncomp:
                raise ValueError(f"{path}: uniform value of patch '{name}' has {data.size} components")
            parts.append(np.tile(data, (n, 1)))
        else:                                   # e.g. zeroGradient: no 'value' written
            if internal is None:
                internal = ff.internal(ncomp)
            if note is not None and (Path(path).name, name) not in note:
                note.add((Path(path).name, name))
                print(f"  Note: '{Path(path).name}' has no 'value' on patch '{name}'; "
                      f"using the adjacent cell value (zeroGradient).")
            ikind, idata = internal
            if ikind == "uniform":
                parts.append(np.tile(idata, (n, 1)))
            else:
                parts.append(idata[owner_getter(start, n)])
    out = np.vstack(parts)
    return out[:, 0] if ncomp == 1 else out


# =====================================================================================
#                                  Case handling
# =====================================================================================
case = FoamCase(CASE_DIR)


def select_times():
    sel = []
    for v, n in case.all_times:
        if v == 0 and not INCLUDE_ZERO:
            continue
        if TIME_MIN is not None and v < TIME_MIN:
            continue
        if TIME_MAX is not None and v > TIME_MAX:
            continue
        if case.field_file(n, "p") is None:
            continue
        sel.append(n)
    sel = sel[::max(1, int(TIME_STRIDE))]
    if LAST_N_STEPS is not None:
        sel = sel[-int(LAST_N_STEPS):]
    return sel


TIME_NAMES = select_times()
if not TIME_NAMES:
    raise SystemExit(f"No time directories with a 'p' field found in {CASE_DIR}.")
TIME_VALUES = [float(t) for t in TIME_NAMES]
N_STEPS = len(TIME_NAMES)
print(f"Case: {CASE_DIR}")
print(f"Using {N_STEPS} time directories: {TIME_NAMES[0]} ... {TIME_NAMES[-1]}")

_notes = set()                    # one-time messages
_visc_logged = False


def resolve_patches(tname):
    bpath = case.mesh_file(tname, "boundary")
    allp = _boundary(str(bpath), _mtime(bpath))
    sel = [(nm, *allp[nm]) for nm in allp
           if any(fnmatch.fnmatchcase(nm, pat) for pat in PATCH_NAMES)]
    if not sel:
        raise SystemExit(f"No patch matching {PATCH_NAMES} in {bpath}. Available: {list(allp)}")
    return sel


def _first_existing(tname, names):
    for nm in names:
        p = case.field_file(tname, nm)
        if p is not None:
            return nm, p
    return None, None


def load_step(i, with_fields=True):
    """Same dictionary as the old CSV loader, built from the OpenFOAM case.

    grad_u    (n,3,3) boundary value of grad(U) or None. (OpenFOAM stores T_ij = d_i U_j, the
              transpose of ParaView's layout; only the symmetric part is used, so it does not matter.)
    visc_force (n,3)  viscous force on the blade from wallShearStress, or None.
    """
    global _visc_logged
    if i >= N_STEPS:
        return None
    tname = TIME_NAMES[i]
    patches = resolve_patches(tname)

    fpath = case.mesh_file(tname, "faces")
    ppath = case.mesh_file(tname, "points")
    area, normals_raw, Cf = _geometry(str(fpath), _mtime(fpath), str(ppath), _mtime(ppath),
                                      tuple(patches))
    out = {
        "area": area,
        "normals_raw": normals_raw,                              # out of the fluid domain (into the blade)
        "points": Cf,
        "grad_u": None, "visc_force": None, "pressure": None,
    }
    if not with_fields:
        return out

    opath = case.mesh_file(tname, "owner")

    def owner_getter(start, n):
        return _owner(str(opath), _mtime(opath))[start:start + n]

    p = read_patch_values(case.field_file(tname, "p"), patches, 1, owner_getter, _notes)
    out["pressure"] = p * p_scale - P_REF_PA                     # gauge, Pa

    src = VISCOUS_SOURCE
    gname, gpath = _first_existing(tname, GRADU_NAMES)
    wname, wpath = _first_existing(tname, WSS_NAMES)
    if src == "auto":
        src = "gradU" if gpath else ("wallShearStress" if wpath else "none")
    if src == "gradU":
        if not gpath:
            raise SystemExit(f"VISCOUS_SOURCE='gradU' but none of {GRADU_NAMES} exists in {tname}/. "
                             f"Run: postProcess -func \"grad(U)\"")
        g = read_patch_values(gpath, patches, 9, owner_getter, _notes)
        out["grad_u"] = g.reshape(-1, 3, 3)
    elif src == "wallShearStress":
        if not wpath:
            raise SystemExit(f"VISCOUS_SOURCE='wallShearStress' but {WSS_NAMES} not found in {tname}/. "
                             f"Run: postProcess -func wallShearStress")
        wss = read_patch_values(wpath, patches, 3, owner_getter, _notes)
        out["visc_force"] = WSS_SIGN * wss * (rho if WSS_IS_KINEMATIC else 1.0) * area[:, None]
    if not _visc_logged:
        print(f"  Viscous force source: {src}"
              + (" (no viscous force; pressure only)" if src == "none" else ""))
        _visc_logged = True
    return out


def save_fig(fig, path, **kw):
    """Save a figure; if the file is locked, save under a new name. Returns the path used."""
    try:
        fig.savefig(path, **kw)
        return path
    except OSError:
        alt = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        fig.savefig(alt, **kw)
        print(f"{path.name} is locked (open in a viewer?). Saved to {alt.name} instead.")
        return alt


def tip_to_tip_axis(points, area, prev_axis=None):
    """Unit vector (x, z) along the rotor from one tip to the other, in the XZ rotor plane.

    Major axis of the area-weighted second-moment matrix of the face centres about the hub.
    The eigenvector sign is arbitrary, so it is fixed to be continuous from the previous
    timestep (needs < 90 deg of rotation per step); this keeps segment 1 on the same tip.
    Returns (axis, quality), quality = major / minor moment.
    """
    dx = points[:, 0] - hub_x
    dz = points[:, 2] - hub_z
    M = np.array([[np.sum(area * dx * dx), np.sum(area * dx * dz)],
                  [np.sum(area * dx * dz), np.sum(area * dz * dz)]])
    w, v = np.linalg.eigh(M)                    # eigenvalues ascending
    axis = v[:, 1]
    quality = w[1] / max(w[0], 1e-300)
    if prev_axis is None:
        k = 0 if abs(axis[0]) >= abs(axis[1]) else 1
        if axis[k] < 0:
            axis = -axis
    elif axis @ prev_axis < 0:
        axis = -axis
    return axis, quality


def step_weights(ang_deg, period=ROTOR_PERIOD_DEG, tol=3.0):
    """Weights proportional to the angle each step represents (sum = 1)."""
    a = np.asarray(ang_deg, dtype=float)
    n = len(a)
    if n < 2:
        return np.ones(n) / max(n, 1)
    da = np.abs(np.diff(a))
    gap = period - abs(a[-1] - a[0])
    if abs(gap - da.mean()) < tol:              # steps cover one full period: wrap around
        inc = np.append(da, gap)
        w = 0.5 * (np.roll(inc, 1) + inc)
    else:                                       # open interval: trapezoid weights
        w = np.empty(n)
        w[0], w[-1] = da[0] / 2, da[-1] / 2
        w[1:-1] = 0.5 * (da[:-1] + da[1:])
    return w / w.sum()


# ---------------- Pass 0: blade axis per step and global segment edges ----------------
span_axes = {}                    # step -> unit vector (x, z) of the tip-to-tip line
s_lo, s_hi = np.inf, -np.inf
prev_axis, prev_angle = None, None
raw_angles = {}
print("Rotor tip-to-tip axis per timestep (angle from +X toward +Z):")
for i in range(N_STEPS):
    d = load_step(i, with_fields=not span_axes)      # fields only needed for the first-step checks
    if not span_axes:
        a_sum = d["area"].sum()
        cx = np.sum(d["area"] * d["points"][:, 0]) / a_sum
        cz = np.sum(d["area"] * d["points"][:, 2]) / a_sum
        print(f"  Area-weighted centroid xz = ({cx:+.5f}, {cz:+.5f}) m vs hub ({hub_x}, {hub_z}) "
              f"(should agree for a symmetric rotor, otherwise the axis, F_t, F_r and torque are biased)")
        gauge_mean = np.sum(d["area"] * d["pressure"]) / a_sum
        print(f"  Area-mean gauge pressure = {gauge_mean:+.1f} Pa "
              f"(should be tens to hundreds of Pa; if ~1e5, P_REF_PA is wrong)")
    axis, quality = tip_to_tip_axis(d["points"], d["area"], prev_axis)
    span_axes[i] = axis
    coord = (d["points"][:, 0] - hub_x) * axis[0] + (d["points"][:, 2] - hub_z) * axis[1]
    s_lo, s_hi = min(s_lo, coord.min()), max(s_hi, coord.max())

    angle = np.degrees(np.arctan2(axis[1], axis[0]))
    raw_angles[i] = angle
    jump = 0.0 if prev_angle is None else (angle - prev_angle + 180.0) % 360.0 - 180.0
    note = ""
    if abs(jump) > MAX_ANGLE_JUMP_DEG:
        note += "  <- large jump: axis tracking may be unreliable"
    if quality < MIN_AXIS_QUALITY:
        note += "  <- rotor not clearly elongated: axis may be ill-defined"
    print(f"  [t{i} = {TIME_NAMES[i]}] angle = {angle:+8.3f} deg  (step change {jump:+7.3f} deg)  "
          f"s range = [{coord.min():+.4f}, {coord.max():+.4f}] m{note}")
    prev_axis, prev_angle = axis, angle

if not np.isfinite(s_lo):
    raise SystemExit("No data found.")

_steps0 = list(raw_angles.keys())
_unwrapped = np.degrees(np.unwrap(np.radians([raw_angles[k] for k in _steps0])))
step_angle = dict(zip(_steps0, _unwrapped))     # continuous angle per step (deg)

S_USER = (S_MIN is not None) or (S_MAX is not None)
S_MIN = s_lo if S_MIN is None else S_MIN
S_MAX = s_hi if S_MAX is None else S_MAX
print(f"Segment span: S_MIN = {S_MIN:.4f} m, S_MAX = {S_MAX:.4f} m along the rotating tip-to-tip line"
      + ("" if S_USER else " (taken from data; set S_MIN / S_MAX explicitly for physical values)"))

edges = np.linspace(S_MIN, S_MAX, N_SEG + 1)    # same for every timestep
ds = np.diff(edges)
s_mid = 0.5 * (edges[:-1] + edges[1:])
s_grid = np.linspace(S_MIN, S_MAX, 400)         # for the cumulative-thrust curve
print(f"Bin width = {ds[0] * 1e3:.2f} mm  (= {ds[0] / FACE_SIZE_EXPECTED:.1f} face widths at "
      f"{FACE_SIZE_EXPECTED * 1e3:.2f} mm)")

# ---------------- Pass 1: compute ----------------
results = []
thrust_history = []
diag = None
for i in range(N_STEPS):
    d = load_step(i)
    if d is None:
        print(f"Warning: step {i} could not be loaded. Skipping.")
        continue

    area, pressure = d["area"], d["pressure"]
    points = d["points"]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radius = np.hypot(x - hub_x, z - hub_z)     # distance from the hub axis (torque / F_t / F_r only)
    e_span = span_axes[i]                       # tip-to-tip direction at this timestep
    span = (x - hub_x) * e_span[0] + (z - hub_z) * e_span[1]   # coordinate used for the segmentation

    # Faces outside the segment range are clipped into the first/last segment (checked every step)
    n_out = int(np.sum((span < edges[0]) | (span > edges[-1])))
    if n_out:
        print(f"  Warning [t{i}]: {n_out} faces lie outside [S_MIN, S_MAX]; clipped into the first/last segment.")

    # ---- Data sanity checks (first step only) ----
    if not results:
        # Origin moved to the area-weighted centroid so the sign test is not dominated by cancellation
        centroid = np.sum(area[:, None] * points, axis=0) / area.sum()
        score_raw = np.sum(area * np.einsum("ij,ij->i", d["normals_raw"], points - centroid))
        suggested = 1.0 if score_raw > 0 else -1.0
        status = "OK" if suggested == NORMAL_SIGN else f"MISMATCH -> set NORMAL_SIGN = {suggested:+.0f}"
        print(f"  Normals check (raw): sum(n.(p-c) dA) = {score_raw:.3e} -> raw normals point "
              f"{'OUTWARD' if score_raw > 0 else 'INWARD'}; NORMAL_SIGN = {NORMAL_SIGN:+.0f}: {status}")
        n_norm = np.linalg.norm(d["normals_raw"], axis=1)
        print(f"  |n| range: {n_norm.min():.4f} .. {n_norm.max():.4f} (should be ~1)")
        edge_len = np.sqrt(4.0 * np.median(area) / np.sqrt(3.0))   # equilateral-triangle edge length
        print(f"  area min/median/max: {area.min():.3e} / {np.median(area):.3e} / {area.max():.3e}, "
              f"total area = {area.sum():.4e} m^2, faces = {len(area)}")
        print(f"  median face edge (if triangles) = {edge_len * 1e3:.2f} mm, "
              f"sqrt(median area) = {np.sqrt(np.median(area)) * 1e3:.2f} mm "
              f"(expected ~{FACE_SIZE_EXPECTED * 1e3:.2f} mm)")

    normals = d["normals_raw"] * NORMAL_SIGN
    area_vec = normals * area[:, None]

    if not results:
        closure = np.linalg.norm(area_vec.sum(axis=0))
        print(f"  Closure |sum(n dA)| = {closure:.3e} m^2 "
              f"({100 * closure / area.sum():.2e}% of total area; ~0 means a closed patch)")

    # Per-face force: (-p I + tau) . n dA
    if d["visc_force"] is not None:
        viscous = d["visc_force"]
    elif d["grad_u"] is not None:
        tau = mu * (d["grad_u"] + np.transpose(d["grad_u"], (0, 2, 1)))
        viscous = np.einsum("ijk,ik->ij", tau, area_vec)
    else:
        viscous = np.zeros_like(area_vec)
    pressure_f = -pressure[:, None] * area_vec
    force = pressure_f + viscous

    # Local frame about the Y axis through the hub
    rr = np.maximum(radius, 1e-12)
    e_r = np.column_stack([(x - hub_x) / rr, np.zeros_like(rr), (z - hub_z) / rr])
    e_t = np.column_stack([(z - hub_z) / rr, np.zeros_like(rr), -(x - hub_x) / rr])
    F_r = np.einsum("ij,ij->i", force, e_r)
    F_t = np.einsum("ij,ij->i", force, e_t)
    torque_y = (z - hub_z) * force[:, 0] - (x - hub_x) * force[:, 2]
    fy = THRUST_DIR * force[:, 1]

    # Segment index from the span coordinate (tip to tip)
    seg_id = np.clip(np.digitize(span, edges) - 1, 0, N_SEG - 1)

    # np.bincount is much faster than np.add.at for summing face values into segments
    seg_thrust = np.bincount(seg_id, weights=fy, minlength=N_SEG)
    seg_Ft = np.bincount(seg_id, weights=F_t, minlength=N_SEG)
    seg_Fr = np.bincount(seg_id, weights=F_r, minlength=N_SEG)
    seg_torque = np.bincount(seg_id, weights=torque_y, minlength=N_SEG)
    seg_Fxyz = np.column_stack([np.bincount(seg_id, weights=force[:, c], minlength=N_SEG)
                                for c in range(3)])

    # Bin-independent cumulative thrust T(s) = sum of face thrust from the start tip up to s
    order = np.argsort(span)
    cum = np.concatenate([[0.0], np.cumsum(fy[order])])
    T_cum = cum[np.searchsorted(span[order], s_grid, side="right")]

    # ---- Totals and consistency checks ----
    total = force.sum(axis=0)
    thrust = THRUST_DIR * total[1]
    assert np.isclose(seg_thrust.sum(), thrust, rtol=1e-9, atol=1e-12), "segment thrust != total thrust"
    assert np.isclose(seg_torque.sum(), np.sum(radius * F_t), rtol=1e-9, atol=1e-12), \
        "torque_y != sum(r * F_t)"
    print(f"  [t{i} = {TIME_NAMES[i]}] total F = ({total[0]:.4f}, {total[1]:.4f}, {total[2]:.4f}) N | "
          f"pressure Fy = {pressure_f.sum(0)[1]:.4f} | viscous Fy = {viscous.sum(0)[1]:.2e} | "
          f"thrust = {thrust:.4f} N | thrust/{THRUST_TARGET} = {thrust / THRUST_TARGET:.3f} | "
          f"torque_y = {seg_torque.sum():.4e} N.m")
    thrust_history.append(thrust)

    # ---- Per-bin diagnostic table for one timestep ----
    if i == DIAG_STEP:
        up = normals[:, 1] > 0     # faces whose outward normal points to +Y (one blade surface)
        print(f"\n  Per-bin diagnostic, step {i}  (Fy_up / Fy_low = thrust from faces with n_y>0 / n_y<0)")
        print(f"  bin   {SPAN_NAME}_mid[m]  faces    area[m^2]   Fy_up[N]  Fy_low[N]  Fy_tot[N]  mean_p_gauge[Pa]")
        for j in range(N_SEG):
            m = seg_id == j
            if not m.any():
                print(f"  {j + 1:>3}  {s_mid[j]:+.4f}      0  (empty bin)")
                continue
            print(f"  {j + 1:>3}  {s_mid[j]:+.4f}  {m.sum():>6d}  {area[m].sum():.3e}  "
                  f"{fy[m & up].sum():+9.3f}  {fy[m & ~up].sum():+9.3f}  {fy[m].sum():+9.3f}  "
                  f"{pressure[m].mean():+10.2f}")
        rms = np.sqrt(np.mean(seg_thrust ** 2))
        asym = np.sqrt(np.mean((seg_thrust - seg_thrust[::-1]) ** 2)) / max(rms, 1e-300)
        print(f"  Mirror check: RMS(bin k - bin N+1-k) / RMS(bin) = {asym:.3f} "
              f"(small for two identical blades about the hub; ~1 means noise or a wrong reference pressure)")
        print()
        diag = {"x": x, "z": z, "span": span, "fy": fy, "area": area, "step": i, "axis": e_span}

    results.append({
        "step": i, "time": TIME_VALUES[i], "thrust": seg_thrust, "Ft": seg_Ft, "Fr": seg_Fr,
        "torque": seg_torque, "Fxyz": seg_Fxyz, "Tcum": T_cum,
    })
    del d, force, viscous, pressure_f, area_vec, normals

if not results:
    raise SystemExit("No data found.")

# ---------------- Time-average weights ----------------
if WEIGHT_BY_ANGLE:
    wts = step_weights([step_angle[r["step"]] for r in results])
    print(f"\nAngle-based step weights: min {wts.min() * len(wts):.3f}, max {wts.max() * len(wts):.3f} "
          f"(1.000 = uniform spacing)")
else:
    wts = np.ones(len(results)) / len(results)


def wmean(arr):
    """Weighted mean over the timestep axis (axis 0)."""
    return np.tensordot(wts, np.asarray(arr), axes=(0, 0))


# ---------------- Summary ----------------
th = np.array(thrust_history)
mean_thrust = float(wmean(th))
print(f"Mean thrust over {len(th)} steps (weighted): {mean_thrust:.4f} N   (plain mean {th.mean():.4f} N)")
print(f"Min / max thrust: {th.min():.4f} / {th.max():.4f} N")
print(f"Last-step thrust: {th[-1]:.4f} N")
print(f"Mean thrust vs target {THRUST_TARGET} N: error {100 * (mean_thrust / THRUST_TARGET - 1):+.2f}%")

if len(th) > 1 and np.isclose(th[0], th[-1], rtol=1e-6):
    print("Note: first and last steps have identical thrust. If they are the same instant "
          "(start and end of the period), drop one before averaging.")

ratio = mean_thrust / THRUST_TARGET
if abs(ratio + 1) < 0.1:
    print("-> Thrust is equal in size but opposite in sign. Flip NORMAL_SIGN (if the normals "
          "check says MISMATCH) or THRUST_DIR (if the normals are fine and thrust is -Y).")
elif abs(ratio - 1) < 0.02:
    print("-> Thrust matches the target (within 2%).")
elif abs(ratio - 1.22) < 0.1:
    print("-> Too high by ~rho: p is probably already in Pa; set PRESSURE_IS_KINEMATIC = False")
elif abs(ratio - 1 / 1.22) < 0.1:
    print("-> Too low by ~1/rho: p is probably kinematic; set PRESSURE_IS_KINEMATIC = True")
elif abs(ratio - round(ratio)) < 0.1 and round(ratio) > 1:
    print(f"-> Ratio ~{round(ratio)}: patch may contain {round(ratio)} blades.")
else:
    print("-> No simple factor found: check rotor axis, patch selection, or reference pressure.")

# ---------------- Save table ----------------
# Fx, Fy, Fz are lab-frame components (Fy is NOT multiplied by THRUST_DIR); "Thrust" is.
rows = []
for r in results:
    for j in range(N_SEG):
        rows.append({
            "timestep": r["step"], "time": r["time"], "segment": j + 1,
            "s_in": edges[j], "s_out": edges[j + 1], "s_mid": s_mid[j],
            "Thrust": r["thrust"][j], "dThrust_ds": r["thrust"][j] / ds[j],
            "F_tangential": r["Ft"][j], "F_radial": r["Fr"][j],
            "Fx": r["Fxyz"][j, 0], "Fy": r["Fxyz"][j, 1], "Fz": r["Fxyz"][j, 2],
            "Torque_about_Y": r["torque"][j],
        })
csv_path = output_dir / "segment_forces.csv"
try:
    pd.DataFrame(rows).to_csv(csv_path, index=False)
except OSError:
    csv_path = output_dir / f"segment_forces_{int(time.time())}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"segment_forces.csv is locked (open in Excel?). Saved to {csv_path.name} instead.")

# ---------------- Pass 2: sectional thrust plots (frames + combined, light-blue bars + thin dotted section mean) ----------------
# Appearance: LaTeX switch with plain fallback, serif fonts, o-markers, jet gradient shared by all
# timestep curves, legend inside the axes. Applied via rc_context, so the summary / cumulative /
# face-map figures further down keep their normal fonts (labels like "m^2" would break under usetex).


def _latex_ok():
    """True if a LaTeX toolchain is present and a test string renders."""
    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        return False
    try:
        with matplotlib.rc_context({"text.usetex": True}):
            f = plt.figure(figsize=(1, 1))
            f.text(0.5, 0.5, r"$y/R$")
            f.canvas.draw()
            plt.close(f)
        return True
    except Exception:
        return False


use_latex = bool(USE_LATEX)
if use_latex and not _latex_ok():
    print("LaTeX not available: falling back to plain text mode for the sectional-thrust plots.")
    use_latex = False

if use_latex:
    RC = {"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
          "font.size": 18, "axes.labelsize": 18, "legend.fontsize": 16,
          "xtick.labelsize": 16, "ytick.labelsize": 16}
    XLABEL = rf"Segment number (0: $s = {S_MIN:.3f}$ m, {N_SEG}: $s = {S_MAX:.3f}$ m)"
else:
    RC = {"text.usetex": False, "font.size": 12, "axes.labelsize": 13}
    XLABEL = f"Segment number (0: s = {S_MIN:.3f} m, {N_SEG}: s = {S_MAX:.3f} m)"
YLABEL = "Segment thrust (N)"


def _lab(text):
    """Escape underscores for LaTeX mode."""
    return text.replace("_", r"\_") if use_latex else text


all_thrust = np.array([r["thrust"] for r in results])       # (n_steps, N_SEG)
seg_mean_thrust = wmean(all_thrust)                          # mean of EACH segment over time (angle-weighted)
total_curves = len(results)
colors = plt.get_cmap(CMAP_NAME)(np.linspace(0, 1, max(total_curves, 1)))   # one gradient for the whole figure
seg_x = np.arange(N_SEG) + 0.5                 # segment centres on the 0..N_SEG segment-number axis
mean_label = _lab("mean of each segment (over time)")
bar_label = _lab("segment thrust, this timestep")
bar_label_mean = _lab("mean segment thrust (bars)")
dots_label = _lab("segment thrust, each timestep (dots)")
ymin = min(0.0, all_thrust.min()) * 1.05
ymax = max(0.0, all_thrust.max()) * 1.05


def _time_label(r):
    return _lab(f"t = {TIME_NAMES[r['step']]} s")


def _finish_axes(ax):
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_xlim(0, N_SEG)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(0, N_SEG + 1, 5))
    ax.grid(False)


frame_paths = []                               # actual saved paths (the video uses these)
with matplotlib.rc_context(RC):
    # ---- 1) one frame per timestep ----
    for k, r in enumerate(results):
        fig, ax = plt.subplots(figsize=(8, 6))
        if SHOW_BARS:                          # light-blue bars: thrust of each segment at this timestep
            ax.bar(seg_x, all_thrust[k], width=1.0, color=BAR_COLOR, edgecolor=BAR_EDGE,
                   linewidth=BAR_EDGE_WIDTH, zorder=1, label=bar_label)
        ax.plot(seg_x, all_thrust[k], marker="o", markersize=2, linestyle="None",
                color="k", markerfacecolor="k", markeredgecolor="k", zorder=3, label=_time_label(r))
        ax.plot(seg_x, seg_mean_thrust, linestyle=":", linewidth=MEAN_LINE_WIDTH, color="k", zorder=4,
                label=mean_label)               # thin dotted line: mean of each segment over time
        ax.set_title(f"Timestep {r['step']} (t = {TIME_NAMES[r['step']]}): "
                     f"Sectional thrust")
        _finish_axes(ax)
        ax.legend(ncol=1)
        fig.tight_layout()
        frame_paths.append(save_fig(fig, output_dir / f"thrust_timestep_{r['step']}.png", dpi=FRAME_DPI))
        plt.close(fig)

    # ---- 2) all timesteps on one axes ----
    fig, ax = plt.subplots(figsize=(8, 6))
    if SHOW_BARS:                              # light-blue bars: mean thrust of each segment over time
        ax.bar(seg_x, seg_mean_thrust, width=1.0, color=BAR_COLOR, edgecolor=BAR_EDGE,
               linewidth=BAR_EDGE_WIDTH, zorder=1, label=bar_label_mean)
    for k, r in enumerate(results):
        ax.plot(seg_x, all_thrust[k], marker="o", markersize=2, linestyle="None",
                color="k", markerfacecolor="k", markeredgecolor="k", zorder=3,
                label=dots_label if k == 0 else "_nolegend_")
    ax.plot(seg_x, seg_mean_thrust, linestyle=":", linewidth=MEAN_LINE_WIDTH, color="k", zorder=4,
            label=mean_label)                   # thin dotted line: mean of each segment over time
    ax.set_title("Sectional thrust")
    _finish_axes(ax)
    ax.legend(ncol=1)
    fig.tight_layout()
    combined_png = output_dir / f"{COMBINED_NAME}.png"
    save_fig(fig, combined_png, dpi=300, bbox_inches="tight")
    save_fig(fig, output_dir / f"{COMBINED_NAME}.eps", dpi=300, bbox_inches="tight")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
print(f"Saved sectional-thrust plot: {combined_png} (+ .eps), {total_curves} timestep curves "
      f"plus the dotted section mean")

# Summary figure
mean_seg_thrust = seg_mean_thrust
mean_seg_Ft = wmean([r["Ft"] for r in results])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].bar(s_mid, mean_seg_thrust, width=0.9 * ds, color="b")
ax[0].set(title="Mean thrust per segment", xlabel=f"{SPAN_NAME} (m)", ylabel="N")
ax[1].bar(s_mid, mean_seg_Ft, width=0.9 * ds, color="r")
ax[1].set(title="Mean tangential force per segment", xlabel=f"{SPAN_NAME} (m)", ylabel="N")
ax[2].plot([r["step"] for r in results], thrust_history, "o-")
ax[2].axhline(THRUST_TARGET, color="k", linestyle="--", label=f"target {THRUST_TARGET} N")
ax[2].axhline(mean_thrust, color="g", linestyle=":", label=f"mean {mean_thrust:.3f} N")
ax[2].set(title="Total thrust vs timestep", xlabel="timestep", ylabel="N")
ax[2].legend()
for a in ax:
    a.grid(True, linestyle="--", alpha=0.6)
fig.tight_layout()
save_fig(fig, output_dir / "summary.png", dpi=200)
plt.close(fig)

# Cumulative thrust (bin-independent): mean over time with min/max band, plus the slope dT/ds
Tc = np.array([r["Tcum"] for r in results])
Tc_mean = wmean(Tc)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].fill_between(s_grid, Tc.min(0), Tc.max(0), alpha=0.25, label="min-max over time")
ax[0].plot(s_grid, Tc_mean, "b", label="mean")
ax[0].axhline(mean_thrust, color="k", linestyle="--", linewidth=0.8, label="mean total thrust")
ax[0].set(title=f"Cumulative thrust T({SPAN_NAME}) from the start tip up to {SPAN_NAME}",
          xlabel=f"{SPAN_NAME} (m)", ylabel="N")
ax[0].legend()
# slope smoothed over a ~bin-wide odd window, edge-padded so the tips do not drop to zero
win = max(3, int(round(len(s_grid) / N_SEG)))
if win % 2 == 0:
    win += 1
dTds = np.gradient(Tc_mean, s_grid)
dTds_s = np.convolve(np.pad(dTds, win // 2, mode="edge"), np.ones(win) / win, mode="valid")
ax[1].plot(s_grid, dTds_s, "r")
ax[1].axhline(0, color="k", linewidth=0.8)
ax[1].set(title=f"Spanwise loading dT/d{SPAN_NAME} (smoothed)", xlabel=f"{SPAN_NAME} (m)", ylabel="N/m")
for a in ax:
    a.grid(True, linestyle="--", alpha=0.6)
fig.tight_layout()
save_fig(fig, output_dir / "cumulative_thrust.png", dpi=200)
plt.close(fig)

# Face map for the diagnostic step: where do the large per-face forces sit?
if diag is not None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    lim = np.percentile(np.abs(diag["fy"]), 99)
    sc = ax[0].scatter(diag["x"], diag["z"], c=diag["fy"], s=1, cmap="coolwarm",
                       vmin=-lim, vmax=lim, rasterized=True)
    ax[0].set(title=f"Face thrust Fy per face, step {diag['step']} (top view)",
              xlabel="x (m)", ylabel="z (m)", aspect="equal")
    fig.colorbar(sc, ax=ax[0], label="N per face")
    ea = diag["axis"]
    ax[0].plot([hub_x + S_MIN * ea[0], hub_x + S_MAX * ea[0]],
               [hub_z + S_MIN * ea[1], hub_z + S_MAX * ea[1]], "k--", linewidth=0.8)
    perp = np.array([-ea[1], ea[0]])             # segment boundaries: ticks across the tip-to-tip line
    half = 0.05 * (S_MAX - S_MIN)
    for e in edges:
        c = np.array([hub_x, hub_z]) + e * ea
        ax[0].plot([c[0] - half * perp[0], c[0] + half * perp[0]],
                   [c[1] - half * perp[1], c[1] + half * perp[1]], "k-", linewidth=0.6)
    ax[1].hist(diag["span"], bins=np.linspace(S_MIN, S_MAX, 120), weights=diag["area"], color="gray")
    for e in edges:
        ax[1].axvline(e, color="r", linewidth=0.5)
    ax[1].set(title=f"Surface area vs {SPAN_NAME} (red = bin edges)", xlabel=f"{SPAN_NAME} (m)",
              ylabel="area (m^2)")
    fig.tight_layout()
    save_fig(fig, output_dir / "face_map.png", dpi=200)
    plt.close(fig)

# ---------------- Video from plots ----------------
try:
    import imageio.v2 as imageio
    from PIL import Image
except ImportError:
    raise SystemExit("Plots and CSV are saved. For the video run:  "
                     "python -m pip install imageio imageio-ffmpeg pillow  and run the script again.")

first = Image.open(frame_paths[0]).convert("RGB")
w, h = first.size
if w > MAX_VIDEO_WIDTH:
    h = int(h * MAX_VIDEO_WIDTH / w)
    w = MAX_VIDEO_WIDTH
w -= w % 2
h -= h % 2

video_path = output_dir / "thrust_timesteps.mp4"
try:
    writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264",
                                quality=8, macro_block_size=None)
except OSError:
    video_path = output_dir / f"thrust_timesteps_{int(time.time())}.mp4"
    writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264",
                                quality=8, macro_block_size=None)

for fp in frame_paths:
    img = Image.open(fp).convert("RGB").resize((w, h), Image.LANCZOS)
    writer.append_data(np.asarray(img))
writer.close()

print(f"Video saved: {video_path} ({len(frame_paths)} frames at {VIDEO_FPS} fps, "
      f"{len(frame_paths) / VIDEO_FPS:.1f} s)")
print(f"Done. Sectional-thrust plots, summary.png, cumulative_thrust.png, face_map.png, CSV and video "
      f"saved in '{output_dir}'.")
