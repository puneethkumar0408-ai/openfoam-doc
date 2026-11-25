#!/usr/bin/env python3
"""
make_rotor_case.py - build a single-rotor (rotor/stator, cyclicAMI) OpenFOAM case for pimpleFoam from rotor.msh

Work is split between the two files in this folder:
  caseSetup.sh        shell: mkdir, constant/ + system/ dictionaries, fluent3DMeshToFoam,
                       sed edits of the boundary file, topoSet, checkMesh
  make_rotor_case.py   python: settings, reads the corrected boundary file, writes the
                       0/ fields (U p k omega nut) patch by patch, drives the shell steps

Order:
  1. dirs + constant/ system/ files            (shell)
  2. fluent3DMeshToFoam                        (shell)
  3. boundary corrections: ami1/ami2, cyclicAMI, transform noOrdering,
     neighbourPatch, open patch -> patch       (shell, sed)
  4. 0/ fields from the corrected boundary     (python)
  5. re-write dictionaries that depend on the mesh: forces patches,
     rotor/stator cellZones, topoSetDict       (shell)
  6. topoSet: cellSets + cellZones rotordomain and statordomain; the
     rotordomain cellZone is the one used in constant/dynamicMeshDict   (shell)
  7. checkMesh, only if requested              (shell)

Usage:   python3 make_rotor_case.py
         asks for mesh file, case directory, unit scale, rpm, number of subdomains and checkMesh.
         (optional shortcuts that pre-answer a question: MESH -c -s --rpm --nprocs --check/--no-check --defaults)
         (-s is optional: unit scaling only; the mesh is always converted)
"""
import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path

# =============================== case settings ================================
CFG = dict(
    # controlDict
    APPLICATION="pimpleFoam", END_TIME="1", DELTA_T="0.000001",
    PURGE_WRITE="36", MAX_CO="50", RHO_INF="1.22",
    # solid-body rotation (dynamicMeshDict + forces CofR)
    ROT_ORIGIN="0 0 0", ROT_AXIS="0 1 0",
    # names of the cellSets/cellZones that topoSet creates (rotordomain goes into dynamicMeshDict)
    ROTOR_ZONE="rotordomain", STATOR_ZONE="statordomain",
    # transport / turbulence
    NU="1.5e-05", RAS_MODEL="kOmegaSST",
)
# asked at run time (defaults offered at the prompt); everything derived from them is computed below
RPM_DEFAULT = "-5000"       # sign = rotation direction about ROT_AXIS (right-hand rule)
NPROCS_DEFAULT = "112"
WRITE_DEG = 5               # one output every WRITE_DEG degrees of rotor rotation
# initial / boundary values for the 0/ files
K, OMEGA, P_IN, P_OUT = "0.0054", "36", "0", "0"
ROUGH_H, ROUGH_C, ROUGH_F = "3.84e-05", "0.75", "1"

# how patches are recognised (case-insensitive substring of the patch name)
INLET_KEY, OUTLET_KEY, OPEN_KEY = "inlet", "outlet", "external"
TOPO_TYPES = {"empty", "symmetry", "symmetryPlane", "wedge", "cyclic", "cyclicSlip"}

VERSION = os.environ.get("WM_PROJECT_VERSION", "2312").lstrip("v")
BANNER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  @VER@                                  |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
""".replace("@VER@", f"{VERSION:<4}")
RULE = "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
FOOTER = "\n// ************************************************************************* //\n"


def die(msg):
    sys.exit(f"ERROR: {msg}")


def ask(prompt, default, cast, given=None, interactive=True, shown=None):
    """Value from the command line if given, else ask on the terminal (Enter = default).
    cast() validates/converts; a bad flag value is fatal, a bad typed answer is asked again."""
    while True:
        from_flag = given is not None
        if from_flag:
            raw, given = str(given), None
        elif interactive:
            try:
                raw = input(f"{prompt} [{shown if shown is not None else default}]: ").strip() or str(default)
            except EOFError:
                raw = str(default)
        else:
            raw = str(default)
        try:
            return cast(raw)
        except ValueError as e:
            if from_flag or not interactive:
                die(f"{prompt}: invalid value '{raw}' ({e})")
            print(f"  -> {e}. Try again.", file=sys.stderr)


def as_mesh(s):
    p = Path(s).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"file not found: {p}")
    return p


def as_scale(s):
    if s.strip().lower() in ("", "none", "no", "n", "-"):
        return ""
    v = float(s)
    if v <= 0:
        raise ValueError("scale must be positive")
    return s.strip()


def as_yes_no(s):
    v = s.strip().lower()
    if v in ("y", "yes"):
        return True
    if v in ("n", "no"):
        return False
    raise ValueError("answer y or n")


def as_rpm(s):
    v = float(s)
    if v == 0:
        raise ValueError("speed must not be 0")
    return v


def as_count(s):
    v = int(s)
    if v < 1:
        raise ValueError("must be at least 1")
    return v


def as_word(s):
    s = s.strip()
    if not s or re.search(r"[\s;]", s):
        raise ValueError("must be a single word (no spaces or ';')")
    return s


def as_pos_num(s):
    if float(s) <= 0:
        raise ValueError("must be positive")
    return s.strip()                                    # keep the user's formatting (e.g. 0.000001)


def as_nonneg_int(s):
    v = int(s)
    if v < 0:
        raise ValueError("must be 0 or more")
    return str(v)


def hier_factors(n):
    """n = a*b*c with the three factors as equal as possible (for the hierarchical coeffs)."""
    best = (n, 1, 1)
    for a in range(1, n + 1):
        if n % a:
            continue
        for b in range(a, n // a + 1):
            if (n // a) % b:
                continue
            t = (a, b, n // a // b)
            if t[2] >= b and max(t) - min(t) < max(best) - min(best):
                best = t
    return " ".join(str(x) for x in sorted(best, reverse=True))


# ============================= boundary file (read) ===========================
def read_boundary(case):
    """[(patchName, patchType), ...] in file order, from constant/polyMesh/boundary."""
    t = (case / "constant/polyMesh/boundary").read_text()
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    t = re.sub(r"FoamFile\s*\{.*?\}", "", t, count=1, flags=re.S)
    out = []
    for m in re.finditer(r"(\S+)\s*\{([^}]*)\}", t):
        ty = re.search(r"\btype\s+(\S+?);", m.group(2))
        if ty:
            out.append((m.group(1), ty.group(1)))
    if not out:
        die("no patches found in constant/polyMesh/boundary")
    return out


def role(name, ptype):
    low = name.lower()
    if name in ("ami1", "ami2"):
        if ptype != "cyclicAMI":
            die(f"patch {name} has type {ptype}, expected cyclicAMI (boundary step failed?)")
        return "ami"
    if ptype in TOPO_TYPES:
        return ptype
    if OPEN_KEY in low:
        return "open"
    if INLET_KEY in low:
        return "inlet"
    if OUTLET_KEY in low:
        return "outlet"
    if ptype == "wall":
        return "wall"
    die(f"cannot decide the boundary condition for patch '{name}' (type {ptype}); "
        f"adjust INLET_KEY / OUTLET_KEY / OPEN_KEY at the top of make_rotor_case.py")


# =============================== 0/ field files ===============================
def block(pairs, width=16):
    return "".join(f"        {k:<{width}}{v};\n" for k, v in pairs)


def patch_bc(field, r):
    zero = {"U": "(0 0 0)", "p": "0", "k": K, "omega": OMEGA, "nut": "0"}[field]
    if r == "ami":
        return block([("type", "cyclicAMI"), ("value", f"uniform {zero}")])
    if r in TOPO_TYPES:
        return block([("type", r)])
    if field == "U":
        return {
            "inlet":  block([("type", "pressureInletVelocity"), ("value", "uniform (0 0 0)")]),
            "outlet": block([("type", "pressureInletOutletVelocity"), ("value", "uniform (0 0 0)")]),
            "open":   block([("type", "pressureInletOutletVelocity"), ("value", "uniform (0 0 0)")]),
            "wall":   block([("type", "movingWallVelocity"), ("value", "uniform (0 0 0)")]),
        }[r]
    if field == "p":
        return {
            "inlet":  block([("type", "totalPressure"), ("p0", f"uniform {P_IN}"), ("value", f"uniform {P_IN}")]),
            "outlet": block([("type", "fixedValue"), ("value", f"uniform {P_OUT}")]),
            "open":   block([("type", "totalPressure"), ("p0", f"uniform {P_OUT}"), ("value", f"uniform {P_OUT}")]),
            "wall":   block([("type", "zeroGradient")]),
        }[r]
    if field in ("k", "omega"):
        v = K if field == "k" else OMEGA
        if r == "wall":
            wall = [("type", "kqRWallFunction")] if field == "k" else \
                   [("type", "omegaWallFunction"), ("blending", "binomial"), ("n", "2")]
            return block(wall + [("value", f"uniform {v}")])
        return block([("type", "inletOutlet"), ("inletValue", f"uniform {v}"), ("value", f"uniform {v}")])
    if field == "nut":
        if r == "wall":
            return block([("type", "nutURoughWallFunction"), ("roughnessHeight", ROUGH_H),
                          ("roughnessConstant", ROUGH_C), ("roughnessFactor", ROUGH_F),
                          ("value", "uniform 0")], width=20)
        return block([("type", "calculated"), ("value", "uniform 0")])


def write_fields(case, patches):
    specs = [("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)"),
             ("p", "volScalarField", "[0 2 -2 0 0 0 0]", "0"),
             ("k", "volScalarField", "[0 2 -2 0 0 0 0]", K),
             ("omega", "volScalarField", "[0 0 -1 0 0 0 0]", OMEGA),
             ("nut", "volScalarField", "[0 2 -1 0 0 0 0]", "0")]
    roles = [(n, role(n, t)) for n, t in patches]
    for f, cls, dims, internal in specs:
        head = (BANNER + "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
                '    arch        "LSB;label=32;scalar=64";\n'
                f'    class       {cls};\n    location    "0";\n    object      {f};\n}}\n' + RULE)
        body = "".join(f"    {n}\n    {{\n{patch_bc(f, r)}    }}\n" for n, r in roles)
        (case / "0" / f).write_text(
            head + f"dimensions      {dims};\n\ninternalField   uniform {internal};\n\n"
            f"boundaryField\n{{\n{body}}}\n" + FOOTER)
    return roles


# ==================================== main ====================================
def main():
    ap = argparse.ArgumentParser(
        description="Build a single-rotor AMI case. Everything is asked on the terminal; "
                    "the options below are optional shortcuts that pre-answer a question.")
    ap.add_argument("mesh", nargs="?", help="Fluent .msh file")
    ap.add_argument("-c", "--case", help="case directory")
    ap.add_argument("-s", "--scale", help="mesh unit scale, e.g. 0.001 for mm")
    ap.add_argument("--rpm", help="rotor speed in rpm")
    ap.add_argument("--nprocs", help="number of decomposition subdomains")
    ap.add_argument("--end-time", help="controlDict endTime [s]")
    ap.add_argument("--delta-t", help="controlDict deltaT [s]")
    ap.add_argument("--write-deg", help="write one output every this many degrees of rotation")
    ap.add_argument("--purge-write", help="controlDict purgeWrite (0 = keep all)")
    ap.add_argument("--max-co", help="controlDict maxCo")
    ap.add_argument("--rho-inf", help="rhoInf for the forces function object")
    ap.add_argument("--check", dest="check", action="store_const", const="y", help="run checkMesh")
    ap.add_argument("--no-check", dest="check", action="store_const", const="n", help="skip checkMesh")
    ap.add_argument("--defaults", action="store_true", help="do not ask; use defaults for anything not given")
    a = ap.parse_args()
    ask_now = not a.defaults
    if ask_now:
        try:                                    # tab-completion of file names at the prompts
            import readline
            readline.set_completer_delims(" \t\n")
            readline.parse_and_bind("tab: complete")
        except ImportError:
            pass

    helper = Path(__file__).resolve().parent / "caseSetup.sh"
    if not helper.is_file():
        die(f"shell helper missing: {helper}")

    print("=== single-rotor case setup (press Enter to accept the value in brackets) ===")
    mesh = ask("Fluent mesh file", "rotor.msh", as_mesh, a.mesh, ask_now)
    case = Path(ask("Case directory", str(mesh.parent), lambda s: str(Path(s).expanduser().resolve()),
                    a.case, ask_now)).resolve()
    scale = ask("Mesh unit scale (0.001 if the mesh is in mm)", "", as_scale, a.scale, ask_now, shown="no scaling")
    rpm = ask("Rotor speed in rpm (negative = reverse direction)", RPM_DEFAULT, as_rpm, a.rpm, ask_now)
    nprocs = ask("Number of subdomains for decomposition", NPROCS_DEFAULT, as_count, a.nprocs, ask_now)
    print(f"--- controlDict settings for {CFG['APPLICATION']} (Enter = default) ---")
    end_time = ask("endTime [s]", CFG["END_TIME"], as_pos_num, a.end_time, ask_now)
    delta_t = ask("deltaT [s] (initial time step)", CFG["DELTA_T"], as_pos_num, a.delta_t, ask_now)
    write_deg = ask("Write output every N degrees of rotor rotation", WRITE_DEG, as_pos_num, a.write_deg, ask_now)
    purge_write = ask("purgeWrite (number of time dirs kept, 0 = keep all)", CFG["PURGE_WRITE"],
                      as_nonneg_int, a.purge_write, ask_now)
    max_co = ask("maxCo", CFG["MAX_CO"], as_pos_num, a.max_co, ask_now)
    rho_inf = ask("rhoInf for forces [kg/m3]", CFG["RHO_INF"], as_pos_num, a.rho_inf, ask_now)
    do_check = ask("Run checkMesh at the end? (y/n)", "y", as_yes_no, a.check, ask_now)

    omega = rpm * 2 * math.pi / 60                      # rpm -> rad/s (sign kept)
    interval = (2 * math.pi / abs(omega)) * float(write_deg) / 360   # time for write_deg degrees of rotation
    CFG.update(ROT_OMEGA=f"{omega:.6g}", NPROCS=str(nprocs), HIER_N=hier_factors(nprocs),
               WRITE_INTERVAL=f"{interval:.4e}", END_TIME=end_time,
               DELTA_T=delta_t, PURGE_WRITE=purge_write, MAX_CO=max_co, RHO_INF=rho_inf)
    print("\n--- summary ---")
    print(f"  mesh          {mesh}")
    print(f"  case dir      {case}" + ("   (files in 0/ constant/ system/ will be overwritten)"
                                      if any((case / d).exists() for d in ("0", "constant", "system")) else ""))
    print(f"  scale         {scale or 'none'}")
    print(f"  rotor         {rpm:g} rpm = {CFG['ROT_OMEGA']} rad/s")
    print(f"  decomposition {nprocs} subdomains")
    print(f"  solver        {CFG['APPLICATION']}  (fixed)")
    print(f"  time          endTime {end_time} s, deltaT {delta_t} s, maxCo {max_co}")
    print(f"  write every   {write_deg} deg of rotation = {CFG['WRITE_INTERVAL']} s, purgeWrite {purge_write}")
    print(f"  rhoInf        {rho_inf}")
    print(f"  topoSet       {CFG['ROTOR_ZONE']}, {CFG['STATOR_ZONE']}  (dynamicMeshDict uses {CFG['ROTOR_ZONE']})")
    print(f"  checkMesh     {'yes' if do_check else 'no'}")
    if ask_now and not ask("Start? (y/n)", "y", as_yes_no, None, True):
        sys.exit("aborted, nothing was written")
    case.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, **CFG, "OPEN_KEY": OPEN_KEY, "FORCE_PATCHES": "walls"}

    def sh(*args):
        r = subprocess.run(["bash", str(helper), *args], cwd=case, env=env)
        if r.returncode:
            sys.exit(r.returncode)

    print(f">>> case: {case}   mesh: {mesh}")
    print(">>> 1/6 directories + constant/ system/ files        [shell]")
    sh("dirs")
    sh("dicts")

    print(">>> 2/6 mesh conversion                              [shell]")
    sh("convert", str(mesh), *([scale] if scale else []))

    print(">>> 3/6 boundary corrections                         [shell/sed]")
    sh("boundary")

    print(">>> 4/6 0/ fields from the boundary file             [python]")
    roles = write_fields(case, read_boundary(case))
    for n, r in roles:
        print(f"      {n:<28} {r}")

    print(">>> 5/6 mesh-dependent dictionaries                  [shell]")
    env["FORCE_PATCHES"] = " ".join(n for n, r in roles if r == "wall") or "walls"
    sh("dicts")

    print(f">>> 6/6 topoSet ({CFG['ROTOR_ZONE']}, {CFG['STATOR_ZONE']})".ljust(58) + "[shell]")
    sh("toposet")

    if do_check:
        sh("check")
    print(">>> case ready (ran fluent3DMeshToFoam, topoSet" + (" and checkMesh)" if do_check else ")"))


if __name__ == "__main__":
    main()
