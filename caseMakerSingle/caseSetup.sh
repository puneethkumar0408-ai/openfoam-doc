#!/usr/bin/env bash
# caseSetup.sh - the shell half of make_rotor_case.py (run it through the .py file)
#   dirs       mkdir 0 constant system
#   dicts      write constant/ and system/ dictionaries
#   convert    fluent3DMeshToFoam <mesh> [scale]
#   boundary   sed edits on constant/polyMesh/boundary (ami1/ami2, cyclicAMI, neighbourPatch, open patch)
#   toposet    topoSet -> cellSets + cellZones "rotordomain" and "statordomain"  (log.topoSet)
#   check      checkMesh -> log.cMesh
# Case values arrive as environment variables set by make_rotor_case.py.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

VER=${WM_PROJECT_VERSION:-2312}; VER=${VER#v}

header() {   # $1 class  $2 object  [$3 location]
  cat <<'EOF' | sed "s/@VER/$(printf '%-4s' "$VER")/"
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  @VER                                  |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
EOF
  printf 'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n' "$1"
  [[ -n ${3:-} ]] && printf '    location    "%s";\n' "$3"
  printf '    object      %s;\n}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n' "$2"
}
footer() { printf '\n// ************************************************************************* //\n'; }

cell_zones() {   # names of the cell zones in constant/polyMesh/cellZones (one per line)
  [[ -f constant/polyMesh/cellZones ]] || return 0
  awk '
    { line=$0; sub(/\r$/, "", line); gsub(/^[ \t]+|[ \t]+$/, "", line) }
    line ~ /^\/\// { next }
    depth==0 && line ~ /^[^ \t{}();\/]+$/ { cand=line }
    depth==0 && line=="{" { blk=cand; if (blk!="FoamFile") all[++na]=blk }
    depth>=1 && blk!="" && blk!="FoamFile" && line ~ /^type[ \t]+cellZone;/ { typed[++nt]=blk; blk="" }
    { o=gsub(/[{]/, "{", line); c=gsub(/[}]/, "}", line); depth+=o-c }
    END {                      # prefer blocks marked "type cellZone"; else every top-level block
      if (nt>0) { for (i=1;i<=nt;i++) print typed[i] }
      else      { for (i=1;i<=na;i++) print all[i] }
    }
  ' constant/polyMesh/cellZones
}

# one cellSet (from a Fluent cellZone) + one cellZone, both called $1, built from the Fluent zone $2
topo_action() {   # $1 new name  $2 source zone in the converted mesh
  printf '    {\n        name    %s;\n        type    cellSet;\n        action  new;\n        source  zoneToCell;\n        zone    %s;\n    }\n\n' "$1" "$2"
  printf '    {\n        name    %s;\n        type    cellZoneSet;\n        action  new;\n        source  setToCellZone;\n        set     %s;\n    }\n\n' "$1" "$1"
}

# ------------------------------------------------------------------------------
step_dirs() { mkdir -p 0 constant system; }

step_dicts() {
  for v in APPLICATION END_TIME DELTA_T WRITE_INTERVAL PURGE_WRITE MAX_CO RHO_INF ROT_ORIGIN ROT_AXIS \
           ROT_OMEGA ROTOR_ZONE STATOR_ZONE NU RAS_MODEL NPROCS HIER_N FORCE_PATCHES; do
    [[ -n ${!v:-} ]] || die "$v not set (run this through make_rotor_case.py)"
  done

  # Which Fluent cell zones become rotordomain / statordomain?
  #   any zone name containing 'rotor'              -> rotordomain
  #   any zone name containing 'stator'/'stationary' -> statordomain   (case-insensitive)
  local zones z src_rotor src_stator have_mesh=no
  mapfile -t zones < <(cell_zones)
  [[ -f constant/polyMesh/cellZones ]] && have_mesh=yes
  src_rotor=$(printf '%s\n' "${zones[@]:-}" | grep -i 'rotor' | head -n1 || true)
  src_stator=$(printf '%s\n' "${zones[@]:-}" | grep -iE 'stator|stationary' | head -n1 || true)
  # no obvious name for one of them: it is any zone that is not the other one
  if [[ -z $src_rotor ]]; then
    for z in "${zones[@]:-}"; do
      if [[ -n $z && $z != "$src_stator" ]]; then src_rotor=$z; break; fi
    done
  fi
  if [[ -z $src_stator ]]; then
    for z in "${zones[@]:-}"; do
      if [[ -n $z && $z != "$src_rotor" ]]; then src_stator=$z; break; fi
    done
  fi
  if [[ $have_mesh == yes ]]; then
    [[ -n $src_rotor  ]] || die "no rotor cellZone found in constant/polyMesh/cellZones (zones: ${zones[*]:-none})"
    [[ -n $src_stator ]] || die "no stator cellZone found in constant/polyMesh/cellZones (zones: ${zones[*]:-none})"
  else                                               # before conversion: placeholders, rewritten after it
    src_rotor=$ROTOR_ZONE; src_stator=$STATOR_ZONE
  fi

  # the rotating zone is always the cellZone created by topoSet
  { header dictionary dynamicMeshDict constant; cat <<EOF
dynamicFvMesh dynamicMotionSolverFvMesh;

motionSolverLibs ( "libfvMotionSolvers.so" );

motionSolver solidBody;

cellZone $ROTOR_ZONE;

solidBodyMotionFunction rotatingMotion;


origin    ($ROT_ORIGIN);
axis      ($ROT_AXIS);
omega     $ROT_OMEGA;

EOF
  footer; } > constant/dynamicMeshDict

  { header dictionary transportProperties constant; cat <<EOF
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] $NU;
EOF
  footer; } > constant/transportProperties

  { header dictionary turbulenceProperties constant; cat <<EOF
simulationType RAS;

RAS
{
    RASModel        $RAS_MODEL;

    turbulence      on;

    printCoeffs     on;
}
EOF
  footer; } > constant/turbulenceProperties

  { header dictionary controlDict system; cat <<EOF
application     $APPLICATION;

startFrom       latestTime;

startTime       0;

stopAt          endTime;

endTime         $END_TIME;

deltaT          $DELTA_T;

writeControl    runTime;

writeInterval   $WRITE_INTERVAL;

purgeWrite      $PURGE_WRITE;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

adjustTimeStep  yes;

maxCo           $MAX_CO;

functions
{

  yPlus1 // Y+ < 300
  {
      type        yPlus;
      libs       ("fieldFunctionObjects");
      writeControl    runTime;
      writeInterval   $WRITE_INTERVAL;
  }


  forces
  {
    type forces;
    libs ("libforces.so");
    patches ($FORCE_PATCHES); // sum the forces (thrust [kg] = force_z/g) and moments on those patches
    writeControl timeStep;
    writeInterval 1;
    p p;
    U U;
    log true;
    rhoInf $RHO_INF;
    rho rhoInf;
    CofR ($ROT_ORIGIN); //centreOfRotation
  }

}

EOF
  footer; } > system/controlDict

  { header dictionary fvSchemes system; cat <<'EOF'
ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwindV grad(U);
    div(phi,k)      Gauss upwind;
    div(phi,omega)  Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear limited 0.5;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited 0.5;
}

wallDist
{
    method meshWave;
}

EOF
  footer; } > system/fvSchemes

  { header dictionary fvSolution system; cat <<'EOF'
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-08;
        relTol          0.01;
        smoother        GaussSeidel;
        nCellsInCoarsestLevel 20;
    }

    pFinal
    {
        solver          GAMG;
        tolerance       1e-08;
        relTol          0;
        smoother        GaussSeidel;
    }


    "pcorr.*"
    {
        solver          GAMG;
        tolerance       1e-2;
        relTol          0;
        smoother                DICGaussSeidel;
        cacheAgglomeration no;
        maxIter                 50;
    }


    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-07;
        relTol          0.001;
    }

    k
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-07;
        relTol          0.001;
    }

    omega
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-07;
        relTol          0.001;
    }


    "(U|k|omega)Final"
    {
        $U;
        tolerance       1e-08;
        relTol          0;
    }
}

PIMPLE
{
    nNonOrthogonalCorrectors 2;
    nCorrectors          2;
    nOuterCorrectors     10;
        relaxOnFinalIter false;

    pRefCell             0;
    pRefValue            0;

    residualControl
    {
        U
        {
            tolerance   1e-05;
            relTol      0;
        }
        p
        {
            tolerance   1e-05;
            relTol      0;
        }
        omega
        {
            tolerance   1e-05;
            relTol      0;
        }
        k
        {
            tolerance   1e-05;
            relTol      0;
        }
    }
}

EOF
  footer; } > system/fvSolution

  { header dictionary decomposeParDict system; cat <<EOF
numberOfSubdomains $NPROCS;

method          scotch; //hierarchical;

coeffs
{
    n           ($HIER_N);
}
EOF
  footer; } > system/decomposeParDict

  # topoSetDict: rotordomain + statordomain, each as cellSet and as cellZone
  { header dictionary topoSetDict system
    echo "actions"; echo "("
    topo_action "$ROTOR_ZONE"  "$src_rotor"
    topo_action "$STATOR_ZONE" "$src_stator"
    echo ");"
    footer; } > system/topoSetDict
  echo "      dictionaries written ($ROTOR_ZONE <- $src_rotor, $STATOR_ZONE <- $src_stator, forces on: $FORCE_PATCHES)"
}

# ------------------------------------------------------------------------------
step_convert() {
  local mesh=${1:?mesh file} scale=${2:-} args=()
  command -v fluent3DMeshToFoam >/dev/null || die "fluent3DMeshToFoam not found - source the OpenFOAM bashrc first"
  [[ -f $mesh ]] || die "mesh file '$mesh' not found in $PWD"
  [[ -n $scale ]] && args=(-scale "$scale")          # no scaling unless asked for
  echo ">>> fluent3DMeshToFoam ${args[*]} $mesh"
  fluent3DMeshToFoam "${args[@]}" "$mesh" > log.fluent3DMeshToFoam 2>&1 \
    || die "fluent3DMeshToFoam failed - see log.fluent3DMeshToFoam"
  [[ -f constant/polyMesh/boundary ]] || die "no constant/polyMesh/boundary was written"
}

step_boundary() {
  local F=constant/polyMesh/boundary open=${OPEN_KEY:?OPEN_KEY not set}
  cp "$F" "$F.orig"
  sed -i '/physicalType/d' "$F"

  # 1. Fluent interface names -> ami1 / ami2   (ami1-..., ami-1-..., ami_2..., and ...src / ...trg as fallback)
  sed -i -E 's/^([[:space:]]*)ami[-_ ]?1[^[:space:]{}();]*[[:space:]]*$/\1ami1/I;
             s/^([[:space:]]*)ami[-_ ]?2[^[:space:]{}();]*[[:space:]]*$/\1ami2/I' "$F"
  grep -qE '^[[:space:]]*ami1[[:space:]]*$' "$F" || \
    sed -i -E 's/^([[:space:]]*)[^[:space:]{}();]*ami[^[:space:]{}();]*src[^[:space:]{}();]*[[:space:]]*$/\1ami1/I' "$F"
  grep -qE '^[[:space:]]*ami2[[:space:]]*$' "$F" || \
    sed -i -E 's/^([[:space:]]*)[^[:space:]{}();]*ami[^[:space:]{}();]*(trg|tgt|target)[^[:space:]{}();]*[[:space:]]*$/\1ami2/I' "$F"
  for p in ami1 ami2; do
    grep -qE "^[[:space:]]*$p[[:space:]]*\$" "$F" || \
      die "no patch could be renamed to $p. Patches in the mesh: $(awk '/^[[:space:]]*[A-Za-z][^[:space:]{}();]*[[:space:]]*$/ && $1!="FoamFile" {printf "%s ", $1}' "$F.orig")"
  done

  # 2. ami1 / ami2 -> cyclicAMI (replace type, drop old inGroups, add the new one)
  for p in ami1 ami2; do
    sed -i "/^[[:space:]]*$p[[:space:]]*\$/,/}/ {
        /inGroups/d
        s/type[[:space:]]*[a-zA-Z]*;/type            cyclicAMI;\n        inGroups        1(cyclicAMI);/
    }" "$F"
  done

  # 3. transform + neighbourPatch after startFace
  sed -i '/^[[:space:]]*ami1[[:space:]]*$/,/}/ s/\(startFace[[:space:]]*[0-9]*;\)/\1\n        transform       noOrdering;\n        neighbourPatch  ami2;/' "$F"
  sed -i '/^[[:space:]]*ami2[[:space:]]*$/,/}/ s/\(startFace[[:space:]]*[0-9]*;\)/\1\n        transform       noOrdering;\n        neighbourPatch  ami1;/' "$F"

  # 4. the open boundary arrives as a wall -> plain patch
  sed -i -E "/^[[:space:]]*[^[:space:]{}();]*${open}[^[:space:]{}();]*[[:space:]]*\$/I,/}/ {
      /inGroups/d
      s/type[[:space:]]*wall;/type            patch;/
  }" "$F"
  echo "      boundary file corrected (original: $F.orig)"
}

step_toposet() {
  command -v topoSet >/dev/null || die "topoSet not found - source the OpenFOAM bashrc first"
  [[ -f system/topoSetDict ]] || die "system/topoSetDict missing"
  echo ">>> topoSet -> log.topoSet"
  topoSet > log.topoSet 2>&1 || die "topoSet failed - see log.topoSet"
  for z in "${ROTOR_ZONE:?ROTOR_ZONE not set}" "${STATOR_ZONE:?STATOR_ZONE not set}"; do
    grep -qE "^[[:space:]]*$z[[:space:]]*\$" constant/polyMesh/cellZones \
      || die "cellZone '$z' was not created - see log.topoSet"
  done
  echo "      cellSets + cellZones created: $ROTOR_ZONE $STATOR_ZONE"
}

step_check() {
  command -v checkMesh >/dev/null || die "checkMesh not found"
  echo ">>> checkMesh -> log.cMesh"
  checkMesh > log.cMesh 2>&1 && tail -n 3 log.cMesh || echo "WARN: checkMesh reported problems (see log.cMesh)"
}

case "${1:-}" in
  dirs)     step_dirs ;;
  dicts)    step_dicts ;;
  convert)  shift; step_convert "$@" ;;
  boundary) step_boundary ;;
  toposet)  step_toposet ;;
  check)    step_check ;;
  *) die "usage: $0 dirs | dicts | convert <mesh> <scale> | boundary | toposet | check" ;;
esac
