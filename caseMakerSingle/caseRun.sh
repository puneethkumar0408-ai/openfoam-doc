#!/usr/bin/env bash
# =============================================================================
# caseRun.sh - run a case prepared by make_rotor_case.py
#
#   topoSet  ->  decomposePar  ->  <solver> -parallel  ->  redistributePar -reconstruct
#
# Every step writes its own log in the case directory:
#   log.topoSet   log.decomposePar   log.<solver>   log.redistributePar
# The script stops at the first step that fails (its log tail is shown).
#
# Number of processors : numberOfSubdomains in system/decomposeParDict
# Solver               : application in system/controlDict
#
# Usage:  ./caseRun.sh [-c caseDir] [-f] [step ...]
#   steps   any of: topoSet decompose solve reconstruct      (default: all four, in order)
#   -c      case directory                                    (default: current directory)
#   -f      pass -force to decomposePar (deletes existing processor* directories first)
# Optional environment:
#   MPI_ARGS    extra mpirun arguments, e.g. MPI_ARGS="--hostfile hosts --bind-to core"
#   RECON_OPTS  extra redistributePar options, e.g. RECON_OPTS="-latestTime"
# Examples:
#   ./caseRun.sh                      # everything
#   ./caseRun.sh solve reconstruct    # continue after an earlier decompose
# =============================================================================
set -euo pipefail

CASE_DIR=.; FORCE=
while getopts "c:fh" o; do
  case $o in
    c) CASE_DIR=$OPTARG ;;
    f) FORCE=-force ;;
    *) sed -n '2,25p' "$0"; exit 0 ;;
  esac
done
shift $((OPTIND - 1))
STEPS=("$@"); (( ${#STEPS[@]} )) || STEPS=(topoSet decompose solve reconstruct)

die() { echo "ERROR: $*" >&2; exit 1; }

cd "$CASE_DIR"
[[ -f system/controlDict && -f system/decomposeParDict ]] || die "$PWD is not a prepared case (system/controlDict or decomposeParDict missing)"
for c in topoSet decomposePar redistributePar mpirun; do
  command -v "$c" >/dev/null || die "$c not found - source the OpenFOAM bashrc (and check mpirun)"
done

NPROCS=$(awk '/^[[:space:]]*numberOfSubdomains/ {gsub(";","",$2); print $2; exit}' system/decomposeParDict)
SOLVER=$(awk '/^[[:space:]]*application/ {gsub(";","",$2); print $2; exit}' system/controlDict)
[[ $NPROCS =~ ^[0-9]+$ ]] || die "could not read numberOfSubdomains from system/decomposeParDict"
[[ -n $SOLVER ]]          || die "could not read application from system/controlDict"
command -v "$SOLVER" >/dev/null || die "solver '$SOLVER' not found"

echo "case: $PWD   solver: $SOLVER   processors: $NPROCS   steps: ${STEPS[*]}"

run() {   # run <title> <logfile> <command...>
  local title=$1 log=$2; shift 2
  echo ">>> $title  ->  $log"
  local t0=$SECONDS
  if "$@" > "$log" 2>&1; then
    echo "    finished in $((SECONDS - t0)) s"
  else
    echo "    FAILED after $((SECONDS - t0)) s - last lines of $log:" >&2
    tail -n 15 "$log" >&2
    exit 1
  fi
}

for step in "${STEPS[@]}"; do
  case $step in
    topoSet)
      [[ -f system/topoSetDict ]] || die "system/topoSetDict missing"
      run "topoSet" log.topoSet topoSet
      ;;
    decompose)
      if [[ -z $FORCE ]] && compgen -G "processor*" >/dev/null; then
        die "processor* directories already exist - remove them or use -f"
      fi
      run "decomposePar" log.decomposePar decomposePar $FORCE
      n=$(find . -maxdepth 1 -type d -name 'processor[0-9]*' | wc -l)
      (( n == NPROCS )) || die "decomposePar made $n processor directories, expected $NPROCS"
      ;;
    solve)
      [[ -d processor0 ]] || die "no processor directories - run the decompose step first"
      # shellcheck disable=SC2086
      run "$SOLVER ($NPROCS processors)" "log.$SOLVER" mpirun ${MPI_ARGS:-} -np "$NPROCS" --oversubscribe "$SOLVER" -parallel
      ;;
    reconstruct)
      [[ -d processor0 ]] || die "no processor directories to reconstruct"
      # shellcheck disable=SC2086
      run "redistributePar -reconstruct" log.redistributePar \
          mpirun ${MPI_ARGS:-} -np "$NPROCS" redistributePar -reconstruct -parallel ${RECON_OPTS:-}
      ;;
    *) die "unknown step '$step' (use: topoSet decompose solve reconstruct)" ;;
  esac
done

echo "all requested steps finished"
