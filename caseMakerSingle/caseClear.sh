#!/usr/bin/env bash
# =============================================================================
# caseClear.sh - clear a case directory, keeping only .msh, .py and .sh files
#
# Deletes everything else in the case directory: 0/ constant/ system/ processor*/
# time directories, log.* files, boundary backups, ... (top-level .msh, .py and
# .sh files stay, including this script).
#
# Usage:  ./caseClear.sh [-c caseDir] [-y] [-n]
#   -c  case directory                       (default: current directory)
#   -y  do not ask for confirmation
#   -n  dry run: only list what would be deleted
#
# Safety: it refuses to run unless the directory contains a .msh file AND looks
# like an OpenFOAM case (0, constant, system, processor*, or log.* present), and
# it never runs in / or your home directory.
# =============================================================================
set -euo pipefail

CASE_DIR=.; YES=no; DRY=no
while getopts "c:ynh" o; do
  case $o in
    c) CASE_DIR=$OPTARG ;; y) YES=yes ;; n) DRY=yes ;;
    *) sed -n '2,/^# =====*$/p' "$0"; exit 0 ;;
  esac
done
die() { echo "ERROR: $*" >&2; exit 1; }

# Files to keep (top level only): .msh, .py, .sh
KEEP_EXPR=( \( -type f \( -iname '*.msh' -o -iname '*.py' -o -iname '*.sh' \) \) )

cd "$CASE_DIR" || die "cannot enter $CASE_DIR"
here=$(pwd -P)
[[ $here != / && $here != "${HOME:-/nonexistent}" ]] || die "refusing to clear $here"

# Safety check still relies on a mesh file being present
mapfile -t mesh < <(find . -maxdepth 1 -type f -iname '*.msh' -printf '%f\n' | sort)
(( ${#mesh[@]} )) || die "no .msh file in $here - refusing (wrong directory?)"

mapfile -t keep < <(find . -maxdepth 1 "${KEEP_EXPR[@]}" -printf '%f\n' | sort)
mapfile -d '' -t del < <(find . -mindepth 1 -maxdepth 1 ! "${KEEP_EXPR[@]}" -printf '%f\0' | sort -z)
if (( ${#del[@]} == 0 )); then
  echo "$here is already clean (only: ${keep[*]})"; exit 0
fi
if ! { [[ -d 0 || -d constant || -d system ]] || compgen -G "processor*" >/dev/null || compgen -G "log.*" >/dev/null; }; then
  die "$here does not look like an OpenFOAM case (no 0, constant, system, processor* or log.*) - refusing"
fi

echo "case: $here"
echo "keeping : ${keep[*]}"
echo "deleting (${#del[@]} items):"
for d in "${del[@]}"; do
  if [[ -d $d && ! -L $d ]]; then echo "   $d/"; else echo "   $d"; fi
done

[[ $DRY == yes ]] && { echo "(dry run - nothing deleted)"; exit 0; }
if [[ $YES != yes ]]; then
  read -r -p "Delete these? [y/N]: " ans || ans=n
  [[ $ans == y || $ans == Y || $ans == yes ]] || { echo "aborted, nothing deleted"; exit 1; }
fi
rm -rf -- "${del[@]}"
echo "done - left: $(ls -A | tr '\n' ' ')"
