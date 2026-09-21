#!/usr/bin/env bash
# Rebuilds the renders of the standard drone of chapter 4:
#
#   standard_drone_actuation.png          fig:drone-actuators, panel b (actuated pose over the neutral one)
#   standard_drone_neutral_3quarter.png   fig:drone-standard, panel a (same camera as the actuation figure)
#   standard_drone_neutral_top.png        fig:drone-standard, panel b (top view, nose up)
#
#   1. Genesis (docker) poses the standard drone and dumps its visual geometry,
#   2. Blender (Cycles) renders the drone, its shadow and, for the actuation figure, the neutral surfaces,
#   3. PIL composes the passes on white.
#
# Needs the mygenesis docker image, python3 with numpy + pillow, and Blender >= 4.2;
# the portable archive of blender.org is enough, no installation:
#
#   BLENDER=/path/to/blender-4.2.x-linux-x64/blender tesis/images/04_Simulation_Environment/make_drone_figures.sh
#
# FIGS="actuation" (or neutral_3quarter, neutral_top) rebuilds only some of them;
# extra arguments go to every Blender run (e.g. --draft).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
BLENDER="${BLENDER:-blender}"
FIGS="${FIGS:-actuation neutral_3quarter neutral_top}"
WORK="$(mktemp -d -t drone_figures_XXXXXX)"
REL="tesis/images/04_Simulation_Environment"

docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$REPO":/workspace/bind -v "$WORK":/work \
    -e PYTHONPATH=/workspace/bind/src -w /workspace/bind \
    mygenesis:latest python "$REL/drone_dump_geometry.py" /work/geometry.npz

for fig in $FIGS; do
    case "$fig" in
        actuation)        view=(--pose actuated --ghost) ;;
        neutral_3quarter) view=(--pose neutral) ;;
        neutral_top)      view=(--pose neutral --top) ;;
        *) echo "unknown figure: $fig" >&2; exit 1 ;;
    esac
    "$BLENDER" -b --python "$HERE/drone_render_blender.py" -- "$WORK/geometry.npz" "$WORK/$fig" "${view[@]}" "$@"
    python3 "$HERE/drone_compose.py" "$WORK/$fig" "$HERE/standard_drone_$fig.png"
done
echo "passes kept in $WORK"
