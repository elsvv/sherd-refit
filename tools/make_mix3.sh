#!/bin/sh
# Build the two coloured mixed collections: three decorated vessels broken separately and put in
# one crate.  This script is the definition of `input/synthetic_mix3_24` and
# `input/synthetic_mix3_60` -- the sets themselves are gitignored like every other collection under
# `input/`, and this file plus `tools/make_synthetic.py` and `tools/merge_collections.py` is what
# reproduces them.
#
#     sh tools/make_mix3.sh [24|60|all]        (default: all)
#
# Needs `python tools/fetch_sources.py` to have downloaded the five scans, and the virtualenv of
# `pyproject.toml`'s `synth` extra (trimesh and pygltflib on top of the runtime dependencies).
#
# The three vessels are chosen for their decoration, which is the point of these two sets: a
# stamped grey-brown decorated vessel, a red terra sigillata goblet and a bowl painted in
# blue-green and cream.  Every one of the three carries a real photograph as a glTF base-colour
# texture, and `--texture` bakes it onto the shell of every fragment while the fracture faces take
# the vessel's own clay-body colour -- so a fragment looks like a sherd, painted skin and plain
# break, and colour is measurable evidence on a set that has object ids.  See
# `docs/superpowers/notes/2026-09-11-s1-coloured-sets.md`.
#
# Wall thickness, voxel, target edge, wear and pose spread are the pingsdorf sets' own, so the
# fragments are the same size and resolution as the sets the gates already use.  Each vessel is
# scaled to the same 8 mm wall, which is what stops wall thickness from being an object label.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

SRC=input/source_models
WORK=${MIX3_WORK:-input/.mix3_parts}
AUTHOR='LWL-Archaeologie fuer Westfalen / Florian Westphal'

# break one vessel: <model> <zenodo url> <fragments> <seed> <part dir>
break_one() {
    echo "=== $1: $3 fragments, seed $4 -> $5"
    python tools/make_synthetic.py "$SRC/$1" --out "$5" --fragments "$3" --seed "$4" \
        --wall 8.0 --voxel 0.6 --edge 0.6 --wear 0.25 --missing 0.0 --spread 500 \
        --texture --body-colour dark-quantile \
        --source-url "$2" --license CC-BY-4.0 --author "$AUTHOR"
}

build_24() {
    break_one 012_verziertes_gefaess.glb https://doi.org/10.5281/zenodo.10311275 8 0 "$WORK/24_V012"
    break_one 049_kelch.glb              https://doi.org/10.5281/zenodo.10354385 8 1 "$WORK/24_V049"
    break_one 094_bemalte_schuessel.glb  https://doi.org/10.5281/zenodo.10330624 8 2 "$WORK/24_V094"
    python tools/merge_collections.py --out input/synthetic_mix3_24 \
        "V012=$WORK/24_V012" "V049=$WORK/24_V049" "V094=$WORK/24_V094"
}

build_60() {
    break_one 012_verziertes_gefaess.glb https://doi.org/10.5281/zenodo.10311275 20 100 "$WORK/60_V012"
    break_one 049_kelch.glb              https://doi.org/10.5281/zenodo.10354385 20 101 "$WORK/60_V049"
    break_one 094_bemalte_schuessel.glb  https://doi.org/10.5281/zenodo.10330624 20 102 "$WORK/60_V094"
    python tools/merge_collections.py --out input/synthetic_mix3_60 \
        "V012=$WORK/60_V012" "V049=$WORK/60_V049" "V094=$WORK/60_V094"
}

case "${1:-all}" in
    24)  build_24 ;;
    60)  build_60 ;;
    all) build_24; build_60 ;;
    *)   echo "usage: sh tools/make_mix3.sh [24|60|all]" >&2; exit 2 ;;
esac

echo "the per-vessel part directories under $WORK are intermediate; remove them once the"
echo "merged collections are in place:  rm -rf $WORK"
