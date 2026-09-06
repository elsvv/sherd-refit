"""Command-line interface: `sherd-refit run INPUT_DIR --out OUT_DIR`."""
from __future__ import annotations

import argparse
import logging
import sys

from .matching import Params


def build_parser():
    ap = argparse.ArgumentParser(prog="sherd-refit", description="Reassemble 3D-scanned fragments of a broken ceramic object.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="full pipeline: segment, match, assemble, write outputs")
    r.add_argument("input_dir")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--target-faces", type=int, default=200000, help="working-mesh face budget per fragment")
    r.add_argument("--workers", type=int, default=None, help="parallel processes (default: cores-1)")
    r.add_argument("--threads", type=int, default=None, help="threads per matching process (default: cores / processes)")
    r.add_argument("--candidates", type=int, default=40, help="candidates refined with full ICP per pair")
    r.add_argument("--stage1", type=int, default=400, help="hypotheses refined with breakline ICP per pair")
    r.add_argument("--min-tight", type=float, default=Params.min_tight, help="min tight-contact fraction to accept a join")
    r.add_argument("--max-gap", type=float, default=Params.max_gap, help="k for the max median fracture gap; the pair's limit is max(k t, m res)")
    r.add_argument("--max-pen", type=float, default=Params.max_pen, help="max penetrating surface fraction")
    r.add_argument("--min-seam", type=float, default=Params.min_seam, help="min seam length (in t)")
    r.add_argument("--thick-ratio", type=float, default=Params.thick_ratio, help="skip a pair whose wall thicknesses differ by more than this factor")
    r.add_argument("--screen-top-k", type=int, default=Params.screen_top_k, help="partners kept per fragment by the partner search (0 disables it)")
    r.add_argument("--screen-points", type=int, default=Params.screen_points, help="breakline points per fragment used by the partner search")
    r.add_argument("--screen-min-pairs", type=int, default=Params.screen_min_pairs, help="the partner search is skipped below this many pairs")
    r.add_argument("--stage1-floor", type=float, default=Params.stage1_floor, help="skip stage 2 when the pair's best stage-1 breakline score is below this")
    r.add_argument("--early-reject-tight", type=float, default=Params.early_reject_tight, help="skip the fracture-only ICPs and the costly verification below this tight-contact fraction")
    r.add_argument("--margin-points", type=int, default=Params.margin_points, help="shell-margin points kept per fragment for ICP and the continuity test")
    r.add_argument("--surface-points", type=int, default=Params.surface_points, help="whole-surface samples per fragment (penetration test and shell margin)")
    r.add_argument("--frac-density", type=float, default=Params.frac_per_t2, help="fracture samples per t^2 of fracture area")
    r.add_argument("--no-preview", action="store_true")
    r.add_argument("--no-refine", action="store_true", help="skip full-resolution refinement")
    r.add_argument("--no-meshes", action="store_true", help="do not write placed/merged meshes")
    r.add_argument("-v", "--verbose", action="store_true")
    s = sub.add_parser("segment", help="only preprocess and segment; writes caches and a segmentation preview")
    s.add_argument("input_dir")
    s.add_argument("--out", required=True)
    s.add_argument("--target-faces", type=int, default=200000)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from . import pipeline
    if args.cmd == "run":
        p = Params(stage1=args.stage1, stage2=args.candidates, min_tight=args.min_tight, max_gap=args.max_gap, max_pen=args.max_pen,
                   min_seam=args.min_seam, thick_ratio=args.thick_ratio, early_reject_tight=args.early_reject_tight, stage1_floor=args.stage1_floor,
                   screen_top_k=args.screen_top_k, screen_points=args.screen_points, screen_min_pairs=args.screen_min_pairs,
                   margin_points=args.margin_points, surface_points=args.surface_points, frac_per_t2=args.frac_density)
        pipeline.run(args.input_dir, args.out, target_faces=args.target_faces, workers=args.workers, params=p,
                     preview=not args.no_preview, refine=not args.no_refine, write_meshes=not args.no_meshes, threads=args.threads)
    elif args.cmd == "segment":
        pipeline.segment_only(args.input_dir, args.out, target_faces=args.target_faces, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
