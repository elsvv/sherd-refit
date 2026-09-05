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
    r.add_argument("--candidates", type=int, default=40, help="candidates refined with full ICP per pair")
    r.add_argument("--stage1", type=int, default=400, help="hypotheses refined with breakline ICP per pair")
    r.add_argument("--min-tight", type=float, default=0.25, help="min tight-contact fraction to accept a join")
    r.add_argument("--max-gap", type=float, default=0.065, help="max median fracture gap (in t) to accept a join")
    r.add_argument("--max-pen", type=float, default=0.005, help="max penetrating surface fraction")
    r.add_argument("--min-seam", type=float, default=3.0, help="min seam length (in t)")
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
        p = Params(stage1=args.stage1, stage2=args.candidates, min_tight=args.min_tight, max_gap=args.max_gap, max_pen=args.max_pen, min_seam=args.min_seam)
        pipeline.run(args.input_dir, args.out, target_faces=args.target_faces, workers=args.workers, params=p,
                     preview=not args.no_preview, refine=not args.no_refine, write_meshes=not args.no_meshes)
    elif args.cmd == "segment":
        pipeline.segment_only(args.input_dir, args.out, target_faces=args.target_faces, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
