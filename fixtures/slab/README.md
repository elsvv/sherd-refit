# Slab parity fixture

The synthetic slab pair of `tests/test_synthetic.py` — a curved 300 × 200 slab of wall thickness
30, cut in two along a bumpy fracture surface, each half given its own random rigid pose — with a
full stage-boundary dump of the reference pipeline run over it. It is the one fixture small
enough to live in the repository (≈ 18 MB); every other set is generated on demand into
`output/fixtures/` and stays out of git (see `docs/superpowers/notes/2026-09-06-p0-fixtures.md`).

```
input/          pieceA.ply, pieceB.ply, ground_truth.json   (the collection, as the pipeline reads it)
dump/           the fixture: fragments/, pairs/, assembly/, refine/, outputs/, manifest.json
```

The two halves are exactly complementary — both are built from the same fracture-surface vertex
grid — so the relative pose the pipeline has to recover is known to machine precision, and the
dump records what the reference actually produced for it: the pair is accepted, one group, the
pose within the test module's 2° / 0.1 t bounds.

## Regenerating

```
python tools/make_slab.py fixtures/slab/input
python tools/dump_fixtures.py fixtures/slab/input fixtures/slab/dump --force --verify-determinism
rm -rf fixtures/slab/dump/_run
```

`--verify-determinism` dumps twice and compares every hash; the run must report all files
byte-identical. `manifest.json` records the commit, the library versions and a SHA-256 per file,
and `tests/test_fixtures.py` checks the committed tree against it.

## Using it

```
python tools/compare_fixtures.py fixtures/slab/dump SOME_OTHER_DUMP            # injected tolerances
python tools/compare_fixtures.py fixtures/slab/dump SOME_OTHER_DUMP --mode native
```

The Rust core's `--dump-fixtures DIR` is meant to write the same layout, so the same command
judges it.
