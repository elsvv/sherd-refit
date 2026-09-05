"""Integration test on the real scanned fragments (input/test_fragments_1).

Skipped when the scans are not present.  The museum's manual assembly is
104 - 094 - 021 with 007 left over, so those are the joins the pipeline must find.
"""
from __future__ import annotations

import os

import pytest

from sherd_refit import pipeline

REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "test_fragments_1", "fragments")
TRUE_JOINS = {frozenset({"FY234021_reduced", "FY234094_reduced"}),
              frozenset({"FY234094_reduced", "FY234104_reduced"})}
LEFTOVER = "FY234007_reduced"


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(REAL_DIR), reason=f"scans not available at {REAL_DIR}")
def test_real_fragments_assemble_as_the_museum_did(tmp_path):
    poses, groups, used, cands = pipeline.run(str(REAL_DIR), str(tmp_path / "out"), workers=4,
                                              preview=False, write_meshes=False)
    assert {frozenset({c.a, c.b}) for c in used} == TRUE_JOINS
    assert [LEFTOVER] in groups, f"{LEFTOVER} should stay unplaced, groups={groups}"
    assert len(groups) == 2
    assert set(max(groups, key=len)) == {"FY234021_reduced", "FY234094_reduced", "FY234104_reduced"}
    for c in used:
        assert c.scores["pen"] <= 0.005, f"{c.a}-{c.b} penetrates ({c.scores['pen']:.4f})"
