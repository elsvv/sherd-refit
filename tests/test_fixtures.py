"""The parity harness: the fixture sink, `tools/dump_fixtures.py` and `tools/compare_fixtures.py`.

The committed dump under `fixtures/slab/` is the reference these tests work on, so most of them
cost nothing: the comparison tool has to pass on two copies of one dump and fail on every
perturbation of it that the design document's §10.2 tolerances are meant to catch.  Two tests do
run the pipeline: one checks that two dumps of the same input are byte-identical, the other that
switching the sink on changes no result.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLAB = os.path.join(ROOT, "fixtures", "slab")
SLAB_DUMP = os.path.join(SLAB, "dump")
SLAB_INPUT = os.path.join(SLAB, "input")

sys.path.insert(0, ROOT)
from sherd_refit import fixture  # noqa: E402


def _tool(name):
    """Load a script from `tools/` (the directory is not an importable package)."""
    path = os.path.join(ROOT, "tools", name + ".py")
    spec = importlib.util.spec_from_file_location("sherd_refit_tool_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cf = _tool("compare_fixtures")

pytestmark = pytest.mark.skipif(not os.path.isdir(SLAB_DUMP),
                                reason=f"the committed slab fixture is missing from {SLAB_DUMP}")


# ---------------------------------------------------------------- the sink itself

def test_sink_is_off_without_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(fixture.ENV_DIR, raising=False)
    assert not fixture.enabled()
    assert fixture.current() is None
    with fixture.scope("fragments/x"):
        assert fixture.put("md.S", np.zeros((3, 3)), "md") is False
        assert fixture.trace("fragments/x", "pair") is None
    assert list(tmp_path.iterdir()) == []


def test_sink_writes_arrays_and_json_with_their_own_dtypes(tmp_path, monkeypatch):
    monkeypatch.setenv(fixture.ENV_DIR, str(tmp_path))
    monkeypatch.setenv(fixture.ENV_LEVEL, "full")
    with fixture.scope("fragments/x"):
        assert fixture.put("md.sp", np.arange(4, dtype=np.int32), "md")
        assert fixture.put("md.params", {"t": 1.5, "seed": 0}, "md")
        with fixture.auto_scope("fragments/y"):          # an explicit scope wins
            fixture.put("md.valid", np.ones(2, bool), "md")
    got = np.load(tmp_path / "fragments" / "x" / "md.sp.npy")
    assert got.dtype == np.int32 and got.tolist() == [0, 1, 2, 3]
    assert json.load(open(tmp_path / "fragments" / "x" / "md.params.json")) == {"t": 1.5, "seed": 0}
    assert (tmp_path / "fragments" / "x" / "md.valid.npy").exists()
    assert not (tmp_path / "fragments" / "y").exists()


def test_level_min_drops_the_detail_groups(tmp_path, monkeypatch):
    monkeypatch.setenv(fixture.ENV_DIR, str(tmp_path))
    monkeypatch.setenv(fixture.ENV_LEVEL, "min")
    assert fixture.want("md") and fixture.want("mesh") and fixture.want("seg_final")
    assert not fixture.want("orig") and not fixture.want("seg") and not fixture.want("pair")


# ---------------------------------------------------------------- the committed fixture

def test_committed_slab_dump_matches_its_manifest():
    with open(os.path.join(SLAB_DUMP, "manifest.json")) as f:
        man = json.load(f)
    files = fixture.iter_files(SLAB_DUMP)
    assert {rel for _, rel in files} == set(man["files"]), "the dump and its manifest disagree"
    for path, rel in files:
        assert fixture.sha256_of(path) == man["files"][rel]["sha256"], f"{rel} changed"


def test_committed_slab_dump_holds_every_documented_stage():
    d = cf.Dump(SLAB_DUMP)
    assert d.fragments() == ["pieceA", "pieceB"]
    assert d.pairs() == ["pieceA__pieceB"]
    for n in d.fragments():
        for key in ("load.n_orig", "thick.t", "mesh.res", "mesh.stats", "seg.info", "md.params", "md.rng"):
            assert d.js(f"fragments/{n}/{key}") is not None, f"{n}/{key}"
        for key in ("load.V0", "load.F0", "thick.idx", "thick.t_hit", "thick.prim", "mesh.V", "mesh.F",
                    "seg.rep", "seg.near", "seg.NS", "seg.good", "seg.frac_raw", "seg.frac_majority",
                    "seg.frac_islands", "seg.ref", "seg.has_ref", "seg.frac_final",
                    "md.S", "md.sp", "md.Pf", "md.fp", "md.brk_P", "md.brk_ns", "md.brk_nf", "md.brk_f",
                    "md.brk_sub", "md.margin_idx", "md.brk_t", "md.brk_dih", "md.valid"):
            assert d.npy(f"fragments/{n}/{key}") is not None, f"{n}/{key}"
    pr = "pairs/pieceA__pieceB"
    for key in ("hyp.ia", "hyp.ib", "hyp.pa", "hyp.pb", "coarse.idx", "coarse.cs", "nms1.kept",
                "s1.T", "s1.score", "nms2.kept", "s2.T_reg1", "s2.T_reg2", "s2.T_frac1", "s2.T_frac2",
                "s2.T", "s2.accepted"):
        assert d.npy(f"{pr}/{key}") is not None, key
    for key in ("scales", "md_used", "s2.scores", "result.candidates"):
        assert d.js(f"{pr}/{key}") is not None, key
    for key in ("assembly/poses", "assembly/groups", "assembly/used", "assembly/rejected",
                "assembly/md_t_median", "refine/joins", "refine/poses_final",
                "outputs/transforms", "outputs/report", "collection", "pairs"):
        assert d.js(key) is not None, key


def test_committed_slab_dump_reproduces_the_synthetic_tests_result():
    """The slab pair is joined, in one group, with the pose the test module asserts."""
    d = cf.Dump(SLAB_DUMP)
    assert [sorted(g) for g in d.js("assembly/groups")] == [["pieceA", "pieceB"]]
    used = d.js("assembly/used")
    assert len(used) == 1 and used[0]["accepted"]
    cands = d.js("pairs/pieceA__pieceB/result.candidates")
    assert cands[0]["accepted"] and cands[0]["tight"] >= 0.4 and cands[0]["seam"] >= 3.0
    tf = d.js("outputs/transforms")
    assert all(tf["fragments"][n]["placed"] for n in ("pieceA", "pieceB"))


# ---------------------------------------------------------------- the comparison tool

@pytest.fixture(scope="module")
def two_copies(tmp_path_factory):
    d = tmp_path_factory.mktemp("compare")
    a, b = str(d / "ref"), str(d / "cand")
    shutil.copytree(SLAB_DUMP, a)
    shutil.copytree(SLAB_DUMP, b)
    return a, b


@pytest.mark.parametrize("mode", ["injected", "native"])
def test_compare_passes_on_two_dumps_of_the_same_run(two_copies, mode, capsys):
    a, b = two_copies
    stages = cf.compare(a, b, mode)
    cf.print_table(stages)
    failed = [s.name for s in stages if not s.ok]
    assert failed == [], capsys.readouterr().out
    ran = {s.name for s in stages if s.checks}
    assert {"load", "working_mesh", "segmentation", "breakline", "pair_result",
            "assembly", "refine", "outputs"} <= ran
    if mode == "injected":
        assert {"hypotheses", "coarse", "stage1", "stage2"} <= ran


def _edit_json(path, fn):
    with open(path) as f:
        obj = json.load(f)
    with open(path, "w") as f:
        json.dump(fn(obj), f)


def _edit_npy(path, fn):
    np.save(path, fn(np.load(path, allow_pickle=False)))


FRAG = "fragments/pieceA"
PAIR = "pairs/pieceA__pieceB"

PERTURBATIONS = [
    ("load", lambda d: _edit_json(f"{d}/{FRAG}/load.n_orig.json",
                                  lambda o: dict(o, n_faces=o["n_faces"] + 7))),
    ("thickness", lambda d: _edit_json(f"{d}/{FRAG}/thick.t.json", lambda o: o * 1.10)),
    ("working_mesh", lambda d: _edit_json(f"{d}/{FRAG}/mesh.stats.json",
                                          lambda o: dict(o, area=o["area"] * 1.02))),
    ("segmentation", lambda d: _edit_npy(f"{d}/{FRAG}/seg.frac_final.npy",
                                         lambda a: np.where(np.arange(len(a)) % 5 == 0, ~a, a))),
    ("breakline", lambda d: _edit_npy(f"{d}/{FRAG}/md.brk_P.npy", lambda a: a + 0.3)),
    ("hypotheses", lambda d: _edit_npy(f"{d}/{PAIR}/hyp.pa.npy", lambda a: a[:-5])),
    ("coarse", lambda d: _edit_npy(f"{d}/{PAIR}/coarse.cs.npy", lambda a: a + 0.05)),
    ("stage1", lambda d: _edit_npy(f"{d}/{PAIR}/s1.score.npy", lambda a: a + 0.05)),
    ("stage2", lambda d: _edit_json(f"{d}/{PAIR}/s2.scores.json",
                                    lambda o: [dict(s, tight=s["tight"] + 0.05) for s in o])),
    ("pair_result", lambda d: _edit_json(f"{d}/{PAIR}/result.candidates.json",
                                         lambda o: [dict(c, accepted=False) for c in o])),
    ("assembly", lambda d: _edit_json(f"{d}/assembly/used.json", lambda o: [])),
    ("refine", lambda d: _edit_json(f"{d}/refine/poses_final.json",
                                    lambda o: _shift(o, "pieceB", 2.0))),
    ("outputs", lambda d: _edit_json(f"{d}/outputs/transforms.json", lambda o: _shift_tf(o, "pieceB", 2.0))),
]


def _shift(poses, name, dx):
    T = np.array(poses[name], float)
    T[0, 3] += dx
    return dict(poses, **{name: T.tolist()})


def _shift_tf(tf, name, dx):
    T = np.array(tf["fragments"][name]["matrix"], float)
    T[0, 3] += dx
    tf["fragments"][name]["matrix"] = T.tolist()
    return tf


@pytest.mark.parametrize("stage,perturb", PERTURBATIONS, ids=[p[0] for p in PERTURBATIONS])
def test_compare_fails_on_a_perturbed_copy(two_copies, tmp_path, stage, perturb):
    ref, _ = two_copies
    cand = str(tmp_path / "cand")
    shutil.copytree(ref, cand)
    perturb(cand)
    stages = cf.compare(ref, cand, "injected")
    failed = {s.name for s in stages if not s.ok}
    assert stage in failed, f"{stage} accepted a perturbed dump; failures were {sorted(failed)}"


def test_compare_exits_non_zero_on_failure(two_copies, tmp_path):
    ref, _ = two_copies
    cand = str(tmp_path / "cand")
    shutil.copytree(ref, cand)
    _edit_json(f"{cand}/assembly/used.json", lambda o: [])
    assert cf.main([ref, cand]) == 1
    assert cf.main([ref, ref]) == 0


# ---------------------------------------------------------------- the sink end to end

@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two dumps of the same input, plus one run with the sink off, for comparison.

    The refinement is switched off: Open3D's point-to-plane ICP sums its normal equations with an
    OpenMP reduction whose order depends on the thread count, and the pool's threads cannot be
    reconfigured once the process has used it (`tools/dump_fixtures.py` pins OMP_NUM_THREADS
    before importing Open3D, which is why a dump made through the CLI is reproducible in full).
    """
    from sherd_refit import pipeline
    df = _tool("dump_fixtures")
    d = tmp_path_factory.mktemp("sink")
    kw = dict(workers=2, refine=False, preview=False, meshes=False)
    df.dump(SLAB_INPUT, str(d / "a"), **kw)
    df.dump(SLAB_INPUT, str(d / "b"), **kw)
    assert fixture.root() is None, "the sink must be switched off again after a dump"
    pipeline.run(SLAB_INPUT, str(d / "off"), workers=2, preview=False, refine=False, write_meshes=False)
    return str(d / "a"), str(d / "b"), str(d / "off")


@pytest.mark.slow
def test_two_dumps_of_the_same_input_are_byte_identical(runs):
    a, b, _ = runs
    ha = {rel: fixture.sha256_of(p) for p, rel in fixture.iter_files(a)}
    hb = {rel: fixture.sha256_of(p) for p, rel in fixture.iter_files(b)}
    assert set(ha) == set(hb)
    differ = sorted(k for k in ha if ha[k] != hb[k])
    assert differ == [], f"{len(differ)} file(s) differ between two runs, e.g. {differ[:5]}"
    assert len(ha) > 100


@pytest.mark.slow
def test_the_sink_does_not_change_the_result(runs):
    a, _, off = runs
    with open(os.path.join(a, "_run", "report.json")) as f:
        on = json.load(f)
    with open(os.path.join(off, "report.json")) as f:
        plain = json.load(f)
    for key in ("groups", "fragments", "params", "joins_used", "joins_rejected", "candidates", "thickness"):
        assert on[key] == plain[key], f"the sink changed {key}"
