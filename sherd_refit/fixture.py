"""Stage-boundary fixture sink.

Switched on by the environment variable ``SHERD_REFIT_FIXTURES=DIR`` (inherited by the worker
processes the pipeline spawns).  When it is unset every call here is a cheap no-op and the
pipeline behaves exactly as it did before the sink existed.

The sink writes one file per array (``.npy``, the array's own dtype, no compression) and one
file per scalar group (``.json``, keys sorted), under a *scope* directory that names the stage's
owner (a fragment, a pair, the assembly).  The layout follows the design document's §10.1; the
per-file inventory is in ``docs/superpowers/notes/2026-09-06-p0-fixtures.md``.

Two properties matter and are tested:

* **determinism** — two runs over the same input produce byte-identical files.  Everything that
  varies between runs (wall-clock timings, worker counts, absolute temporary paths) is either
  left out or normalised before it is written.
* **process safety** — preprocessing and matching run in worker processes and, inside a pair,
  in threads.  Every file is written to a temporary name and renamed into place, so a reader
  never sees a half-written file and two workers that compute the same deduplicated array
  cannot interleave.  Per-candidate arrays produced by a thread pool are collected in memory by
  index (`Trace`) and written once, in job order, by the thread that owns the pair.

Levels (``SHERD_REFIT_FIXTURES_LEVEL``) trade completeness for size; see ``LEVELS``.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
from contextlib import contextmanager

import numpy as np

ENV_DIR = "SHERD_REFIT_FIXTURES"
ENV_LEVEL = "SHERD_REFIT_FIXTURES_LEVEL"

#: every group a `put` can be tagged with; a level is a subset of these
ALL_GROUPS = ("counts", "orig", "thick", "mesh", "seg", "seg_final", "md", "pair", "result",
              "screen", "assembly", "refine", "outputs")

LEVELS = {
    # everything, including the cleaned original mesh (the largest arrays by far)
    "full": set(ALL_GROUPS),
    # everything except the original mesh: enough to inject every stage from the working mesh on
    "slim": set(ALL_GROUPS) - {"orig"},
    # the design document's reduced set for very large collections
    "min": {"counts", "mesh", "seg_final", "md", "result", "assembly", "refine", "outputs"},
}

_scope: str | None = None       # process-global: one pair / one fragment at a time per process
_counter = itertools.count()


# ---------------------------------------------------------------- switch

def root() -> str | None:
    d = os.environ.get(ENV_DIR)
    return d if d else None


def enabled() -> bool:
    return root() is not None


def level() -> str:
    lv = (os.environ.get(ENV_LEVEL) or "full").lower()
    return lv if lv in LEVELS else "full"


def want(group: str) -> bool:
    """True if a `put` tagged `group` would be written."""
    return enabled() and group in LEVELS[level()]


# ---------------------------------------------------------------- scopes

@contextmanager
def scope(name: str):
    """Set the scope directory (relative to the sink root) for the puts made inside."""
    global _scope
    if not enabled():
        yield None
        return
    prev = _scope
    _scope = name
    try:
        yield name
    finally:
        _scope = prev


@contextmanager
def auto_scope(name: str):
    """Set the scope only if none is set: an explicit outer scope always wins."""
    if not enabled() or _scope is not None:
        yield _scope
        return
    with scope(name) as s:
        yield s


def current() -> str | None:
    return _scope if enabled() else None


def t_key(t: float, surface_points: int) -> str:
    """Directory name for match arrays rebuilt at a wall thickness other than the fragment's own.

    A pair is matched at `min(t_a, t_b)`, so the thicker fragment rebuilds its arrays; many pairs
    ask for the same (fragment, t, surface_points) and write the same bytes, so they share one
    directory instead of one per pair.
    """
    return f"t{t:.9g}_sp{int(surface_points)}"


# ---------------------------------------------------------------- writing

def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def _write_atomic(path: str, payload: bytes | None = None, save: np.ndarray | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}_{next(_counter)}"
    with open(tmp, "wb") as f:
        if save is not None:
            np.save(f, np.ascontiguousarray(save), allow_pickle=False)
        else:
            f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def put(key: str, value, group: str = "counts", scope_name: str | None = None) -> bool:
    """Write one array (`.npy`) or one JSON value (`.json`) named `key` into the current scope.

    Returns True if something was written.  `key` may contain dots (`"seg.frac_final"`), which is
    how the design document names the files; it may not contain a path separator.
    """
    if not want(group):
        return False
    sc = _scope if scope_name is None else scope_name
    if sc is None:
        return False
    base = os.path.join(root(), *[p for p in sc.split("/") if p])
    if isinstance(value, np.ndarray):
        _write_atomic(os.path.join(base, key + ".npy"), save=value)
    else:
        text = json.dumps(value, sort_keys=True, indent=1, default=_plain) + "\n"
        _write_atomic(os.path.join(base, key + ".json"), payload=text.encode())
    return True


class Trace:
    """Thread-safe per-index collector for arrays produced by a thread pool.

    Stage 2 refines its candidates in parallel; writing from the workers would make the file set
    depend on the thread schedule.  Each worker records its arrays under its job index and the
    owning thread stacks them in job order at the end.
    """

    def __init__(self, scope_name: str, group: str):
        self.scope_name = scope_name
        self.group = group
        self._lock = threading.Lock()
        self._data: dict[str, dict[int, np.ndarray]] = {}

    def put(self, i: int, key: str, value):
        with self._lock:
            self._data.setdefault(key, {})[int(i)] = np.asarray(value)

    def write(self, n: int, prefix: str = ""):
        """Stack each key over `range(n)` and write it; missing entries become NaN rows."""
        for key in sorted(self._data):
            d = self._data[key]
            if not d:
                continue
            shape = next(iter(d.values())).shape
            dtype = next(iter(d.values())).dtype
            out = np.full((n,) + shape, np.nan if dtype.kind == "f" else 0, dtype=dtype)
            for i, v in d.items():
                if 0 <= i < n:
                    out[i] = v
            put(prefix + key, out, self.group, self.scope_name)


def trace(scope_name: str | None, group: str) -> Trace | None:
    """A `Trace` when the sink would write this group, else None."""
    sc = _scope if scope_name is None else scope_name
    if not want(group) or sc is None:
        return None
    return Trace(sc, group)


# ---------------------------------------------------------------- manifest

def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(dump_dir: str, skip=("_run",)):
    """Every fixture file under `dump_dir`, as (absolute path, posix-relative path), sorted."""
    out = []
    for base, dirs, names in os.walk(dump_dir):
        rel_base = os.path.relpath(base, dump_dir)
        parts = [] if rel_base == "." else rel_base.split(os.sep)
        if parts and parts[0] in skip:
            dirs[:] = []
            continue
        dirs.sort()
        for n in sorted(names):
            if n == "manifest.json" or ".tmp" in n:
                continue
            p = os.path.join(base, n)
            out.append((p, "/".join(parts + [n])))
    out.sort(key=lambda x: x[1])
    return out


def write_manifest(dump_dir: str, extra: dict | None = None) -> dict:
    """Hash every file under `dump_dir` and write `manifest.json`.

    Built after the run rather than incrementally, because the files come from several worker
    processes.  Nothing time-dependent goes in, so two runs give the same manifest.
    """
    files = {}
    for path, rel in iter_files(dump_dir):
        info = {"size": os.path.getsize(path), "sha256": sha256_of(path)}
        if rel.endswith(".npy"):
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            info["shape"] = list(a.shape)
            info["dtype"] = str(a.dtype)
        files[rel] = info
    man = dict(extra or {})
    man["files"] = files
    text = json.dumps(man, sort_keys=True, indent=1, default=_plain) + "\n"
    _write_atomic(os.path.join(dump_dir, "manifest.json"), payload=text.encode())
    return man
