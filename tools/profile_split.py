#!/usr/bin/env python3
"""Aggregate a `samply` profile into self time per symbol and inclusive time per pipeline stage.

Phase 1e, task E1. `samply record --save-only --unstable-presymbolicate -o P.json.gz -- CMD`
writes two files: `P.json.gz`, whose frames are bare relative addresses, and `P.json.syms.json`,
which carries the demangled symbol table of every library a frame landed in. Neither has inline
frames, and the port is built with `lto = "thin"` and `codegen-units = 1`, so most of what this
note wants to separate — `ladder::climb`, the two `icp` estimators, `verify::fracture_scores` —
lives inside its caller. `atos -i` against the `.dSYM` recovers that chain, so every address of
the profiled binary is expanded to the inline stack it stands for and the attribution runs on the
expanded stacks.

Time is `threadCPUDelta`: microseconds of CPU actually burnt on that thread since the previous
sample, summed over every thread of the process. The totals are therefore **core**-seconds, and a
stage's share is its share of the machine's work rather than of the wall clock.

    python3 tools/profile_split.py P.json.gz --binary target/release/sherd-refit-rs \
        [--top 15] [--stages STAGES.json] [--min-share 0.05]

`--stages` is a JSON list of `[name, regex]` pairs, outermost first; a sample is charged to the
**outermost** frame of its stack that matches, which is what makes the number inclusive. Without
it the built-in list of R §3, R §5 and R §11 stages is used.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# The port's stages, outermost first: a sample is charged to the first of these its stack enters.
# The patterns are matched against `normalise`d frames, so they are the module paths of
# `crates/sherd-core/src/` with every generic argument already gone.
DEFAULT_STAGES: list[tuple[str, str]] = [
    # R §3's per-fragment work. In a cold run these rows *are* preprocessing; in a warm one
    # preprocessing is a cache read and every microsecond here is R §4.2's rebuild of one
    # fragment's arrays at `t_pair`, i.e. the other half of "pair setup" below. The stack cannot
    # tell them apart once `rayon` has stolen the chunk, so the rows stay separate and the reader
    # adds them according to which run it is.
    ("R §3.1 load + clean", r"sherd_core::(io::|mesh::clean|mesh::components)"),
    ("R §3.2 thickness", r"sherd_core::fragment::thickness"),
    ("R §3.3 decimate + Taubin", r"sherd_core::mesh::(decimate|taubin)|meshopt::"),
    ("R §3.4 segmentation", r"sherd_core::fragment::segment"),
    ("R §3.5.3-5 breaklines", r"sherd_core::fragment::breakline"),
    ("R §3.5.1-2 match arrays", r"sherd_core::fragment::samples"),
    ("R §3.7 cache", r"sherd_core::fragment::cache"),
    # R §5 — one pair, in the order `match_pair` runs them.
    ("pair setup: R §4.2 clouds + KD-trees", r"sherd_core::matching::pair::Pair::build"),
    ("hypotheses", r"sherd_core::matching::(hypotheses|pair::Pair::hypotheses)"),
    ("coarse score", r"sherd_core::matching::(coarse|pair::Pair::(coarse|probe))"),
    ("NMS", r"sherd_core::matching::(nms|pair::Pair::suppress)"),
    ("stage 1: breakline ICP", r"sherd_core::matching::pair::Pair::stage1"),
    ("verify scenes (BVH build)", r"sherd_core::matching::(verify::Surfaces::of|pair::Pair::surfaces)"),
    ("stage 2 ladder setup (KD-trees)", r"sherd_core::matching::pair::SurfaceLadder::of"),
    ("stage 2 + verify", r"sherd_core::matching::pair::Pair::stage2_candidate"),
    # R §8, §9, §11.
    ("assembly", r"sherd_core::assembly::"),
    ("refine", r"sherd_core::(refine::|pipeline::refine)"),
    ("write meshes", r"sherd_core::report::(write_placed_meshes|write_merged)|sherd_core::io::writer"),
    ("write previews", r"sherd_core::(render::|pipeline::write_previews)"),
    ("write report/transforms", r"sherd_core::report::(write_report|write_transforms)"),
]

# Coarse containers, consulted only when no stage above matched: they sit *higher* on the stack
# than every stage, so letting them into the pass above would swallow all of it. What they collect
# is the stage's own residue — the loop bodies, the collects, the allocation around the work.
CONTAINERS: list[tuple[str, str]] = [
    ("match_pair: the rest", r"sherd_core::matching::pair::(match_pair|Pair::match_pair)"),
    ("R §3 preprocessing, the rest", r"sherd_core::(pipeline::preprocess|fragment::Fragment::(build|load))"),
    ("run: the rest", r"sherd_core::pipeline::run"),
]

# Inside "stage 2 + verify": which half of a candidate. Same rule, applied below `stage2_candidate`.
DEFAULT_SUBSTAGES: list[tuple[str, str]] = [
    ("R §5.6 ICP rungs", r"sherd_core::matching::(ladder::|icp::)"),
    ("R §6 verify", r"sherd_core::matching::verify::"),
]

# Where a `rayon` worker picked up a *different* job. Under nested parallelism a thread that waits
# inside one `join` steals work from anywhere in the pool, and the stolen job's frames then sit
# below the victim's on the same stack: a `coarse` par_iter of one pair can carry the whole of
# another pair's `Pair::build` under it. Attribution therefore starts below the last of these.
STEAL_BOUNDARY = (
    r"rayon_core::registry::WorkerThread::wait_until|rayon_core::job::StackJob::execute"
)

# What a sample that entered no stage was doing, decided on its leaf frame.
FALLBACK: list[tuple[str, str]] = [
    (
        "rayon idle / work stealing",
        r"rayon_core::|no_work_found|yield_now|swtch_pri|cthread_yield|__psynch"
        r"|libsystem_kernel|libsystem_pthread|_pthread_cond",
    ),
    ("allocator", r"malloc|free|libsystem_malloc|__rust_(alloc|dealloc|realloc)"),
]


def load_symbols(syms_path: Path) -> dict[str, tuple[list[int], list[int], list[str]]]:
    """`{lib debug name: (sorted starts, sizes, names)}` from samply's presymbolicate sidecar."""
    blob = json.loads(syms_path.read_text())
    strings = blob["string_table"]
    out: dict[str, tuple[list[int], list[int], list[str]]] = {}
    for entry in blob["data"]:
        table = sorted(entry["symbol_table"], key=lambda s: s["rva"])
        out[entry["debug_name"]] = (
            [s["rva"] for s in table],
            [s["size"] for s in table],
            [strings[s["symbol"]] for s in table],
        )
    return out


def symbol_at(table: tuple[list[int], list[int], list[str]], rva: int) -> str | None:
    """The symbol whose `[rva, rva + size)` contains `rva`, or `None`."""
    starts, sizes, names = table
    i = bisect.bisect_right(starts, rva) - 1
    if i < 0 or rva >= starts[i] + sizes[i]:
        return None
    return names[i]


# The `__TEXT` vmaddr of a macOS arm64 executable; `atos -l` undoes it again.
MACHO_TEXT_BASE = 0x100000000


def inline_chains(binary: Path, rvas: list[int]) -> dict[int, list[str]]:
    """`{rva: [outermost, …, innermost]}` from `atos -i` against the binary's `.dSYM`.

    Empty when there is no `.dSYM` or no `atos`; the caller then falls back to the sidecar's one
    name per address, which is the containing function without its inlined callees.
    """
    dsym = Path(f"{binary}.dSYM/Contents/Resources/DWARF/{binary.name}")
    if not dsym.exists() or not rvas:
        if not dsym.exists():
            print(f"no {dsym}; run `dsymutil {binary}` for inline frames", file=sys.stderr)
        return {}
    query = "\n".join(hex(MACHO_TEXT_BASE + r) for r in rvas)
    try:
        proc = subprocess.run(
            ["atos", "-o", str(dsym), "-l", hex(MACHO_TEXT_BASE), "-i"],
            input=query,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"atos unavailable ({exc}); no inline frames", file=sys.stderr)
        return {}
    blocks = proc.stdout.split("\n\n")  # one block per address, innermost frame first
    chains: dict[int, list[str]] = {}
    for rva, block in zip(rvas, blocks):
        names = [line.split(" (in ")[0].strip() for line in block.splitlines() if line.strip()]
        if names:
            chains[rva] = list(reversed(names))
    return chains


def demangle(names: set[str]) -> dict[str, str]:
    """Rust v0 and legacy symbols through `rustfilt`; identity when it is not installed."""
    ordered = sorted(names)
    if not ordered:
        return {}
    try:
        proc = subprocess.run(
            ["rustfilt"], input="\n".join(ordered), capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return {n: n for n in ordered}
    out = proc.stdout.splitlines()
    return dict(zip(ordered, out)) if len(out) == len(ordered) else {n: n for n in ordered}


def normalise(name: str) -> str:
    """One function, one row: `<A as B>::c::<T>` becomes `A::c`.

    Two things force this. Rust v0 demangles a method as `<Type>::method`, so a plain path match
    would miss every method; and `rayon`'s plumbing carries the *closure it is driving* in its own
    type arguments, so a frame like `bridge_producer_consumer::helper::<…match_pair::{closure#1}>`
    names a stage it is not itself in. Lifting the qualifier fixes the first, dropping generic
    arguments the second.
    """
    if name.startswith("<"):
        depth = 0
        for i, ch in enumerate(name):
            depth += (ch == "<") - (ch == ">")
            if depth == 0:
                name = name[1:i].split(" as ")[0] + name[i + 1 :]
                break
    out, depth = [], 0
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return re.sub(r"::+", "::", "".join(out)).strip(":")


# A crate path inside a generic argument: `…::helper::<…, sherd_core::…::{closure#0}>`.
DRIVEN = re.compile(r"sherd_core(?:::[A-Za-z0-9_]+|::\{[^}]*\})+")


def driven_paths(full_name: str) -> list[str]:
    """The crate paths a frame's generic arguments name, in order, normalised and de-duplicated."""
    seen: list[str] = []
    inside = full_name[full_name.index("<") :] if "<" in full_name else ""
    for hit in DRIVEN.findall(inside):
        cleaned = normalise(hit)
        if cleaned not in seen:
            seen.append(cleaned)
    return seen


class Profile:
    """A samply profile with every frame resolved to a normalised inline chain."""

    def __init__(self, profile_path: Path, binary: Path) -> None:
        with gzip.open(profile_path, "rt") as handle:
            self.raw = json.load(handle)
        stem = str(profile_path)
        for suffix in (".gz", ".json"):
            stem = stem.removesuffix(suffix)
        syms = Path(f"{stem}.json.syms.json")
        self.symbols = load_symbols(syms) if syms.exists() else {}
        if not self.symbols:
            print(f"no {syms}; record with --unstable-presymbolicate", file=sys.stderr)
        self.libs = [lib["debugName"] for lib in self.raw["libs"]]
        self.binary_name = binary.name
        self._resolve(binary)

    def _frame_lib(self, thread: dict, index: int) -> str | None:
        resource = thread["funcTable"]["resource"][thread["frameTable"]["func"][index]]
        if resource is None or resource < 0:
            return None
        return self.libs[thread["resourceTable"]["lib"][resource]]

    def _resolve(self, binary: Path) -> None:
        """Fill `self.chain[t][frame]` (outermost first) and `self.symbol[t][frame]`."""
        own: set[int] = set()
        for thread in self.raw["threads"]:
            for i in range(thread["frameTable"]["length"]):
                if self._frame_lib(thread, i) == self.binary_name:
                    own.add(thread["frameTable"]["address"][i])
        chains = inline_chains(binary, sorted(own))
        raw = {n for chain in chains.values() for n in chain}
        for table in self.symbols.values():
            raw.update(table[2])
        pretty = demangle(raw)
        cache: dict[str, str] = {}

        def clean(name: str) -> str:
            if name not in cache:
                cache[name] = normalise(pretty.get(name, name))
            return cache[name]

        # A frame becomes its inline chain, outermost first, followed by the crate paths its own
        # generic arguments name. That tail is what identifies a stolen `rayon` chunk: the
        # plumbing that drives one carries the closure it is driving in its type arguments and
        # nowhere else, and `normalise` has just thrown those away.
        self.chain: list[list[list[str]]] = []
        self.symbol: list[list[str]] = []
        for thread in self.raw["threads"]:
            chain_of: list[list[str]] = []
            symbol_of: list[str] = []
            for i in range(thread["frameTable"]["length"]):
                rva = thread["frameTable"]["address"][i]
                lib = self._frame_lib(thread, i)
                raw_outer = symbol_at(self.symbols[lib], rva) if lib in self.symbols else None
                outer = clean(raw_outer) if raw_outer else f"{lib or '?'}!0x{rva:x}"
                own = lib == self.binary_name and rva in chains
                names = [clean(n) for n in chains[rva]] if own else [outer]
                driven = [d for d in driven_paths(pretty.get(raw_outer, "")) if d not in names]
                chain_of.append(names + driven)
                symbol_of.append(outer)
            self.chain.append(chain_of)
            self.symbol.append(symbol_of)

    def stacks(self):
        """Yield `(cpu_µs, stack outermost first, leaf symbol)` for every sample with CPU on it."""
        for t, thread in enumerate(self.raw["threads"]):
            table, samples = thread["stackTable"], thread["samples"]
            prefix, frame = table["prefix"], table["frame"]
            cpu = samples.get("threadCPUDelta") or [0] * samples["length"]
            chain_of, symbol_of = self.chain[t], self.symbol[t]
            memo: dict[int, list[str]] = {}

            def expand(node: int) -> list[str]:
                seen, path = [], []
                while node is not None and node not in memo:
                    seen.append(node)
                    path.append(chain_of[frame[node]])
                    node = prefix[node]
                head = memo[node] if node is not None else []
                for chunk, at in zip(reversed(path), reversed(seen)):
                    head = head + chunk
                    memo[at] = head
                return head

            for i in range(samples["length"]):
                node = samples["stack"][i]
                weight = (cpu[i] or 0) if node is not None else 0
                if weight:
                    yield weight, expand(node), symbol_of[frame[node]]


def charge(stack: list[str], stages: list[tuple[str, re.Pattern[str]]], start: int = 0):
    """The outermost frame of `stack[start:]` matching a stage, and the index it was found at."""
    for i in range(start, len(stack)):
        for name, pattern in stages:
            if pattern.search(stack[i]):
                return name, i
    return None, len(stack)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("profile", type=Path, help="the .json.gz of `samply record --save-only`")
    ap.add_argument("--binary", type=Path, required=True, help="the executable (its .dSYM is used)")
    ap.add_argument("--top", type=int, default=15, help="how many self-time rows to print")
    ap.add_argument("--stages", type=Path, help="JSON list of [name, regex], outermost first")
    ap.add_argument("--min-share", type=float, default=0.05, help="hide stages below this percent")
    args = ap.parse_args()

    profile = Profile(args.profile, args.binary)
    stages = [
        (n, re.compile(p))
        for n, p in (json.loads(args.stages.read_text()) if args.stages else DEFAULT_STAGES)
    ]
    containers = [(n, re.compile(p)) for n, p in CONTAINERS]
    substages = [(n, re.compile(p)) for n, p in DEFAULT_SUBSTAGES]
    fallback = [(n, re.compile(p)) for n, p in FALLBACK]
    steal = re.compile(STEAL_BOUNDARY)

    total = 0
    self_time: dict[str, int] = defaultdict(int)
    stage_time: dict[str, int] = defaultdict(int)
    sub_time: dict[str, int] = defaultdict(int)
    for weight, stack, leaf in profile.stacks():
        total += weight
        self_time[leaf] += weight
        cut = 0
        for i, frame in enumerate(stack):
            if steal.search(frame):
                cut = i + 1
        name, at = charge(stack, stages, cut)
        if name is None:
            name, _ = charge(stack, containers, cut)
        if name is None:
            name = next(
                (n for n, p in fallback if p.search(leaf) or any(p.search(f) for f in stack[cut:])),
                "unattributed",
            )
        stage_time[name] += weight
        if name == "stage 2 + verify":
            sub, _ = charge(stack, substages, at + 1)
            sub_time[sub or "(setup and dispatch)"] += weight

    if not total:
        print("no CPU samples", file=sys.stderr)
        return 1
    print(f"total CPU {total / 1e6:.2f} core-s over {len(profile.raw['threads'])} threads\n")
    print(f"top {args.top} symbols by self time")
    print(f"{'core-s':>9}  {'share':>7}  symbol")
    for name, weight in sorted(self_time.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"{weight / 1e6:9.2f}  {100 * weight / total:6.2f}%  {name}")

    print("\ninclusive time per stage (outermost match wins)")
    print(f"{'core-s':>9}  {'share':>7}  stage")
    for name, weight in sorted(stage_time.items(), key=lambda kv: -kv[1]):
        if 100 * weight / total >= args.min_share:
            print(f"{weight / 1e6:9.2f}  {100 * weight / total:6.2f}%  {name}")
    if sub_time:
        print("\n  inside 'stage 2 + verify'")
        for name, weight in sorted(sub_time.items(), key=lambda kv: -kv[1]):
            print(f"{weight / 1e6:9.2f}  {100 * weight / total:6.2f}%    {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
