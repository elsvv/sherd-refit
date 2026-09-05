"""Download the source vessel scans used by the synthetic benchmark from Zenodo.

    python tools/fetch_sources.py [--out input/source_models]

All five models are photogrammetry scans of real vessels published by LWL-Archäologie für
Westfalen under CC BY 4.0 (model author: LWL-Archäologie für Westfalen / Florian Westphal).
Keep the attribution when redistributing anything derived from them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

RECORDS = {
    # local name                 Zenodo record id   title
    "074_tongefaess.glb":        (10332909, "074 Tongefäß / Potter Vessel (Pingsdorf ware, ~900 AD)"),
    "049_kelch.glb":             (10354385, "049 Kelch / Goblet"),
    "012_verziertes_gefaess.glb": (10311275, "012 Verziertes Gefäß / Decorated Vessel"),
    "094_bemalte_schuessel.glb": (10330624, "094 Bemalte Schüssel / Painted Bowl"),
    "025_zylinderhalsgefaess.glb": (10327711, "025 Zylinderhalsgefäß / Cylinder-necked beaker"),
}


def fetch(record_id: int, dest: str) -> str:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record_id}", timeout=60) as r:
        rec = json.load(r)
    glb = [f for f in rec.get("files", []) if f["key"].lower().endswith(".glb")]
    if not glb:
        raise RuntimeError(f"record {record_id} has no .glb file")
    url = glb[0]["links"]["self"]
    lic = rec.get("metadata", {}).get("license", {}).get("id", "?")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return lic


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="input/source_models")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for name, (rid, title) in RECORDS.items():
        dest = os.path.join(args.out, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"{name}: already present")
            continue
        print(f"{name}: downloading {title} (zenodo.{rid}) ...", flush=True)
        lic = fetch(rid, dest)
        print(f"{name}: {os.path.getsize(dest) / 1e6:.1f} MB, licence {lic}, https://doi.org/10.5281/zenodo.{rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
