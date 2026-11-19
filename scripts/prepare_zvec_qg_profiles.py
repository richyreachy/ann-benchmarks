#!/usr/bin/env python3
"""Resolve the checked-in g9i presets to a new run directory.

Set ZVEC_QG_SIFT_INDEX, ZVEC_QG_GIST_INDEX and ZVEC_QG_GLOVE_INDEX first.
No graph is built and no query parameter is tuned by this script.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    presets = Path(__file__).resolve().parents[1] / "ann_benchmarks/algorithms/zvec_qg/configs/g9i"
    profiles = {}
    for path in sorted(presets.glob("*.json")):
        profile = json.loads(path.read_text())
        index = os.path.expandvars(profile["index_path"])
        if "$" in index or not Path(index).is_file():
            raise SystemExit("Set an existing index path: " + profile["index_path"])
        profile["index_path"] = str(Path(index).resolve())
        profiles[path.name] = profile
    args.output.mkdir(parents=True, exist_ok=False)
    for name, profile in profiles.items():
        output = args.output / name
        output.write_text(json.dumps(profile, indent=2) + "\n")
        print(output)


if __name__ == "__main__":
    main()
