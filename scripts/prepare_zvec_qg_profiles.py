#!/usr/bin/env python3
"""Resolve the checked-in g9i presets to a new run directory.

Optional ZVEC_QG_*_INDEX values select independent snapshot prefixes.
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
        variable = profile["index_path"][2:-1]
        prefix = os.environ.get(variable)
        if prefix:
            prefix = str(Path(prefix).resolve())
            for suffix in profile["snapshot_sha256"]:
                if not Path(prefix + suffix).is_file():
                    raise SystemExit("Missing snapshot file: " + prefix + suffix)
        profile["index_path"] = prefix or ""
        profiles[path.name] = profile
    args.output.mkdir(parents=True, exist_ok=False)
    for name, profile in profiles.items():
        output = args.output / name
        output.write_text(json.dumps(profile, indent=2) + "\n")
        print(output)


if __name__ == "__main__":
    main()
