#!/usr/bin/env python3
"""Convert frozen regression profiles to the independent QG adapter schema.

Inputs must describe existing FP32 indexes, not the historical g9i preset schema.
No graph is built and no query parameter is tuned by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for dataset in ["sift", "gist", "glove"]:
        original = args.reference_run / "config" / (dataset + ".json")
        cfg = json.loads(original.read_text())
        md, data = cfg["index_metadata"], cfg["data"]
        storage = dict(md["construction"])
        storage.pop("use_quantized_graph", None)
        if storage["data_type"] != "DT_FP32":
            raise ValueError("This frozen-container adapter supports FP32 only")
        storage["params"] = {
            "proxima.quantized_graph.thp_policy": "require",
            "proxima.quantized_graph.thp_min_coverage": 0.95,
            "proxima.quantized_graph.thp_min_coverage_floor": 0.80,
        }
        fingerprint = hashlib.sha256()
        with h5py.File(data["path"]) as stream:
            for start in range(0, len(stream["train"]), 4096):
                block = np.ascontiguousarray(stream["train"][start:start + 4096], dtype=np.float32)
                fingerprint.update(memoryview(block))
        profile = dict(
            dataset=data["ann_dataset"], metric=data["metric"], dimension=storage["dimension"],
            train_shape=data["train"], train_sha256=fingerprint.hexdigest(),
            index_path=md["index_path"], index_sha256=md["index_sha256"],
            storage=storage, construction_provenance=md, settings=[],
            configuration_source=str(original.resolve()),
        )
        for item in cfg["settings"]:
            params = item["params"]
            options = {"capacity": params["ef_search"]}
            options.update({key: params["quantized_graph_" + key] for key in
                            ["margin", "bucket_width", "max_expansions", "route_beam", "uncertainty_bonus"]})
            profile["settings"].append(dict(target=item["target"], options=options))
        output = args.output / (dataset + ".json")
        output.write_text(json.dumps(profile, indent=2) + "\n")
        print(output)


if __name__ == "__main__":
    main()
