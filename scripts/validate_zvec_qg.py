#!/usr/bin/env python3
"""Exercise registered zvec_qg definitions on a complete ANN dataset; emit correctness records."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def recall(train, queries, ids, truth, thresholds, metric):
    hits = 0
    for start in range(0, len(queries), 128):
        batch = slice(start, start + 128)
        vectors = train[ids[batch]].astype(np.float64)
        q = queries[batch].astype(np.float64)
        if metric == "angular":
            dot = np.einsum("ijk,ik->ij", vectors, q)
            distances = 1 - dot / (np.linalg.norm(vectors, axis=2) * np.linalg.norm(q, axis=1)[:, None])
        else:
            distances = np.linalg.norm(vectors - q[:, None], axis=2)
        hits += np.count_nonzero(distances <= thresholds[batch, None] + .001)
    strict = np.any(ids[:, :, None] == truth[:, None, :], axis=2).mean()
    return hits / ids.size, float(strict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Full ANN-Benchmarks HDF5 file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from ann_benchmarks.definitions import get_definitions, instantiate_algorithm

    with h5py.File(args.dataset) as f:
        train, queries = f["train"][:], f["test"][:]
        metric = f.attrs["distance"]
        truth, thresholds = f["neighbors"][:, :10], f["distances"][:, 9].astype(np.float64)
    definitions = get_definitions(train.shape[1], "float", metric, 10,
                                  str(args.repo / "ann_benchmarks/algorithms"))
    records = []
    for definition in definitions:
        if definition.algorithm != "zvec_qg" or definition.disabled:
            continue
        algo = instantiate_algorithm(definition)
        try:
            algo.fit(train)
            for group in definition.query_argument_groups:
                algo.set_query_arguments(*group)
                ids = np.asarray([algo.query(q, 10) for q in queries], dtype="<i8")
                if (ids.shape != truth.shape or (ids < 0).any() or (ids >= len(train)).any()
                        or (np.diff(np.sort(ids, axis=1), axis=1) == 0).any()):
                    raise ValueError("Invalid or duplicate top-10 IDs")
                if any(not np.array_equal(algo.query(q, 10), expected) for q, expected in zip(queries, ids)):
                    raise ValueError("Unstable repeated query")
                ann, strict = recall(train, queries, ids, truth, thresholds, metric)
                records.append(dict(**algo.get_additional(), ann_recall=ann, strict_recall=strict,
                                    queries=len(queries), ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
        finally:
            algo.done()
    floors = sorted(record["target_recall"] for record in records)
    if floors != [.30, .50, .70, .85, .90, .95, .99]:
        raise ValueError(f"Incomplete or duplicate target recall coverage: {floors}")
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
