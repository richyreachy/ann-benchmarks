"""Run with an installed zvec exposing the public QuantizedGraph API.

Use run.py --local --algorithm zvec_qg. By default fit builds from FP32 rows.
For snapshot replay set ZVEC_QG_{SIFT,GIST,GLOVE}_INDEX to the matching prefix.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from ..base.module import BaseANN


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_config(path):
    return json.loads(Path(os.path.expandvars(str(path))).read_text())


def load_profile(profile, metric, dimension):
    if profile in ("g9i", "g9i-independent", "g9i-succinct"):
        presets = Path(__file__).parent / "configs" / "g9i"
        candidates = [read_config(path) for path in sorted(presets.glob("*.json"))]
    else:
        candidates = [read_config(profile)]
    matches = [p for p in candidates if p["metric"] == metric and p["dimension"] == dimension]
    if len(matches) != 1:
        raise ValueError("Expected one g9i preset for this metric and dimension")
    preset = matches[0]
    if preset["graph_format"] != "independent-fp32-v1":
        raise ValueError("Expected an independent FP32 graph")
    prefix = preset.get("index_path", "")
    if prefix.startswith("${") and prefix.endswith("}"):
        prefix = os.environ.get(prefix[2:-1], "")
    prefix = os.path.expandvars(prefix)
    if "$" in prefix:
        raise ValueError("Unresolved snapshot prefix: " + prefix)
    preset["index_path"] = prefix
    return preset


class ZvecQuantizedGraph(BaseANN):
    def __init__(self, metric, dimension, profile):
        import zvec

        self.profile = load_profile(profile, metric, dimension)
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.graph_type = zvec.QuantizedGraph
        if not self.graph_type.supported():
            raise RuntimeError("This zvec build or CPU does not support QuantizedGraph")
        self.runtime_sha256 = digest(sys.modules[self.graph_type.__module__].__file__)
        self.zvec_version = zvec.__version__
        self.index = None
        self.name = "zvec_qg " + self.profile["dataset"]

    def fit(self, X):
        if self.index is not None:
            raise RuntimeError("Close the current index before fitting again")
        if X.dtype != np.float32 or not X.flags.c_contiguous or list(X.shape) != self.profile["train_shape"]:
            raise ValueError("Expected the full contiguous FP32 training corpus")
        if hashlib.sha256(memoryview(X)).hexdigest() != self.profile["train_sha256"]:
            raise ValueError("Training corpus fingerprint mismatch")

        prefix = self.profile["index_path"]
        if not prefix:
            self.index = self.graph_type.build(
                X, metric=self.metric, options=self.profile["build"], page_params=self.profile["page_params"]
            )
            return
        parts = self.profile["snapshot_sha256"]
        if set(parts) != {".vectors", ".offsets", ".edges", ".router", ".qg"}:
            raise ValueError("Incomplete independent snapshot manifest")
        for ext, expected in parts.items():
            if digest(prefix + ext) != expected:
                raise ValueError("Snapshot fingerprint mismatch: " + prefix + ext)
        self.index = self.graph_type.open(
            prefix, count=len(X), dimension=X.shape[1], metric=self.metric, page_params=self.profile["page_params"]
        )

    def set_query_arguments(self, setting):
        if self.index is None:
            raise RuntimeError("Fit the index before configuring queries")
        if isinstance(setting, dict):
            self.index.configure(setting)
            return
        if type(setting) is not int or not 0 <= setting < len(self.profile["settings"]):
            raise ValueError("Expected a query setting index or options dictionary")
        self.index.configure(self.profile["settings"][setting]["options"])

    def query(self, q, n):
        return self.index.search(q, n)

    def get_additional(self):
        prefix = self.profile["index_path"]
        metadata = {
            "zvec_version": self.zvec_version,
            "runtime_sha256": self.runtime_sha256,
            "entry": "zvec.QuantizedGraph",
            "base_graph": "independent",
            "fit_mode": "load" if prefix else "build",
            "query_threads": 1,
            "borrowed_corpus_bytes": 0,
        }
        if prefix:
            metadata["index_disk_bytes"] = sum(Path(prefix + ext).stat().st_size
                                               for ext in self.profile["snapshot_sha256"])
        return metadata

    def done(self):
        self.index = None
