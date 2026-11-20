"""g9i succinct FP32 adapter for ANN-Benchmarks (--local --algorithm zvec_qg).

Set ZVEC_QG_ARTIFACT to the matching _independent_qg runtime manifest.
fit builds from database rows. To replay the recommended frozen graphs, set
ZVEC_QG_{SIFT,GIST,GLOVE}_INDEX to their respective five-file snapshot prefixes.
"""

import hashlib
import importlib.util
import json
import os
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
    def __init__(self, metric, dimension, profile, artifact):
        self.profile = load_profile(profile, metric, dimension)
        self.artifact = read_config(artifact)
        self.index = None
        self.name = "zvec_qg succinct " + self.profile["dataset"]
        for key in ("source_commit", "bridge_sha256", "options_sha256"):
            if self.artifact.get(key) != self.profile[key]:
                raise ValueError("Succinct runtime differs from the preset: " + key)
        module = "_independent_qg"
        extension = self.artifact["extension"]
        hashes = self.artifact["runtime_sha256"]
        if self.artifact["module"] != module or extension not in hashes:
            raise ValueError("Expected a fingerprinted independent QG extension")
        for path, expected in hashes.items():
            if digest(path) != expected:
                raise ValueError("Runtime fingerprint mismatch: " + path)
        spec = importlib.util.spec_from_file_location(module, extension)
        self.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.core)

    def fit(self, X):
        if self.index is not None:
            raise RuntimeError("Close the current index before fitting again")
        if X.dtype != np.float32 or not X.flags.c_contiguous or list(X.shape) != self.profile["train_shape"]:
            raise ValueError("Expected the full contiguous FP32 training corpus")
        if hashlib.sha256(memoryview(X)).hexdigest() != self.profile["train_sha256"]:
            raise ValueError("Training corpus fingerprint mismatch")

        cosine = self.profile["metric"] == "angular"
        prefix = self.profile["index_path"]
        if not prefix:
            self.index = self.core.Index(X, cosine, json.dumps(self.profile))
            return
        parts = self.profile["snapshot_sha256"]
        if set(parts) != {".vectors", ".offsets", ".edges", ".router", ".qg"}:
            raise ValueError("Incomplete independent snapshot manifest")
        for ext, expected in parts.items():
            if digest(prefix + ext) != expected:
                raise ValueError("Snapshot fingerprint mismatch: " + prefix + ext)
        self.index = self.core.Index(prefix, len(X), X.shape[1], cosine,
                                     json.dumps(self.profile["page_params"]))

    def set_query_arguments(self, setting):
        if self.index is None:
            raise RuntimeError("Fit the index before configuring queries")
        if type(setting) is not int or not 0 <= setting < len(self.profile["settings"]):
            raise ValueError("Expected a recommended query setting index")
        self.index.configure(self.profile["settings"][setting]["options"])

    def query(self, q, n):
        return self.index.search(q, n)

    def get_additional(self):
        prefix = self.profile["index_path"]
        metadata = {
            "source_commit": self.artifact["source_commit"],
            "runtime_sha256": self.artifact["runtime_sha256"][self.artifact["extension"]],
            "entry": "QuantizedGraphIndex",
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
