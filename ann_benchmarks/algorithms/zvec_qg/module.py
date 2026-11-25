"""Build r7i independent graphs with the installed zvec.QuantizedGraph API.

Optional single-file index paths: ZVEC_QG_SIFT_INDEX, ZVEC_QG_GIST_INDEX,
ZVEC_QG_GLOVE_N32_INDEX and ZVEC_QG_GLOVE_N64_INDEX.
Settings list overrides to the common r7i search defaults below.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from ..base.module import BaseANN

# Pin the measured defaults even when a future native wheel changes its own.
SEARCH_DEFAULTS = {"capacity": 128, "route_beam": 1, "uncertainty_bonus": 0.0}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_profile(metric, dimension, preset):
    directory = Path(os.environ.get(
        "ZVEC_QG_PRESET_DIR", Path(__file__).parent / "configs/r7i-native-builder"))
    if preset == "euclidean-n64":
        datasets = {128: "sift-n64.json", 960: "gist-n64.json"}
        if dimension not in datasets:
            raise ValueError("No r7i Euclidean preset for this dimension")
        preset = datasets[dimension]
    profile = json.loads((directory / preset).read_text())
    if (metric, dimension) != (profile["metric"], profile["dimension"]):
        raise ValueError("Dataset metric/dimension differs from the QG preset")
    return Path(preset).stem, profile


class ZvecQuantizedGraph(BaseANN):
    def __init__(self, metric, dimension, preset):
        import zvec

        preset, self.profile = load_profile(metric, dimension, preset)
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.graph_type = getattr(zvec, "QuantizedGraph", None)
        if self.graph_type is None or not self.graph_type.supported():
            raise RuntimeError("Install a zvec build exposing QuantizedGraph on a supported CPU")
        self.runtime_sha256 = digest(sys.modules[self.graph_type.__module__].__file__)
        self.zvec_version = zvec.__version__
        self.index_path = os.environ.get(self.profile["index_env"])
        self.index = None
        self.setting = None
        self.name = f"zvec_qg {self.profile['dataset']} {preset}"

    def fit(self, X):
        if self.index is not None:
            raise RuntimeError("Close the current QG index before fitting again")
        if X.dtype != np.float32 or not X.flags.c_contiguous or list(X.shape) != self.profile["train_shape"]:
            raise ValueError("Expected the full contiguous FP32 training corpus")
        if hashlib.sha256(memoryview(X)).hexdigest() != self.profile["train_sha256"]:
            raise ValueError("Training corpus fingerprint mismatch")
        pages = self.profile["page_params"]
        if self.index_path:
            if digest(self.index_path) != self.profile["snapshot_sha256"]:
                raise ValueError("Snapshot fingerprint mismatch: " + self.index_path)
            self.index = self.graph_type.open(self.index_path, page_params=pages)
        else:
            self.index = self.graph_type.build(X, metric=self.metric, options=self.profile["build"],
                                               page_params=pages)
        self.set_query_arguments(0)

    def set_query_arguments(self, setting):
        if self.index is None:
            raise RuntimeError("Fit the graph before configuring queries")
        settings = self.profile["settings"]
        if type(setting) is not int or not 0 <= setting < len(settings):
            raise ValueError("Expected a query setting index for this graph")
        self.index.configure({**SEARCH_DEFAULTS, **settings[setting]["options"]})
        self.setting = setting

    def query(self, q, n):
        return self.index.search(q, n)

    def get_additional(self):
        return dict(entry="zvec.QuantizedGraph", base_graph="independent",
                    fit_mode="load" if self.index_path else "build", query_threads=1,
                    target_recall=self.profile["settings"][self.setting]["target"],
                    max_neighbors=self.profile["build"]["encoding"]["max_neighbors"],
                    zvec_version=self.zvec_version, runtime_sha256=self.runtime_sha256)

    def done(self):
        self.index = None
        self.setting = None
