"""ANN-Benchmarks adapter for read-only Zvec Collection internal-ID queries."""
import hashlib
import inspect
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ..base.module import BaseANN
from ..zvec_qg.module import SEARCH_DEFAULTS, digest, load_profile


class ZvecCollectionInternalIds(BaseANN):
    batch_size = 32

    def __init__(self, metric, dimension, preset):
        import zvec
        from zvec import _zvec

        if "vectors" not in inspect.signature(zvec.Collection.query_internal_ids).parameters:
            raise RuntimeError("Install a zvec build with the NumPy overload of Collection.query_internal_ids")
        self.zvec = zvec
        self.preset, self.profile = load_profile(metric, dimension, preset)
        self.metric = {"angular": zvec.MetricType.COSINE, "euclidean": zvec.MetricType.L2}[metric]
        self.name = f"zvec_qg_internal_ids {self.profile['dataset']} {self.preset}"
        self.runtime_sha256 = digest(_zvec.__file__)
        self.collection = None
        self.temp = None
        self.setting = None

    def _build(self, path, X):
        zvec = self.zvec
        schema = zvec.CollectionSchema("benchmark", vectors=zvec.VectorSchema(
            "v", zvec.DataType.VECTOR_FP32, X.shape[1], index_param=zvec.FlatIndexParam(self.metric)))
        writer = zvec.create_and_open(str(path), schema)
        try:
            for start in range(0, len(X), 1024):
                docs = [zvec.Doc(id=str(start + j), vectors={"v": row})
                        for j, row in enumerate(X[start:start + 1024])]
                for result in writer.insert(docs):
                    if not result.ok():
                        raise RuntimeError(str(result))
            writer.optimize()
            writer.create_index("v", zvec.QuantizedGraphIndexParam(self.metric, self.profile["build"]))
        finally:
            writer.close()

    def fit(self, X):
        if self.collection is not None:
            raise RuntimeError("Close the current collection before fitting again")
        if (X.dtype != np.float32 or not X.flags.c_contiguous or
                list(X.shape) != self.profile["train_shape"] or
                hashlib.sha256(memoryview(X)).hexdigest() != self.profile["train_sha256"]):
            raise ValueError("Training corpus differs from the FP32 preset")
        pin = json.loads(Path(__file__).with_name("collections.json").read_text())[self.preset]
        cached = os.environ.get(pin["env"])
        self.fit_mode = "load" if cached else "build"
        try:
            if cached:
                path = Path(cached)
            else:
                self.temp = TemporaryDirectory()
                path = Path(self.temp.name) / "collection"
                self._build(path, X)
            blocks = list(path.glob("*/v.index.*.proxima"))
            if len(blocks) != 1:
                raise ValueError(f"Expected one sealed vector index in {path}")
            self.block_sha256 = digest(blocks[0])
            if cached and self.block_sha256 != pin["sha256"]:
                raise ValueError(f"Collection index fingerprint mismatch in {path}")
            self.collection = self.zvec.open(str(path), self.zvec.CollectionOption(read_only=True))
            field = self.collection.schema.vector("v")
            if (self.collection.stats.doc_count != len(X) or field is None or
                    field.dimension != X.shape[1] or field.data_type != self.zvec.DataType.VECTOR_FP32 or
                    field.index_param.type != self.zvec.IndexType.QUANTIZED_GRAPH or
                    field.index_param.metric_type != self.metric or field.index_param.options != self.profile["build"]):
                raise ValueError("Collection schema, graph options or row count differs from the preset")
            for start in range(0, len(X), 8192):
                rows = range(start, min(start + 8192, len(X)))
                if self.collection.resolve_internal_ids(list(rows)) != [str(i) for i in rows]:
                    raise ValueError(f"Collection internal IDs differ from training ordinals at block {start}")
            self.set_query_arguments(0)
        except Exception:
            self.done()
            raise

    def set_query_arguments(self, setting):
        settings = self.profile["settings"]
        if self.collection is None or type(setting) is not int or not 0 <= setting < len(settings):
            raise ValueError("Expected an open collection and a valid query setting")
        self.param = self.zvec.QuantizedGraphQueryParam(**{**SEARCH_DEFAULTS, **settings[setting]["options"]})
        self.setting = setting

    def query(self, q, n):
        ids = self.collection.query_internal_ids("v", q, topk=n, param=self.param)
        return ids if ids[-1] >= 0 else ids[ids >= 0]

    def batch_query(self, X, n):
        ids = np.empty((len(X), n), dtype=np.int64)
        for start in range(0, len(X), self.batch_size):
            chunk = slice(start, start + self.batch_size)
            ids[chunk] = self.collection.query_internal_ids("v", X[chunk], topk=n, param=self.param)
        self.res = ids if np.all(ids[:, -1] >= 0) else [row[row >= 0] for row in ids]

    def get_additional(self):
        return dict(entry="Collection.query_internal_ids", fit_mode=self.fit_mode, preset=self.preset,
                    target_recall=self.profile["settings"][self.setting]["target"], query_threads=1,
                    batch_chunk_size=self.batch_size, runtime_sha256=self.runtime_sha256,
                    index_sha256=self.block_sha256)

    def done(self):
        if self.collection is not None:
            self.collection.close()
            self.collection = None
        if self.temp is not None:
            self.temp.cleanup()
            self.temp = None
        self.res = None
