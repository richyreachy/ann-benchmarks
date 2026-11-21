# Zvec QG on r7i

Install the native Zvec wheel into the same Python environment as ANN-Benchmarks.
The GloVe N64 Gaussian preset was verified with succinct `b4d4c6e6` and exploration `0f11d208`. It requires support for two-dimensional Gaussian blocks.
Use a separate environment for each wheel. Set `ZVEC_WHEEL` to the wheel file path, then run
these commands from the ANN-Benchmarks checkout.

```bash
python -m pip install "$ZVEC_WHEEL"
python -c 'import zvec; print(zvec.__version__, zvec.QuantizedGraph.supported())'
for dataset in sift-128-euclidean gist-960-euclidean glove-100-angular; do
    python run.py --local --algorithm zvec_qg --dataset "$dataset" --parallelism 1 --runs 5
done
```

The adapter passes the original FP32 training matrix to `zvec.QuantizedGraph.build`.
Each run group builds its own graph. Each query group calls `configure` once and
each timed ANN query calls `search(q, 10)`. The wheel handles angular normalization.
The default recommended presets in `configs/r7i-native-builder/` cover recall floors
30%, 50%, 70%, 85%, 90%, 95% and 99%. SIFT and GIST use N64 throughout.
GloVe uses N32 for 30–70% and N64 for 85–99%. Each preset specifies topology,
encoding, router training, page policy and search options. Actual recall is measured.

For an exploration replay on the test branch, install the exploration wheel and select
its corresponding presets before running the same commands.

```bash
export ZVEC_QG_PRESET_DIR="$PWD/ann_benchmarks/algorithms/zvec_qg/configs/r7i-native-builder-exploration"
```

Use separate ANN output working directories for independent measurements of two wheels.
The ANN runner skips result files with a matching definition and query group.
When running outside the checkout, link the dataset directory and logging.conf into
the run directory, and pass `--definitions "$ANN/ann_benchmarks/algorithms"` to
`python "$ANN/run.py"`, where `ANN` is the absolute checkout path.

## Replay a measured graph

Leave the index environment variables unset for fresh builds. To replay, set
`ZVEC_QG_SIFT_INDEX`, `ZVEC_QG_GIST_INDEX`, `ZVEC_QG_GLOVE_N32_INDEX`
or `ZVEC_QG_GLOVE_N64_INDEX` to that profile's snapshot prefix. For example,
prefix `/bench/sift/index` identifies `index.vectors`, `index.edges`,
`index.offsets`, `index.router` and `index.qg` in `/bench/sift/`.
The adapter verifies the corpus and all five files against the preset before
calling `zvec.QuantizedGraph.open`. The result attributes include the entry,
build/load mode, wheel version and native extension SHA-256.

The correctness check exercises all seven target floors twice on the full query set,
checks valid unique top-10 IDs, and records ordered-ID digests and both recall definitions.

```bash
python scripts/validate_zvec_qg.py data/glove-100-angular.hdf5 --output glove-check.json
```

The [November 19 report](VALIDATION.md) and archived configurations document the earlier
C++ bridge. Use test revision `80f980c` to reproduce those runs with their original scripts.

## Native wheel verification on 2026-11-21

The succinct `685c21eb` and exploration `49ee8f73` wheels replayed all seven registered settings
per dataset on SIFT (10,000 queries), GIST (1,000) and GloVe (10,000).
All 42 ordered top-10 ID digests matched the corresponding native run and each
setting reached its ANN recall floor. A separate succinct `run.py --local` run
built SIFT from its full training matrix and wrote all seven HDF5 results.
These runs verified the integration and did not produce a new formal QPS ranking.
