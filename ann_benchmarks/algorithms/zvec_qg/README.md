# Zvec quantized graph on g9i

Use the same Python interpreter for ANN-Benchmarks and a zvec package exposing
`zvec.QuantizedGraph`. Check the installed package and CPU support before a run.

```sh
python -c 'import zvec; assert zvec.QuantizedGraph.supported()'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python run.py --local --algorithm zvec_qg --dataset sift-128-euclidean --parallelism 1 --runs 3
```

The other dataset names are `gist-960-euclidean` and `glove-100-angular`.
The [presets](configs/g9i) contain the FP32 graph construction parameters,
THP settings, snapshot hashes and three query settings per dataset.

By default, `fit` calls `zvec.QuantizedGraph.build` with the training rows.
To replay a fixed graph, set `ZVEC_QG_SIFT_INDEX`, `ZVEC_QG_GIST_INDEX` or
`ZVEC_QG_GLOVE_INDEX` to the dataset's snapshot prefix (such as `/data/sift/index`).
The adapter checks the corpus and all five files before calling
`zvec.QuantizedGraph.open`. Each query calls `search` with a single FP32 vector.
The result metadata records the loaded runtime SHA-256, package version, and
whether the graph was built or loaded.

Use a fresh results directory for a new comparison. For an index from a different
build, use an explicit profile with its five matching hashes. Historical measurements
and their original entry points are recorded in [VALIDATION.md](VALIDATION.md)
and [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md).
