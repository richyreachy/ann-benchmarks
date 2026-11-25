# Zvec Collection internal-ID benchmark on r7i

Use the algorithm name `zvec_qg_internal_ids`. The adapter calls
`Collection.query_internal_ids("v", vectors, topk=10, param=...)` on a
read-only Collection. It requires a Zvec Python build that accepts a NumPy
float32 vector and a two-dimensional float32 array through this overload.
The r7i run used Zvec exploration commit `b1bfadd5` with the NumPy binding
patch applied (native extension SHA-256
`97e0e51c5eec80380cd07b53efdd78a46ea7cd91823062fec8d476b5699d4de4`).
For a historical standalone comparison, select `--algorithm zvec_qg --run-disabled`.

From the ANN-Benchmarks checkout, with its dependencies and the matching Zvec
build available in the Python environment, run

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for dataset in sift-128-euclidean gist-960-euclidean glove-100-angular; do
  taskset -c 0 python run.py --local --algorithm zvec_qg_internal_ids --dataset "$dataset" --count 10 --runs 5
done
```

Use a working copy with the ANN-Benchmarks HDF5 files in `data/`. The r7i preset uses
the ANN-Benchmarks batch runner by default for all three datasets. It passes up
to 32 FP32 vectors per native call and also processes the final partial batch.
The saved QPS is throughput over the complete query set. Use `plot.py --batch`
to read the `zvec_qg_internal_ids-batch` results. Build and search settings
are in [the r7i presets](../zvec_qg/configs/r7i-native-builder).

By default, each fit builds a temporary Collection from the supplied training
rows, seals it, creates the QG index and reopens it read-only. To reuse a
specific measured Collection, set the corresponding environment variable in
[collections.json](collections.json) to the Collection directory. The adapter
verifies the index SHA-256 on reuse. At fit time it verifies the corpus,
index options and the mapping of every internal ID to its training-row number
before the query timer starts. Results record the index and runtime fingerprints.
