# Independent quantized graph adapter

Select `--local --algorithm zvec-qg-native`. Queries call
`QuantizedGraphIndex` / `QuantizedGraphQuery` from the independent C++ API.
The provider remains private to the bridge translation unit to preserve the
validated template inlining behavior.

This adapter reopens fingerprinted FP32 historical containers. HNSW storage
classes supply original vectors, ID mappings and snapshot segments; queries do
not call HNSW navigation or its retired quantized-graph dispatch. It supports
the frozen SIFT, GIST and GloVe profiles with TopK=10. It does not construct a new
graph, expose the Collection API, or claim FP16 adapter validation. ANN's `fit`
time includes fingerprint verification and loading, not graph construction.

## Prepare a runtime

Use the exploration environment's registered Python. Replace all paths below
with their locations on the target server. Keep improved's runtime independent.

```bash
PYTHON=/path/to/exploration/.venv-leaders/bin/python
ANN=/path/to/ann-benchmarks
SOURCE=/path/to/performance_succinct
REFERENCE=/path/to/runs/succinct_independent_qg_20261119
RUN=/path/to/exploration/runs/new_native_run

"$PYTHON" "$ANN/scripts/build_zvec_native.py" \
  --validated-run "$REFERENCE" --source "$SOURCE" --output "$RUN/runtime"
"$PYTHON" "$ANN/scripts/prepare_zvec_native_profiles.py" \
  --reference-run "$REFERENCE" --output "$RUN/profiles"
```

The bridge build reuses the reference's version-pinned libraries, compiler flags
and dependency paths. It verifies their fingerprints and requires the current
source `src` tree to match the validated commit exactly. It freshly compiles and
links the bridge, serially, into the new run. If core code changes, first produce
a fresh validated core runtime; this script intentionally refuses to silently
combine incompatible headers and libraries. Large data, indexes, libraries and
their archived build dependencies must be made available separately.

The generated profiles retain the frozen graph/query settings, use canonical
THP Params with `require`, target 95%, floor 80%, and verify corpus/index hashes
on loading. Readiness failures propagate as errors; any bounded external retry
must retain failures and must not select attempts by QPS.

## Run through ANN-Benchmarks

```bash
export ZVEC_QG_NATIVE_ARTIFACT="$RUN/runtime/artifact.json"
export ZVEC_QG_NATIVE_PROFILE="$RUN/profiles/sift.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$RUN/cli"
ln -s "$ANN/data" "$RUN/cli/data"
ln -s "$ANN/logging.conf" "$RUN/cli/logging.conf"
cd "$RUN/cli"
taskset -c 0 "$PYTHON" "$ANN/run.py" --local \
  --definitions "$ANN/ann_benchmarks/algorithms" \
  --algorithm zvec-qg-native --dataset sift-128-euclidean \
  --parallelism 1 --runs 3
```

For GIST/GloVe select `gist.json` / `glove.json` and the matching ANN dataset.
Run timed jobs serially, with no concurrent compilation. Build/algorithm job
concurrency may not exceed two. This query adapter uses one thread; it does not
retain a second borrowed corpus. Report total RSS and the pre-load baseline
separately; ANN's incremental memory field is not exclusive deployed memory.

ANN's normal CLI retains its best-of-runs reporting convention. The paired
regression archive instead stores every timing and reports pooled medians and
independent paired blocks; do not mix those summaries. The controlled regression
uses `hint` plus a run-local page controller to obtain identical PMD offsets,
not the generated profiles' natural `require` admission. Record actual coverage
when comparing naturally admitted processes.

## Provenance boundary

`benchmark_configs/g9i/` remains the immutable historical configuration archive
measured with `b44f0e09+python-qg-binding`. Its 21 presets and new-API mappings do
not become revalidated merely by switching branches or adding this adapter.
The current server's nine regression settings use separately fingerprinted
indexes and results. Keep new measurements in a new role-specific run directory.
