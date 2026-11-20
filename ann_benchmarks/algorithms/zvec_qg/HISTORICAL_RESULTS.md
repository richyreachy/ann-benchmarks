# Historical G9i results

Current native-entry validation and memory measurements are in [VALIDATION.md](VALIDATION.md).
The CSV below preserves all 21 historical summary rows without rounding or changing their values.

These measurements used Zvec `b44f0e09+python-qg-binding`, full SIFT/GIST/GloVe
training and query sets, FP32, TopK=10, and one query thread. QPS is the median
of three repeats, each using the best of five timed rounds. Recall uses the
ANN distance threshold (the tenth exact distance plus 0.001). Public queries
were used to select parameters; these results are not an independent holdout.
They have not been revalidated through the current native adapter. Do not mix
this timing protocol or these versions with the current validation table.

## Original evidence

The complete historical construction/query recipes, three-repeat measurements,
machine metadata and file fingerprints remain in Git at
[`e754035`](https://github.com/richyreachy/ann-benchmarks/tree/e7540351ef5a05944052355e4da41f967ca9cd7a/benchmark_configs).
The former `benchmark_configs/g9i/` contains the original measurement archive;
`benchmark_configs/legacy-qg-20261106/` contains retired HNSW-QG YAML recipes.
Those recipes are not current executable defaults. Removing them from the
working tree does not change the original source, indexes or measurements.

For example, inspect the original detailed SIFT results with:

```bash
git show e7540351ef5a05944052355e4da41f967ca9cd7a:benchmark_configs/g9i/sift.json
```

Raw data, indexes, timings and environment-specific reports remain in their
original server run directories. An additional Git bundle and working-copy
backup were saved under the exploration run `ann_cleanup_20261120` before this
cleanup; no active benchmark data or runtime was moved.

## Unmodified historical summary

```csv
dataset,target_ann_recall,profile,qps_median,mean_query_time_us,actual_ann_recall,measured_revision,new_api_revalidated
gist-960-euclidean,0.3,hadamard_low_rank,68664.81743158014,14.563499,0.3056,b44f0e09+python-qg-binding,False
gist-960-euclidean,0.5,full,48550.07624789474,20.59729,0.5003,b44f0e09+python-qg-binding,False
gist-960-euclidean,0.7,no_router,32937.80084975573,30.360254000000005,0.7193,b44f0e09+python-qg-binding,False
gist-960-euclidean,0.85,full,18291.674654031263,54.66968000000001,0.9052,b44f0e09+python-qg-binding,False
gist-960-euclidean,0.9,full,18291.674654031263,54.66968000000001,0.9052,b44f0e09+python-qg-binding,False
gist-960-euclidean,0.95,full,11940.659222302476,83.74747,0.9559,b44f0e09+python-qg-binding,False
gist-960-euclidean,0.99,no_router,4016.6422986350576,248.96416599999998,0.9937,b44f0e09+python-qg-binding,False
glove-100-angular,0.3,full,95980.2630794056,10.418808700000001,0.35348,b44f0e09+python-qg-binding,False
glove-100-angular,0.5,no_router,64766.635094875055,15.4400487,0.53785,b44f0e09+python-qg-binding,False
glove-100-angular,0.7,no_router,44784.360326866175,22.3292237,0.73153,b44f0e09+python-qg-binding,False
glove-100-angular,0.85,one_view,19009.294282687093,52.6058456,0.87336,b44f0e09+python-qg-binding,False
glove-100-angular,0.9,full,18082.755193895304,55.30130720000001,0.90685,b44f0e09+python-qg-binding,False
glove-100-angular,0.95,no_router,10589.731717894045,94.43109860000001,0.95575,b44f0e09+python-qg-binding,False
glove-100-angular,0.99,no_router,3494.2990824956,286.180426,0.9933,b44f0e09+python-qg-binding,False
sift-128-euclidean,0.3,no_router,129348.42567314795,7.7310566000000005,0.32093,b44f0e09+python-qg-binding,False
sift-128-euclidean,0.5,full,104523.91962482687,9.5671881,0.51758,b44f0e09+python-qg-binding,False
sift-128-euclidean,0.7,full,76901.31207711343,13.003679300000002,0.73709,b44f0e09+python-qg-binding,False
sift-128-euclidean,0.85,full,62511.978857948656,15.996934,0.88805,b44f0e09+python-qg-binding,False
sift-128-euclidean,0.9,full,59400.618298659836,16.834841600000004,0.90301,b44f0e09+python-qg-binding,False
sift-128-euclidean,0.95,full,47210.384192158046,21.1817806,0.95178,b44f0e09+python-qg-binding,False
sift-128-euclidean,0.99,no_router,28695.66851796772,34.848465000000004,0.99018,b44f0e09+python-qg-binding,False
```

The original CSV SHA256 is `708888b8ef7349fb3388a9f47dac71f480bdc2d663114c0fe9f95ea7435a5f95`.
For current runs use only the colocated [g9i presets](configs/g9i/) and retain
new measurements with their own source, binary, index and dataset fingerprints.
