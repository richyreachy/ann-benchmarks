# r7ibench_test

[Run Zvec QG through its native Python wheel](ann_benchmarks/algorithms/zvec_qg/README.md).
The same ANN algorithm name, `zvec_qg`, supports succinct and exploration. The wheel
selects the implementation. The algorithm README lists the corresponding preset directories,
snapshot replay and full-query correctness commands.

The [historical campaign archive](benchmark-configs/r7i-new-api/README.txt) records
the earlier independent API and its timing protocol on the Intel Xeon Platinum 8369B.
Its measurements retain their original entry point and date. To audit archived configurations, run
`python scripts/check_zvec_qg_configs.py`.
