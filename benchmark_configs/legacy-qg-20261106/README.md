# Historical HNSW-QG configurations

These 20 YAML files preserve the former exploration benchmark configurations,
including the three locally edited GloVe M64/800 files. They were moved byte for
byte from Zvec's `tools/core/bench/config/` on 2026-11-20. The source checkout and
per-file hashes are recorded in [provenance.json](provenance.json); its source
HEAD is the migration location, not a claim about the measured code revision.

The recipes and comparison metadata date from 2026-10-30 through 2026-11-06.
They include the retired HNSW-QG entry and have **not** been revalidated against
the independent QuantizedGraphIndex API. Do not pass these recipes to the current
local_builder or treat their FP16 settings as current ANN adapter coverage.

Current SIFT, GIST and GloVe FP32 testing uses the
[zvec_qg adapter and colocated presets](../../ann_benchmarks/algorithms/zvec_qg/README.md).
The separate [g9i measurement archive](../g9i/README.txt) retains its own code,
index and timing provenance. The r7i campaign is not modified by this migration.
Large datasets, indexes and raw timings remain in their original run directories.
