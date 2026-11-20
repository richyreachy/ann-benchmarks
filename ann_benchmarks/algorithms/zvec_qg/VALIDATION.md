# Naming and entrypoint validation (2026-11-19)

Core commits: exploration `5702bc79241ecd70e1d64a98065e173f1e795453`; succinct `3cb02f17b7ad929cd84af4f149bf0c11e11516da`. The installed/runtime artifact manifest identifies which branch is loaded. The current default validation manifest uses succinct.

Project-owned Turbo modules now use `quantized_graph_*` names. HNSW source, arithmetic, index wire format and frozen query settings are unchanged. Existing FP32 containers reopen through the independent graph API. The ANN selector is `zvec_qg`; executable presets reside in `configs/g9i/`. [Historical measurements](HISTORICAL_RESULTS.md) keep their original provenance.

Each branch: two balanced AB/BA process blocks, three complete-query timings per setting/process, single-thread CPU0, serial. Total: 24 accepted processes, 216 timings and 72 audited official HDF5 results. All ordered IDs and official ANN Recall@10 match. C++ tests: 29 exploration + 30 succinct; unsupported-feature compilation passes. Full performance validation covers the three frozen FP32 indexes; FP16 has functional coverage only.

QPS changes below use unrounded pooled medians versus the respective branch before renaming. Keep the small negative values; this is not a claim of zero regression on every process or hardware.

| Dataset | Target | Actual recall | Exploration QPS change | Succinct QPS change |
|---|---:|---:|---:|---:|
| SIFT | 90% | 90.007% | +0.43% | +0.02% |
| SIFT | 95% | 95.158% | +0.26% | -0.52% |
| SIFT | 99% | 99.019% | +0.27% | -0.38% |
| GIST | 90% | 90.250% | -0.38% | +0.06% |
| GIST | 95% | 95.740% | +0.11% | -0.28% |
| GIST | 99% | 99.430% | +0.67% | +0.17% |
| GLOVE | 90% | 90.721% | -0.05% | +0.68% |
| GLOVE | 95% | 96.034% | -0.02% | +0.20% |
| GLOVE | 99% | 99.189% | -0.19% | -0.21% |

Matched graph PMD sets: SIFT/GloVe 100%, GIST 2555/2838 (90.028189%). A process-local controller prepared these layouts before timing; no global kernel changes. Two exploration readiness rejections were retained; no succinct rejection occurred. Natural `require`/95%/80%-floor deployment may admit other coverage and is validated separately through ordinary `run.py`.

| Dataset | Succinct total query RSS (GiB) | Pre-load RSS (GiB) | Change vs pre-rename (MiB) | Index disk (GiB) |
|---|---:|---:|---:|---:|
| SIFT | 4.253 | 0.552 | -0.72 | 3.568 |
| GIST | 13.230 | 3.650 | -0.11 | 9.518 |
| GLOVE | 5.107 | 0.515 | -0.07 | 4.430 |

RSS includes Python, evaluation state and the retained complete training array; borrowed corpus = 0. These are medians of process query-time maxima, not whole-process peak RSS or exclusive deployed index memory. Disk size is reported separately.

Archive: `environments/exploration/runs/qg_naming_layout_20261119/`. It contains full source patches/fingerprints, pinned build commands/libraries, per-process page maps, failures, every timing, official HDF5, CLI logs, and audit/report generators. Large data, indexes and binaries are not included in Git. See [README](README.md) for path-replaceable build and run commands.

The freshly built bridge was also exercised through the checked-in default `zvec_qg` definition with ordinary `run.py --local --parallelism 1 --runs 3`: all three datasets and nine HDF5 outputs passed ordered-ID and recall comparisons. All three natural-admission loads succeeded on their first attempt. These CLI best-of-three results are kept separate from the paired median table.
