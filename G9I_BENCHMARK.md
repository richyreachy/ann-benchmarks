# G9i benchmark 配置归档

机器相关的评测配置和证据在 ann-benchmarks 的 `g9ibench` 分支维护；算法实现仍在 Zvec 仓库维护。

## 当前执行配置

算法入口统一为 [`zvec_qg`](ann_benchmarks/algorithms/zvec_qg/README.md)。使用 `--algorithm zvec_qg`，配置自动从算法目录选择：

- [SIFT](ann_benchmarks/algorithms/zvec_qg/configs/g9i/sift.json)
- [GIST](ann_benchmarks/algorithms/zvec_qg/configs/g9i/gist.json)
- [GloVe](ann_benchmarks/algorithms/zvec_qg/configs/g9i/glove.json)

设置 `ZVEC_QG_ARTIFACT` 及三个数据集的 `ZVEC_QG_*_INDEX` 路径变量。查询配置、构图来源和索引/训练数据 SHA256 均随配置记录。

## 历史配置归档

- [使用说明及新 API 迁移边界](benchmark_configs/g9i/README.txt)
- [SIFT](benchmark_configs/g9i/sift.json)
- [GIST](benchmark_configs/g9i/gist.json)
- [GloVe](benchmark_configs/g9i/glove.json)
- [21个召回档配置摘要](benchmark_configs/g9i/summary.csv)
- [数据来源和验证证据](benchmark_configs/g9i/validation.json)

`benchmark_configs/g9i/` 归档已完成的 g9i 全量实验配置，没有打包数据、索引、二进制或原始大结果文件。

独立原生入口另由 [`zvec_qg`](ann_benchmarks/algorithms/zvec_qg/README.md) 适配器提供。
它加载有指纹的既有索引，通过 `QuantizedGraphIndex` 查询，使用独立的运行配置和结果目录；不会把本目录的历史测量重新标记为新版本结果。

## 验证归档

在仓库根目录执行：

```bash
python3 benchmark_configs/g9i/validate.py
```

该命令检查配置与摘要一致性，不运行 benchmark。

## 版本边界

原始归档来自 Zvec 仓库提交 `3eb98231`。测量版本为 Zvec `b44f0e09+python-qg-binding`，不是新 API 提交 `3634b4a6` 的复验结果。

JSON 中保留旧接口参数，同时提供尚未经过新 API 运行验证的映射。新提交拒绝旧 HNSW QG JSON 和 local_builder YAML；这些历史 JSON 不能直接作为新接口的一键复测配置。新适配器使用算法目录中的冻结配置，历史 21 个点的全量复验仍应另外记录版本与测量证据。

证据路径以原评测工作区根目录为基准；全新 checkout 中不存在 `bench_runs/` 是正常的。相应 NPZ、数据及索引须从原测试机器另行获取，不能把配置一致性检查当作运行正确性验证。
