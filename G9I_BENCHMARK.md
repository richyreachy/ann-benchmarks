# G9i benchmark 配置归档

机器相关的评测配置和证据在 ann-benchmarks 的 `g9ibench` 分支维护；算法实现仍在 Zvec 仓库维护。

## 配置

- [使用说明及新 API 迁移边界](benchmark_configs/g9i/README.txt)
- [SIFT](benchmark_configs/g9i/sift.json)
- [GIST](benchmark_configs/g9i/gist.json)
- [GloVe](benchmark_configs/g9i/glove.json)
- [21个召回档配置摘要](benchmark_configs/g9i/summary.csv)
- [数据来源和验证证据](benchmark_configs/g9i/validation.json)

本分支只归档已完成的 g9i 全量实验配置，不包含当前工作区其他未提交的算法和调度修改，也没有打包数据、索引、二进制或原始大结果文件。

## 验证归档

在仓库根目录执行：

```bash
python3 benchmark_configs/g9i/validate.py
```

该命令检查配置与摘要一致性，不运行 benchmark。

## 版本边界

原始归档来自 Zvec 仓库提交 `3eb98231`。测量版本为 Zvec `b44f0e09+python-qg-binding`，不是新 API 提交 `3634b4a6` 的复验结果。

JSON 中保留旧接口参数，同时提供尚未经过新 API 运行验证的映射。新提交拒绝旧 HNSW QG JSON 和 local_builder YAML；本分支目前不是新接口的一键复测包。完成新入口适配及全量复验后，应另外记录版本与测量证据。

证据路径以原评测工作区根目录为基准；全新 checkout 中不存在 `bench_runs/` 是正常的。相应 NPZ、数据及索引须从原测试机器另行获取，不能把配置一致性检查当作运行正确性验证。
