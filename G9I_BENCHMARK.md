# G9i benchmark

测试使用 `g9ibench_test`，发布使用 `g9ibench`。算法实现位于 Zvec 仓库。

当前入口为 [`zvec_qg`](ann_benchmarks/algorithms/zvec_qg/README.md)，使用 `--local --algorithm zvec_qg`。
执行配置集中在算法目录中：

- [SIFT](ann_benchmarks/algorithms/zvec_qg/configs/g9i/sift.json)
- [GIST](ann_benchmarks/algorithms/zvec_qg/configs/g9i/gist.json)
- [GloVe](ann_benchmarks/algorithms/zvec_qg/configs/g9i/glove.json)

运行前安装提供 `zvec.QuantizedGraph` 的 Zvec Python 包。默认从 FP32 训练向量构建独立图。复测已有索引时设置对应数据集的索引前缀，适配器会校验训练数据和五个快照文件的指纹。

已测结果保留在以下文档中，代码版本、计时协议和内存口径分别标明：

- [历史原生入口验证：三个数据集、召回/QPS 回归及内存表](ann_benchmarks/algorithms/zvec_qg/VALIDATION.md)
- [历史 21 个召回档结果及完整原始配置的 Git 位置](ann_benchmarks/algorithms/zvec_qg/HISTORICAL_RESULTS.md)

仓库根目录 `results/` 中现有 PNG 沿用上游 main，不是这次 Zvec 验证生成的图。
原始测量、报告、大型数据、索引和二进制保留在对应服务器 run 目录中，未因本次清理移动或覆盖。

查询计时固定 CPU 0、单线程且串行进行；构建/算法任务并发最多为 2，计时期间不编译。
新增测量应使用独立 run 目录，记录实际召回、QPS、总 RSS、加载前 RSS、索引磁盘大小及代码/运行时指纹。
