# CAGRA：树莓派共享表示入口

当前正式入口与 HNSW、NSG、NSSG、VAMANA 使用相同的数据及共享编码约定。
本部署目标为 Linux/树莓派，支持 96 维 DEEP 与 128 维 SIFT。
本机仓库还保留 Android 构建配置，见 [BUILDING.md](BUILDING.md)。
下列路径和参考模型对应当前树莓派；本机源码未包含数据、模型和实验结果。
仓库版 `python_pi.sh` 使用仓库根目录 `.venv`；树莓派部署版使用该设备已有的专用环境。

## 构建

```bash
cd /home/pi-6/ann/CAGRA
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

程序位于 `build/bin/`：`ann_knn`、`ann_build`、`ann_query`（128维）、
`ann_query96`（96维）。查询启动器按输入维度选择程序。
128维直接包含共享评分头文件；96维在构建目录中按其他四索引 DEEP 实验相同的
规则生成共享头文件的96维版本，来源哈希见 `build/shared96/source_manifest.json`。
共享目录和其他索引源码不需要修改。

## DEEP1M 完整实验

先查看参数；加 `--execute` 才执行。输出必须是 `CAGRA/results` 中的新目录。

```bash
cd /home/pi-6/ann/CAGRA
bash python_pi.sh run_experiment.py --output results/deep1m-shared-run1
bash python_pi.sh run_experiment.py --output results/deep1m-shared-run1 --execute
```

默认读取 `../data/deep1M/` 的原始 fvecs 和 ivecs，不生成底库副本。
使用 `../HNSW/experiments/deep1m-20261109/` 的 PQ M48/PCA80 模型；
INT8/Local 按其他四索引相同的 HNSW 内部顺序抽样训练，编码则按原始数据ID写入。
基线直接读取 fvecs。四种压缩表示逐种准备、查询后删除本轮临时编码，
保留每种表示的编码来源记录。已有模型、数据和其他索引图只读使用。

默认构图4线程；查询单线程、Top-10、W=50、预热200、重复1次、不重排。
构图仍使用 CAGRA 风格 detour 剪枝，反向边处理阶段保持串行。
四线程不代表每个阶段都有四倍加速。
完整流程的磁盘检查按原生图和单份最大编码计算，约需700 MiB额外可用空间。
若需重复3次查询，加 `--repeats 3`。

输出包括 `graph/final.graph`、`graph/build.json`、各阶段原始日志、
`queries/<method>.json/.labels/.stdout/.time`、编码来源记录和 `summary.json`。
构图速度统计候选图和最终图两个完整进程，不再有 CSR 转换。
峰值内存主字段来自 GNU time，与既有四索引 DEEP 脚本一致；查询程序的 VmHWM
单独保存在 `vmhwm_peak_rss_mib`。QPM 使用查询循环时间，不含模型训练、文件加载和写盘。

## 单独构图与查询

```bash
bash python_pi.sh run_build.py   --data /home/pi-6/ann/data/deep1M/deep_base.fvecs   --output /home/pi-6/ann/CAGRA/results/deep-build-run1 --threads 4 --execute

bash python_pi.sh run_search.py --method exact   --graph /home/pi-6/ann/CAGRA/results/deep-build-run1/final.graph   --base /home/pi-6/ann/data/deep1M/deep_base.fvecs   --query /home/pi-6/ann/data/deep1M/deep_query.fvecs   --truth /home/pi-6/ann/data/deep1M/deep_groundtruth.ivecs   --output results/deep-query-run1/exact --execute
```

`run_build.py` 默认在成功记录最终图后清理本轮候选 KNN 图；需要保留时加
`--keep-candidate`。各个阶段失败后保留日志，换新输出目录重试。

压缩查询使用 `--method int8|pq48|pq64|pq|pca80|pca|local`、
`--codes <含codes.npy的目录>`、`--model <native目录>`。
编码必须按原始数据ID排列，不能直接传 HNSW 内部顺序编码。
量化查询不访问 `--base`，不做全精度重排。
独立准备编码可使用 `prepare_codes.py --help`。

## 历史记录

旧 `pipeline.py` 入口已停用。原始修改备份位于 `.backups/`。
旧 SIFT 查询结果及日志仍保留在原来的 `results/sift1m-m64-pca80-run1/`。
历史文档和旧独立距离核不参与当前 CMake 构建。
当前修改经过96/128维真实子集验证；不代表已经完成本版本的百万级全量复测。
