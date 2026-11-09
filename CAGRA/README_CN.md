# CAGRA：共享表示与 DEEP1M 实验

本目录使用 EFANNA 候选图和 CAGRA 风格 detour-count 剪枝，生成原生邻接图，直接查询，不经过 CSR 转换。支持 96 维 DEEP 与 128 维 SIFT。Linux 和 Android 使用相同的构图、查询源码；构建方式见 [BUILDING.md](BUILDING.md)。

## 入口

- `run_build.py`：读取原始 fvecs 或 EGF32001 底库，构建候选图与最终图；成功后默认删除候选图，`--keep-candidate` 可保留。
- `run_search.py`：根据查询维度选择 `ann_query96` 或 `ann_query`；支持 exact、INT8、PQ、PCA、Local。
- `prepare_codes.py`：调用同级共享 Python 量化实现，按原始数据 ID 顺序写入 `codes.npy` 和 `native/` 模型。
- `run_experiment.py`：完成 DEEP 构图、exact 查询、四种压缩表示的准备和查询；每种临时编码在该方法查询完成后删除。

构图、查询和完整实验入口默认只检查输入或展示命令，显式加 `--execute` 才开始运行。`prepare_codes.py` 是直接执行的编码工具。旧 `pipeline.py` 已停用；`run_local.sh` 转发到 `run_experiment.py`。

## 数据与默认参数

底库和查询支持完整的 96/128 维 fvecs 或 EGF32001；真值支持 ivecs 或 EGI32001。图和编码的行数及节点顺序必须一致。底库子集需要重新计算对应真值。

构图默认四线程，K=64、L=80、迭代8轮，最终图 R=32、L=64、C=132。查询默认单线程、Top-10、W=50、预热200、重复1次、不做全精度重排。DEEP 默认 PQ M48，SIFT 默认 PQ M64，PCA 均为80维。

128维查询直接包含同级共享评分头文件；96维查询在构建目录 `shared96/` 生成与其他四索引 DEEP 实验相同的维度特化，不修改共享源文件。压缩查询只读取图、编码、模型和查询，不加载 float32 底库。

完整 DEEP1M 实验使用树莓派现有的 HNSW DEEP 参考模型和训练采样顺序，具体路径、命令、磁盘需求及输出见 [PI_RUN_CN.md](PI_RUN_CN.md)。本地仓库保留可移植的 `python_pi.sh`，使用仓库根目录 `.venv`，与树莓派部署版的专用 Python 环境包装不同。

## 输出与统计

构图输出 `graph/final.graph`、`graph/build.json` 及候选图/最终图原始日志；查询输出各方法的 `.json/.labels/.stdout/.time`。完整实验另外保留编码来源记录和逐步更新的 `summary.json`；只有终端显示 `COMPLETE` 才表示全部五种查询结束。

构图时间统计候选图和最终图两个完整进程；峰值内存主字段来自 GNU time，查询自身的 VmHWM 单独保留。QPM 使用查询循环时间，不包含模型训练、文件加载和写盘。

已有96/128维真实子集验证通过；百万级新版本复测结果应以实际运行输出为准。
