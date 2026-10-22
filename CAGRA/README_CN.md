# CPU 图检索：NSG + CAGRA 风格剪枝

本项目是非 Faiss 的 CPU 研究实现，不是 NVIDIA CUDA/cuVS CAGRA。使用 EFANNA NN-Descent 产生候选图，在 NSG 初次剪枝和反向边溢出剪枝中使用 detour-count 规则，并保留连通性修复。

Linux（包括树莓派、WSL）和 Android ARM64 共用一套构图、查询源码。构图默认 4 线程，可通过参数修改；初次候选收集与剪枝使用 OpenMP，反向边阶段及连通性修复仍为串行，不表示所有阶段都能同时使用 4 核。查询为单线程 Top-10，无全精度重排。

## 目录

```text
ann/
├── ann_quantization/             # 共享训练、编码和原生评分实现
└── CAGRA/
    ├── CMakeLists.txt
    ├── cmake/
    │   ├── linux.cmake
    │   ├── android.cmake
    │   └── build_targets.cmake
    ├── src/                     # 共用 C++ 源码和执行封装
    ├── include/                 # I/O、距离与剪枝头文件
    ├── vendor/                  # NSG / EFANNA 源码及许可证
    ├── run_build.py
    ├── run_search.py
    ├── build_parameters.json
    ├── experiment.json
    └── pipeline.py              # 本机准备数据与调用共享量化的辅助流程
```

编译和正式运行见 [BUILDING.md](BUILDING.md)。本机 HDF5 数据准备、共享编码生成及完整辅助流程见 [PI_RUN_CN.md](PI_RUN_CN.md)。请克隆完整仓库，保留 `CAGRA` 与 `ann_quantization` 同级。

## 默认参数

| 内容 | 默认值 |
| --- | --- |
| KNN | K=64、L=80、iterations=8、S=10、reverse R=100 |
| 图剪枝 | R=32、L=64、C=132 |
| 构图 | 4 线程，随机种子 20261014 |
| 查询 | L=50、Top-10、预热最多 200 条、重复 3 次、单线程 |
| INT8 | 每维对称 INT8 |
| PCA | 128 → 80 维 float32 |
| PQ | M=64、每个子空间 256 个码字、ADC 查询 |
| Local | 32 个簇、簇号及 INT8 残差 |

`build_parameters.json` 和 `experiment.json` 为两个平台共用的配置。数据规模从输入文件推导，不固定百万规模。仅支持当前实验的 128 维输入。使用底库子集时，必须提供针对该子集重新计算的 Top-10 真值和与图节点顺序一致的编码。

## 图与输入格式

构图程序直接输出 `final.graph`，查询直接读取它，不需要 Python 或服务器转换 CSR：

```text
128D 底库 → ann_knn → candidate.knn → ann_build → final.graph → ann_query
```

- 底库/查询：标准 `fvecs`，也兼容 `EGF32001` + uint32 N/D + N×D float32 的旧数据矩阵。
- 真值：标准 `ivecs`，每行至少 10 个 ID；也兼容 `EGI32001` 矩阵。
- 候选图：逐行 uint32 K + K 个 uint32 邻居。
- 最终图：uint32 最大出度/入口 ID，随后逐节点 uint32 出度 + 邻居 ID。文件不含 N，运行时从底库或编码推导；直接运行 C++ 时须传正确 N。
- 量化：共享实现生成的 `codes.npy`，以及原生模型目录中的 `scale.bin`、`mean.bin`、`basis.bin`、`codebooks.bin` 或 `centers.bin`。

统一版本不读取历史 `EGCSR001` 图和 `EGCODE01` 单文件编码。旧结果可作为历史记录，但不能直接作为新入口的输入。历史 `raw.nsg` 如果是同样的原生邻接格式，则可直接指定，不必改名。

## 内存与评测

量化查询直接调用 `../ann_quantization/native_search_scorer.hpp`。量化模式不打开全精度底库，不复制 float32 表示，不做候选精确重排。共享评分器以只读 mmap 加载编码和模型；图使用内存中的原生邻接表。

查询结果写入 `<prefix>.json` 和 `<prefix>.labels`。JSON 包括 Recall@10、QPM、查询中位耗时、各次耗时、加载耗时与 `/proc/self/status` 的 VmHWM 峰值 RSS。查询计时包含投影、PQ 查表和图遍历，排除文件加载、预热、召回计算及写盘。

统一 Python 入口执行时还保存 `.stdout`、`.time`，后者来自 `time -v`，可查看构图/查询进程总耗时与峰值内存；`time -v` 的进程总耗时与 JSON 的纯查询耗时口径不同。各方法、各次运行均应选择新输出前缀。

剪枝规则测试可通过 `-DBUILD_TESTING=ON` 单独启用。旧 CSR / 单文件编码专用的验证与 PQ 探针已移除，避免调用不兼容的旧接口。

来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
