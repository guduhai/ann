# 来源与修改说明

本文件说明 `CAGRA/` 中使用的上游源码、项目改动及其与同级共享量化代码的关系。下文路径除特别标注外均相对于 `CAGRA/`。

## 上游源码

本项目保留了 NSG 和 EFANNA graph 的部分 C++ 源码，并在此基础上修改构图流程。历史引入版本记录如下：

| 项目 | 仓库 | 引入提交 | 本地位置与用途 |
| --- | --- | --- | --- |
| NSG | [ZJULearning/nsg](https://github.com/ZJULearning/nsg) | [`5ec8fadf6af71b6b48a486bbaa28c8b7cae34275`](https://github.com/ZJULearning/nsg/tree/5ec8fadf6af71b6b48a486bbaa28c8b7cae34275) | `vendor/nsg/`，候选收集、反向边处理及连通性修复 |
| EFANNA graph | [ZJULearning/efanna_graph](https://github.com/ZJULearning/efanna_graph) | [`6d741ad44a3c565507f86d955917dc028ffe67bc`](https://github.com/ZJULearning/efanna_graph/tree/6d741ad44a3c565507f86d955917dc028ffe67bc) | `vendor/efanna_graph/`，通过 NN-Descent 构建初始候选图 |

- [UPSTREAM.json](UPSTREAM.json) 保存历史引入提交及下载归档的 SHA-256；归档哈希不是修改后本地源码的哈希。
- [VENDORED_FILES.json](VENDORED_FILES.json) 记录引入时保留及排除的文件；上游可选的 Faiss/PQ 组件不参与当前构建。
- [vendor_changes.patch](vendor_changes.patch) 保存历史上游修改补丁。当前编译使用本仓库中的实际源码；该历史补丁不保证覆盖后续全部改动。

NSG 的完整 MIT 许可保留在 [vendor/nsg/LICENSE](vendor/nsg/LICENSE)，其中版权署名为 `Copyright (c) 2018 Cong Fu, Changxu Wang, Deng Cai`。部分 NSG 源文件还保留 `Copyright (c) 2017 ZJULearning` 和 MIT 许可声明。

EFANNA graph 源文件中保留 `Copyright (c) 2017 ZJULearning` 和 MIT 许可声明；当前裁剪目录没有单独的上游 `LICENSE` 文件。原有作者声明和第三方许可随源码保留，不以本项目的修改说明替代。

## 构图与查询改动

本项目是在 NSG 框架中使用 CAGRA 风格 detour-count 剪枝的 CPU 实验实现，不是 NVIDIA CUDA/cuVS CAGRA 的完整实现。主要改动与新增部分包括：

- 在初次剪枝和反向边溢出剪枝中使用 `include/detour_prune.h`：按距离排列候选，统计经更靠前候选的两跳路径，优先保留 detour 较少的边。
- 反向边阶段使用初次剪枝图的只读快照，并串行更新；保留 NSG 的 `tree_grow` 连通性修复。
- 以 `include/portable_distance.h` 提供通用 C++ / ARM NEON L2 距离；两个上游 `distance.h` 当前作为该实现的转接头。
- 使用 `include/bitset.h` 替换原有 Boost 位图依赖，修复数组释放问题，并增加短候选池等输入检查。
- 禁用上游携带全精度底库的 `OptimizeGraph` 路径，保存原生邻接图，并通过共用的 `src/query.cpp` 直接读取；不再转换 CSR。
- 构图入口默认配置 4 个 OpenMP 线程，可通过参数覆盖；反向边更新和连通性修复仍为串行。Linux 与 Android 共用构图、查询源码，平台选项在 `cmake/` 中分别维护。
- `src/knn_pooled_new_android.cpp` 沿用手机全量实验的 KNN 小分配内存池，仅链接 Android 的 `ann_knn`，避免大量小分配耗尽 Scudo 映射。

## 共享量化源码

量化实现位于仓库根目录下的 [ann_quantization/](../ann_quantization/)，与 `CAGRA/` 同级：

```text
ann/
├── ann_quantization/
│   ├── __init__.py
│   └── codecs.py
└── CAGRA/
    └── ann_codecs.py
```

- `../ann_quantization/codecs.py` 提供 float32、INT8、PCA、PQ、Local adaptive INT8 的模型及统一接口。
- `ann_codecs.py` 是适配入口，不包含量化算法副本。它根据自身位置加载同级共享包，并转出模型类、`make_codec`、`save_codec` 和 `load_codec`。
- `pipeline.py` 使用共享实现进行训练和编码，并导出其原生模型；正式图搜索由 C++ 程序完成，量化评分直接复用 `../ann_quantization/native_search_scorer.hpp`，不保存私有算法副本。
- 查询以只读 mmap 加载共享编码和模型；量化方法不读取全精度底库，也不做精确重排。旧 `include/distance_kernels.h` 和旧单文件编码专用入口已移除。
- 原说明中“`ann_codecs.py` 为本地算法副本”已不适用于当前结构；已移除对不存在的 `LOCAL_CODE_ORIGIN.json` 的引用。

## 运行依赖与实验数据

NumPy、h5py 及编译工具的统一安装方法见[仓库根目录说明](../README.md)。它们是运行或构建依赖，来源和许可分别由各自项目维护。

SIFT 实验使用 [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) 提供的 [SIFT-128 Euclidean HDF5](https://ann-benchmarks.com/sift-128-euclidean.hdf5)，也支持设备已有的标准 fvecs/ivecs。输入格式见 [README_CN.md](README_CN.md)，编译与运行见 [BUILDING.md](BUILDING.md)，本机 HDF5 辅助流程见 [PI_RUN_CN.md](PI_RUN_CN.md)。数据集及生成的图、编码、实验结果不属于上游源码归档。
