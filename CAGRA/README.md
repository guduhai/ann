# CAGRA CPU：Linux / Android 统一实现

使用 NSG/EFANNA 构图与 CAGRA 风格 detour-count 剪枝，支持 96 维 DEEP 与 128 维 SIFT，以及 float32、INT8、PCA80、PQ、Local 查询。DEEP 默认 PQ M48，SIFT 默认 PQ M64。构图默认 4 线程，查询单线程，无全精度重排，不依赖 Faiss。

- [实现、数据格式与运行说明](README_CN.md)
- [Linux / Android 编译与统一启动入口](BUILDING.md)
- [树莓派 DEEP1M 完整实验及资源统计](PI_RUN_CN.md)
- [第三方源码与修改说明](THIRD_PARTY_NOTICES.md)

两个平台编译同一套 `src/`、`include/`、`vendor/`。量化直接复用同级 `ann_quantization`，不保存私有算法副本。最新统一入口读取原生邻接图及共享 `codes.npy` / 原生模型；历史 CSR / `EGCODE01` 编码不兼容。

构图使用 `run_build.py`，查询使用 `run_search.py`，编码使用 `prepare_codes.py`；`run_experiment.py` 串行完成 DEEP 构图和五种表示查询，每次只保留一种临时压缩编码。旧 `pipeline.py` 已停用，`run_local.sh` 转发到新的完整实验入口。
