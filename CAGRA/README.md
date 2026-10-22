# CAGRA CPU：Linux / Android 统一实现

使用 NSG/EFANNA 构图与 CAGRA 风格 detour-count 剪枝，支持 float32、INT8、PCA80、PQ M64、Local 查询。构图默认 4 线程，查询单线程，无全精度重排，不依赖 Faiss。

- [实现、数据格式与运行说明](README_CN.md)
- [Linux / Android 编译与统一启动入口](BUILDING.md)
- [本机数据准备与共享量化辅助流程](PI_RUN_CN.md)
- [第三方源码与修改说明](THIRD_PARTY_NOTICES.md)

两个平台编译同一套 `src/`、`include/`、`vendor/`。量化直接复用同级 `ann_quantization`，不保存私有算法副本。最新统一入口读取原生邻接图及共享 `codes.npy` / 原生模型；历史 CSR / `EGCODE01` 编码不兼容。
