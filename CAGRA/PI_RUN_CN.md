# CAGRA 本机数据准备与完整流程

文件名保留历史名称；本页适用于 Linux、WSL 和树莓派。环境依赖按仓库根目录 README 安装，平台编译及已有 fvecs/编码的运行方法见 [BUILDING.md](BUILDING.md)。Android 不运行本页的 Python 数据准备流程，直接使用已有数据和共享编码。

## 编译与入口

克隆完整仓库，保留 `CAGRA/` 与 `ann_quantization/` 同级。以下命令从 `CAGRA/` 执行：

```bash
cmake -S . -B build/linux-arm64 -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux-arm64 -j2
```

`linux-arm64` 是目录名称，在 WSL 可改为 `linux-x86_64`，并同步修改下方 `BIN`。安装系统 `time` 软件包后，构图/查询入口会记录进程耗时和峰值内存。

`python_pi.sh` 根据自身位置调用根目录 `.venv/bin/python`，清除外部 Python/动态库路径并关闭字节码写入。它切换到 `CAGRA/` 执行，相对路径按该目录解释。其 `OMP_NUM_THREADS=1` 不限制构图：C++ 构图入口会显式设置 `--threads` 指定的线程数，默认 4。

## SIFT1M 全量流程

将 `DATASET` 替换为已有 HDF5 文件的绝对路径；需要下载时可使用 [ANN-Benchmarks SIFT](https://ann-benchmarks.com/sift-128-euclidean.hdf5)。输出目录必须尚不存在：

```bash
DATASET=/absolute/path/sift-128-euclidean.hdf5
OUT=results/sift1m-m64-pca80
BIN=build/linux-arm64/bin

bash python_pi.sh pipeline.py prepare --output "$OUT" \
  --dataset "$DATASET" --vectors 0 --queries 10000
bash python_pi.sh pipeline.py build --output "$OUT" --bin "$BIN" \
  --threads 4 --gk 64 --knn-l 80 --iterations 8 --sample 10 --reverse 100 \
  --degree 32 --build-l 64 --build-c 132
bash python_pi.sh pipeline.py encode --output "$OUT" \
  --train 10000 --pca-dim 80 --pq-m 64 --codec-iterations 10
bash python_pi.sh pipeline.py query --output "$OUT" --bin "$BIN" \
  --methods float32,int8,pca,pq,local --search-l 50 --warmup 200 --repeats 3
```

PQ 固定每个子空间 256 个码字，Local 固定 32 个簇，与共享原生评分器一致。仅支持 128 维输入；`--pq-m` 必须整除 128，`--pca-dim` 为 1–128。训练样本至少 256 条。

构图与训练需要额外内存，底库、中间图、模型和编码需要数 GB 磁盘。构图配置 4 线程，其中反向边更新和连通性修复仍串行；查询保持单线程、无全精度重排。

需要小规模数据时，在新输出目录准备 `--vectors 20000 --queries 500`，其余阶段不变。子集真值会针对实际底库重新计算。已有全量真值不能直接用于底库子集。

## 无需下载的示例

以下生成固定种子的合成数据并计算 Top-10，展示完整流程，不代表 SIFT 性能：

```bash
OUT=results/synthetic-1024
BIN=build/linux-arm64/bin
bash python_pi.sh pipeline.py prepare --output "$OUT" \
  --vectors 1024 --queries 32
bash python_pi.sh pipeline.py build --output "$OUT" --bin "$BIN" --threads 4
bash python_pi.sh pipeline.py encode --output "$OUT" \
  --train 1024 --pca-dim 80 --pq-m 64 --codec-iterations 10
bash python_pi.sh pipeline.py query --output "$OUT" --bin "$BIN" \
  --search-l 50 --warmup 20 --repeats 3
```

## 产物与再次查询

所有辅助流程产物均在 `CAGRA/results/` 内：

- `base.f32`、`queries.f32`、`truth.i32`、`dataset.json`：数据矩阵、真值与哈希。
- `graph/candidate.knn`、`graph/final.graph`：候选图与原生邻接图。
- `graph/knn.stdout`、`graph/knn.time`、`graph/build.stdout`、`graph/build.time`、`build.json`：构图日志、时间/内存和参数。
- `codecs/<method>/codes.npy`、`native/*.bin`、`codec.json`、`codec.npz`：共享编码和模型。
- `codecs/manifest.json`：图、编码及模型校验信息。
- `query-results/summary.json`、`query-results/RESULTS_CN.md`：指标与结果表；每组另有 `.json/.labels/.stdout/.time`。

复用本次新格式的图和编码时，指定尚不存在的结果子目录：

```bash
bash python_pi.sh pipeline.py query --output "$OUT" --bin "$BIN" \
  --run-name query-next --search-l 20,50,100,200 --warmup 20 --repeats 3
```

`--warmup` 不得超过实际查询条数。更改训练/编码参数请使用新的实验目录。本页流程与历史 CSR / `EGCODE01` 编码不兼容；历史报告可保留，但旧产物不能直接交给新入口。所有生成产物均被 Git 忽略。
