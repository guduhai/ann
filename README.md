# ANN 图索引与共享量化实验

五种 CPU 图索引（CAGRA、HNSW、NSG、NSSG、VAMANA）配合共享的 `ann_quantization` 量化表示：INT8、PQ（M64）、PCA（80 维）、Local adaptive INT8，并保留 float32 基线。量化查询只使用编码和模型，不加载 float32 底库，也不做全精度重排。

本文只讲两个目标平台的部署：

- **树莓派**：在树莓派上直接编译并运行（aarch64 Linux）。
- **高通骁龙**：在 Ubuntu 主机上用 Android NDK 交叉编译，经 ADB 推送到手机 CPU 上运行。

数据、向量维度、编码和模型都与平台无关，两个平台共用同一套数据准备流程。

## 仓库布局

```text
ann/
├── README.md
├── requirements.txt       # 统一 Python 依赖
├── .venv/                 # 在目标机器上创建，不随仓库分发
├── ann_quantization/      # Python 量化模型与 C++ 评分头文件
├── CAGRA/                 # EFANNA 候选图 + CAGRA 风格剪枝 + CSR 查询
├── HNSW/                  # hnswlib 构图与查询
├── NSG/                   # EFANNA 候选图 + NSG 构图与查询
├── NSSG/                  # EFANNA 候选图 + NSSG 构图与查询
└── VAMANA/                # DiskANN/VAMANA 内存图构图与查询
```

各索引的平台编译细节见 [CAGRA](CAGRA/BUILDING.md)、[HNSW](HNSW/BUILDING.md)、[NSG](NSG/BUILDING.md)、[NSSG](NSSG/BUILDING.md)、[VAMANA](VAMANA/BUILDING.md)。

## 数据与表示

实验数据为 SIFT1M（100 万条 128 维向量，10,000 条查询，Top-10 真值）。当前表示与查询距离只支持 **128 维**。

| 用途 | 格式与位置 |
| --- | --- |
| 底库、查询 | HNSW、NSG、NSSG、CAGRA 用 `.fvecs`；VAMANA 用 `.fbin` |
| 真值 | `.ivecs`，必须针对实际底库计算，不能截取子集 |
| 量化编码与模型 | `ann_quantization/prepared/<method>/`，导出模型在其中的 `native/` |
| HNSW 量化编码 | `ann_quantization/prepared/hnsw-order/<method>/`，须与紧凑图的内部节点顺序一致 |

查询方法名：`exact`（float32 基线）、`int8`、`pq64`、`pca80`、`local`。PQ M64 与 PCA80 是两种独立表示，不是先降维再量化。

查询固定单线程、Top-10、无重排。搜索宽度默认取各索引 `experiment.json` 中的 `W`（实验报告中为 50）。

### 准备数据与编码

HDF5 转 fvecs/ivecs：

```bash
.venv/bin/python NSG/src/prepare_dataset.py --help
```

VAMANA 需要的 `.fbin` 由编译出的 `fvecs_to_bin` 转换。

四种量化编码和原生模型由 HNSW 的构图结果生成，一次产出两套编码：按原始数据集 ID 排序（供 NSG、NSSG、VAMANA 共用）和按 HNSW 内部节点顺序排序（供 HNSW 量化查询）：

```bash
.venv/bin/python HNSW/src/prepare_codes.py \
  --index HNSW/indexes/hnsw-1m.bin \
  --output ann_quantization/prepared
```

底库至少 256 条（PQ 的 K256 要求）。输出目录必须尚不存在。HNSW 量化查询还要用 `HNSW/src/export_compact_graph.py` 导出与该编码顺序对应的紧凑图。

CAGRA 有自己的端到端流程（HDF5 准备、构图、编码、查询），见 [CAGRA/PI_RUN_CN.md](CAGRA/PI_RUN_CN.md)。

## 共同约定

- 所有 Python 入口使用仓库根目录的 `.venv/bin/python`，不需要 `activate`。
- `run_build.py`、`run_search.py` 默认只打印命令，加 `--execute` 才真正运行。
- 构图输出目录和查询输出都不能覆盖已有结果，每次运行使用新路径。
- 图、编码、模型和底库必须对应同一份数据的同一节点顺序。
- 四个新增索引的 Linux 构建默认按本机 CPU 优化（x86-64 用 `-march=native`，ARM64 用 `-mcpu=native`），请在目标机器上编译，不要把二进制拷到别的机器上运行。

---

# 树莓派部署

在树莓派本机（aarch64，Debian/Ubuntu 系）编译并运行。以下命令在树莓派的 Bash 中、从仓库根目录执行。

## 1. 安装依赖并创建环境

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake libgomp1 time \
  python3 python3-dev python3-venv python3-pip \
  libboost-dev libboost-program-options-dev \
  libaio-dev liblapacke-dev libopenblas-dev \
  git curl ca-certificates
```

| 依赖 | 用途 |
| --- | --- |
| CMake 3.20+、GCC（C++17）、OpenMP | 全部五个索引 |
| `time` | CAGRA 入口用它记录耗时和峰值内存 |
| Boost 头文件 | NSG、NSSG、VAMANA |
| Boost program_options、OpenBLAS、LAPACKE、libaio | 仅 VAMANA；MKL 不需要，`src/compat/mkl.h` 已转接到 OpenBLAS |
| Python 3 及开发头文件 | 启动脚本与 hnswlib 绑定 |

不需要 Faiss、CUDA、cuVS，也不需要单独安装 Tessil（头文件已随 `VAMANA/vendor/diskann/include/tsl/` 提供）。

```bash
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
CC=/usr/bin/gcc CXX=/usr/bin/g++ HNSWLIB_NO_NATIVE=1 \
  .venv/bin/python -m pip install --no-build-isolation ./HNSW/vendor/hnswlib
.venv/bin/python -m pip check
.venv/bin/python -c 'import numpy, h5py, psutil, hnswlib, ann_quantization; print("依赖导入成功")'
```

换机器时重新创建 `.venv/` 并重新编译，不要复制旧的 `.venv/` 或 `build/`。

## 2. 编译

```bash
for index in CAGRA HNSW NSG NSSG VAMANA; do
  cmake -S "$index" -B "$index/build/linux-arm64" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$index/build/linux-arm64" -j2
done
```

内存紧张时把 `-j2` 改为 `-j1`。`linux-arm64` 只是目录名，架构由本机决定。产物位于各索引的 `build/linux-arm64/bin/`：

| 索引 | 构图程序 | 查询程序 |
| --- | --- | --- |
| CAGRA | `ann_knn`、`ann_build` | `ann_query` |
| HNSW | `ann_build` | `hnsw_search` |
| NSG | `ann_knn`、`ann_build` | `nsg_search` |
| NSSG | `ann_knn`、`ann_build` | `nssg_search` |
| VAMANA | `ann_build`、`fvecs_to_bin` | `vamana_search` |

## 3. 构图

`--data` 指向底库，`--output` 是尚不存在的输出目录，`--binary-dir` 指向上一步的产物目录。

```bash
BIN=HNSW/build/linux-arm64/bin
.venv/bin/python HNSW/run_build.py \
  --data data/sift1M/sift_base.fvecs \
  --output HNSW/indexes/run1 \
  --binary-dir "$BIN" --execute
```

NSG、NSSG 同样接受 `.fvecs`，输出分别为 `sift.nsg`、`sift.ssg`（另有候选图 `sift_200nn.graph`）。VAMANA 接受 `.fbin`，图前缀为 `sift1m_r50_l70_a2`。HNSW 输出 `hnsw-1m.bin`。构图线程数取各自 `build_parameters.json`。

CAGRA 的构图输出必须位于 `CAGRA/` 内，且支持 `--threads`：

```bash
.venv/bin/python CAGRA/run_build.py \
  --data data/sift1M/sift_base.fvecs \
  --output CAGRA/indexes/run1 \
  --binary-dir CAGRA/build/linux-arm64/bin --threads 4 --execute
```

产物为 `candidate.knn` 和 `final.graph`，以及 `.stdout`、`.time` 日志。

## 4. 查询

`--method` 取 `exact`、`int8`、`pq64`、`pca80`、`local`。量化方法需要 `--codes` 和 `--model`，不读取底库；`exact` 需要 `--base`（HNSW 除外，它的 float32 基线直接使用图中保存的向量）。`--output` 的相对路径按索引目录解释，且必须落在该索引的 `results/` 内。

```bash
BIN=NSG/build/linux-arm64/bin
.venv/bin/python NSG/run_search.py --method pq64 \
  --graph NSG/indexes/run1/sift.nsg \
  --query data/sift1M/sift_query.fvecs \
  --codes ann_quantization/prepared/pq64 \
  --model ann_quantization/prepared/pq64/native \
  --binary-dir "$BIN" \
  --output results/run1/pq64 --execute
```

HNSW 的量化查询必须改用紧凑图和内部顺序编码：

```bash
.venv/bin/python HNSW/run_search.py --method pq64 \
  --graph HNSW/indexes/hnsw-1m-codeids.bin \
  --query data/sift1M/sift_query.fvecs \
  --codes ann_quantization/prepared/hnsw-order/pq64 \
  --model ann_quantization/prepared/pq64/native \
  --binary-dir HNSW/build/linux-arm64/bin \
  --output results/run1/pq64 --execute
```

CAGRA 的查询额外需要 `--truth`，输出是前缀而非单个文件（生成 `.json`、`.labels`、`.stdout`、`.time`）：

```bash
.venv/bin/python CAGRA/run_search.py --method pq64 \
  --graph CAGRA/indexes/run1/final.graph \
  --query data/sift1M/sift_query.fvecs \
  --truth data/sift1M/groundtruth.ivecs \
  --codes ann_quantization/prepared/pq64 \
  --model ann_quantization/prepared/pq64/native \
  --binary-dir CAGRA/build/linux-arm64/bin \
  --output results/run1/pq64 --execute
```

常用覆盖参数：`--queries N`（0 为全部）、`--W`（搜索宽度，至少 10）、`--warmup`、`--repeats`。

树莓派上的全量实验结果见 [树莓派完整实验结果.md](树莓派完整实验结果.md)。

---

# 高通骁龙部署

在 Ubuntu 主机（含 WSL2）上交叉编译，通过 ADB 在手机 CPU 上运行。构图、图遍历、量化距离计算和计时全部在手机上完成，主机只负责编译、传文件和取结果。手机不需要安装 Python、CMake 或编译器。

已验证的设备为 Android 14、骁龙 8 Gen 3，目标 ABI 为 `arm64-v8a`。Linux ARM64 二进制不能在 Android 上运行，两边的构建目录也不能混用。

## 1. 主机准备

主机需要：

- 仓库根目录的 `.venv`（按上文树莓派一节的 Python 步骤创建；该步骤同样适用于 Ubuntu 主机）。
- Android SDK Platform-Tools，提供 `adb`。
- Android NDK r26c。

确认手机已开启 USB 调试且被识别：

```bash
adb devices -l          # 目标设备状态应为 device
export ANDROID_NDK="$HOME/Android/Sdk/ndk/26.2.11394342"
```

## 2. 交叉编译

五个索引使用同一组 NDK 参数：

```bash
cmake -S CAGRA -B CAGRA/build/android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release
cmake --build CAGRA/build/android-arm64 -j2
```

把 `CAGRA` 换成 `HNSW`、`NSG`、`NSSG`、`VAMANA` 即编译其余索引。额外的依赖只在 Android 构建时需要，Linux 系统库不能替代：

| 索引 | 额外 CMake 参数 |
| --- | --- |
| CAGRA、HNSW | 无 |
| NSG、NSSG | `-DANN_BOOST_INCLUDE_DIR=/path/to/boost-1.71/include` |
| VAMANA | 上述 Boost 头文件，另加 `-DANN_SIMDE_ROOT=/path/to/simde-0.8.2` 和 `-DANN_BOOST_PROGRAM_OPTIONS_SOURCE=/path/to/program_options-boost-1.71.0` |

Boost 1.71 与 SIMDe 0.8.2 是移植时使用的版本，可放在被 Git 忽略的 `.deps/` 或仓库外部。VAMANA 的 Android 构建只支持 float 内存图（`--build_PQ_bytes 0`），不包含上游的磁盘索引与 PQ/OPQ 构图。

CAGRA 的 Android 构图默认静态链接 OpenMP，且 `ann_knn` 带小块内存池以避免 Scudo 分配器映射耗尽，可用 `-DANN_ANDROID_KNN_POOL=OFF` 关闭。

## 3. 推送程序与数据

每个索引推到手机上各自的目录，避免同名的 `ann_build`、`ann_knn` 互相覆盖：

```bash
DEVICE=手机的ADB序列号
adb -s "$DEVICE" shell mkdir -p /data/local/tmp/ann/bin/cagra
adb -s "$DEVICE" push CAGRA/build/android-arm64/bin/. /data/local/tmp/ann/bin/cagra/
adb -s "$DEVICE" shell chmod 755 \
  /data/local/tmp/ann/bin/cagra/ann_knn \
  /data/local/tmp/ann/bin/cagra/ann_build \
  /data/local/tmp/ann/bin/cagra/ann_query
```

其余索引同理，目标目录分别为 `bin/hnsw`、`bin/nsg`、`bin/nssg`、`bin/vamana`，这些也是启动脚本在未指定 `--binary-dir` 时的默认查找位置。

数据、编码和模型用 `adb push` 传到手机。实验中沿用的目录约定：

```text
/data/local/tmp/ann/datasets/sift1m/   # 底库与查询；VAMANA 用 fbin，其余用 fvecs
/data/local/tmp/ann/assets/            # groundtruth.ivecs
/data/local/tmp/ann/codes/sift1m/quant/<method>/     # codes.npy
/data/local/tmp/ann/models/sift1m/quant/<method>/native/
/data/local/tmp/ann/indexes/<索引名>/  # 构图输出（运行时新建）
/data/local/tmp/ann/results/<索引名>/  # 查询输出（运行时新建）
```

## 4. 在手机上构图和查询

启动脚本仍在主机上运行，通过 `--target android --device` 经 ADB 执行。所有数据与输出参数都必须是**手机上的绝对路径**，不是主机路径。构图输出必须是 `indexes/<索引名>/` 下尚不存在的目录，查询输出必须位于 `results/<索引名>/` 内。`--device-root` 可整体改掉 `/data/local/tmp/ann` 这个根。

```bash
DEVICE=手机的ADB序列号
P=/data/local/tmp/ann

.venv/bin/python CAGRA/run_build.py --target android --device "$DEVICE" \
  --data "$P/datasets/sift1m/sift_base.fvecs" \
  --output "$P/indexes/cagra/run1" --threads 4 --execute

.venv/bin/python CAGRA/run_search.py --target android --device "$DEVICE" \
  --method pq64 \
  --graph "$P/indexes/cagra/run1/final.graph" \
  --query "$P/datasets/sift1m/sift_query.fvecs" \
  --truth "$P/assets/groundtruth.ivecs" \
  --codes "$P/codes/sift1m/quant/pq64" \
  --model "$P/models/sift1m/quant/pq64/native" \
  --output "$P/results/cagra/run1/pq64" --execute
```

其他索引把脚本路径换成对应目录即可，参数差异与树莓派一节相同：NSG/NSSG/VAMANA 的 `exact` 需要 `--base`，量化方法需要 `--codes` 和 `--model`，HNSW 量化查询使用紧凑图与内部顺序编码。不加 `--execute` 时只打印命令，但会通过 ADB 只读检查设备上的文件。

图和结果保存在手机上，用 `adb pull` 按实际路径取回。

手机上的全量实验结果见 [高通完整实验结果.md](高通完整实验结果.md)。

---

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 缺少 `boost/dynamic_bitset.hpp` | 树莓派安装 `libboost-dev`；Android 构建检查 `ANN_BOOST_INCLUDE_DIR` |
| VAMANA 找不到 aio、lapacke、openblas 或 boost_program_options | 安装对应开发包后重新配置 CMake |
| 缺少 `tsl/robin_map.h` | 确认 `VAMANA/vendor/diskann/include/tsl/` 完整，并用本仓库的 CMake 构建 |
| 缺少 `native_search_scorer.hpp` | 头文件在 `ann_quantization/` 根目录，保持仓库完整 |
| `ensurepip is not available` | 安装 `python3-venv` 后重新创建 `.venv` |
| hnswlib 绑定构建失败 | 先装好 `requirements.txt`，再按上文命令加 `--no-build-isolation` 安装仓库自带源码 |
| 用了错误环境的 Python 或动态库 | 清掉 `PYTHONPATH`、`LD_LIBRARY_PATH`，显式使用 `.venv/bin/python` |
| Android 上构图中途因内存映射失败退出 | 确认使用的是 Android 构建（CAGRA 的 `ann_knn` 默认带分配池），不要把 Linux 二进制推到手机 |
| 脚本报 `$'\r'` 换行错误 | `.gitattributes` 已指定 `.sh` 用 LF，拉取最新版本后重试 |
