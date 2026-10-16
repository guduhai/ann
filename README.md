# ANN 图索引与共享量化实验

本仓库包含 CAGRA、HNSW、NSG、NSSG、VAMANA 五种 CPU 图索引方案，结合共享 `ann_quantization` 中的 INT8、PCA、PQ、Local adaptive INT8 四种量化表示，并保留 float32 基线。正式量化查询不进行全精度重排。

本文是全仓库统一的环境配置入口。各子目录保留算法、文件格式、运行和结果说明。

## 仓库布局

克隆完整仓库并保留以下同级关系。仓库可以改名或移动，不依赖固定用户名和绝对路径。

```text
ann/
├── README.md
├── requirements.txt       # 统一 Python 依赖
├── .venv/                 # 在目标机器创建，共享虚拟环境
├── ann_quantization/      # Python 量化模型、C++ 评分头文件
├── CAGRA/                 # NSG/EFANNA 构图、CAGRA 风格剪枝、CSR 查询
├── HNSW/                  # hnswlib 原生构图与查询
├── NSG/                   # EFANNA 候选图、NSG 构图与查询
├── NSSG/                  # EFANNA 候选图、NSSG 构图与查询
└── VAMANA/                # DiskANN/VAMANA 内存图构建与查询
```

CAGRA 是 CPU 研究实现，不是 NVIDIA CUDA/cuVS CAGRA。共享头文件位于 `ann_quantization/` 根目录，四个索引的 CMake 已直接引用该目录，不需要 `ann_quantization/include/`。

## 平台与版本要求

以下安装、编译和运行命令面向 Ubuntu 24.04 的 Bash，包括 Windows 上的 Ubuntu WSL2。代码包含 Linux x86-64 和 ARM64 分支；已有 CAGRA 实验记录不能替代其余方案或最新构图代码在目标机器上的验证。Windows 原生、Android 和骁龙设备环境未在本文中验证。

| 依赖 | 要求与用途 |
| --- | --- |
| Python | 使用 Ubuntu 24.04 的系统 Python 3.12；其他版本需另行验证固定 Python 依赖的兼容性 |
| GCC/G++ | C++17、OpenMP、pthread；同时用于构建 hnswlib Python 扩展 |
| CMake | **3.20 或更新**，统一满足五种方案；CAGRA 单独要求 3.16 |
| Boost | NSG、NSSG、VAMANA 使用头文件；VAMANA 命令行工具还链接 program_options |
| OpenBLAS、LAPACKE、Linux AIO | VAMANA 原生构图及上游查询目标的链接依赖 |
| Tessil robin-map、sparse-map | 已在 `VAMANA/vendor/diskann/include/tsl/` 中自带头文件，无需另装系统包 |
| NumPy、h5py | 共享量化、HDF5 数据准备和实验工具 |
| psutil、hnswlib | HNSW Python 基准、资源统计及原生索引对照工具 |
| pybind11、setuptools、wheel、Python 开发头文件 | 构建仓库自带的 hnswlib Python 绑定 |

当前顶层 CMake 构建目标不要求 Faiss、CUDA 或 cuVS。VAMANA 的 `src/compat/mkl.h` 将所需接口转接到 OpenBLAS/LAPACKE，无需单独安装 Intel MKL；当前目标未启用上游 TCMalloc 内存释放宏。

四个新增索引项目在 x86-64 使用 `-march=native`，在 ARM64 使用 `-mcpu=native`。请在目标机器本地编译；这些选项不能直接用于跨平台交叉编译，也不应把生成的可执行程序当作其他 CPU 通用版本。

`vendor/` 中上游原始文档保留作来源记录；其中针对完整上游工程的历史安装步骤不能替代本文。

## 统一安装环境

Windows 用户先在 **Windows PowerShell** 中执行 `wsl -l -v` 查看发行版，再用 `wsl -d ANN-Repro` 进入目标环境（替换为实际发行版名称）。如果提示符类似 `user@host:~$`，说明已在 Ubuntu 中，不要再输入裸 `wsl` 命令。

下面的命令均在同一个 Ubuntu Bash 环境执行。先用 `cat /etc/os-release` 确认 Ubuntu 24.04。每个阶段成功后再继续；括号中的 `set -e` 会在错误时停止当前阶段。

```bash
(
set -e
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake libgomp1 \
  python3 python3-dev python3-venv python3-pip \
  libboost-dev libboost-program-options-dev \
  libaio-dev liblapacke-dev libopenblas-dev \
  git curl ca-certificates
/usr/bin/cmake --version
/usr/bin/g++ --version
/usr/bin/python3 --version
)
```

无需安装 Tessil 的系统开发包，相关头文件已经随仓库提供。如果 apt 提示找不到某个包，该安装阶段没有完成；先解决错误，不要直接继续创建虚拟环境或编译。

尚未克隆时，在希望保存项目的父目录执行；已有源码则直接进入现有仓库根目录：

```bash
git clone https://github.com/guduhai/ann.git
cd ann
```

以下命令都从仓库根目录执行。路径带空格或括号时，手动执行 `cd` 应将路径加引号，例如 `cd '/mnt/e/ann(1)'`。

```bash
(
set -e
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip --version
.venv/bin/python -m pip install -r requirements.txt
# 构建工具已由上一条命令安装，使用同一环境构建随仓库提供的绑定。
CC=/usr/bin/gcc CXX=/usr/bin/g++ HNSWLIB_NO_NATIVE=1 \
  .venv/bin/python -m pip install --no-build-isolation ./HNSW/vendor/hnswlib
.venv/bin/python -m pip check
.venv/bin/python -c 'import numpy, h5py, psutil, hnswlib, ann_quantization; print("依赖导入成功")'
)
```

根目录 `requirements.txt` 包含所有自有 Python 工具的第三方包依赖；`hnswlib` 使用上述单独命令安装仓库内源码，避免意外使用其他版本。`HNSWLIB_NO_NATIVE=1` 仅关闭该 Python 扩展的 CPU 特定优化，不改变原生 CMake 目标的编译选项。

全仓库共用根目录 `.venv/`，不在每个索引下创建环境，也无需 `activate`。旧 `CAGRA/.venv/` 不再被启动脚本使用。更换机器时重新创建环境并编译，不复制旧 `.venv/`、`build/` 或 CMake 缓存，不使用 `--system-site-packages`。

NumPy、h5py、psutil 沿用新版本各子项目声明的固定版本；构建工具仍含版本范围，因此这不是完整锁定环境。最终复现记录还应保存实际工具链与安装包版本。

## 编译全部原生目标

配置完环境后，从仓库根目录执行：

```bash
(
set -e
for index in CAGRA HNSW NSG NSSG VAMANA; do
  /usr/bin/cmake -S "$index" -B "$index/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++
  /usr/bin/cmake --build "$index/build" -j2
done
)
```

内存有限时将 `-j2` 改为 `-j1`。每个子目录保留自己的构建目录，避免同名程序互相覆盖：

| 方案 | 构建产物 |
| --- | --- |
| CAGRA | `CAGRA/build/ann_knn`、`ann_build`、`ann_query`、`test_prune` |
| HNSW | `HNSW/build/bin/ann_build`、`hnsw_search`、`ann_query` |
| NSG | `NSG/build/bin/ann_knn`、`ann_build`、`nsg_search`、`ann_query`、`ann_query_official` |
| NSSG | `NSSG/build/bin/ann_knn`、`ann_build`、`nssg_search`、`ann_query`、`ann_query_official` |
| VAMANA | `VAMANA/build/bin/ann_build`、`vamana_search`、`ann_query`、`ann_query_official`、`fvecs_to_bin` |

四个新增索引中的 `ann_query` 是量化查询程序的副本，保留原有 `*_search` 名称兼容现有入口；`ann_query_official` 是额外的上游查询对照程序。它们不能仅凭名称混用。

VAMANA 的 CMake 会查找系统 AIO、LAPACKE、OpenBLAS、Boost program_options。`vendor/deps/` 中的路径是历史本地依赖搜索提示，安装上述 Ubuntu 开发包后由 CMake 查找系统库；无需复制另一台机器的 sysroot 或二进制库。

## 解释器与运行入口

所有 Python 入口使用根目录虚拟环境。从仓库根目录调用：

```bash
# CAGRA 启动器自动切换到 CAGRA/，相对参数按 CAGRA/ 解析。
bash CAGRA/python_pi.sh pipeline.py --help

# 其他四个方案的构图和查询入口。
.venv/bin/python HNSW/run_build.py --help
.venv/bin/python NSG/run_build.py --help
.venv/bin/python NSSG/run_build.py --help
.venv/bin/python VAMANA/run_build.py --help
.venv/bin/python NSG/run_search.py --help

# 共享量化模型导出工具使用模块入口。
.venv/bin/python -m ann_quantization.export_native_model --help
```

`CAGRA/python_pi.sh` 使用根目录 `.venv/bin/python`，清除外部 Python 和动态库路径，并设置单线程环境变量。`CAGRA/run_local.sh` 默认也使用根目录环境，可用 `PYTHON` 显式指定解释器；该自动流程包含编译和测试步骤。

## 数据准备、构图与量化查询

CAGRA 的合成示例、SIFT1M 流程和文件格式见[中文说明](CAGRA/README_CN.md)、[English reference](CAGRA/README.md) 和[运行入口](CAGRA/PI_RUN_CN.md)。它采用独立 CSR 图；其他四种方案保留各自原生图格式，不能互换图文件。

其他四个方案新增 `run_build.py`，参数来自各自 `build_parameters.json`。`--data` 必须指向已存在的数据文件，`--output` 必须是尚不存在的构图输出目录；默认打印命令，加 `--execute` 才会构图。输入格式为：

| 方案 | 构图输入 |
| --- | --- |
| HNSW、NSG、NSSG | 底库 `.fvecs`，HNSW 原生构图也读取此格式 |
| VAMANA | 底库 `.fbin`，可用构建出的 `fvecs_to_bin` 转换 |

`NSG/src/prepare_dataset.py` 提供 HDF5 到 fvecs/ivecs 的全量和子集转换；HNSW 的 `src/prepare_codes.py`、`src/export_compact_graph.py` 提供共享编码准备及紧凑索引导出。后者要求原始 HNSW 索引与编码节点顺序一致。上述工具均使用根目录 `.venv/bin/python` 启动，具体参数见源码中的命令行定义。

`run_build.py` 解析相对 `--data`、`--output` 时以调用者当前目录为基准。`run_search.py` 的默认输入则按仓库结构定位：

| 查询输入 | 默认位置 |
| --- | --- |
| HNSW、NSG、NSSG 查询向量 | `data/sift1M/sift_query.fvecs` |
| NSG、NSSG float32 底库 | `data/sift1M/sift_base.fvecs` |
| VAMANA 查询/float32 底库 | `data/sift1M/sift1m_query.fbin`、`sift1m_base.fbin` |
| 原生图 | 各自 `experiment.json` 的 `graph` 字段指定，通常位于各自 `indexes/` |
| 量化编码和模型 | `ann_quantization/prepared/<method>/`；导出模型位于其中 `native/` |
| HNSW 量化编码 | `ann_quantization/prepared/hnsw-order/<method>/`，另需配置中指定的紧凑图 |

四个查询入口的方法名为 `exact`、`int8`、`pq64`、`pca80`、`local`。PQ M64 和 PCA80 是两种独立表示，不是先 PCA80 再 PQ M64。查询程序从实际图推导底库数量、从查询文件推导查询数量，不再限定百万底库或 10,000 条查询。当前表示与查询距离仍要求 **128 维**，不是任意维度实现。

四个 `run_search.py` 支持以下覆盖参数：

| 参数 | 含义 |
| --- | --- |
| `--graph` | 本次构图文件；HNSW 量化模式必须指定紧凑图 |
| `--base`、`--query` | 底库与查询文件；量化模式忽略底库参数，不打开全精度底库 |
| `--codes` | 当前方法包含 `codes.npy` 的目录；HNSW 使用内部 ID 顺序 |
| `--model` | 当前方法包含原生模型 `.bin` 文件的目录 |
| `--queries N` | 读取前 N 条查询；0 表示全部，随仓库提供的配置默认为 0 |
| `--W` / `--search-l` | 搜索宽度，至少 10，且不超过实际节点数；NSSG 的随机初始化要求严格小于节点数 |
| `--warmup` | 预热条数；省略时取配置值与实际查询条数的较小值，显式指定超出范围时报错 |
| `--repeats` | 重复次数，至少 1；省略时读取配置的 repeats |
| `--config` | 可选实验 JSON，省略时读取当前索引的 experiment.json |

显式输入路径按调用者当前目录解析；配置中的默认图路径仍相对索引目录。`--output` 的相对路径也相对索引目录，输出必须位于该索引的 `results/` 内，且不能覆盖已有文件。入口固定单线程、Top-10、无重排。

例如，已有 128 维小规模 fvecs 数据时，可从仓库根目录执行：

```bash
.venv/bin/python NSG/run_build.py \
  --data data/siftsmall/siftsmall_base.fvecs --output NSG/indexes/small --execute
.venv/bin/python NSG/run_search.py --method exact \
  --graph NSG/indexes/small/sift.nsg \
  --base data/siftsmall/siftsmall_base.fvecs \
  --query data/siftsmall/siftsmall_query.fvecs \
  --queries 0 --W 50 --warmup 20 --repeats 3 \
  --output results/small-exact.ids --execute
```

编码准备工具 `HNSW/src/prepare_codes.py` 按实际节点数量工作，各方法的训练样本数不会超过底库条数；PQ K256 要求底库至少 256 条。紧凑图导出不再要求百万节点。构图参数也必须适合数据量：例如 NSG/NSSG 默认候选邻居数为 200，小数据测试建议从 1,000 条以上开始。子集真值必须针对该子集重新计算，不能直接截取百万底库的真值 ID。

构图输出目录和查询配置不会自动联动：应让图路径、编码节点顺序、模型和数据相互匹配。HNSW float32 基线加载原生索引中保存的全精度向量；量化模式使用紧凑索引。正式量化查询不恢复全精度重排。完整实验还需准备格式匹配的图、编码与模型，安装环境不会自动生成它们。

## 环境检查

安装和编译完成后，可执行以下检查：

```bash
.venv/bin/python -m pip check
.venv/bin/python -c 'import numpy, h5py, psutil, hnswlib, ann_quantization'
ctest --test-dir CAGRA/build --output-on-failure
```

这些命令只检查依赖加载和 CAGRA 剪枝功能；完整实验还需分别验证构图、查询、召回和内存。CAGRA 的运行入口见其运行说明。

## 常见环境问题

| 问题 | 处理 |
| --- | --- |
| 缺少 `boost/dynamic_bitset.hpp` | 在实际编译的 Ubuntu/WSL 安装 `libboost-dev` |
| VAMANA 找不到 aio/lapacke/openblas/boost_program_options | 安装对应开发包，重新配置该项目的 CMake |
| 缺少 `tsl/robin_map.h` 或 `tsl/sparse_map.h` | 检查 `VAMANA/vendor/diskann/include/tsl/` 是否完整，并使用本项目顶层 CMake 构建 |
| 缺少 `native_search_scorer.hpp` | 保留完整仓库；头文件在 `ann_quantization/` 根目录 |
| `ensurepip is not available` 或虚拟环境中没有 pip | 确认 apt 阶段成功；系统 Python 3.12 可安装 `python3.12-venv` 后重新运行 `/usr/bin/python3 -m venv .venv`，再检查 `.venv/bin/python -m pip --version` |
| 找不到 CMake 或 C++ 编译器检测失败 | 确认 `cmake build-essential python3-dev` 已安装，检查系统工具版本；hnswlib 的通用“至少 C++11”报错不能单独证明编译器版本过旧 |
| hnswlib Python 绑定构建缺包 | 先安装根目录 requirements，再按上述命令使用 `--no-build-isolation` 安装仓库自带源码 |
| 混入其他软件环境的 Python、CMake 或动态库 | 清除外部环境变量，显式使用根目录 `.venv/bin/python`，必要时指定 `/usr/bin/cmake`、`/usr/bin/g++` |
| Bash 提示 `$'\r'` 等换行错误 | 根目录 `.gitattributes` 已指定 `.sh` 使用 LF；运行时使用更新后的脚本 |
