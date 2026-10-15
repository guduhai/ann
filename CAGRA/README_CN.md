# CPU 图检索：NSG + CAGRA 风格剪枝

[English](README.md) | [仓库首页](../README.md)

本项目使用原生 NSG/EFANNA 构建近邻图，并在 NSG 的初次剪枝和反向边溢出剪枝中使用 CAGRA 风格的 detour-count 规则。查询由独立 C++ 程序执行，支持 float32、INT8、PCA、PQ 和 Local adaptive INT8。

这是 CPU 上的研究实现，不是 NVIDIA CUDA/cuVS CAGRA。构图和查询不依赖 Faiss，不读取 `.faiss` 索引；量化训练与编码使用本仓库的 `ann_quantization` 和 NumPy。量化查询不执行全精度重排。

## 平台与依赖

面向 Ubuntu Linux x86-64、ARM64，以及 Windows 上的 Ubuntu WSL2。已有实验记录覆盖 Ubuntu 24.04 / WSL x86-64 和树莓派 Ubuntu 24.04 ARM64；Python 环境和 C++ 可执行程序需在目标机器重新创建、编译。

Windows 原生运行和 Android 尚未验证。查询内存统计使用 Linux `/proc/self/status`，测试还使用 `ldd`，因此以下命令应在 Linux 或 WSL 的 Bash 中执行。

| 依赖 | 要求 |
| --- | --- |
| Python | 3.11 或更新版本；推荐 Ubuntu 24.04 自带的 Python 3.12 |
| C++ 编译器 | 支持 C++17 和 OpenMP 的 GCC/G++ |
| CMake | 3.16 或更新版本 |
| Python 包 | `requirements.txt` 中的 NumPy、h5py |

无需安装 Faiss、CUDA、cuVS 或 Boost。当前依赖清单使用版本范围，并非锁定环境；跨机器的耗时及浮点结果可能存在差异。

## 仓库布局

请克隆完整仓库，并保留以下同级关系。仓库本身可以放在任意目录，也可以改名。

```text
ann/
├── ann_quantization/
│   ├── __init__.py
│   └── codecs.py
└── CAGRA/
    ├── ann_codecs.py
    ├── pipeline.py
    ├── python_pi.sh
    ├── requirements.txt
    ├── CMakeLists.txt
    ├── include/
    ├── src/
    ├── tests/
    └── vendor/
```

`ann_codecs.py` 根据自身位置加载 `../ann_quantization/codecs.py`，不使用固定的树莓派路径。不能仅下载 `CAGRA/` 后遗漏共享量化源码。

## 从零安装

Windows 用户先进入已安装的 Ubuntu WSL2，例如在 PowerShell 中执行 `wsl -d Ubuntu-24.04`。以下安装和运行命令均在 Ubuntu Bash 中执行。

安装工具；Ubuntu 的 GCC 提供 OpenMP 编译支持，`libgomp1` 提供运行库：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libgomp1 python3 python3-venv python3-pip curl ca-certificates
```

进入克隆后的仓库根目录，再创建 CAGRA 自己的虚拟环境：

```bash
cd CAGRA
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

如果已有 `.venv` 是从另一台机器复制来的，应先移走，再重新创建。不要使用 `--system-site-packages`；`.venv/pyvenv.cfg` 中的 `include-system-site-packages` 应为 `false`。

编译 C++ 程序：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j2
```

下面的命令都从 `CAGRA/` 目录执行。`python_pi.sh` 虽然保留历史名称，现在是通用入口：它直接使用脚本所在目录的 `.venv/bin/python`，清除外部 Python 包路径并切换到 `CAGRA/`，无需执行 `activate`。也可通过脚本的绝对路径从其他目录启动；传入的相对文件路径会相对于 `CAGRA/` 解析。

## 无需下载数据的合成示例

生成 1,024 条 128 维向量、32 条查询，构图、生成五种表示并查询：

```bash
OUT=results/synthetic-1024
bash python_pi.sh pipeline.py prepare --output "$OUT" \
  --vectors 1024 --queries 32 --dim 128
bash python_pi.sh pipeline.py build --output "$OUT"
bash python_pi.sh pipeline.py encode --output "$OUT" \
  --train 1024 --pca-dim 80 --pq-m 64 --pq-k 256 \
  --clusters 32 --codec-iterations 10
bash python_pi.sh pipeline.py query --output "$OUT" \
  --search-l 20,50,100,200 --warmup 20 --repeats 3
```

不指定 `--dataset` 时，`prepare` 使用固定随机种子 `20261014` 生成数据并计算精确 Top-10。合成数据用于检查完整流程，不代表 SIFT1M 的召回率或速度。

也可使用 `run_local.sh` 自动编译并运行完整流程。该脚本默认使用 `python3`，因此必须显式指定项目虚拟环境：

```bash
PYTHON="$PWD/.venv/bin/python" DATASET= \
OUTPUT=results/synthetic-next VECTORS=1024 QUERIES=32 WARMUP=20 REPEATS=3 \
bash run_local.sh
```

输出必须位于 `CAGRA/` 内，且新实验的输出目录必须尚不存在；重新运行时换一个目录名。编码和查询输出也不覆盖已有结果。

## SIFT1M 数据集与完整实验

使用 [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) 提供的 [SIFT-128 Euclidean HDF5](https://ann-benchmarks.com/sift-128-euclidean.hdf5)。它对应 1,000,000 条 128 维底库向量和 10,000 条查询。

将数据集下载到仓库外的缓存目录，避免把它提交到 Git：

```bash
DATASET="${XDG_CACHE_HOME:-$HOME/.cache}/ann-datasets/sift-128-euclidean.hdf5"
mkdir -p "$(dirname "$DATASET")"
curl --fail --location --output "$DATASET" \
  https://ann-benchmarks.com/sift-128-euclidean.hdf5
```

`prepare` 读取以下 HDF5 数据集：

| 键 | 形状与含义 |
| --- | --- |
| `train` | `(N, D)`，底库向量；读取后转为 float32 |
| `test` | `(Q, D)`，查询向量；维度须与底库相同 |
| `neighbors` | `(Q, K)`，`K >= 10`，按精确欧氏距离排序的邻居 ID；ID 从 0 开始，对应 `train` 的行号 |

ANN-Benchmarks 文件中的 `distances` 和其他附加数据集不被本流程读取。输入向量应为有限数值。全量实验直接使用 `neighbors` 的前 10 列；子集实验会重新计算子集的精确 Top-10，不能沿用全量邻居 ID。

完成上面的环境配置、编译和下载后，在同一 Bash 会话中运行：

```bash
OUT=results/sift1m-m64-pca80
bash python_pi.sh pipeline.py prepare --output "$OUT" \
  --dataset "$DATASET" --vectors 0 --queries 10000
bash python_pi.sh pipeline.py build --output "$OUT" \
  --gk 64 --knn-l 80 --iterations 8 --sample 10 --reverse 100 \
  --degree 32 --build-l 64 --build-c 132
bash python_pi.sh pipeline.py encode --output "$OUT" \
  --train 10000 --pca-dim 80 --pq-m 64 --pq-k 256 \
  --clusters 32 --codec-iterations 10
bash python_pi.sh pipeline.py query --output "$OUT" \
  --methods float32,int8,pca,pq,local \
  --search-l 20,50,100,200 --warmup 200 --repeats 3
```

`--vectors 0` 表示使用全部 `train`。完整建图当前为单线程。数据准备、候选图、最终图和五种表示会占用数 GB 磁盘，构图与编码训练还需要额外内存；按目标机器可用资源选择全量或子集实验。

先尝试较小的 SIFT 子集时，可将输出目录改为新名称，将 `prepare` 参数改成 `--vectors 20000 --queries 500`，其余阶段继续使用该输出目录。PQ 要求维度能被子空间数整除：128 维可使用 M64，96 维可显式指定 `--pq-m 32`。训练样本数不能少于 PQ 码字数或 Local 簇数。

已有图和编码可重复查询，使用新的查询目录名：

```bash
bash python_pi.sh pipeline.py query --output "$OUT" \
  --run-name query-again --search-l 20,50,100,200 --warmup 200 --repeats 3
```

## 测试命令

以下命令供使用者自行验证。`ctest` 检查剪枝规则；`verify.py` 针对已完成构图和编码的小数据集，检查五种距离与 NumPy 对照、穷举搜索、隐藏全精度底库后的量化查询，以及异常文件拒绝行为：

```bash
ctest --test-dir build --output-on-failure
bash python_pi.sh tests/verify.py \
  --case results/synthetic-1024 --bin build --output-name verification
```

`verify.py` 限制底库不超过 4,096 条；不要对 SIFT1M 全量目录运行。重复验证应选择新的 `--output-name`。

需要 AddressSanitizer / UndefinedBehaviorSanitizer 时，可另建目录，再将验证命令的 `--bin` 改为 `build-asan`：

```bash
cmake -S . -B build-asan -DANN_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
cmake --build build-asan -j2
```

## 输出与评测口径

| 路径（相对于实验输出目录） | 内容 |
| --- | --- |
| `dataset.json` | 数据来源、规模及输入哈希 |
| `base.f32`、`queries.f32`、`truth.i32` | 准备后的底库、查询和精确 Top-10 |
| `candidate.knn`、`raw.nsg` | 构图中间产物 |
| `index.graph`、`build.json` | CSR 图、构图参数、耗时、度数和可达性统计 |
| `codecs/`、`codecs/manifest.json` | 编码模型、查询表示和关联哈希 |
| `query-results/summary.json` | 平台、查询程序哈希及各组原始指标 |
| `query-results/RESULTS_CN.md` | 自动生成的结果表 |
| `query-results/*.labels` | 每组查询的 Top-10 ID |

每个方法、每个 L 启动独立 C++ 子进程。查询耗时包含 PCA 投影、PQ 查表和图遍历，排除文件加载、预热、召回率计算及写盘；速度使用重复查询耗时的中位数。峰值内存使用 Linux `VmHWM`，`ru_maxrss` 仅作诊断。同一 L 不等于同一召回率，方法间速度比较应同时参考 Recall@10。

历史树莓派输出若保留在本地，可查看 `results/sift1m-m64-pca80-run1/`；这些生成文件被 `.gitignore` 排除，不是克隆仓库后自带的数据。自动报告中仍可能出现历史固定措辞，平台应以 `summary.json` 的 `machine`、`platform` 字段为准。

## 算法与内存策略

- EFANNA 的 NN-Descent 产生初始候选图，NSG 负责候选收集、反向边和连通性修复。
- detour-count 按到当前节点的距离排序候选，统计经更靠前候选的两跳路径，优先保留 detour 较少的边；同数时按距离和节点 ID 排序。
- 反向边阶段读取初次剪枝图的快照，并串行更新；保留 NSG 的 `tree_grow`，因此修复边可能使部分节点的出度超过目标 R，实际统计写入 `build.json`。
- 查询加载 CSR 图、查询向量、评测真值及一种表示。量化模式不打开 `base.f32` 或 `codecs/float32.bin`，不做精确重排；float32 基线单独加载自己的全精度表示。
- 查询表示完整读入 RAM，不使用 mmap。节省的是查询阶段内存，不承诺降低构图峰值，也未测量冷缓存磁盘 I/O。

| 表示 | 默认配置 |
| --- | --- |
| float32 | 原始 128 维 float32 基线 |
| INT8 | 每维对称 INT8 及缩放 |
| PCA | 128 → 80 维 float32，附均值和投影矩阵 |
| PQ | 64 个子空间，每个 256 个码字，使用 ADC 查询 |
| Local | 32 个簇、簇号与 INT8 残差，附中心和缩放 |

旧编码不会因默认参数变化自动更新；改变参数时应在新的实验目录重新编码。

## 源码与文件格式

`src/` 提供构图和查询入口，`include/` 包含剪枝、距离与 I/O 实现，`vendor/` 保存裁剪后的 NSG/EFANNA 源码，`tests/` 保存规则测试、距离验证和 PQ 审计工具。来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[UPSTREAM.json](UPSTREAM.json)、[VENDORED_FILES.json](VENDORED_FILES.json) 和 [vendor_changes.patch](vendor_changes.patch)；保留上游许可证及作者声明。

二进制格式为小端、版本 1：

- `*.f32`：`EGF32001`、uint32 N/D、N×D float32。
- `truth.i32`：`EGI32001`、uint32 Q/10、Q×10 uint32 ID。
- `index.graph`：`EGCSR001`、uint32 N/D/entry/flags、uint64 E、N+1 个 uint64 行偏移及 E 个 uint32 邻居 ID。
- `<method>.bin`：`EGCODE01`、uint32 N/D/mode/width/centroids/clusters、uint64 编码字节数、编码及适用的 float32 模型数组。
- mode：float32=0、INT8=1、PCA=2、PQ=3、Local=4。

`pipeline.py query` 检查清单中的输入身份哈希；直接运行底层 `ann_query` 的格式检查不替代这一步。

`.venv/`、`build*/`、`results/`、`logs/` 和 Python 缓存不应提交到 Git。复现依赖源码、安装步骤、参数和数据集，而不是复制某台机器的环境或可执行文件。
