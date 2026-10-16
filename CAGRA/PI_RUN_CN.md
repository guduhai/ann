# CAGRA 通用运行入口说明

文件名保留历史名称 `PI_RUN_CN.md`，运行入口不要求固定用户名或部署路径。统一平台要求、依赖安装和编译见[仓库根目录说明](../README.md)，算法和数据格式见 [README_CN.md](README_CN.md)。以下运行命令在 Linux 或 WSL 的 Bash 中执行。

## 仓库位置与入口行为

克隆完整的 `ann` 仓库，保留 `CAGRA/` 和 `ann_quantization/` 同级：

```text
ann/
├── ann_quantization/
│   ├── __init__.py
│   └── codecs.py
└── CAGRA/
    ├── ann_codecs.py
    ├── python_pi.sh
    └── pipeline.py
```

- 仓库可以放在任意位置。`ann_codecs.py` 根据自身位置只读加载 `../ann_quantization/codecs.py`，不向共享目录写字节码。
- `python_pi.sh` 直接调用仓库根目录的 `.venv/bin/python`，不依赖终端当前激活的 Python，也不设置系统 Python 包路径。
- 入口清除 `PYTHONPATH`、`PYTHONHOME`、`LD_LIBRARY_PATH` 和 `LD_PRELOAD`，关闭用户级 Python 包加载，并设置单线程相关环境变量。
- 入口切换到 `CAGRA/` 后执行命令，并设置 `TMPDIR=CAGRA/build/tmp`。传入的相对路径相对于 `CAGRA/` 解析；外部数据集建议使用绝对路径。
- 缺少仓库根目录的 `.venv/bin/python` 时，入口会提示创建环境并退出；它不会自动安装依赖。

## 不下载数据的完整示例

以下命令均从 `CAGRA/` 执行，输出目录须尚不存在：

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

`prepare` 未指定 `--dataset` 时生成固定种子的合成数据，并计算精确 Top-10；该示例用于演示完整流程，不代表 SIFT1M 性能。

## SIFT1M 全量实验

先按 [README_CN.md](README_CN.md) 下载 ANN-Benchmarks 的 `sift-128-euclidean.hdf5`。下面使用与 README 相同的缓存位置；若数据集位于其他位置，将 `DATASET` 改为该文件的绝对路径。

```bash
DATASET="${XDG_CACHE_HOME:-$HOME/.cache}/ann-datasets/sift-128-euclidean.hdf5"
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

完整建图当前为单线程，底库、中间图和五种表示会占用数 GB 磁盘，构图和训练还需要额外内存。资源有限时可选择新输出目录，先以 `--vectors 20000 --queries 500` 准备 SIFT 子集。输出必须位于 `CAGRA/` 内，不覆盖已有实验。

128 维 SIFT 可以使用 PQ M64；96 维输入应显式使用兼容的子空间数，例如 `--pq-m 32`。训练样本数不能少于 PQ 码字数或 Local 簇数。

## 复用已有图和编码

已有实验的 `index.graph`、`codecs/`、输入查询和清单完整保留时，可以只执行查询。将 `OUT` 设为该实验目录，并选择尚不存在的 `--run-name`：

```bash
bash python_pi.sh pipeline.py query --output "$OUT" \
  --run-name query-next --search-l 20,50,100,200 --warmup 20 --repeats 3
```

`--warmup` 不得超过实际查询条数；上例适用于本文的 32 条合成查询或 10,000 条 SIFT 查询。更改训练或编码参数时，应在新的实验目录重新生成所需产物。

## 历史树莓派结果

历史 ARM64 SIFT1M 实验目录为 `results/sift1m-m64-pca80-run1/`，对应 100 万条底库向量、10,000 条查询。若本地仍保留，可查看：

- `index.graph`、`build.json`：图及构图记录。
- `codecs/`、`codecs/manifest.json`：编码与清单。
- `query-results/summary.json`：平台、查询程序哈希和各组原始指标。
- `query-results/RESULTS_CN.md`：可读结果表。

这些生成文件被 Git 忽略，不是克隆源码后自带的产物。旧报告中的固定平台措辞以 `summary.json` 的 `machine`、`platform` 字段为准。历史部署环境与当前通用入口配置应分别看待。
