# CAGRA：Linux / Android 编译与运行

两个平台均从本目录 `CMakeLists.txt` 编译同一套 C++ 源码。`cmake/linux.cmake` 与 `cmake/android.cmake` 只负责平台编译选项和依赖。以下命令从仓库根目录执行。

## Linux / 树莓派

Python 环境使用根目录 `.venv`，安装方法见根目录 README。CAGRA 本身需要 C++17、CMake 3.20+、OpenMP；不需要 Boost、Faiss、BLAS 或 AIO。统一启动入口使用系统 `time` 记录资源，在 Debian/Ubuntu 上可安装：

```bash
sudo apt-get install build-essential cmake libgomp1 time
cmake -S CAGRA -B CAGRA/build/linux-arm64 -DCMAKE_BUILD_TYPE=Release
cmake --build CAGRA/build/linux-arm64 -j2
```

目录名 `linux-arm64` 只是产物分类，不决定目标架构。在 x86 WSL 中可改为 `linux-x86_64`。本机优化默认开启，交叉编译时不会添加 `native` 选项，也可显式 `-DANN_NATIVE_CPU=OFF`。

```bash
.venv/bin/python -B CAGRA/run_build.py \
  --data /absolute/path/sift_base.fvecs \
  --output "$PWD/CAGRA/results/run1/graph" \
  --binary-dir CAGRA/build/linux-arm64/bin --threads 4 --execute

.venv/bin/python -B CAGRA/run_search.py \
  --method pq64 --graph CAGRA/results/run1/graph/final.graph \
  --query /absolute/path/sift_query.fvecs --truth /absolute/path/groundtruth.ivecs \
  --codes /absolute/path/pq64 --model /absolute/path/pq64/native \
  --binary-dir CAGRA/build/linux-arm64/bin --output results/run1/pq64 --execute
```

`--data`、`--graph`、`--query`、`--truth`、`--codes`、`--model`、`--binary-dir` 以及构图 `--output` 的相对路径按当前终端目录解释。查询 `--output` 的相对路径按 `CAGRA/` 解释，必须位于其 `results/` 内。

## Android ARM64

在电脑/服务器上使用 NDK 交叉编译，不能直接复制 Linux ARM64 程序到 Android。建议使用与其他索引相同的 NDK r26c、API 28+，目标设备为 arm64-v8a：

```bash
cmake -S CAGRA -B CAGRA/build/android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release
cmake --build CAGRA/build/android-arm64 -j2
```

Android 构图静态链接 OpenMP；仅 `ann_knn` 默认加入小分配内存池，沿用全量手机实验中避免 Scudo 映射耗尽的处理。查询不链接该分配池。可通过 `-DANN_ANDROID_KNN_POOL=OFF` 关闭。

生成 `ann_knn`、`ann_build`、`ann_query`；`ann_query_phone` 是 `ann_query` 的兼容副本，两者使用同一份源码。编译不会自动部署，也不会启动实验。

下面示例把新程序和输出全部放在手机 CAGRA 目录内；共享数据、编码与模型只读使用。把设备序列号和输入路径替换成实际值：

```bash
DEVICE=71ad4
PHONE_ROOT=/data/local/tmp/ann/CAGRA
adb -s "$DEVICE" shell mkdir -p "$PHONE_ROOT/bin/cagra"
adb -s "$DEVICE" push CAGRA/build/android-arm64/bin/ann_knn "$PHONE_ROOT/bin/cagra/"
adb -s "$DEVICE" push CAGRA/build/android-arm64/bin/ann_build "$PHONE_ROOT/bin/cagra/"
adb -s "$DEVICE" push CAGRA/build/android-arm64/bin/ann_query "$PHONE_ROOT/bin/cagra/"
adb -s "$DEVICE" shell chmod 755 "$PHONE_ROOT/bin/cagra/ann_knn" "$PHONE_ROOT/bin/cagra/ann_build" "$PHONE_ROOT/bin/cagra/ann_query"

.venv/bin/python -B CAGRA/run_build.py --target android --device "$DEVICE" \
  --device-root "$PHONE_ROOT" \
  --data /data/local/tmp/ann/datasets/sift1m/sift_base.fvecs \
  --output "$PHONE_ROOT/indexes/cagra/run1" --threads 4 --execute

.venv/bin/python -B CAGRA/run_search.py --target android --device "$DEVICE" \
  --device-root "$PHONE_ROOT" --method pq64 \
  --graph "$PHONE_ROOT/indexes/cagra/run1/final.graph" \
  --query /data/local/tmp/ann/datasets/sift1m/sift_query.fvecs \
  --truth "$PHONE_ROOT/assets/sift1m-groundtruth.ivecs" \
  --codes /data/local/tmp/ann/codes/sift1m/quant/pq64 \
  --model /data/local/tmp/ann/models/sift1m/quant/pq64/native \
  --output "$PHONE_ROOT/results/cagra/run1/pq64" --execute
```

Python 仅在电脑/服务器发起命令，C++ 构图和查询均在手机运行；没有 CSR 转换或服务器代算阶段。也可以在手机 shell 直接调用这三个原生程序，不依赖手机 Python。

## 通用参数和输出

- 不加 `--execute` 只展示命令，但可能读取输入文件头或通过 ADB 只读检查文件。
- 默认 `--target linux`；旧构建目录可用 `--binary-dir CAGRA/build/bin`。不同平台构建目录请显式指定。
- Android 默认运行根目录为 `/data/local/tmp/ann`；示例显式改为其 `CAGRA` 子目录。`--device-root` 控制程序、索引和结果默认位置。所有 Android 输入/输出参数均为设备绝对路径，不是服务器路径。
- 构图 `--vectors 0` 表示全部，可指定前 N 条；`--threads` 默认取配置中的 4。
- 查询方法：`exact`/`float32`、`int8`、`pca80`/`pca`、`pq64`/`pq`、`local`。float32 需要 `--base`；量化方法需要 `--codes`、`--model`，不访问 `--base`。
- 查询 `--vectors 0` 从底库/编码推导 N；`--queries 0` 表示全部查询。可用 `--width`、`--warmup`、`--repeats` 覆盖配置。
- 量化编码行数和节点顺序必须匹配图；使用底库子集必须重新准备对应编码和真值。PQ/PCA 维度由 `experiment.json` 决定，默认 M64/PCA80，不会自动修改已有编码。
- 构图输出：`candidate.knn`、`final.graph`、`knn.stdout`、`knn.time`、`build.stdout`、`build.time`。
- 查询输出：`<prefix>.json`、`.labels`、`.stdout`、`.time`。构图目录与查询前缀均不能覆盖既有结果；失败后查看日志并选择新路径重试。

原生参数顺序：

```text
ann_knn BASE KNN K L ITER S REVERSE_R [N=0] [THREADS=4]
ann_build BASE KNN GRAPH R L C [N=0] [THREADS=4]
ann_query GRAPH BASE QUERY TRUTH N Q KIND CODES MODEL L WARMUP REPEATS PREFIX CODE_SIZE
```

`ann_query` 中 KIND 使用 `float32/int8/pca/pq/local`，CODE_SIZE 对 PCA/PQ 分别为 80/64，其他为 0；量化模式 BASE 可传 `NO_FLOAT_DATABASE`，float32 模式 CODES/MODEL 可传 `unused`。直接运行程序时须预先创建输出父目录，计时和日志可用设备的 `time -v` 包裹。
