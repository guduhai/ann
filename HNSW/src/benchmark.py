"""Cross-platform SIFT1M HNSW baseline using hnswlib 0.8.0.

The same Python source runs on Windows and Linux. Native Python packages must be
installed separately on each platform. All measured build and query phases run
in separate child processes, so their peak resident memory is distinguishable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import hnswlib
import numpy as np
import psutil


K = 10
HNSWLIB_VERSION = "0.8.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(path: Path) -> dict:
    with h5py.File(path, "r") as source:
        train = source["train"]
        test = source["test"]
        neighbors = source["neighbors"]
        if train.ndim != 2 or test.ndim != 2 or neighbors.ndim != 2:
            raise ValueError("Expected two-dimensional train, test and neighbors datasets")
        if train.shape[1] != 128 or test.shape[1] != 128:
            raise ValueError("This benchmark requires 128-dimensional vectors")
        if test.shape[0] != neighbors.shape[0] or neighbors.shape[1] < K:
            raise ValueError("Queries and top-10 ground truth do not match")
        if str(source.attrs.get("distance", "")) != "euclidean":
            raise ValueError("This benchmark requires Euclidean ground truth")
        return {
            "train_count": int(train.shape[0]),
            "query_count": int(test.shape[0]),
            "dimension": int(train.shape[1]),
            "groundtruth_width": int(neighbors.shape[1]),
        }


def recall_metrics(labels: np.ndarray, truth: np.ndarray) -> dict:
    if labels.shape != truth.shape or labels.ndim != 2 or labels.shape[1] != K:
        raise ValueError("Expected matching [queries, 10] result and truth arrays")
    matches = labels[:, :, None] == truth[:, None, :]
    overlap = np.any(matches, axis=2).sum(axis=1) / K
    first_hit = np.any(labels == truth[:, :1], axis=1)
    return {
        "recall_overlap_at_10": float(np.mean(overlap)),
        "first_neighbor_hit_at_10": float(np.mean(first_hit)),
    }


def build_phase(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    with h5py.File(args.dataset, "r") as source:
        shape = source["train"].shape
        if shape != (1_000_000, 128):
            raise ValueError(f"Formal build requires SIFT1M [1000000,128], got {shape}")
        vectors = np.asarray(source["train"][:], dtype=np.float32, order="C")
    t_loaded = time.perf_counter()

    index = hnswlib.Index(space="l2", dim=128)
    index.init_index(
        max_elements=1_000_000,
        M=args.m,
        ef_construction=args.ef_construction,
        random_seed=args.seed,
    )
    ids = np.arange(1_000_000, dtype=np.int32)
    t_insert_start = time.perf_counter()
    index.add_items(vectors, ids, num_threads=args.build_threads)
    t_insert_done = time.perf_counter()
    if index.get_current_count() != 1_000_000:
        raise RuntimeError("Index did not contain all one million vectors")

    args.index.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(args.index))
    t_saved = time.perf_counter()
    result = {
        "phase": "build",
        "timestamp_utc": utc_now(),
        "vectors": 1_000_000,
        "dimensions": 128,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "seed": args.seed,
        "build_threads": args.build_threads,
        "load_seconds": t_loaded - t0,
        "insert_seconds": t_insert_done - t_insert_start,
        "save_seconds": t_saved - t_insert_done,
        "total_seconds": t_saved - t0,
        "insert_vectors_per_second": 1_000_000 / (t_insert_done - t_insert_start),
        "total_vectors_per_second": 1_000_000 / (t_saved - t0),
        "index_bytes": args.index.stat().st_size,
    }
    save_json(args.result, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def query_phase(args: argparse.Namespace) -> None:
    with h5py.File(args.dataset, "r") as source:
        queries = np.asarray(source["test"][:], dtype=np.float32, order="C")
        truth = np.asarray(source["neighbors"][:, :K], dtype=np.int32, order="C")
    if queries.shape != (10_000, 128) or truth.shape != (10_000, K):
        raise ValueError("Formal query requires 10000 queries and top-10 ground truth")

    t_load_start = time.perf_counter()
    index = hnswlib.Index(space="l2", dim=128)
    index.load_index(str(args.index), max_elements=1_000_000)
    t_load_done = time.perf_counter()
    if index.get_current_count() != 1_000_000:
        raise RuntimeError("Loaded index did not contain all one million vectors")

    runs = []
    for ef in args.ef_values:
        if ef < K:
            raise ValueError("ef_search must be at least 10")
        index.set_ef(ef)
        index.knn_query(queries[: args.warmup], k=K, num_threads=args.query_threads)
        for repeat in range(args.repeats):
            start = time.perf_counter()
            labels, _ = index.knn_query(queries, k=K, num_threads=args.query_threads)
            elapsed = time.perf_counter() - start
            if labels.min() < 0 or labels.max() >= 1_000_000:
                raise RuntimeError("Index returned an out-of-range label")
            metrics = recall_metrics(labels, truth)
            runs.append(
                {
                    "ef_search": ef,
                    "repeat": repeat + 1,
                    "queries": len(queries),
                    "query_seconds": elapsed,
                    "qps": len(queries) / elapsed,
                    "qpm": 60 * len(queries) / elapsed,
                    **metrics,
                }
            )

    result = {
        "phase": "query",
        "timestamp_utc": utc_now(),
        "queries": len(queries),
        "k": K,
        "query_threads": args.query_threads,
        "warmup_queries": args.warmup,
        "repeats": args.repeats,
        "index_load_seconds": t_load_done - t_load_start,
        "runs": runs,
    }
    save_json(args.result, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def smoke_phase(args: argparse.Namespace) -> None:
    count = args.smoke_vectors
    count_queries = args.smoke_queries
    with h5py.File(args.dataset, "r") as source:
        vectors = np.asarray(source["train"][:count], dtype=np.float32, order="C")
        queries = np.asarray(source["test"][:count_queries], dtype=np.float32, order="C")
    index = hnswlib.Index(space="l2", dim=128)
    index.init_index(max_elements=count, M=16, ef_construction=200, random_seed=100)
    index.add_items(vectors, np.arange(count, dtype=np.int32), num_threads=2)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(args.index))
    restored = hnswlib.Index(space="l2", dim=128)
    restored.load_index(str(args.index), max_elements=count)
    restored.set_ef(100)
    labels, _ = restored.knn_query(queries, k=K, num_threads=1)
    truth = np.empty((count_queries, K), dtype=np.int32)
    for row, query in enumerate(queries):
        differences = vectors - query
        distances = np.einsum("ij,ij->i", differences, differences)
        truth[row] = np.argpartition(distances, K - 1)[:K]
    result = {
        "phase": "smoke",
        "timestamp_utc": utc_now(),
        "vectors": count,
        "queries": count_queries,
        "index_bytes": args.index.stat().st_size,
        **recall_metrics(labels, truth),
    }
    save_json(args.result, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def run_child(command: list[str], log_path: Path) -> tuple[float, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    peak_rss = 0
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.05)
        exit_code = child.wait()
    if exit_code:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
        raise RuntimeError(f"Child exited {exit_code}; see {log_path}\n{tail}")
    return float(peak_rss / (1024**3)), exit_code


def run_full(args: argparse.Namespace) -> None:
    info = inspect_dataset(args.dataset)
    if info["train_count"] != 1_000_000 or info["query_count"] != 10_000:
        raise ValueError("Full run requires the complete SIFT1M data")
    args.output.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(args.index.parent).free
    if free_before < args.min_free_gib * (1024**3):
        raise RuntimeError("Insufficient free disk space before build")
    script = str(Path(__file__).resolve())
    common = [sys.executable, script]
    build_result = args.output / "build.json"
    query_result = args.output / "query.json"
    build_cmd = common + [
        "build", "--dataset", str(args.dataset), "--index", str(args.index),
        "--result", str(build_result), "--m", str(args.m),
        "--ef-construction", str(args.ef_construction),
        "--seed", str(args.seed), "--build-threads", str(args.build_threads),
    ]
    build_peak, _ = run_child(build_cmd, args.output / "build.log")
    query_cmd = common + [
        "query", "--dataset", str(args.dataset), "--index", str(args.index),
        "--result", str(query_result), "--ef-values", ",".join(map(str, args.ef_values)),
        "--query-threads", str(args.query_threads), "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
    ]
    query_peak, _ = run_child(query_cmd, args.output / "query.log")
    build = json.loads(build_result.read_text(encoding="utf-8"))
    query = json.loads(query_result.read_text(encoding="utf-8"))
    summary = {
        "timestamp_utc": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "hnswlib": HNSWLIB_VERSION,
        "dataset_sha256": sha256_file(args.dataset),
        "dataset": info,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(args.index.parent).free,
        "build_peak_rss_gib": build_peak,
        "query_peak_rss_gib": query_peak,
        "build": build,
        "query": query,
    }
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def int_list(text: str) -> list[int]:
    return [positive_int(item.strip()) for item in text.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("smoke", "build", "query", "run"):
        selected = modes.add_parser(mode)
        selected.add_argument("--dataset", type=Path, required=True)
        selected.add_argument("--index", type=Path, required=True)
        if mode == "run":
            selected.add_argument("--output", type=Path, required=True)
            selected.add_argument("--min-free-gib", type=float, default=0.5)
        else:
            selected.add_argument("--result", type=Path, required=True)
        if mode in ("smoke",):
            selected.add_argument("--smoke-vectors", type=positive_int, default=10_000)
            selected.add_argument("--smoke-queries", type=positive_int, default=100)
        if mode in ("build", "run"):
            selected.add_argument("--m", type=positive_int, default=16)
            selected.add_argument("--ef-construction", type=positive_int, default=200)
            selected.add_argument("--seed", type=int, default=100)
            selected.add_argument("--build-threads", type=positive_int, default=4)
        if mode in ("query", "run"):
            selected.add_argument("--ef-values", type=int_list, default=[20, 50, 100, 200])
            selected.add_argument("--query-threads", type=positive_int, default=1)
            selected.add_argument("--repeats", type=positive_int, default=3)
            selected.add_argument("--warmup", type=positive_int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "smoke":
        smoke_phase(args)
    elif args.mode == "build":
        build_phase(args)
    elif args.mode == "query":
        query_phase(args)
    else:
        run_full(args)
