"""Search-time HNSW quantization experiment; never rebuilds or edits the graph.

The four reusable codecs live at the project root in ann_quantization. This
adapter reads an existing hnswlib 0.8.0 index and scores graph candidates from
encoded vectors. Run on Windows first; the same Python sources run on Linux.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# A Conda interpreter launched by its full path still inherits the caller's
# PATH. Put this environment's native DLLs first before importing NumPy.
if sys.platform == "win32":
    native = Path(sys.prefix) / "Library" / "bin"
    if native.is_dir():
        os.environ["PATH"] = str(native) + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(str(native))

import h5py
import numpy as np
import psutil

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "ann_quantization" / "__init__.py").exists():
        sys.path.insert(0, str(ancestor))
        break
else:
    raise RuntimeError("ann_quantization package not found in a parent directory")

from ann_quantization import load_codec, make_codec, save_codec
from hnsw_graph import HnswGraph


KINDS = ("int8", "pca", "pq", "local")
SAMPLE_SIZES = {"int8": 100_000, "pca": 30_000, "pq": 8_192, "local": 20_000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def prepare(args) -> None:
    graph = HnswGraph(args.index)
    if graph.count != 1_000_000 or graph.dimension != 128:
        raise ValueError("This experiment requires the existing SIFT1M 128D graph")
    codec = make_codec(args.kind, 128)
    sample_count = SAMPLE_SIZES[args.kind]
    sample_ids = np.linspace(0, graph.count - 1, sample_count, dtype=np.int32)
    start = time.perf_counter()
    codec.fit(np.asarray(graph.vectors[sample_ids]))
    fit_seconds = time.perf_counter() - start
    output = args.output / args.kind
    output.mkdir(parents=True, exist_ok=True)
    codes = np.lib.format.open_memmap(
        output / "codes.npy", mode="w+", dtype=codec.code_dtype(),
        shape=(graph.count,) + codec.code_shape()
    )
    start = time.perf_counter()
    for begin in range(0, graph.count, args.batch_size):
        end = min(begin + args.batch_size, graph.count)
        codes[begin:end] = codec.encode(np.asarray(graph.vectors[begin:end]))
    codes.flush()
    encode_seconds = time.perf_counter() - start
    save_codec(codec, output)
    manifest = {
        "kind": args.kind, "config": codec.config(), "vectors": graph.count,
        "dimension": graph.dimension, "sample_count": sample_count,
        "fit_seconds": fit_seconds, "encode_seconds": encode_seconds,
        "codes_bytes": int(codes.nbytes), "index_sha256": sha256(args.index),
        "index_path": str(args.index.resolve()),
        "search_only": True, "graph_rebuilt": False,
    }
    save_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def recall(labels: np.ndarray, truth: np.ndarray) -> float:
    return float(np.any(labels[:, :, None] == truth[:, None, :], axis=2).sum() /
                 (len(labels) * labels.shape[1]))


def query(args) -> None:
    output = args.output / args.kind
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["index_sha256"] != sha256(args.index):
        raise ValueError("The HNSW index differs from the one used for encoding")
    graph = HnswGraph(args.index)
    codec = load_codec(output)
    codes = np.load(output / "codes.npy", mmap_mode="r", allow_pickle=False)
    if len(codes) != graph.count or codec.dimension != graph.dimension:
        raise ValueError("Code array and graph do not match")
    with h5py.File(args.dataset, "r") as source:
        count = min(args.queries, len(source["test"]))
        queries = np.asarray(source["test"][:count], dtype=np.float32)
        truth = np.asarray(source["neighbors"][:count, :10], dtype=np.int32)
    if len(queries) == 0 or truth.shape != (count, 10):
        raise ValueError("Missing SIFT1M queries or top-10 ground truth")
    for ef in args.ef_values:
        for q in queries[:min(args.warmup, count)]:
            graph.search(codec, codes, q, ef, rerank=args.rerank)
        runs = []
        measured_recall = None
        measured_first_hit = None
        for repeat in range(args.repeats):
            start = time.perf_counter()
            labels = np.stack([graph.search(codec, codes, q, ef, rerank=args.rerank) for q in queries])
            elapsed = time.perf_counter() - start
            current_recall = recall(labels, truth)
            current_first_hit = float(np.mean(np.any(labels == truth[:, :1], axis=1)))
            if measured_recall is not None and (current_recall != measured_recall or
                                                current_first_hit != measured_first_hit):
                raise RuntimeError("Search results changed between repeats")
            measured_recall, measured_first_hit = current_recall, current_first_hit
            runs.append({"repeat": repeat + 1, "query_seconds": elapsed,
                         "qpm": 60 * count / elapsed})
        result = {
            "kind": args.kind, "ef_search": ef, "queries": count,
            "exact_rerank": args.rerank,
            "warmup_queries": min(args.warmup, count), "repeats": args.repeats,
            "qpm": statistics.median(run["qpm"] for run in runs),
            "recall_overlap_at_10": measured_recall,
            "first_neighbor_hit_at_10": measured_first_hit,
            "index_sha256": manifest["index_sha256"],
            "codes_bytes": manifest["codes_bytes"],
            "runs": runs,
        }
        tag = "-rerank" if args.rerank else ""
        destination = args.results / args.kind / f"ef{ef}-q{count}{tag}.json"
        save_json(destination, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


def child(command: list[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    peak = 0
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        monitor = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak, monitor.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.05)
        status = process.wait()
    if status:
        raise RuntimeError(f"Child failed ({status}): {log}\n" +
                           log.read_text(encoding="utf-8", errors="replace")[-2000:])
    return peak / (1024 ** 3)


def run(args) -> None:
    script = str(Path(__file__).resolve())
    for kind in args.kinds:
        common = [sys.executable, script]
        if args.reuse_prepared:
            manifest = json.loads((args.output / kind / "manifest.json").read_text(encoding="utf-8"))
            if manifest["index_sha256"] != sha256(args.index):
                raise ValueError("Prepared codes use another graph")
            prepare_peak = None
        else:
            preparation = common + ["prepare", "--index", str(args.index),
                                    "--output", str(args.output), "--kind", kind]
            prepare_peak = child(preparation, args.results / kind / "prepare.log")
        request = common + ["query", "--index", str(args.index),
                            "--dataset", str(args.dataset), "--output", str(args.output),
                            "--results", str(args.results), "--kind", kind,
                            "--queries", str(args.queries), "--ef-values",
                            ",".join(map(str, args.ef_values)),
                            "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        if args.rerank:
            request.append("--rerank")
        query_peak = child(request, args.results / kind / "query.log")
        tag = "-rerank" if args.rerank else ""
        results = [json.loads((args.results / kind / f"ef{ef}-q{args.queries}{tag}.json").read_text(
            encoding="utf-8")) for ef in args.ef_values]
        summary = {
            "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "kind": kind, "queries": args.queries,
            "prepare_peak_rss_gib": prepare_peak, "query_peak_rss_gib": query_peak,
            "runs": results,
        }
        save_json(args.results / kind / f"summary-q{args.queries}{tag}.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "query", "run"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ef-values", default="20,50,100,200")
    parser.add_argument("--rerank", action="store_true", help="Final exact L2 rerank of ef candidates")
    parser.add_argument("--reuse-prepared", action="store_true", help="Use existing codes; verify index checksum")
    args = parser.parse_args()
    args.kinds = args.kinds.split(",")
    args.ef_values = [int(value) for value in args.ef_values.split(",")]
    if any(kind not in KINDS for kind in args.kinds) or any(ef < 10 for ef in args.ef_values):
        parser.error("Invalid kind or ef value")
    if args.repeats < 1 or args.queries < 1 or args.warmup < 0:
        parser.error("Invalid repeat, query or warmup count")
    if args.phase in ("query", "run") and (args.dataset is None or args.results is None):
        parser.error("query/run requires --dataset and --results")
    if args.phase in ("prepare", "query") and args.kind is None:
        parser.error("prepare/query requires --kind")
    {"prepare": prepare, "query": query, "run": run}[args.phase](args)


if __name__ == "__main__":
    main()
