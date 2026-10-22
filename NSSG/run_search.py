"""Run native no-rerank search on 128-dimensional data of any supported row count.

Default paths come from the repository layout. Explicit input paths are relative
to the caller; relative output paths remain relative to this index directory.
The default action prints the command. Add --execute to run it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
from src.execution import Runtime, add_execution_arguments

ROOT = Path(__file__).resolve().parent
REPRESENTATIONS = {
    "exact": ("exact", 0), "int8": ("int8", 128), "pq64": ("pq", 64),
    "pca80": ("pca", 80), "local": ("local", 129),
}


def uint(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 0xffffffff:
        raise ValueError(f"{name} must be an integer in [{minimum}, 4294967295]")
    return value


def make_command(args: argparse.Namespace) -> list[str]:
    runtime = Runtime(ROOT, args)
    config_path = args.config.resolve() if args.config else ROOT / "experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = config["index"]
    if index != ROOT.name:
        raise ValueError("Configuration index must match this launcher directory")
    if config.get("k", 10) != 10 or config.get("threads", 1) != 1 or config.get("rerank", False):
        raise ValueError("This launcher requires Top-10, one search thread and no rerank")
    kind, size = REPRESENTATIONS[args.method]
    output = runtime.output(args.output)
    if runtime.android:
        required = ['graph', 'query']
        if args.method == 'exact' and index != 'HNSW':
            required.append('base')
        if args.method != 'exact':
            required.extend(['codes', 'model'])
        for name in required:
            if not getattr(args, name):
                raise ValueError('Android requires explicit --' + name + ' device path')
    compact = index == "HNSW" and args.method != "exact"
    graph_key = "compact_graph" if compact else "graph"
    graph = runtime.path(args.graph) if args.graph else ROOT / config["graph"][graph_key]
    shared = ROOT.parent / "ann_quantization/prepared" / args.method
    codes_default = ROOT.parent / "ann_quantization/prepared/hnsw-order" / args.method if compact else shared
    codes = runtime.path(args.codes) if args.codes else codes_default
    model = runtime.path(args.model) if args.model else shared / "native"
    query_default = ROOT.parent / "data/sift1M" / ("sift1m_query.fbin" if index == "VAMANA" else "sift_query.fvecs")
    query = runtime.path(args.query) if args.query else query_default
    available = runtime.query_rows(query, index)
    requested = uint(args.queries if args.queries is not None else config.get("queries", 0), "queries")
    count = available if requested == 0 else requested
    if count > available:
        raise ValueError(f"Requested {count} queries, but file contains {available}")
    width = uint(args.width if args.width is not None else config["W"], "W", 10)
    repeats = uint(args.repeats if args.repeats is not None else config.get("repeats", 1), "repeats", 1)
    # Automatic warmup is bounded by the chosen query count; explicit values are checked.
    warmup = uint(args.warmup, "warmup") if args.warmup is not None else min(uint(config.get("warmup", 200), "warmup"), count)
    if warmup > count:
        raise ValueError("warmup must not exceed the chosen query count")
    base_default = ROOT.parent / "data/sift1M" / ("sift1m_base.fbin" if index == "VAMANA" else "sift_base.fvecs")
    base = runtime.path(args.base) if args.base else base_default
    if args.method != "exact":
        base = runtime.device_root / "NO_FLOAT_DATABASE" if runtime.android else ROOT / "NO_FLOAT_DATABASE"
        if not runtime.android and base.exists():
            raise ValueError("The no-float sentinel path must not exist")
    binary = runtime.binary_dir() / f"{index.lower()}_search"
    common = [width, count, warmup, repeats, output, size]
    if index == "HNSW":
        command = [binary, "compact" if compact else "float", graph, query, kind, codes, model, *common]
    elif index == "VAMANA":
        command = [binary, "--base", base, "--query", query, "--graph", graph,
                   "--kind", kind, "--codes", codes, "--model", model, "--size", size,
                   "--L", width, "--K", 10, "--limit", count, "--warmup", warmup,
                   "--repeats", repeats, "--rerank", 0, "--output", output]
    else:
        command = [binary, graph, base, query, kind, codes, model, *common]
    return list(map(str, command))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(REPRESENTATIONS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", type=Path, help="Experiment JSON; defaults to this index's experiment.json")
    parser.add_argument("--graph", help="HNSW quantized methods require a compact graph")
    parser.add_argument("--base", help="Float32 baseline only; ignored by quantized methods")
    parser.add_argument("--query", help="Query fvecs (VAMANA: fbin)")
    parser.add_argument("--codes", help="Method directory containing codes.npy")
    parser.add_argument("--model", help="Method directory containing native model .bin files")
    parser.add_argument("--queries", type=int, help="0 selects all query rows; otherwise select the first N")
    parser.add_argument("--W", "--search-l", dest="width", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--execute", action="store_true")
    add_execution_arguments(parser)
    args = parser.parse_args()
    try:
        command = make_command(args)
        runtime = Runtime(ROOT, args)
        print(shlex.join(runtime.wrapped(command, 1)), flush=True)
        if args.execute:
            runtime.execute([command], runtime.output(args.output), 1)

    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
