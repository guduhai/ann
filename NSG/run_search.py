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

ROOT = Path(__file__).resolve().parent
REPRESENTATIONS = {
    "exact": ("exact", 0), "int8": ("int8", 128), "pq64": ("pq", 64),
    "pca80": ("pca", 80), "local": ("local", 129),
}


def query_rows(path: Path, index: str) -> int:
    size = path.stat().st_size
    with path.open("rb") as source:
        if index == "VAMANA":
            header = source.read(8)
            if len(header) != 8:
                raise ValueError("Truncated query fbin header")
            count, dimension = struct.unpack("<II", header)
            valid = dimension == 128 and size == 8 + count * dimension * 4
        else:
            header = source.read(4)
            dimension = struct.unpack("<I", header)[0] if len(header) == 4 else 0
            count = size // (129 * 4)
            valid = dimension == 128 and size % (129 * 4) == 0
    if not valid or not 0 < count <= 0xffffffff:
        raise ValueError("Expected a nonempty 128-dimensional query file with complete records")
    return count


def uint(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 0xffffffff:
        raise ValueError(f"{name} must be an integer in [{minimum}, 4294967295]")
    return value


def make_command(args: argparse.Namespace) -> list[str]:
    config_path = args.config.resolve() if args.config else ROOT / "experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = config["index"]
    if index != ROOT.name:
        raise ValueError("Configuration index must match this launcher directory")
    if config.get("k", 10) != 10 or config.get("threads", 1) != 1 or config.get("rerank", False):
        raise ValueError("This launcher requires Top-10, one search thread and no rerank")
    kind, size = REPRESENTATIONS[args.method]
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Output must be inside this index's results directory")
    compact = index == "HNSW" and args.method != "exact"
    graph_key = "compact_graph" if compact else "graph"
    graph = args.graph.resolve() if args.graph else ROOT / config["graph"][graph_key]
    shared = ROOT.parent / "ann_quantization/prepared" / args.method
    codes_default = ROOT.parent / "ann_quantization/prepared/hnsw-order" / args.method if compact else shared
    codes = args.codes.resolve() if args.codes else codes_default
    model = args.model.resolve() if args.model else shared / "native"
    query_default = ROOT.parent / "data/sift1M" / ("sift1m_query.fbin" if index == "VAMANA" else "sift_query.fvecs")
    query = args.query.resolve() if args.query else query_default
    available = query_rows(query, index)
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
    base = args.base.resolve() if args.base else base_default
    if args.method != "exact":
        base = ROOT / "NO_FLOAT_DATABASE"
        if base.exists():
            raise ValueError("The no-float sentinel path must not exist")
    binary = ROOT / "build/bin" / f"{index.lower()}_search"
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Experiment JSON; defaults to this index's experiment.json")
    parser.add_argument("--graph", type=Path, help="HNSW quantized methods require a compact graph")
    parser.add_argument("--base", type=Path, help="Float32 baseline only; ignored by quantized methods")
    parser.add_argument("--query", type=Path, help="Query fvecs (VAMANA: fbin)")
    parser.add_argument("--codes", type=Path, help="Method directory containing codes.npy")
    parser.add_argument("--model", type=Path, help="Method directory containing native model .bin files")
    parser.add_argument("--queries", type=int, help="0 selects all query rows; otherwise select the first N")
    parser.add_argument("--W", "--search-l", dest="width", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        command = make_command(args)
        print(shlex.join(command), flush=True)
        if args.execute:
            output = (ROOT / args.output).resolve()
            if output.exists():
                raise ValueError("Refusing to overwrite an existing result file")
            output.parent.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", LC_ALL="C")
            subprocess.run(command, env=environment, check=True)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
