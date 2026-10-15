"""Run one CPU search using the frozen no-rerank experiment configuration.

The default action prints the command. Use --execute explicitly to run it.
Results require an explicit output path inside this workspace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parent
REPRESENTATIONS = {
    "exact": ("exact", 0),
    "int8": ("int8", 128),
    "pq64": ("pq", 64),
    "pca80": ("pca", 80),
    "local": ("local", 129),
}


def make_command(args: argparse.Namespace) -> list[str]:
    """Resolve runtime paths without embedding a device address or username."""
    config = json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))
    index = config["index"]
    method = args.method
    kind, size = REPRESENTATIONS[method]
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("The output path must be inside the workspace results directory.")

    binary = ROOT / "build/bin" / f"{index.lower()}_search"
    graph = ROOT / config["graph"]["graph"]
    shared = ROOT.parent / "ann_quantization/prepared" / method
    codes = ROOT.parent / "ann_quantization/prepared/hnsw-order" / method if index == "HNSW" else shared
    model = shared / "native"
    base = ROOT.parent / "data/sift1M/sift_base.fvecs"
    query = ROOT.parent / "data/sift1M/sift_query.fvecs"
    if method != "exact":
        base = ROOT / "NO_FLOAT_DATABASE"
        if base.exists():
            raise ValueError("The no-float sentinel path must not exist.")

    common = [config["W"], config["queries"], config["warmup"], 1, output, size]
    if index == "HNSW":
        mode = "float" if method == "exact" else "compact"
        if mode == "compact":
            graph = ROOT / config["graph"]["compact_graph"]
        command = [binary, mode, graph, query, kind, codes, model, *common]
    elif index == "VAMANA":
        if method == "exact":
            base = ROOT.parent / "data/sift1M/sift1m_base.fbin"
        command = [
            binary, "--base", base, "--query", ROOT.parent / "data/sift1M/sift1m_query.fbin",
            "--graph", graph, "--kind", kind, "--codes", codes, "--model", model,
            "--size", size, "--L", config["W"], "--K", 10,
            "--limit", config["queries"], "--warmup", config["warmup"],
            "--repeats", 1, "--rerank", 0, "--output", output,
        ]
    else:
        command = [binary, graph, base, query, kind, codes, model, *common]
    return [str(value) for value in command]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(REPRESENTATIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    command = make_command(args)
    print(shlex.join(command), flush=True)
    if args.execute:
        output = (ROOT / args.output).resolve()
        if output.exists():
            parser.error("Refusing to overwrite an existing result file.")
        output.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", LC_ALL="C")
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
