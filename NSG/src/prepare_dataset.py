"""Convert the project's SIFT HDF5 file to WEAVESS fvecs/ivecs in WSL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def write_fvecs(source, path: Path, rows: int, batch_size: int = 8192) -> None:
    dimension = int(source.shape[1])
    with path.open("wb") as target:
        for start in range(0, rows, batch_size):
            values = np.asarray(source[start : min(start + batch_size, rows)], dtype="<f4")
            records = np.empty((len(values), dimension + 1), dtype="<f4")
            records.view("<i4")[:, 0] = dimension
            records[:, 1:] = values
            records.tofile(target)


def write_ivecs(values: np.ndarray, path: Path) -> None:
    values = np.asarray(values, dtype="<i4")
    records = np.empty((len(values), values.shape[1] + 1), dtype="<i4")
    records[:, 0] = values.shape[1]
    records[:, 1:] = values
    with path.open("wb") as target:
        records.tofile(target)


def subset_truth(base: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    if len(base) < k:
        raise ValueError(f"smoke base must contain at least {k} vectors")
    truth = np.empty((len(queries), k), dtype=np.int32)
    for query_id, query in enumerate(queries):
        delta = base - query
        distances = np.einsum("ij,ij->i", delta, delta)
        candidates = np.argpartition(distances, k - 1)[:k]
        truth[query_id] = candidates[np.argsort(distances[candidates])]
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-base", type=int, default=10000)
    parser.add_argument("--smoke-queries", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    name = "siftsmall" if args.mode == "smoke" else "sift1M"
    prefix = "siftsmall" if args.mode == "smoke" else "sift"
    output = args.output_root / name
    files = {
        "base": output / f"{prefix}_base.fvecs",
        "query": output / f"{prefix}_query.fvecs",
        "groundtruth": output / f"{prefix}_groundtruth.ivecs",
        "manifest": output / "manifest.json",
    }
    existing = [str(path) for path in files.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError("Already exists; pass --force to overwrite: " + ", ".join(existing))
    output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as dataset:
        base = dataset["train"]
        queries = dataset["test"]
        if args.mode == "smoke":
            count = min(args.smoke_base, len(base))
            query_count = min(args.smoke_queries, len(queries))
            base_values = np.asarray(base[:count], dtype=np.float32)
            query_values = np.asarray(queries[:query_count], dtype=np.float32)
            truth = subset_truth(base_values, query_values)
            write_fvecs(base_values, files["base"], count)
            write_fvecs(query_values, files["query"], query_count)
        else:
            count = len(base)
            query_count = len(queries)
            truth = np.asarray(dataset["neighbors"], dtype=np.int32)
            if len(truth) != query_count or np.any(truth < 0) or np.any(truth >= count):
                raise ValueError("Full dataset ground truth IDs do not match the base vectors")
            write_fvecs(base, files["base"], count)
            write_fvecs(queries, files["query"], query_count)
        write_ivecs(truth, files["groundtruth"])

    manifest = {
        "mode": args.mode,
        "source": str(args.input),
        "base_count": int(count),
        "query_count": int(query_count),
        "dimension": int(base.shape[1]),
        "groundtruth_width": int(truth.shape[1]),
        "files": {key: str(value) for key, value in files.items() if key != "manifest"},
    }
    files["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
