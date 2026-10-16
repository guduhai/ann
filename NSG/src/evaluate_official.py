"""Check the original NSG command-line search output against SIFT ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        raise ValueError(f"empty ivecs: {path}")
    width = int(raw[0])
    if width < 1 or raw.size % (width + 1):
        raise ValueError(f"invalid ivecs shape: {path}")
    rows = raw.reshape(-1, width + 1)
    if not np.all(rows[:, 0] == width):
        raise ValueError(f"inconsistent ivecs widths: {path}")
    return rows[:, 1:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--base-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = read_ivecs(args.results)
    truth = read_ivecs(args.truth)
    if len(results) > len(truth) or results.shape[1] < 10 or truth.shape[1] < 10:
        raise ValueError("query count or Top-10 width mismatch")
    truth = truth[: len(results)]
    if np.any(results < 0) or np.any(results >= args.base_count):
        raise ValueError("result ID outside base range")
    top10 = results[:, :10]
    exact = truth[:, :10]
    hits = sum(len(set(map(int, a)) & set(map(int, b))) for a, b in zip(top10, exact))
    report = {
        "queries": int(len(results)),
        "base_count": args.base_count,
        "top10_intersection_recall": hits / (10 * len(results)),
        "index_bytes": args.index.stat().st_size,
        "index_sha256": sha256(args.index),
        "results_sha256": sha256(args.results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
