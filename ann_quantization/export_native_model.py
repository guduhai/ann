"""Export existing Python codec models for read-only native CPU search.

This never refits, re-encodes, copies the million codes, or modifies a graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ann_quantization import load_codec


MODEL_NAMES = {
    "int8": ("scale",),
    "pca": ("mean", "basis"),
    "pq": ("codebooks",),
    "local": ("centers", "scale"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; refusing to overwrite")
    manifest = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    kind = manifest["kind"]
    if kind not in MODEL_NAMES:
        parser.error("Unsupported codec")
    recorded_hash = manifest.get("index_sha256", manifest.get("graph_sha256"))
    if recorded_hash != args.graph_sha256 or not manifest["search_only"] or manifest["graph_rebuilt"]:
        parser.error("Prepared code does not refer to the approved unchanged graph")
    if manifest["vectors"] != 1_000_000 or manifest["dimension"] != 128:
        parser.error("Expected SIFT1M codes")
    codes_path = args.prepared / "codes.npy"
    codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
    if len(codes) != 1_000_000 or codes.nbytes != manifest["codes_bytes"]:
        parser.error("Code count or byte size mismatch")
    codec = load_codec(args.prepared)
    arrays = codec.arrays()
    args.output.mkdir(parents=True)
    files = {}
    for name in MODEL_NAMES[kind]:
        path = args.output / f"{name}.bin"
        np.asarray(arrays[name], dtype="<f4", order="C").tofile(path)
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    exported = {
        "kind": kind,
        "graph_sha256": recorded_hash,
        "source_codes_npy": str(codes_path.resolve()),
        "source_codes_npy_sha256": sha256(codes_path),
        "source_model_manifest_sha256": sha256(args.prepared / "manifest.json"),
        "files": files,
        "search_only": True,
        "graph_rebuilt": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(exported, ensure_ascii=False))


if __name__ == "__main__":
    main()
