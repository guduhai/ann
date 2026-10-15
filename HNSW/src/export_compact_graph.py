"""Export hnswlib 0.8.0 links with 4-byte internal IDs instead of float vectors.

This only changes the stored payload/layout. Header graph parameters, ordered
adjacency lists, node levels, entry point and external labels are preserved.
The output is search-only and must be opened with the supplied ID distance space.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<QQQQQQiIQQQdQ")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_exact(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Truncated index")
    return value


def export(source, target, expected_hash):
    source_hash = sha256(source)
    if source_hash != expected_hash:
        raise ValueError("Source graph does not match the prepared codec manifest")
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, target.open("xb") as output:
        header = list(HEADER.unpack(read_exact(original, HEADER.size)))
        offset, capacity, count, stride, labels, vectors = header[:6]
        max_m, max_m0 = header[8:10]
        if (offset != 0 or count != 1_000_000 or capacity != count or
                vectors != 4 + max_m0 * 4 or labels != vectors + 128 * 4 or
                stride != labels + 8 or max_m0 != 2 * max_m):
            raise ValueError("Unsupported original SIFT1M hnswlib layout")
        compact_stride = vectors + 4 + 8
        compact_header = header.copy()
        compact_header[3:5] = [compact_stride, vectors + 4]
        output.write(HEADER.pack(*compact_header))
        for start in range(0, count, 8192):
            length = min(8192, count - start)
            raw = np.frombuffer(read_exact(original, length * stride), np.uint8).reshape(length, stride)
            compact = np.empty((length, compact_stride), dtype=np.uint8)
            compact[:, :vectors] = raw[:, :vectors]
            compact[:, vectors:vectors + 4] = np.arange(start, start + length, dtype="<u4").view(np.uint8).reshape(length, 4)
            compact[:, vectors + 4:] = raw[:, labels:labels + 8]
            output.write(compact.tobytes())
        for block in iter(lambda: original.read(8 * 1024 * 1024), b""):
            output.write(block)

    # Independently reread both files and compare every link/label byte and ID.
    original_links = hashlib.sha256()
    compact_links = hashlib.sha256()
    with source.open("rb") as original, target.open("rb") as compact:
        if list(HEADER.unpack(read_exact(original, HEADER.size))) != header:
            raise ValueError("Original header changed")
        if list(HEADER.unpack(read_exact(compact, HEADER.size))) != compact_header:
            raise ValueError("Compact header differs")
        for start in range(0, count, 8192):
            length = min(8192, count - start)
            src = np.frombuffer(read_exact(original, length * stride), np.uint8).reshape(length, stride)
            dst = np.frombuffer(read_exact(compact, length * compact_stride), np.uint8).reshape(length, compact_stride)
            if not np.array_equal(src[:, :vectors], dst[:, :vectors]):
                raise ValueError("Level-0 graph links changed")
            if not np.array_equal(src[:, labels:labels + 8], dst[:, vectors + 4:]):
                raise ValueError("External labels changed")
            ids = np.ascontiguousarray(dst[:, vectors:vectors + 4]).view("<u4").ravel()
            if not np.array_equal(ids, np.arange(start, start + length, dtype=np.uint32)):
                raise ValueError("Internal code IDs differ")
            for digest, block, label_offset in ((original_links, src, labels), (compact_links, dst, vectors + 4)):
                digest.update(block[:, :vectors].tobytes())
                digest.update(block[:, label_offset:label_offset + 8].tobytes())
        while True:
            src = original.read(8 * 1024 * 1024)
            dst = compact.read(8 * 1024 * 1024)
            if src != dst:
                raise ValueError("Upper-level links changed")
            original_links.update(src)
            compact_links.update(dst)
            if not src:
                break
    if sha256(source) != source_hash:
        raise ValueError("Source index changed during export")
    report = {
        "source": str(source), "source_sha256": source_hash,
        "output": str(target), "output_sha256": sha256(target),
        "nodes": count, "M": header[10], "ef_construction": header[12],
        "entry": header[7], "max_level": header[6],
        "source_bytes": source.stat().st_size, "compact_bytes": target.stat().st_size,
        "original_vector_bytes_in_compact": 0,
        "internal_id_bytes": count * 4,
        "links_and_labels_sha256": original_links.hexdigest(),
        "compact_links_and_labels_sha256": compact_links.hexdigest(),
        "all_links_labels_levels_preserved": True, "graph_rebuilt": False,
        "format": "hnswlib-0.8.0-layout-with-uint32-internal-id-payload",
        "search_only": True,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    export(args.source, args.output, args.expected_sha256)
