"""Read-only hnswlib 0.8.0 graph adapter for interchangeable distance codecs.

The hnswlib index is memory mapped. Graph links and entry point are reused;
only candidate distance evaluation is delegated to a codec. This parser is
specific to hnswlib 0.8.0's little-endian saveIndex layout.
"""

from __future__ import annotations

import heapq
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<QQQQQQiIQQQdQ")


class HnswGraph:
    def __init__(self, path: Path, dimension: int = 128):
        self.path = Path(path)
        self.raw = np.memmap(self.path, dtype=np.uint8, mode="r")
        if len(self.raw) < HEADER.size:
            raise ValueError("Truncated HNSW index")
        (level0_offset, capacity, count, stride, label_offset, vector_offset,
         max_level, entry, max_m, max_m0, m, _mult, _efc) = HEADER.unpack_from(self.raw)
        if (level0_offset != 0 or capacity != count or count == 0 or
                stride != vector_offset + dimension * 4 + 8 or
                label_offset != vector_offset + dimension * 4 or
                vector_offset != 4 + max_m0 * 4 or max_m0 != 2 * max_m or
                m != max_m or entry >= count or max_level < 0):
            raise ValueError("Unsupported hnswlib index layout")
        self.count = count
        self.dimension = dimension
        self.max_level = max_level
        self.entry = entry
        self.max_m = max_m
        self.max_m0 = max_m0
        self.stride = stride
        self.base = HEADER.size
        self.upper_stride = 4 + max_m * 4
        base_end = self.base + stride * count
        if base_end > len(self.raw):
            raise ValueError("Truncated level-0 graph")
        self.counts = np.ndarray((count,), dtype="<u2", buffer=self.raw,
                                 offset=self.base, strides=(stride,))
        self.links = np.ndarray((count, max_m0), dtype="<u4", buffer=self.raw,
                                offset=self.base + 4, strides=(stride, 4))
        self.vectors = np.ndarray((count, dimension), dtype="<f4", buffer=self.raw,
                                  offset=self.base + vector_offset, strides=(stride, 4))
        self.labels = np.ndarray((count,), dtype="<u8", buffer=self.raw,
                                 offset=self.base + label_offset, strides=(stride,))
        if np.any(self.counts > max_m0):
            raise ValueError("Invalid level-0 link count")
        self.upper_offsets = np.full(count, -1, dtype=np.int64)
        position = base_end
        for item in range(count):
            if position + 4 > len(self.raw):
                raise ValueError("Truncated upper graph")
            size = struct.unpack_from("<I", self.raw, position)[0]
            position += 4
            if size:
                if size % self.upper_stride:
                    raise ValueError("Invalid upper graph block")
                self.upper_offsets[item] = position
            position += size
        if position != len(self.raw):
            raise ValueError("Unexpected trailing index bytes")
        self.visited = np.zeros(count, dtype=np.uint32)
        self.epoch = np.uint32(0)

    def upper_links(self, item: int, level: int) -> np.ndarray:
        start = int(self.upper_offsets[item]) + (level - 1) * self.upper_stride
        if start < 0 or start + self.upper_stride > len(self.raw):
            raise ValueError("Missing upper-level links")
        degree = struct.unpack_from("<H", self.raw, start)[0]
        if degree > self.max_m:
            raise ValueError("Invalid upper-level link count")
        return np.ndarray((degree,), dtype="<u4", buffer=self.raw, offset=start + 4)

    def search(self, codec, codes: np.ndarray, query: np.ndarray,
               ef: int, k: int = 10, rerank: bool = False) -> np.ndarray:
        if ef < k or len(codes) != self.count:
            raise ValueError("ef must be >= k and code count must match graph")
        state = codec.query_state(query)
        current = int(self.entry)
        distance = float(codec.distances(state, codes[current:current + 1])[0])
        for level in range(self.max_level, 0, -1):
            improved = True
            while improved:
                improved = False
                ids = self.upper_links(current, level)
                if len(ids):
                    if np.any(ids >= self.count):
                        raise ValueError("Invalid graph neighbor")
                    scores = codec.distances(state, codes[ids])
                    for item, score in zip(ids, scores):
                        if score < distance:
                            current = int(item)
                            distance = float(score)
                            improved = True

        self.epoch = np.uint32(self.epoch + 1)
        if self.epoch == 0:
            self.visited.fill(0)
            self.epoch = np.uint32(1)
        epoch = self.epoch
        self.visited[current] = epoch
        candidates = [(distance, current)]
        nearest = [(-distance, current)]
        lower_bound = distance
        while candidates:
            candidate_distance, candidate = heapq.heappop(candidates)
            if candidate_distance > lower_bound:
                break
            ids = self.links[candidate, :self.counts[candidate]]
            fresh = ids[self.visited[ids] != epoch]
            if not len(fresh):
                continue
            self.visited[fresh] = epoch
            scores = codec.distances(state, codes[fresh])
            for item, score in zip(fresh, scores):
                score = float(score)
                if len(nearest) < ef or score < lower_bound:
                    item = int(item)
                    heapq.heappush(candidates, (score, item))
                    heapq.heappush(nearest, (-score, item))
                    if len(nearest) > ef:
                        heapq.heappop(nearest)
                    lower_bound = -nearest[0][0]
        if rerank:
            ids = np.fromiter((item for _, item in nearest), dtype=np.int32)
            diff = self.vectors[ids] - np.asarray(query, dtype=np.float32)
            exact = np.einsum("ij,ij->i", diff, diff)
            selected = ids[np.argsort(exact)[:k]]
        else:
            selected = [item for _, item in sorted(nearest, key=lambda pair: -pair[0])[:k]]
        return np.asarray(self.labels[selected], dtype=np.int32)
