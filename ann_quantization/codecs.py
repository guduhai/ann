"""CPU distance codecs. Fitting/encoding never changes an index's graph edges.

Every codec has the same four operations: fit on representative vectors,
encode stored vectors, prepare one query, and score encoded candidates. An
index adapter supplies candidate IDs; the codec knows nothing about HNSW.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _matrix(values: np.ndarray, dimension: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != dimension or not np.isfinite(result).all():
        raise ValueError(f"Expected finite matrix [n,{dimension}]")
    return result


def _nearest(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return nearest center IDs without allocating a [n,k,d] tensor."""
    points = np.asarray(points, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    center_norm = np.einsum("ij,ij->i", centers, centers)
    scores = center_norm[None, :] - 2.0 * (points @ centers.T)
    return np.argmin(scores, axis=1).astype(np.int32)


def _kmeans(points: np.ndarray, k: int, iterations: int, seed: int) -> np.ndarray:
    if points.shape[0] < k:
        raise ValueError("Training sample smaller than centroid count")
    rng = np.random.default_rng(seed)
    centers = np.array(points[rng.choice(len(points), k, replace=False)], dtype=np.float32)
    for _ in range(iterations):
        labels = _nearest(points, centers)
        counts = np.bincount(labels, minlength=k)
        updated = np.empty_like(centers)
        for column in range(points.shape[1]):
            updated[:, column] = np.bincount(
                labels, weights=points[:, column], minlength=k
            ) / np.maximum(counts, 1)
        missing = np.flatnonzero(counts == 0)
        if len(missing):
            updated[missing] = points[rng.integers(len(points), size=len(missing))]
        centers = updated
    return centers


class Codec:
    kind = "base"

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)

    def fit(self, sample: np.ndarray) -> None:
        raise NotImplementedError

    def code_dtype(self) -> np.dtype:
        raise NotImplementedError

    def code_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def query_state(self, query: np.ndarray):
        raise NotImplementedError

    def distances(self, state, codes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def config(self) -> dict:
        return {"dimension": self.dimension}

    def arrays(self) -> dict[str, np.ndarray]:
        return {}

    def restore(self, arrays: dict[str, np.ndarray]) -> None:
        if arrays:
            raise ValueError("Unexpected model arrays")


class ExactFloat(Codec):
    """Validation reference for the graph adapter; not a quantization method."""

    kind = "exact"

    def fit(self, sample: np.ndarray) -> None:
        _matrix(sample, self.dimension)

    def code_dtype(self) -> np.dtype:
        return np.dtype("<f4")

    def code_shape(self) -> tuple[int, ...]:
        return (self.dimension,)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        return _matrix(vectors, self.dimension)

    def query_state(self, query: np.ndarray) -> np.ndarray:
        return np.asarray(query, dtype=np.float32)

    def distances(self, state, codes: np.ndarray) -> np.ndarray:
        diff = np.asarray(codes, dtype=np.float32) - state
        return np.einsum("ij,ij->i", diff, diff)


class SymmetricInt8(Codec):
    kind = "int8"

    def fit(self, sample: np.ndarray) -> None:
        sample = _matrix(sample, self.dimension)
        self.scale = np.maximum(np.max(np.abs(sample), axis=0) / 127.0, 1e-8).astype(np.float32)

    def code_dtype(self) -> np.dtype:
        return np.dtype("i1")

    def code_shape(self) -> tuple[int, ...]:
        return (self.dimension,)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        vectors = _matrix(vectors, self.dimension)
        return np.clip(np.rint(vectors / self.scale), -127, 127).astype(np.int8)

    def query_state(self, query: np.ndarray) -> np.ndarray:
        return np.asarray(query, dtype=np.float32)

    def distances(self, state, codes: np.ndarray) -> np.ndarray:
        diff = codes.astype(np.float32) * self.scale - state
        return np.einsum("ij,ij->i", diff, diff)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"scale": self.scale}

    def restore(self, arrays: dict[str, np.ndarray]) -> None:
        self.scale = np.asarray(arrays["scale"], dtype=np.float32)


class PCA(Codec):
    kind = "pca"

    def __init__(self, dimension: int, components: int = 64):
        super().__init__(dimension)
        if not 0 < components <= dimension:
            raise ValueError("components must be within the input dimension")
        self.components = int(components)

    def fit(self, sample: np.ndarray) -> None:
        sample = _matrix(sample, self.dimension)
        self.mean = sample.mean(axis=0).astype(np.float32)
        centered = sample - self.mean
        covariance = centered.T @ centered / max(len(sample) - 1, 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        self.basis = np.ascontiguousarray(eigenvectors[:, -self.components:][:, ::-1], dtype=np.float32)

    def code_dtype(self) -> np.dtype:
        return np.dtype("<f4")

    def code_shape(self) -> tuple[int, ...]:
        return (self.components,)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray((_matrix(vectors, self.dimension) - self.mean) @ self.basis, dtype=np.float32)

    def query_state(self, query: np.ndarray) -> np.ndarray:
        return (np.asarray(query, dtype=np.float32) - self.mean) @ self.basis

    def distances(self, state, codes: np.ndarray) -> np.ndarray:
        diff = codes - state
        return np.einsum("ij,ij->i", diff, diff)

    def config(self) -> dict:
        return {"dimension": self.dimension, "components": self.components}

    def arrays(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean, "basis": self.basis}

    def restore(self, arrays: dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(arrays["mean"], dtype=np.float32)
        self.basis = np.asarray(arrays["basis"], dtype=np.float32)


class ProductQuantization(Codec):
    kind = "pq"

    def __init__(self, dimension: int, subspaces: int = 16, centroids: int = 256, iterations: int = 10):
        super().__init__(dimension)
        if dimension % subspaces or not 2 <= centroids <= 256:
            raise ValueError("dimension must divide subspaces; centroids must be 2..256")
        self.subspaces = int(subspaces)
        self.centroids = int(centroids)
        self.iterations = int(iterations)
        self.width = dimension // subspaces

    def fit(self, sample: np.ndarray) -> None:
        sample = _matrix(sample, self.dimension)
        self.codebooks = np.stack([
            _kmeans(sample[:, j * self.width:(j + 1) * self.width], self.centroids,
                    self.iterations, 100 + j)
            for j in range(self.subspaces)
        ]).astype(np.float32)

    def code_dtype(self) -> np.dtype:
        return np.dtype("u1")

    def code_shape(self) -> tuple[int, ...]:
        return (self.subspaces,)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        vectors = _matrix(vectors, self.dimension)
        codes = np.empty((len(vectors), self.subspaces), dtype=np.uint8)
        for j in range(self.subspaces):
            codes[:, j] = _nearest(
                vectors[:, j * self.width:(j + 1) * self.width], self.codebooks[j]
            )
        return codes

    def query_state(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        lookup = np.empty((self.subspaces, self.centroids), dtype=np.float32)
        for j in range(self.subspaces):
            diff = self.codebooks[j] - query[j * self.width:(j + 1) * self.width]
            lookup[j] = np.einsum("ij,ij->i", diff, diff)
        return lookup

    def distances(self, state, codes: np.ndarray) -> np.ndarray:
        return state[np.arange(self.subspaces)[None, :], codes].sum(axis=1)

    def config(self) -> dict:
        return {"dimension": self.dimension, "subspaces": self.subspaces,
                "centroids": self.centroids, "iterations": self.iterations}

    def arrays(self) -> dict[str, np.ndarray]:
        return {"codebooks": self.codebooks}

    def restore(self, arrays: dict[str, np.ndarray]) -> None:
        self.codebooks = np.asarray(arrays["codebooks"], dtype=np.float32)


class LocalAdaptiveInt8(Codec):
    """Coarse clusters with per-cluster, per-dimension int8 residual scales."""

    kind = "local"

    def __init__(self, dimension: int, clusters: int = 32, iterations: int = 10):
        super().__init__(dimension)
        if not 2 <= clusters <= 256:
            raise ValueError("clusters must be 2..256")
        self.clusters = int(clusters)
        self.iterations = int(iterations)

    def fit(self, sample: np.ndarray) -> None:
        sample = _matrix(sample, self.dimension)
        self.centers = _kmeans(sample, self.clusters, self.iterations, 200)
        group = _nearest(sample, self.centers)
        self.scale = np.empty((self.clusters, self.dimension), dtype=np.float32)
        fallback = np.maximum(np.max(np.abs(sample - self.centers[group]), axis=0) / 127.0, 1e-8)
        for j in range(self.clusters):
            members = sample[group == j]
            self.scale[j] = (np.maximum(np.max(np.abs(members - self.centers[j]), axis=0) / 127.0,
                                        1e-8) if len(members) else fallback)

    def code_dtype(self) -> np.dtype:
        return np.dtype([("cluster", "u1"), ("residual", "i1", (self.dimension,))])

    def code_shape(self) -> tuple[int, ...]:
        return ()

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        vectors = _matrix(vectors, self.dimension)
        group = _nearest(vectors, self.centers)
        residual = np.clip(np.rint((vectors - self.centers[group]) / self.scale[group]),
                           -127, 127).astype(np.int8)
        codes = np.empty(len(vectors), dtype=self.code_dtype())
        codes["cluster"] = group
        codes["residual"] = residual
        return codes

    def query_state(self, query: np.ndarray) -> np.ndarray:
        return np.asarray(query, dtype=np.float32)

    def distances(self, state, codes: np.ndarray) -> np.ndarray:
        group = codes["cluster"]
        diff = self.centers[group] + self.scale[group] * codes["residual"].astype(np.float32) - state
        return np.einsum("ij,ij->i", diff, diff)

    def config(self) -> dict:
        return {"dimension": self.dimension, "clusters": self.clusters,
                "iterations": self.iterations}

    def arrays(self) -> dict[str, np.ndarray]:
        return {"centers": self.centers, "scale": self.scale}

    def restore(self, arrays: dict[str, np.ndarray]) -> None:
        self.centers = np.asarray(arrays["centers"], dtype=np.float32)
        self.scale = np.asarray(arrays["scale"], dtype=np.float32)


KINDS = {cls.kind: cls for cls in (ExactFloat, SymmetricInt8, PCA, ProductQuantization, LocalAdaptiveInt8)}


def make_codec(kind: str, dimension: int, **options) -> Codec:
    try:
        return KINDS[kind](dimension=dimension, **options)
    except KeyError as exc:
        raise ValueError(f"Unknown codec: {kind}") from exc


def save_codec(codec: Codec, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "codec.json").write_text(
        json.dumps({"kind": codec.kind, "config": codec.config()}, indent=2), encoding="utf-8"
    )
    np.savez(directory / "codec.npz", **codec.arrays())


def load_codec(directory: Path) -> Codec:
    meta = json.loads((directory / "codec.json").read_text(encoding="utf-8"))
    codec = make_codec(meta["kind"], **meta["config"])
    with np.load(directory / "codec.npz", allow_pickle=False) as archive:
        codec.restore({name: archive[name] for name in archive.files})
    return codec
