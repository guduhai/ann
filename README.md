# ANN graph search and quantization

CPU approximate nearest neighbor experiments with shared vector quantization code.

```text
ann/
├── ann_quantization/  # Shared Python codecs
└── CAGRA/             # NSG/EFANNA graph construction and C++ search
```

Keep both directories when cloning or moving the repository. `CAGRA/ann_codecs.py`
loads the sibling `ann_quantization` package relative to its own file location.

## CAGRA experiment

The experiment combines native NSG/EFANNA graph construction with CAGRA-style
detour-count pruning on the CPU. Search supports float32, INT8, PCA, PQ and local
adaptive INT8 representations. It does not require Faiss, CUDA or cuVS.
This is a CPU research implementation, not NVIDIA's CUDA CAGRA implementation.

- [Installation and reproduction instructions](CAGRA/README.md)
- [中文安装与复现说明](CAGRA/README_CN.md)

The instructions cover Ubuntu Linux on x86-64 and ARM64, including Ubuntu under
WSL2 and Ubuntu on Raspberry Pi. Native Windows and Android execution have not
been validated. Build the C++ programs and recreate the Python virtual environment
on the target machine; do not copy an existing `build/` or `.venv/` across machines.

Do not commit generated environments, build output, datasets or large experiment
artifacts. The CAGRA instructions include a synthetic example that requires no
dataset download, followed by the full SIFT1M experiment.

## Source attribution

See [third-party notices](CAGRA/THIRD_PARTY_NOTICES.md),
[upstream revisions](CAGRA/UPSTREAM.json), and the licenses retained under
`CAGRA/vendor/` for the bundled NSG and EFANNA sources.
