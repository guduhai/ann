"""The historical CSR pipeline has been retired from the active deployment."""
raise SystemExit(
    "旧 CSR pipeline 已停用。构图使用 run_build.py，查询使用 run_search.py；"
    "完整 DEEP 实验使用 run_experiment.py --output results/<新目录> --execute。"
)
