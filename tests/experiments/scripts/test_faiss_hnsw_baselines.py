import numpy as np

from experiments.scripts.run_faiss_hnsw_baselines import run_hnsw_curve, run_ivfflat_curve


def test_faiss_baseline_rows_include_paper_curve_fields() -> None:
    database = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
    queries = np.array([[0.1], [10.1]], dtype=np.float32)
    ground_truth = np.array(
        [[0, 1, 2, 3, -1, -1, -1, -1, -1, -1], [2, 3, 1, 0, -1, -1, -1, -1, -1, -1]], dtype=np.int64
    )

    rows = run_ivfflat_curve(
        database=database,
        queries=queries,
        ground_truth=ground_truth,
        metric="euclidean",
        nlist=2,
        nprobe_values=[1],
        config_name="tiny",
    ) + run_hnsw_curve(
        database=database,
        queries=queries,
        ground_truth=ground_truth,
        metric="euclidean",
        degree=2,
        ef_construction=10,
        ef_search_values=[4],
        config_name="tiny",
    )

    assert {row["method"] for row in rows} == {"faiss_ivfflat", "faiss_hnsw"}
    for row in rows:
        assert row["recall@10"] >= 0.0
        assert row["mean_distcomp"] >= 0.0
        assert row["std_n_distcomp"] >= 0.0
        assert row["n_queries"] == 2
