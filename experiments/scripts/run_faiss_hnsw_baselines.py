"""Run FAISS-IVFFlat and HNSW baseline curves for small-scale BATL benchmarks."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import faiss
import numpy as np

from batl.distance import l2_normalize
from batl.utils.config_parsing import load_config_with_device
from batl.utils.io import load_run_data, write_rows
from batl.utils.metrics import recall_at_k


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FAISS baseline Recall@10 curves.")
    parser.add_argument("config", help="Experiment YAML config.")
    parser.add_argument(
        "--methods", nargs="+", choices=["ivfflat", "hnsw"], default=["ivfflat", "hnsw"]
    )
    parser.add_argument("--nprobe", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--ef-search", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--hnsw-degree", type=int, default=8)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--n-queries", type=int, default=None)
    parser.add_argument("--result-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config_with_device(args.config, args, device_override="cpu")
    database, queries, ground_truth = load_run_data(
        cfg, max_queries=args.n_queries, copy_arrays=True
    )
    metric = cfg.dataset_metric or "euclidean"
    rows: list[dict] = []

    if "ivfflat" in args.methods:
        rows.extend(
            run_ivfflat_curve(
                database=database,
                queries=queries,
                ground_truth=ground_truth,
                metric=metric,
                nlist=args.nlist,
                nprobe_values=args.nprobe,
                config_name=cfg.name,
            )
        )
    if "hnsw" in args.methods:
        rows.extend(
            run_hnsw_curve(
                database=database,
                queries=queries,
                ground_truth=ground_truth,
                metric=metric,
                degree=args.hnsw_degree,
                ef_construction=args.ef_construction,
                ef_search_values=args.ef_search,
                config_name=cfg.name,
            )
        )

    output_dir = Path(args.result_dir or cfg.output_dir)
    write_rows(output_dir, "baseline_rows", rows)


def run_ivfflat_curve(
    *,
    database: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    metric: str,
    nlist: int,
    nprobe_values: list[int],
    config_name: str,
) -> list[dict]:
    index_data, search_queries, faiss_metric = _faiss_inputs(database, queries, metric)
    quantizer = _flat_index(index_data.shape[1], faiss_metric)
    index = faiss.IndexIVFFlat(  # type: ignore[reportCallIssue]
        quantizer, index_data.shape[1], nlist, faiss_metric
    )
    index.train(index_data)
    index.add(index_data)
    rows = []
    for nprobe in nprobe_values:
        index.nprobe = nprobe
        retrieved, distcomp, elapsed = _search_with_ivf_stats(index, search_queries, top_k=10)
        rows.append(
            _row(
                method="faiss_ivfflat",
                config_name=config_name,
                knob_name="nprobe",
                knob_value=nprobe,
                retrieved=retrieved,
                ground_truth=ground_truth,
                distcomp=distcomp,
                elapsed_s=elapsed,
            )
        )
    return rows


def run_hnsw_curve(
    *,
    database: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    metric: str,
    degree: int,
    ef_construction: int,
    ef_search_values: list[int],
    config_name: str,
) -> list[dict]:
    index_data, search_queries, faiss_metric = _faiss_inputs(database, queries, metric)
    index = faiss.IndexHNSWFlat(index_data.shape[1], degree, faiss_metric)  # type: ignore[reportCallIssue]
    index.hnsw.efConstruction = ef_construction
    index.add(index_data)
    rows = []
    for ef_search in ef_search_values:
        index.hnsw.efSearch = ef_search
        retrieved, distcomp, elapsed = _search_with_hnsw_stats(index, search_queries, top_k=10)
        rows.append(
            _row(
                method="faiss_hnsw",
                config_name=config_name,
                knob_name="efSearch",
                knob_value=ef_search,
                retrieved=retrieved,
                ground_truth=ground_truth,
                distcomp=distcomp,
                elapsed_s=elapsed,
            )
        )
    return rows


def _faiss_inputs(
    database: np.ndarray,
    queries: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if metric in {"angular", "cosine"}:
        return (
            np.ascontiguousarray(l2_normalize(database), dtype=np.float32),
            np.ascontiguousarray(l2_normalize(queries), dtype=np.float32),
            faiss.METRIC_INNER_PRODUCT,
        )
    return (
        np.ascontiguousarray(database, dtype=np.float32),
        np.ascontiguousarray(queries, dtype=np.float32),
        faiss.METRIC_L2,
    )


def _flat_index(dim: int, faiss_metric: int) -> faiss.Index:
    if faiss_metric == faiss.METRIC_INNER_PRODUCT:
        return faiss.IndexFlatIP(dim)
    return faiss.IndexFlatL2(dim)


def _search_with_ivf_stats(
    index: faiss.Index,
    queries: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    ids = []
    distcomp = []
    start = time.perf_counter()
    for query in queries:
        faiss.cvar.indexIVF_stats.reset()
        _, found = index.search(query.reshape(1, -1), top_k)  # type: ignore[reportCallIssue]
        ids.append(found[0].astype(np.int64, copy=False))
        distcomp.append(int(faiss.cvar.indexIVF_stats.ndis))
    return np.vstack(ids), np.asarray(distcomp, dtype=np.int64), time.perf_counter() - start


def _search_with_hnsw_stats(
    index: faiss.Index,
    queries: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    ids = []
    distcomp = []
    start = time.perf_counter()
    for query in queries:
        faiss.cvar.hnsw_stats.reset()
        _, found = index.search(query.reshape(1, -1), top_k)  # type: ignore[reportCallIssue]
        ids.append(found[0].astype(np.int64, copy=False))
        distcomp.append(int(faiss.cvar.hnsw_stats.ndis))
    return np.vstack(ids), np.asarray(distcomp, dtype=np.int64), time.perf_counter() - start


def _row(
    *,
    method: str,
    config_name: str,
    knob_name: str,
    knob_value: int,
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    distcomp: np.ndarray,
    elapsed_s: float,
) -> dict:
    per_query_recall = recall_at_k(retrieved, ground_truth, 10)
    return {
        "method": method,
        "config": config_name,
        "model_id": method,
        "knob_name": knob_name,
        "knob_value": knob_value,
        "recall@10": float(np.mean(per_query_recall)),
        "mean_distcomp": float(np.mean(distcomp)),
        "std_n_distcomp": float(np.std(distcomp)),
        "n_queries": int(retrieved.shape[0]),
        "search_time_s": float(elapsed_s),
        "qps": float(retrieved.shape[0] / elapsed_s) if elapsed_s > 0 else float("inf"),
    }


if __name__ == "__main__":
    main()
