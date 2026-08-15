from experiments.scripts.plot_curve import render_svg


def test_render_svg_accepts_batl_and_baseline_curve_rows() -> None:
    svg = render_svg(
        [
            {"method": "batl", "model_id": "K64_H2", "mean_distcomp": "1000", "recall@10": "0.4"},
            {
                "method": "faiss_ivfflat",
                "model_id": "faiss_ivfflat",
                "mean_distcomp": "2000",
                "recall@10": "0.5",
            },
        ]
    )

    assert svg.startswith("<svg")
    assert "Mean distance computations" in svg
    assert "Recall@10" in svg
    assert "batl:K64_H2" in svg
