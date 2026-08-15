import torch

from batl.model import BATLDecoder, BATLEncoder, BATLModel, batl_loss
from batl.utils.config import ModelConfig


def _small_config() -> ModelConfig:
    return ModelConfig(
        branching_factor=4,
        tree_height=2,
        embedding_dim=8,
        encoder_hidden=16,
        embed_dim=12,
        num_heads=3,
        ff_dim=24,
        dropout=0.0,
    )


def test_encoder_forward_shape() -> None:
    cfg = _small_config()
    encoder = BATLEncoder(
        input_dim=cfg.embedding_dim,
        hidden_dim=cfg.encoder_hidden,
        embed_dim=cfg.embed_dim,
    )
    x = torch.randn(5, cfg.embedding_dim)

    encoded = encoder(x)

    assert encoded.shape == (5, cfg.embed_dim)


def test_decoder_forward_shape_and_positional_embedding() -> None:
    cfg = _small_config()
    decoder = BATLDecoder(
        K=cfg.branching_factor,
        H=cfg.tree_height,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    )
    path_ids = torch.tensor([[4, 0], [4, 1]], dtype=torch.long)
    memory = torch.randn(2, cfg.embed_dim)

    probs = decoder(path_ids, memory)

    assert decoder.pos_embedding.num_embeddings == cfg.tree_height
    assert probs.shape == (2, cfg.tree_height, cfg.branching_factor)


def test_decoder_reuses_position_and_mask_cache() -> None:
    cfg = _small_config()
    decoder = BATLDecoder(
        K=cfg.branching_factor,
        H=cfg.tree_height,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    )
    path_ids = torch.tensor([[4, 0], [4, 1]], dtype=torch.long)
    memory = torch.randn(2, cfg.embed_dim)

    decoder(path_ids, memory)
    positions = decoder._position_cache[(path_ids.device, path_ids.shape[1])]
    mask = decoder._causal_mask_cache[(path_ids.device, path_ids.shape[1])]
    decoder(path_ids, memory)

    assert decoder._position_cache[(path_ids.device, path_ids.shape[1])] is positions
    assert decoder._causal_mask_cache[(path_ids.device, path_ids.shape[1])] is mask


def test_decoder_recreates_cache_for_older_pickled_models() -> None:
    cfg = _small_config()
    decoder = BATLDecoder(
        K=cfg.branching_factor,
        H=cfg.tree_height,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    )
    del decoder._position_cache
    del decoder._causal_mask_cache
    path_ids = torch.tensor([[4, 0], [4, 1]], dtype=torch.long)
    memory = torch.randn(2, cfg.embed_dim)

    probs = decoder(path_ids, memory)

    assert probs.shape == (2, cfg.tree_height, cfg.branching_factor)
    assert (path_ids.device, path_ids.shape[1]) in decoder._position_cache
    assert (path_ids.device, path_ids.shape[1]) in decoder._causal_mask_cache


def test_model_forward_teacher_forcing_shape() -> None:
    cfg = _small_config()
    model = BATLModel(cfg)
    x = torch.randn(3, cfg.embedding_dim)
    target_paths = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.long)

    probs = model(x, target_paths)

    assert model.K == cfg.branching_factor
    assert model.START_TOKEN == cfg.branching_factor
    assert probs.shape == (3, cfg.tree_height, cfg.branching_factor)


def test_decode_node_probs_returns_last_step_probabilities() -> None:
    cfg = _small_config()
    model = BATLModel(cfg)
    vectors = torch.randn(6, cfg.embedding_dim)

    probs = model.decode_node_probs(vectors, path_prefix=(1,))

    assert probs.shape == (6, cfg.branching_factor)
    assert torch.allclose(probs.sum(dim=1), torch.ones(6), atol=1e-6)
    assert torch.all(probs >= 0)


def test_decode_node_probs_from_embeddings_matches_raw_vector_path() -> None:
    cfg = _small_config()
    model = BATLModel(cfg)
    model.eval()
    vectors = torch.randn(6, cfg.embedding_dim)

    raw_probs = model.decode_node_probs(vectors, path_prefix=(1,))
    embeddings = model.encode(vectors)
    cached_probs = model.decode_node_probs_from_embeddings(embeddings, path_prefix=(1,))

    assert torch.allclose(cached_probs, raw_probs, atol=1e-6)


def test_batl_loss_matches_cross_entropy_over_all_levels() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]],
            [[1.0, 0.0, 2.0], [2.0, 1.0, 0.0]],
        ]
    )
    targets = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)

    loss = batl_loss(logits, targets)

    # Paper Equation 1: sum over H levels per example, average over batch.
    per_pred = torch.nn.functional.cross_entropy(
        logits.reshape(4, 3), targets.reshape(4), reduction="none"
    )
    expected = per_pred.reshape(2, 2).sum(dim=1).mean()
    assert torch.isclose(loss, expected, atol=1e-5)
