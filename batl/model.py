"""Neural routing model for BATL."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from batl.utils.config import ModelConfig


class BATLEncoder(nn.Module):
    """Two-layer MLP encoder for query and vector embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BATLDecoder(nn.Module):
    """Transformer decoder that predicts the next branch in a path prefix."""

    def __init__(
        self,
        K: int,
        H: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.K = K
        self.H = H
        self.token_embedding = nn.Embedding(K + 1, embed_dim)
        self.pos_embedding = nn.Embedding(H, embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_head = nn.Linear(embed_dim, K)
        self._position_cache: dict[tuple[torch.device, int], Tensor] = {}
        self._causal_mask_cache: dict[tuple[torch.device, int], Tensor] = {}

    def forward(self, path_ids: Tensor, memory: Tensor) -> Tensor:
        if path_ids.ndim != 2:
            raise ValueError("path_ids must have shape (batch, seq_len).")
        if memory.ndim != 2:
            raise ValueError("memory must have shape (batch, embed_dim).")

        batch_size, seq_len = path_ids.shape
        if memory.shape[0] != batch_size:
            raise ValueError("path_ids and memory batch sizes must match.")
        if seq_len > self.H:
            raise ValueError(f"seq_len must be <= H ({self.H}), got {seq_len}.")

        positions = self._positions(path_ids.device, seq_len)
        x = self.token_embedding(path_ids) + self.pos_embedding(positions).unsqueeze(0)
        causal_mask = self._causal_mask(path_ids.device, seq_len)
        decoded = self.transformer(
            tgt=x,
            memory=memory.unsqueeze(1),
            tgt_mask=causal_mask,
        )
        return self.output_head(decoded)

    def _positions(self, device: torch.device, seq_len: int) -> Tensor:
        self._ensure_inference_caches()
        key = (device, seq_len)
        if key not in self._position_cache:
            self._position_cache[key] = torch.arange(seq_len, device=device)
        return self._position_cache[key]

    def _causal_mask(self, device: torch.device, seq_len: int) -> Tensor:
        self._ensure_inference_caches()
        key = (device, seq_len)
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = nn.Transformer.generate_square_subsequent_mask(
                seq_len,
                device=device,
            )
        return self._causal_mask_cache[key]

    def _ensure_inference_caches(self) -> None:
        # Older pickled indexes predate these non-parameter cache attributes.
        if "_position_cache" not in self.__dict__:
            self._position_cache = {}
        if "_causal_mask_cache" not in self.__dict__:
            self._causal_mask_cache = {}


class BATLModel(nn.Module):
    """Encoder-decoder BATL routing model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = replace(config)
        self.K = config.branching_factor
        self.H = config.tree_height
        self.START_TOKEN = self.K
        self.encoder = BATLEncoder(
            input_dim=config.embedding_dim,
            hidden_dim=config.encoder_hidden,
            embed_dim=config.embed_dim,
        )
        self.decoder = BATLDecoder(
            K=config.branching_factor,
            H=config.tree_height,
            embed_dim=config.embed_dim,
            num_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor, target_paths: Tensor) -> Tensor:
        if target_paths.ndim != 2 or target_paths.shape[1] != self.H:
            raise ValueError(f"target_paths must have shape (batch, {self.H}).")

        target_paths = target_paths.to(device=x.device, dtype=torch.long)

        if target_paths.min() < 0 or target_paths.max() >= self.K:
            raise ValueError(f"target_paths must contain branch IDs in [0, {self.K}).")

        start = torch.full(
            (target_paths.shape[0], 1),
            self.START_TOKEN,
            dtype=torch.long,
            device=x.device,
        )
        decoder_input = torch.cat([start, target_paths[:, :-1]], dim=1)
        return self.decoder(decoder_input, self.encode(x))

    @torch.no_grad()
    def decode_node_probs(self, vectors: Tensor, path_prefix: tuple[int, ...]) -> Tensor:
        if len(path_prefix) >= self.H:
            raise ValueError(f"path_prefix length must be < H ({self.H}).")
        if any(branch < 0 or branch >= self.K for branch in path_prefix):
            raise ValueError("path_prefix must contain branch IDs in [0, K).")

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            vectors = vectors.to(device=device, dtype=torch.float32)
            return self.decode_node_probs_from_embeddings(self.encode(vectors), path_prefix)
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def decode_node_probs_from_embeddings(
        self,
        embeddings: Tensor,
        path_prefix: tuple[int, ...],
    ) -> Tensor:
        """Decode next-branch probabilities from precomputed encoder outputs."""
        if len(path_prefix) >= self.H:
            raise ValueError(f"path_prefix length must be < H ({self.H}).")
        if any(branch < 0 or branch >= self.K for branch in path_prefix):
            raise ValueError("path_prefix must contain branch IDs in [0, K).")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (batch, embed_dim).")

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            embeddings = embeddings.to(device=device, dtype=torch.float32)
            prefix = torch.tensor(
                [self.START_TOKEN, *path_prefix],
                dtype=torch.long,
                device=device,
            )
            path_ids = prefix.unsqueeze(0).expand(embeddings.shape[0], -1)
            logits = self.decoder(path_ids, embeddings)
            return F.softmax(logits[:, -1, :], dim=-1)
        finally:
            if was_training:
                self.train()


def batl_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Path prediction loss summed over path levels and averaged over the batch.

    Matches paper Equation 1: L = (1/B) Σ_i Σ_h CE(logits_i,h, target_i,h).
    Expects raw logits (pre-softmax) from the decoder.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, H, K).")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must have shape (batch, H).")
    targets_long = targets.to(device=logits.device, dtype=torch.long)
    per_pred = F.cross_entropy(
        logits.transpose(1, 2),
        targets_long,
        reduction="none",
    )
    return per_pred.sum(dim=1).mean()
