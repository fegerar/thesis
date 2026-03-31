"""
GNN Transformer for football trajectory prediction.

Architecture:
    1. Spatial GNN (per frame): TransformerConv processes each graph independently,
       using pairwise distance as edge features. Produces per-node embeddings.
    2. Temporal Transformer (per node): Causal self-attention across time steps
       for each of the 23 nodes independently, capturing motion patterns.
    3. Prediction head: Projects temporal embeddings to next-frame node features.

Input:  sequence of graphs (B, T, 23, 6) + edge distances (B, T, E, 1)
Output: predicted next frame (B, 23, 6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class SpatialGNN(nn.Module):
    """Per-frame graph neural network using TransformerConv with edge features."""

    def __init__(self, node_dim: int, hidden_dim: int, edge_dim: int = 1,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                TransformerConv(
                    hidden_dim, hidden_dim // heads,
                    heads=heads, edge_dim=edge_dim,
                    dropout=dropout, concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (N, node_dim) node features (all nodes in batch)
            edge_index: (2, E) edges
            edge_attr:  (E, 1) edge distances

        Returns:
            h: (N, hidden_dim) node embeddings
        """
        h = self.input_proj(x)
        for conv, norm in zip(self.conv_layers, self.norms):
            h = norm(h + conv(h, edge_index, edge_attr))
        return h


class TemporalTransformer(nn.Module):
    """Causal temporal attention applied per-node across time steps."""

    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "norm1": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
                "norm2": nn.LayerNorm(hidden_dim),
            }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B * N_nodes, T, hidden_dim) per-node temporal sequences.

        Returns:
            (B * N_nodes, T, hidden_dim) — attended temporal embeddings.
        """
        T = x.size(1)
        # Causal mask: prevent attending to future time steps
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        for layer in self.layers:
            h, _ = layer["attn"](x, x, x, attn_mask=causal_mask)
            x = layer["norm1"](x + h)
            x = layer["norm2"](x + layer["ffn"](x))
        return x


class GNNTransformer(nn.Module):
    """Spatial GNN + Temporal Transformer for next-frame prediction.

    Given a sequence of T graph snapshots, predicts the next graph.
    """

    def __init__(self, node_dim: int = 6, hidden_dim: int = 128,
                 num_nodes: int = 23, edge_dim: int = 1,
                 spatial_layers: int = 3, spatial_heads: int = 4,
                 temporal_layers: int = 3, temporal_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Learnable positional encoding for time steps
        self.temporal_pos = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        self.spatial_gnn = SpatialGNN(
            node_dim=node_dim, hidden_dim=hidden_dim, edge_dim=edge_dim,
            num_layers=spatial_layers, heads=spatial_heads, dropout=dropout,
        )
        self.temporal_transformer = TemporalTransformer(
            hidden_dim=hidden_dim, num_heads=temporal_heads,
            num_layers=temporal_layers, dropout=dropout,
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, input_frames: torch.Tensor, edge_index: torch.Tensor,
                edge_attr_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_frames:  (B, T, N, F) node features per frame.
            edge_index:    (2, E) shared fully-connected edges.
            edge_attr_seq: (B, T, E, 1) edge distances per frame.

        Returns:
            pred: (B, N, F) predicted next-frame node features.
        """
        B, T, N, F = input_frames.shape
        E = edge_index.size(1)

        # --- Spatial GNN: process each (batch, time) frame independently ---
        # Flatten batch and time → treat each frame as a separate graph
        # We need to offset edge_index per graph in the mega-batch
        x_flat = input_frames.reshape(B * T, N, F)          # (B*T, N, F)
        edge_attr_flat = edge_attr_seq.reshape(B * T, E, 1)  # (B*T, E, 1)

        # Build batched edge_index: offset each graph's node indices
        offsets = torch.arange(B * T, device=edge_index.device) * N  # (B*T,)
        # edge_index_batched: (2, B*T*E)
        edge_index_exp = edge_index.unsqueeze(0).expand(B * T, -1, -1)  # (B*T, 2, E)
        offsets_exp = offsets.view(B * T, 1, 1)
        edge_index_batched = (edge_index_exp + offsets_exp).reshape(2, B * T * E)
        edge_attr_batched = edge_attr_flat.reshape(B * T * E, 1)

        x_nodes = x_flat.reshape(B * T * N, F)  # (B*T*N, F)
        h_nodes = self.spatial_gnn(x_nodes, edge_index_batched, edge_attr_batched)
        # h_nodes: (B*T*N, hidden_dim)

        # --- Temporal Transformer: per-node across time ---
        h = h_nodes.reshape(B, T, N, self.hidden_dim)  # (B, T, N, H)
        # Rearrange to (B*N, T, H) so each node has its own temporal sequence
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)

        # Add temporal positional encoding
        h = h + self.temporal_pos[:, :T, :]

        h = self.temporal_transformer(h)  # (B*N, T, H)

        # Take the last time step's embedding for prediction
        h_last = h[:, -1, :]  # (B*N, H)
        h_last = h_last.reshape(B, N, self.hidden_dim)

        # --- Prediction head ---
        pred = self.prediction_head(h_last)  # (B, N, F)
        return pred
