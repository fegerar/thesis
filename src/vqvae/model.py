"""
VQ-VAE model components for shapegraph tokenization.

Components:
    - ShapegraphEncoder: GAT-based graph encoder -> fixed-size embedding (CLS token pooling)
    - VectorQuantizer: discrete bottleneck with EMA or gradient-based codebook updates
    - ShapegraphDecoder: cross-attention decoder -> reconstructed node features + adjacency
    - VQVAE: full pipeline combining encoder, quantizer, decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ShapegraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, embed_dim: int,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # Learnable CLS token — one vector shared across all graphs, broadcast per batch
        self.cls_token = nn.Parameter(torch.randn(1, hidden_dim))

        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                    concat=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x:          (N_total, node_dim)   — all nodes across batch
            edge_index: (2, E_total)           — all edges across batch
            batch:      (N_total,)             — graph index per node

        Returns:
            z_e: (B, embed_dim)
        """
        B = int(batch.max().item()) + 1
        device = x.device

        # Project input nodes to hidden_dim
        h = self.input_proj(x)  # (N_total, hidden_dim)

        # --- Prepend one CLS node per graph ---
        # CLS embeddings: one per graph in the batch
        cls_tokens = self.cls_token.expand(B, -1)  # (B, hidden_dim)

        # New node tensor: [cls_0, cls_1, ..., cls_{B-1}, node_0, node_1, ...]
        h = torch.cat([cls_tokens, h], dim=0)  # (B + N_total, hidden_dim)

        # Update batch vector: CLS nodes belong to graphs 0..B-1
        cls_batch = torch.arange(B, device=device)          # (B,)
        batch_new = torch.cat([cls_batch, batch], dim=0)    # (B + N_total,)

        # Update edge_index: original node indices shift by B
        edge_index_shifted = edge_index + B  # (2, E_total)

        # Connect each CLS node bidirectionally to all nodes in its graph
        # For graph g, CLS index = g; node indices in new tensor = B + (positions where batch==g)
        node_indices = torch.arange(len(batch), device=device) + B  # original nodes, shifted
        cls_indices = batch  # for node i, its graph's CLS token is at index batch[i]

        # Edges: CLS -> node and node -> CLS for every node
        cls_to_node = torch.stack([cls_indices, node_indices], dim=0)   # (2, N_total)
        node_to_cls = torch.stack([node_indices, cls_indices], dim=0)   # (2, N_total)
        cls_edges = torch.cat([cls_to_node, node_to_cls], dim=1)        # (2, 2*N_total)

        edge_index_new = torch.cat([edge_index_shifted, cls_edges], dim=1)  # (2, E_total + 2*N_total)

        # --- GAT layers ---
        for gat, norm in zip(self.gat_layers, self.norm_layers):
            h = norm(h + gat(h, edge_index_new))

        # --- Extract CLS token for each graph (indices 0..B-1) ---
        cls_out = h[:B]                      # (B, hidden_dim)
        z_e = self.output_proj(cls_out)      # (B, embed_dim)
        return z_e


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, use_ema: bool = True,
                 ema_decay: float = 0.99, restart_threshold: float = 1.0):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.restart_threshold = restart_threshold

        self.codebook = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.codebook.weight, -1 / self.K, 1 / self.K)
        # Initialize on unit sphere
        self.codebook.weight.data.copy_(
            F.normalize(self.codebook.weight.data, dim=-1)
        )

        if use_ema:
            self.codebook.weight.requires_grad = False
            self.register_buffer("ema_cluster_size", torch.zeros(self.K))
            self.register_buffer("ema_embed_sum", self.codebook.weight.clone())

    def forward(self, z_e):
        # z_e: (B, D)
        # L2-normalize encoder output and codebook to prevent unbounded drift
        z_e = F.normalize(z_e, dim=-1)
        w = F.normalize(self.codebook.weight, dim=-1)

        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_e @ w.T
            + w.pow(2).sum(dim=1)
        )  # (B, K)

        k = distances.argmin(dim=1)  # (B,)
        z_q = w[k]                   # (B, D)

        if self.training and self.use_ema:
            self._ema_update(z_e, k)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Losses
        if self.use_ema:
            # With EMA the codebook updates outside the compute graph.
            # Only the commitment loss is needed: push z_e toward sg(z_q)
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            vq_loss = self.beta * commitment_loss
        else:
            # Without EMA both terms are needed
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            vq_loss = codebook_loss + self.beta * commitment_loss

        # Codebook utilization
        unique_codes = k.unique().numel()
        utilization = unique_codes / self.K

        return z_q_st, k, vq_loss, utilization

    def _ema_update(self, z_e, k):
        one_hot = F.one_hot(k, self.K).float()  # (B, K)
        cluster_size = one_hot.sum(dim=0)        # (K,)
        embed_sum = one_hot.T @ z_e              # (K, D)

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        self.ema_embed_sum.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        # Laplace smoothing
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
        updated = self.ema_embed_sum / smoothed.unsqueeze(1)
        # Re-normalize codebook entries to unit sphere
        self.codebook.weight.data.copy_(F.normalize(updated, dim=-1))

    def restart_unused_codes(self, z_e):
        """Reinitialize codebook entries that are rarely used."""
        with torch.no_grad():
            z_e = F.normalize(z_e, dim=-1)
            usage = self.ema_cluster_size
            dead = usage < self.restart_threshold
            n_dead = dead.sum().item()
            if n_dead > 0:
                idx = torch.randint(0, z_e.size(0), (n_dead,), device=z_e.device)
                self.codebook.weight.data[dead] = z_e[idx].detach()
                self.ema_cluster_size[dead] = 1.0
                self.ema_embed_sum[dead] = z_e[idx].detach()
            return n_dead


class ShapegraphDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_roles: int, node_out_dim: int,
                 hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_roles = num_roles
        self.role_queries = nn.Parameter(torch.randn(num_roles, embed_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.self_norm = nn.LayerNorm(embed_dim)

        self.node_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_out_dim),
        )

        self.edge_head = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_q):
        B = z_q.size(0)
        queries = self.role_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        z_q_ctx = z_q.unsqueeze(1)                                   # (B, 1, D)

        # Cross-attention: role queries attend to z_q
        h, _ = self.cross_attn(queries, z_q_ctx, z_q_ctx)
        h = self.cross_norm(queries + h)

        # Self-attention among role slots
        h2, _ = self.self_attn(h, h, h)
        h = self.self_norm(h + h2)

        node_feats = self.node_head(h)  # (B, N, node_out_dim)

        # Pairwise edge prediction
        N = self.num_roles
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        pair = torch.cat([h_i, h_j], dim=-1)          # (B, N, N, 2D)
        adj_logits = self.edge_head(pair).squeeze(-1)  # (B, N, N)

        return node_feats, adj_logits


class VQVAE(nn.Module):
    def __init__(self, node_dim: int, encoder_cfg: dict, quantizer_cfg: dict,
                 decoder_cfg: dict):
        super().__init__()
        embed_dim = encoder_cfg["embed_dim"]

        self.encoder = ShapegraphEncoder(
            node_dim=node_dim,
            hidden_dim=encoder_cfg["hidden_dim"],
            embed_dim=embed_dim,
            num_layers=encoder_cfg["num_layers"],
            heads=encoder_cfg["heads"],
            dropout=encoder_cfg.get("dropout", 0.1),
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=quantizer_cfg["num_embeddings"],
            embedding_dim=embed_dim,
            commitment_cost=quantizer_cfg["commitment_cost"],
            use_ema=quantizer_cfg.get("use_ema", True),
            ema_decay=quantizer_cfg.get("ema_decay", 0.99),
            restart_threshold=quantizer_cfg.get("codebook_restart_threshold", 1.0),
        )
        self.decoder = ShapegraphDecoder(
            embed_dim=embed_dim,
            num_roles=decoder_cfg["num_roles"],
            node_out_dim=node_dim,
            hidden_dim=decoder_cfg["hidden_dim"],
            num_heads=decoder_cfg.get("num_cross_attn_heads", 4),
        )

    def forward(self, x, edge_index, batch):
        z_e = self.encoder(x, edge_index, batch)
        z_q, tokens, vq_loss, utilization = self.quantizer(z_e)
        node_feats, adj_logits = self.decoder(z_q)
        return node_feats, adj_logits, z_e, z_q, tokens, vq_loss, utilization

    def encode(self, x, edge_index, batch):
        """Encode graphs to codebook indices (for inference)."""
        z_e = self.encoder(x, edge_index, batch)
        _, tokens, _, _ = self.quantizer(z_e)
        return tokens

    def decode_from_tokens(self, tokens):
        """Decode from codebook indices."""
        z_q = F.normalize(self.quantizer.codebook(tokens), dim=-1)
        return self.decoder(z_q)