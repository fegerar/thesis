"""
VQ-VAE model components for shapegraph tokenization.

Components:
    - ShapegraphEncoder: GAT-based graph encoder -> fixed-size embedding (CLS token pooling)
    - VectorQuantizer: discrete bottleneck with EMA or gradient-based codebook updates
    - ShapegraphDecoder: cross-attention decoder -> reconstructed node features
    - VQVAE: full pipeline combining encoder, quantizer, decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ShapegraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, embed_dim: int,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.1,
                 num_summary_tokens: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_summary_tokens = num_summary_tokens
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # Learnable summary tokens — T vectors per graph for multi-token bottleneck
        self.summary_tokens = nn.Parameter(
            torch.randn(num_summary_tokens, hidden_dim)
        )

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
            z_e: (B, T, embed_dim)  where T = num_summary_tokens
        """
        B = int(batch.max().item()) + 1
        T = self.num_summary_tokens
        device = x.device

        # Project input nodes to hidden_dim
        h = self.input_proj(x)  # (N_total, hidden_dim)

        # --- Prepend T summary nodes per graph ---
        # Summary tokens: T per graph, total B*T prepended nodes
        summary = self.summary_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, T, hidden_dim)
        summary_flat = summary.reshape(B * T, -1)                      # (B*T, hidden_dim)

        # New node tensor: [sum_0_0..sum_0_{T-1}, sum_1_0..., node_0, node_1, ...]
        h = torch.cat([summary_flat, h], dim=0)  # (B*T + N_total, hidden_dim)

        # Update batch vector: summary nodes belong to their respective graphs
        summary_batch = torch.arange(B, device=device).repeat_interleave(T)  # (B*T,)
        batch_new = torch.cat([summary_batch, batch], dim=0)

        # Update edge_index: original node indices shift by B*T
        offset = B * T
        edge_index_shifted = edge_index + offset

        # Connect each summary node bidirectionally to all nodes in its graph
        node_indices = torch.arange(len(batch), device=device) + offset
        node_graph_ids = batch  # graph id for each original node

        # For each node, connect to all T summary tokens of its graph
        # Summary token j of graph g is at index g*T + j
        summary_base = node_graph_ids * T  # (N_total,) — base index for each node's graph
        summary_edges_src = []
        summary_edges_dst = []
        for j in range(T):
            s_indices = summary_base + j  # (N_total,)
            summary_edges_src.append(s_indices)
            summary_edges_dst.append(node_indices)
            summary_edges_src.append(node_indices)
            summary_edges_dst.append(s_indices)

        summary_src = torch.cat(summary_edges_src, dim=0)
        summary_dst = torch.cat(summary_edges_dst, dim=0)
        summary_edges = torch.stack([summary_src, summary_dst], dim=0)

        # Also connect summary tokens to each other within the same graph
        if T > 1:
            s2s_src = []
            s2s_dst = []
            graph_bases = torch.arange(B, device=device) * T  # (B,)
            for i in range(T):
                for j in range(T):
                    if i != j:
                        s2s_src.append(graph_bases + i)
                        s2s_dst.append(graph_bases + j)
            s2s_src = torch.cat(s2s_src, dim=0)
            s2s_dst = torch.cat(s2s_dst, dim=0)
            s2s_edges = torch.stack([s2s_src, s2s_dst], dim=0)
            edge_index_new = torch.cat([edge_index_shifted, summary_edges, s2s_edges], dim=1)
        else:
            edge_index_new = torch.cat([edge_index_shifted, summary_edges], dim=1)

        # --- GAT layers ---
        for gat, norm in zip(self.gat_layers, self.norm_layers):
            h = norm(h + gat(h, edge_index_new))

        # --- Extract summary tokens: indices 0..B*T-1 ---
        summary_out = h[:B * T].reshape(B, T, -1)    # (B, T, hidden_dim)
        z_e = self.output_proj(summary_out)            # (B, T, embed_dim)
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

        if use_ema:
            self.codebook.weight.requires_grad = False
            self.register_buffer("ema_cluster_size", torch.zeros(self.K))
            self.register_buffer("ema_embed_sum", self.codebook.weight.clone())

    def forward(self, z_e):
        # z_e: (B, D) or (B, T, D) for multi-token
        input_shape = z_e.shape
        if z_e.dim() == 3:
            B, T, D = z_e.shape
            z_e = z_e.reshape(B * T, D)
        else:
            T = None

        w = self.codebook.weight

        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_e @ w.T
            + w.pow(2).sum(dim=1)
        )  # (B*T, K)

        k = distances.argmin(dim=1)  # (B*T,)
        z_q = w[k]                   # (B*T, D)

        if self.training and self.use_ema:
            self._ema_update(z_e, k)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Losses
        if self.use_ema:
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            vq_loss = self.beta * commitment_loss
        else:
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            vq_loss = codebook_loss + self.beta * commitment_loss

        # Codebook utilization
        unique_codes = k.unique().numel()
        utilization = unique_codes / self.K

        # Reshape back to (B, T, D) if multi-token
        if T is not None:
            z_q_st = z_q_st.reshape(B, T, D)
            k = k.reshape(B, T)

        return z_q_st, k, vq_loss, utilization

    def _ema_update(self, z_e, k):
        z_e = z_e.detach()
        k = k.detach()
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
        self.codebook.weight.data.copy_(updated)

    def restart_unused_codes(self, z_e):
        """Reinitialize codebook entries that are rarely used."""
        with torch.no_grad():
            usage = self.ema_cluster_size
            dead = usage < self.restart_threshold
            n_dead = dead.sum().item()
            if n_dead > 0:
                idx = torch.randint(0, z_e.size(0), (n_dead,), device=z_e.device)
                self.codebook.weight.data[dead] = z_e[idx].detach()
                self.ema_cluster_size[dead] = 1.0
                self.ema_embed_sum[dead] = z_e[idx].detach()
            return n_dead


class ProductQuantizer(nn.Module):
    """Independent codebook per token slot — structurally prevents slot overlap.

    Instead of one shared codebook (K, D), maintains T separate codebooks
    each of size (K_per_slot, D). Each slot quantizes independently.
    """

    def __init__(self, num_slots: int, num_embeddings_per_slot: int,
                 embedding_dim: int, commitment_cost: float = 0.25,
                 use_ema: bool = True, ema_decay: float = 0.99,
                 restart_threshold: float = 1.0):
        super().__init__()
        self.T = num_slots
        self.K = num_embeddings_per_slot
        self.D = embedding_dim
        self.beta = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.restart_threshold = restart_threshold

        self.codebooks = nn.ModuleList([
            nn.Embedding(self.K, self.D) for _ in range(self.T)
        ])
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1 / self.K, 1 / self.K)

        if use_ema:
            for cb in self.codebooks:
                cb.weight.requires_grad = False
            self.ema_cluster_sizes = nn.ParameterList([
                nn.Parameter(torch.zeros(self.K), requires_grad=False)
                for _ in range(self.T)
            ])
            self.ema_embed_sums = nn.ParameterList([
                nn.Parameter(cb.weight.clone(), requires_grad=False)
                for cb in self.codebooks
            ])

    def forward(self, z_e):
        """
        Args:
            z_e: (B, T, D)
        Returns:
            z_q_st: (B, T, D) — straight-through quantized
            tokens:  (B, T)   — codebook indices per slot
            vq_loss: scalar
            utilization: float
        """
        B, T, D = z_e.shape
        assert T == self.T, f"Expected {self.T} slots, got {T}"

        z_q_list = []
        k_list = []
        total_vq_loss = 0.0
        total_unique = 0

        for t in range(self.T):
            z_e_t = z_e[:, t]  # (B, D)
            w = self.codebooks[t].weight  # (K, D)

            distances = (
                z_e_t.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_e_t @ w.T
                + w.pow(2).sum(dim=1)
            )  # (B, K)

            k_t = distances.argmin(dim=1)  # (B,)
            z_q_t = w[k_t]                 # (B, D)

            if self.training and self.use_ema:
                self._ema_update_slot(t, z_e_t, k_t)

            # Straight-through
            z_q_st_t = z_e_t + (z_q_t - z_e_t).detach()

            # Loss
            if self.use_ema:
                commitment = F.mse_loss(z_e_t, z_q_t.detach())
                total_vq_loss = total_vq_loss + self.beta * commitment
            else:
                codebook_loss = F.mse_loss(z_q_t, z_e_t.detach())
                commitment = F.mse_loss(z_e_t, z_q_t.detach())
                total_vq_loss = total_vq_loss + codebook_loss + self.beta * commitment

            total_unique += k_t.unique().numel()
            z_q_list.append(z_q_st_t)
            k_list.append(k_t)

        z_q_st = torch.stack(z_q_list, dim=1)  # (B, T, D)
        tokens = torch.stack(k_list, dim=1)     # (B, T)
        vq_loss = total_vq_loss / self.T
        utilization = total_unique / (self.T * self.K)

        return z_q_st, tokens, vq_loss, utilization

    def _ema_update_slot(self, t, z_e, k):
        z_e = z_e.detach()
        k = k.detach()
        one_hot = F.one_hot(k, self.K).float()
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.T @ z_e

        self.ema_cluster_sizes[t].data.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        self.ema_embed_sums[t].data.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        n = self.ema_cluster_sizes[t].data.sum()
        smoothed = (
            (self.ema_cluster_sizes[t].data + 1e-5)
            / (n + self.K * 1e-5) * n
        )
        updated = self.ema_embed_sums[t].data / smoothed.unsqueeze(1)
        self.codebooks[t].weight.data.copy_(updated)

    def restart_unused_codes(self, z_e):
        """Reinitialize unused codes across all slot codebooks."""
        total_restarted = 0
        with torch.no_grad():
            for t in range(self.T):
                usage = self.ema_cluster_sizes[t].data
                dead = usage < self.restart_threshold
                n_dead = dead.sum().item()
                if n_dead > 0:
                    z_e_t = z_e[:, t] if z_e.dim() == 3 else z_e
                    idx = torch.randint(0, z_e_t.size(0), (n_dead,), device=z_e.device)
                    self.codebooks[t].weight.data[dead] = z_e_t[idx].detach()
                    self.ema_cluster_sizes[t].data[dead] = 1.0
                    self.ema_embed_sums[t].data[dead] = z_e_t[idx].detach()
                total_restarted += n_dead
        return total_restarted


class ShapegraphDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_roles: int, node_out_dim: int,
                 hidden_dim: int, num_heads: int = 4, num_self_attn_layers: int = 1):
        super().__init__()
        self.num_roles = num_roles
        self.role_queries = nn.Parameter(torch.randn(num_roles, embed_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_self_attn_layers)
        ])
        self.self_attn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_self_attn_layers)
        ])
        self.self_attn_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            for _ in range(num_self_attn_layers)
        ])
        self.self_attn_ffn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_self_attn_layers)
        ])

        self.node_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_out_dim),
        )

    def forward(self, z_q):
        B = z_q.size(0)
        queries = self.role_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # z_q can be (B, D) single-token or (B, T, D) multi-token
        z_q_ctx = z_q.unsqueeze(1) if z_q.dim() == 2 else z_q      # (B, T, D)

        # Cross-attention: role queries attend to z_q tokens
        h, _ = self.cross_attn(queries, z_q_ctx, z_q_ctx)
        h = self.cross_norm(queries + h)

        # Self-attention layers among role slots
        for attn, norm, ffn, ffn_norm in zip(
            self.self_attn_layers, self.self_attn_norms,
            self.self_attn_ffns, self.self_attn_ffn_norms,
        ):
            h2, _ = attn(h, h, h)
            h = norm(h + h2)
            h = ffn_norm(h + ffn(h))

        node_feats = self.node_head(h)  # (B, N, node_out_dim)

        return node_feats


class VQVAE(nn.Module):
    def __init__(self, node_dim: int, encoder_cfg: dict, quantizer_cfg: dict,
                 decoder_cfg: dict, bypass_vq: bool = False):
        super().__init__()
        self.bypass_vq = bypass_vq
        embed_dim = encoder_cfg["embed_dim"]
        num_summary_tokens = encoder_cfg.get("num_summary_tokens", 1)

        self.encoder = ShapegraphEncoder(
            node_dim=node_dim,
            hidden_dim=encoder_cfg["hidden_dim"],
            embed_dim=embed_dim,
            num_layers=encoder_cfg["num_layers"],
            heads=encoder_cfg["heads"],
            dropout=encoder_cfg.get("dropout", 0.1),
            num_summary_tokens=num_summary_tokens,
        )

        quantizer_type = quantizer_cfg.get("type", "shared")
        if quantizer_type == "product":
            self.quantizer = ProductQuantizer(
                num_slots=num_summary_tokens,
                num_embeddings_per_slot=quantizer_cfg["num_embeddings"],
                embedding_dim=embed_dim,
                commitment_cost=quantizer_cfg["commitment_cost"],
                use_ema=quantizer_cfg.get("use_ema", True),
                ema_decay=quantizer_cfg.get("ema_decay", 0.99),
                restart_threshold=quantizer_cfg.get("codebook_restart_threshold", 1.0),
            )
        else:
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
            num_self_attn_layers=decoder_cfg.get("num_self_attn_layers", 1),
        )

    def forward(self, x, edge_index, batch):
        z_e = self.encoder(x, edge_index, batch)
        if self.bypass_vq:
            # Skip quantization — pure autoencoder for diagnostic
            node_feats = self.decoder(z_e)
            vq_loss = torch.tensor(0.0, device=z_e.device)
            tokens = torch.zeros(z_e.shape[:-1], dtype=torch.long, device=z_e.device)
            utilization = 0.0
            return node_feats, z_e, z_e, tokens, vq_loss, utilization
        z_q, tokens, vq_loss, utilization = self.quantizer(z_e)
        node_feats = self.decoder(z_q)
        return node_feats, z_e, z_q, tokens, vq_loss, utilization

    def encode(self, x, edge_index, batch):
        """Encode graphs to codebook indices (for inference)."""
        z_e = self.encoder(x, edge_index, batch)
        _, tokens, _, _ = self.quantizer(z_e)
        return tokens

    def decode_from_tokens(self, tokens):
        """Decode from codebook indices.

        Args:
            tokens: (B,) for single-token or (B, T) for multi-token
        """
        if isinstance(self.quantizer, ProductQuantizer):
            # tokens: (B, T) — look up each slot in its own codebook
            z_q_list = []
            for t in range(tokens.shape[1]):
                z_q_list.append(self.quantizer.codebooks[t](tokens[:, t]))
            z_q = torch.stack(z_q_list, dim=1)  # (B, T, D)
        else:
            z_q = self.quantizer.codebook(tokens)
        return self.decoder(z_q)