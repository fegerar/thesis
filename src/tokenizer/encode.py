"""
Step 3: Encode deduplicated shapegraphs through a pre-trained VQ-VAE.

Loads a Lightning checkpoint, converts NetworkX graphs to PyG batches,
and extracts codebook indices (integer tokens).
"""

import logging

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from vqvae.model import VQVAE
from vqvae.dataset import nx_to_pyg, ShapegraphDataset
from vqvae.lightning_module import VQVAELightningModule

logger = logging.getLogger(__name__)


class VQVAEEncoder:
    """Wrapper for encoding shapegraphs through a trained VQ-VAE."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        batch_size: int = 512,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Load Lightning checkpoint to extract the VQVAE model
        logger.info("Loading VQ-VAE from %s", checkpoint_path)
        lit_module = VQVAELightningModule.load_from_checkpoint(
            checkpoint_path, map_location=self.device
        )
        self.model: VQVAE = lit_module.model
        self.model.to(self.device)
        self.model.eval()

        # Extract codebook size from the quantizer
        self.codebook_size = self.model.quantizer.K
        self.num_summary_tokens = self.model.encoder.num_summary_tokens
        logger.info(
            "VQ-VAE loaded: codebook_size=%d, summary_tokens=%d",
            self.codebook_size, self.num_summary_tokens,
        )

    def encode_graphs(
        self,
        frame_ids: list[int],
        frames: dict[int, nx.Graph],
    ) -> list[list[int]]:
        """Encode a sequence of shapegraphs to codebook indices.

        Args:
            frame_ids: ordered frame numbers to encode
            frames: dict mapping frame_number -> NetworkX graph

        Returns:
            List of token lists, one per frame. Each inner list has
            num_summary_tokens integers (codebook indices).
        """
        # Convert NetworkX graphs to PyG Data objects
        pyg_data = []
        valid_indices = []
        for i, fn in enumerate(frame_ids):
            G = frames[fn]
            data = nx_to_pyg(G)
            if data is not None:
                pyg_data.append(data)
                valid_indices.append(i)
            else:
                logger.warning("Frame %d produced empty PyG data, skipping", fn)

        if not pyg_data:
            return []

        dataset = ShapegraphDataset(pyg_data)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        all_tokens = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                tokens = self.model.encode(batch.x, batch.edge_index, batch.batch)
                # tokens: (B,) for single-token or (B, T) for multi-token
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(1)  # (B, 1)
                all_tokens.append(tokens.cpu())

        all_tokens = torch.cat(all_tokens, dim=0)  # (N, T)

        # Build result list, inserting empty lists for invalid frames
        result = [[] for _ in frame_ids]
        for idx, token_row in zip(valid_indices, all_tokens):
            result[idx] = token_row.tolist()

        return result
