"""
Dataset and dataloader utilities for shapegraph VQ-VAE training.

Loads shapegraphs from pickle, converts to PyG Data objects, and builds
train/val/test dataloaders.
"""

import pickle
from pathlib import Path

import torch
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx


class ShapegraphDataset(Dataset):
    """PyG dataset wrapping shapegraphs loaded from a pickle file.

    Each item is a PyG Data object with:
        - x: node features [N, F]
        - edge_index: graph edges [2, E]
    """

    def __init__(self, graphs: list[Data]):
        super().__init__()
        self._graphs = graphs

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]


def nx_to_pyg(G: nx.Graph) -> Data | None:
    """Convert a NetworkX shapegraph to a PyG Data object.

    Extracts node features: x, y, vx, vy, team, has_ball.
    Returns None if the graph has no nodes.
    """
    if G.number_of_nodes() == 0:
        return None

    node_attrs = []
    for _, attrs in G.nodes(data=True):
        feats = [
            attrs.get("x", 0.0),
            attrs.get("y", 0.0),
            attrs.get("vx", 0.0),
            attrs.get("vy", 0.0),
            float(attrs.get("team", 0)),
            float(attrs.get("has_ball", 0)),
        ]
        node_attrs.append(feats)

    x = torch.tensor(node_attrs, dtype=torch.float)

    # Build edge_index
    edges = list(G.edges())
    if len(edges) > 0:
        # Map node IDs to consecutive indices
        node_ids = list(G.nodes())
        node_map = {nid: i for i, nid in enumerate(node_ids)}
        src = [node_map[e[0]] for e in edges]
        dst = [node_map[e[1]] for e in edges]
        # Undirected: add both directions
        edge_index = torch.tensor(
            [src + dst, dst + src], dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def load_shapegraphs(path: str | Path) -> list[Data]:
    """Load shapegraphs.pkl and convert all frames to PyG Data objects."""
    with open(path, "rb") as f:
        games = pickle.load(f)

    data_list = []
    for game in games:
        for G in game:
            data = nx_to_pyg(G)
            if data is not None:
                data_list.append(data)
    return data_list


def build_dataloaders(
    data_path: str | Path,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders from shapegraphs.pkl."""
    all_data = load_shapegraphs(data_path)
    n = len(all_data)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = [all_data[i] for i in indices[:n_train]]
    val_data = [all_data[i] for i in indices[n_train:n_train + n_val]]
    test_data = [all_data[i] for i in indices[n_train + n_val:]]

    train_ds = ShapegraphDataset(train_data)
    val_ds = ShapegraphDataset(val_data)
    test_ds = ShapegraphDataset(test_data)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader
