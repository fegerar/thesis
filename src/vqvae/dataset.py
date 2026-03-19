"""
Dataset and dataloader utilities for shapegraph VQ-VAE training.

Loads shapegraphs from pickle, converts to PyG Data objects, and builds
train/val/test dataloaders.

Pickle format: list[dict[int, {"original": nx.Graph, "nominal": nx.Graph}]]
  - Each entry is a game (dict mapping frame_number -> pair of shapegraphs)
  - We use the "original" graph (actual player positions)
  - Node attrs: x, y, team ("home"/"away"), inferred_role (str),
                has_ball (bool), shirt, name, index
  - Edge attrs: distance, cross_team

Node features: [x_norm, y_norm, team, has_ball]
  - Positions normalized to [-1, 1] using pitch dimensions (105m x 68m)
  - Roles and edges are derived post-hoc via the shapegraph algorithm
"""

import pickle
from pathlib import Path

import torch
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# Standard pitch dimensions (meters) used for normalization
PITCH_X = 105.0
PITCH_Y = 68.0


class ShapegraphDataset(Dataset):
    def __init__(self, graphs: list[Data]):
        super().__init__()
        self._graphs = graphs

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]


def nx_to_pyg(G: nx.Graph) -> Data | None:
    """Convert a NetworkX shapegraph to a PyG Data object.

    Node features: [x_norm, y_norm, team, has_ball]
      - Positions normalized to [-1, 1]
    """
    if G.number_of_nodes() == 0:
        return None

    node_attrs = []
    for _, attrs in G.nodes(data=True):
        # Positions normalized to [-1, 1]
        px = float(attrs.get("x", 0.0)) / (PITCH_X / 2) - 1.0
        py = float(attrs.get("y", 0.0)) / (PITCH_Y / 2) - 1.0
        # Binary: team (home=0, away=1)
        team = 1.0 if attrs.get("team", "") == "away" else 0.0
        # Binary: has_ball
        has_ball = 1.0 if attrs.get("has_ball", False) else 0.0

        node_attrs.append([px, py, team, has_ball])

    x = torch.tensor(node_attrs, dtype=torch.float)

    # Build edge_index (still needed for GAT encoder message passing)
    edges = list(G.edges())
    if len(edges) > 0:
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


def load_shapegraphs(path: str | Path) -> tuple[list[Data], int]:
    """Load shapegraphs.pkl and convert all frames to PyG Data objects.

    Returns (data_list, node_dim).
    """
    with open(path, "rb") as f:
        games = pickle.load(f)

    data_list = []
    for game in games:
        for frame_data in game.values():
            G = frame_data["original"]
            data = nx_to_pyg(G)
            if data is not None:
                data_list.append(data)

    node_dim = 4  # x_norm, y_norm, team, has_ball
    print(f"Loaded {len(data_list)} shapegraphs (node_dim={node_dim})")
    return data_list, node_dim


def build_dataloaders(
    data_path: str | Path,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Build train/val/test dataloaders from shapegraphs.pkl.

    Returns (train_loader, val_loader, test_loader, node_dim).
    """
    all_data, node_dim = load_shapegraphs(data_path)
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

    return train_loader, val_loader, test_loader, node_dim
