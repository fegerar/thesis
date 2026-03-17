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
"""

import pickle
from pathlib import Path

import torch
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


# Collect all unique roles across the dataset for one-hot encoding
def build_role_vocab(games: list[dict]) -> dict[str, int]:
    """Scan all graphs to build a role -> index mapping."""
    roles = set()
    for game in games:
        for frame_data in game.values():
            G = frame_data["original"]
            for _, attrs in G.nodes(data=True):
                roles.add(attrs.get("inferred_role", "?"))
    roles = sorted(roles)
    return {r: i for i, r in enumerate(roles)}


class ShapegraphDataset(Dataset):
    def __init__(self, graphs: list[Data]):
        super().__init__()
        self._graphs = graphs

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]


def nx_to_pyg(G: nx.Graph, role_vocab: dict[str, int]) -> Data | None:
    """Convert a NetworkX shapegraph to a PyG Data object.

    Node features: [x, y, team_binary, has_ball_binary, role_one_hot...]
    """
    if G.number_of_nodes() == 0:
        return None

    num_roles = len(role_vocab)
    node_attrs = []
    for _, attrs in G.nodes(data=True):
        # Continuous: position
        px = float(attrs.get("x", 0.0))
        py = float(attrs.get("y", 0.0))
        # Binary: team (home=0, away=1)
        team = 1.0 if attrs.get("team", "") == "away" else 0.0
        # Binary: has_ball
        has_ball = 1.0 if attrs.get("has_ball", False) else 0.0
        # One-hot: inferred role
        role = attrs.get("inferred_role", "?")
        role_oh = [0.0] * num_roles
        if role in role_vocab:
            role_oh[role_vocab[role]] = 1.0

        node_attrs.append([px, py, team, has_ball] + role_oh)

    x = torch.tensor(node_attrs, dtype=torch.float)

    # Build edge_index
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

    role_vocab = build_role_vocab(games)
    print(f"Role vocabulary ({len(role_vocab)} roles): {list(role_vocab.keys())}")

    data_list = []
    for game in games:
        for frame_data in game.values():
            G = frame_data["original"]
            data = nx_to_pyg(G, role_vocab)
            if data is not None:
                data_list.append(data)

    # node_dim = 4 (x, y, team, has_ball) + num_roles
    node_dim = 4 + len(role_vocab)
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
