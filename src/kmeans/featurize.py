"""
Convert shapegraphs to fixed-size feature vectors for k-means clustering.

Each graph is flattened to a vector by:
1. Sorting nodes canonically (by team, then by role alphabetically)
2. Extracting node features: [x, y, team, has_ball, role_one_hot...]
3. Padding/truncating to exactly `max_nodes` (22) slots
4. Flattening the adjacency matrix (upper triangle of the max_nodes x max_nodes matrix)
5. Concatenating: [flat_node_features | flat_adj_upper_triangle]
"""

import pickle
from pathlib import Path

import numpy as np
import networkx as nx


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


def graph_to_vector(G: nx.Graph, role_vocab: dict[str, int],
                    max_nodes: int = 22) -> np.ndarray | None:
    """Convert a single shapegraph to a fixed-size feature vector.

    Returns None if the graph is empty.
    """
    if G.number_of_nodes() == 0:
        return None

    num_roles = len(role_vocab)
    node_dim = 4 + num_roles  # x, y, team, has_ball, role_one_hot

    # Extract node data with canonical ordering key
    nodes = []
    for nid, attrs in G.nodes(data=True):
        team_val = 1.0 if attrs.get("team", "") == "away" else 0.0
        role = attrs.get("inferred_role", "?")
        nodes.append({
            "id": nid,
            "sort_key": (team_val, role),
            "x": float(attrs.get("x", 0.0)),
            "y": float(attrs.get("y", 0.0)),
            "team": team_val,
            "has_ball": 1.0 if attrs.get("has_ball", False) else 0.0,
            "role": role,
        })

    # Sort: home team first, then by role alphabetically
    nodes.sort(key=lambda n: n["sort_key"])

    # Truncate to max_nodes
    nodes = nodes[:max_nodes]
    n = len(nodes)

    # Build node feature matrix (max_nodes x node_dim), zero-padded
    node_feats = np.zeros((max_nodes, node_dim), dtype=np.float32)
    id_to_idx = {}
    for i, nd in enumerate(nodes):
        node_feats[i, 0] = nd["x"]
        node_feats[i, 1] = nd["y"]
        node_feats[i, 2] = nd["team"]
        node_feats[i, 3] = nd["has_ball"]
        if nd["role"] in role_vocab:
            node_feats[i, 4 + role_vocab[nd["role"]]] = 1.0
        id_to_idx[nd["id"]] = i

    # Build adjacency matrix (max_nodes x max_nodes)
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    for u, v in G.edges():
        if u in id_to_idx and v in id_to_idx:
            i, j = id_to_idx[u], id_to_idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # Extract upper triangle of adjacency (avoid redundancy)
    adj_upper = adj[np.triu_indices(max_nodes, k=1)]

    # Concatenate flat node features + flat adjacency upper triangle
    flat = np.concatenate([node_feats.ravel(), adj_upper])
    return flat


def featurize_graphs(games: list[dict], role_vocab: dict[str, int],
                     max_nodes: int = 22) -> np.ndarray:
    """Convert all shapegraphs to a feature matrix.

    Returns:
        X: (N_samples, feature_dim) float32 array
    """
    vectors = []
    for game in games:
        for frame_data in game.values():
            G = frame_data["original"]
            vec = graph_to_vector(G, role_vocab, max_nodes)
            if vec is not None:
                vectors.append(vec)

    return np.stack(vectors, axis=0)


def load_and_featurize(data_path: str | Path, max_nodes: int = 22
                       ) -> tuple[np.ndarray, dict[str, int]]:
    """Load shapegraphs.pkl and return the feature matrix + role vocab.

    Returns:
        X: (N_samples, feature_dim) feature matrix
        role_vocab: mapping from role name to index
    """
    with open(data_path, "rb") as f:
        games = pickle.load(f)

    role_vocab = build_role_vocab(games)
    print(f"Role vocabulary ({len(role_vocab)} roles): {list(role_vocab.keys())}")

    X = featurize_graphs(games, role_vocab, max_nodes)
    print(f"Featurized {X.shape[0]} graphs -> {X.shape[1]}-dim vectors")

    return X, role_vocab
