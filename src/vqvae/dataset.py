"""
Dataset and dataloader utilities for VQ-VAE training.

Parses DFL tracking data (XML) directly:
  - DFL_02_01: match information (team rosters, home/away)
  - DFL_04_03: position tracking (25 Hz player + ball positions)

Node features: [x_norm, y_norm, team, is_ball]
  - Players: team ∈ {0, 1}, is_ball = 0
  - Ball: team = 0.5, is_ball = 1
  - Positions normalized to [-1, 1] using pitch dimensions (105m × 68m)
  - Players sorted by team (home first), then by x-coordinate; ball appended last
"""

import os
import pickle
from pathlib import Path
from itertools import combinations
from xml.etree.ElementTree import iterparse

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# Standard pitch dimensions (meters)
PITCH_X = 105.0
PITCH_Y = 68.0


class FrameDataset(Dataset):
    def __init__(self, graphs: list[Data]):
        super().__init__()
        self._graphs = graphs

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]


def parse_match_info(path: str | Path) -> dict:
    """Parse DFL match information XML.

    Returns dict with:
        home_team_id, away_team_id, player_team (PersonId -> 0.0/1.0)
    """
    home_team_id = None
    away_team_id = None
    player_team: dict[str, float] = {}

    current_team_id = None
    current_role = None

    for event, elem in iterparse(path, events=("start", "end")):
        if event == "start" and elem.tag == "Team":
            current_team_id = elem.get("TeamId")
            current_role = elem.get("Role")
            if current_role == "home":
                home_team_id = current_team_id
            else:
                away_team_id = current_team_id

        if event == "end" and elem.tag == "Player":
            person_id = elem.get("PersonId")
            team_val = 0.0 if current_role == "home" else 1.0
            player_team[person_id] = team_val

        if event == "end" and elem.tag == "Team":
            current_team_id = None
            current_role = None

    return {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "player_team": player_team,
    }


def parse_positions(
    position_path: str | Path,
    match_info: dict,
    subsample: int = 25,
) -> list[Data]:
    """Parse DFL position tracking XML and build PyG Data objects.

    Streams the XML to keep memory usage low (~300-400 MB files).

    Args:
        position_path: path to DFL_04_03 XML file
        match_info: output of parse_match_info
        subsample: take every Nth frame (25 = 1 fps at 25 Hz)

    Returns:
        list of PyG Data objects
    """
    player_team = match_info["player_team"]
    home_team_id = match_info["home_team_id"]
    away_team_id = match_info["away_team_id"]

    # First pass: collect all frame data indexed by frame number
    # Each entry: frame_n -> {person_id: (x, y, team_val)} + ball: (x, y)
    frames: dict[int, dict] = {}

    current_person = None
    current_team_id = None
    is_ball = False

    for event, elem in iterparse(position_path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current_person = elem.get("PersonId")
            current_team_id = elem.get("TeamId")
            is_ball = current_team_id == "BALL"

        if event == "end" and elem.tag == "Frame":
            n = int(elem.get("N"))

            # Subsample: only keep every Nth frame
            if n % subsample != 0:
                elem.clear()
                continue

            x = float(elem.get("X"))
            y = float(elem.get("Y"))

            if n not in frames:
                frames[n] = {"players": {}, "ball": None}

            if is_ball:
                frames[n]["ball"] = (x, y)
            elif current_team_id in (home_team_id, away_team_id):
                # Skip referee and non-player FrameSets
                team_val = player_team.get(current_person)
                if team_val is not None:
                    frames[n]["players"][current_person] = (x, y, team_val)

            elem.clear()

        if event == "end" and elem.tag == "FrameSet":
            current_person = None
            current_team_id = None
            is_ball = False
            elem.clear()

    # Build PyG Data objects from complete frames
    data_list = []
    for n in sorted(frames.keys()):
        frame = frames[n]
        if frame["ball"] is None:
            continue
        if len(frame["players"]) < 2:
            continue

        data = _frame_to_pyg(frame)
        if data is not None:
            data_list.append(data)

    return data_list


def _frame_to_pyg(frame: dict) -> Data | None:
    """Convert a single frame dict to a PyG Data object.

    Node layout: [home players sorted by x, away players sorted by x, ball]
    """
    players = frame["players"]
    ball_x, ball_y = frame["ball"]

    # Build player list: (person_id, x, y, team_val)
    player_list = [
        (pid, x, y, team) for pid, (x, y, team) in players.items()
    ]

    # Sort: home (0) first, then away (1); within each team sort by x
    player_list.sort(key=lambda p: (p[3], p[1]))

    # Normalize positions
    node_attrs = []
    for _, px, py, team in player_list:
        node_attrs.append([
            px / (PITCH_X / 2),
            py / (PITCH_Y / 2),
            team,
            0.0,  # is_ball = 0
        ])

    # Append ball node
    node_attrs.append([
        ball_x / (PITCH_X / 2),
        ball_y / (PITCH_Y / 2),
        0.5,  # neutral team
        1.0,  # is_ball = 1
    ])

    x = torch.tensor(node_attrs, dtype=torch.float)
    n_nodes = x.size(0)

    # Fully connected edges (N is small: 23 nodes → 253 undirected edges)
    src, dst = [], []
    for i, j in combinations(range(n_nodes), 2):
        src.extend([i, j])
        dst.extend([j, i])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def load_dfl_match(
    match_info_path: str | Path,
    position_path: str | Path,
    subsample: int = 25,
) -> list[Data]:
    """Load a single DFL match into PyG Data objects."""
    match_info = parse_match_info(match_info_path)
    data_list = parse_positions(position_path, match_info, subsample=subsample)
    print(f"  Loaded {len(data_list)} frames from {Path(position_path).name}")
    return data_list


def load_all_matches(
    data_dir: str | Path,
    subsample: int = 25,
) -> tuple[list[Data], int]:
    """Scan data directory for DFL XML pairs and load all matches.

    Returns (data_list, node_dim).
    """
    data_dir = Path(data_dir)
    files = os.listdir(data_dir)

    # Group files by match ID (last segment before .xml)
    match_ids = {f.rsplit("_", 1)[-1].replace(".xml", "") for f in files}

    data_list = []
    for mid in sorted(match_ids):
        match_files = [f for f in files if mid in f]
        info_files = [f for f in match_files if "matchinformation" in f]
        pos_files = [f for f in match_files if "positions_raw" in f]

        if not info_files or not pos_files:
            continue

        print(f"Loading match {mid}...")
        match_data = load_dfl_match(
            data_dir / info_files[0],
            data_dir / pos_files[0],
            subsample=subsample,
        )
        data_list.append(match_data)

    all_data = [d for match in data_list for d in match]
    node_dim = 4  # x_norm, y_norm, team, is_ball
    print(f"Total: {len(all_data)} frames across {len(data_list)} matches (node_dim={node_dim})")
    return all_data, node_dim


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    subsample: int = 25,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Build train/val/test dataloaders from DFL data directory.

    Returns (train_loader, val_loader, test_loader, node_dim).
    """
    all_data, node_dim = load_all_matches(data_dir, subsample=subsample)
    n = len(all_data)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = [all_data[i] for i in indices[:n_train]]
    val_data = [all_data[i] for i in indices[n_train:n_train + n_val]]
    test_data = [all_data[i] for i in indices[n_train + n_val:]]

    train_ds = FrameDataset(train_data)
    val_ds = FrameDataset(val_data)
    test_ds = FrameDataset(test_data)

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
