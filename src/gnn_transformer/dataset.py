"""
Dataset for GNN Transformer trajectory prediction.

Parses raw DFL tracking data (positions XML + events XML + matchinfo XML),
computes velocity, adds ball as a separate node, and builds temporal
sequences of fully connected graphs.

Nodes (23 total):
    - 22 players: (x, y, vx, vy, team, is_ball=0)  —  team: 0=home, 1=away
    - 1 ball:     (x, y, vx, vy, team=0, is_ball=1)

Edges: fully connected (23*22 directed), edge feature = Euclidean distance.
"""

import logging
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

PITCH_X = 105.0
PITCH_Y = 68.0
NUM_PLAYERS = 22
NUM_NODES = 23  # 22 players + 1 ball
NODE_DIM = 6    # x, y, vx, vy, team, is_ball

# Pre-compute fully connected edge index (shared across all graphs)
_FC_EDGES = None


def _get_fc_edges() -> torch.Tensor:
    """Fully connected edge index for 23 nodes, cached."""
    global _FC_EDGES
    if _FC_EDGES is None:
        src, dst = [], []
        for i in range(NUM_NODES):
            for j in range(NUM_NODES):
                if i != j:
                    src.append(i)
                    dst.append(j)
        _FC_EDGES = torch.tensor([src, dst], dtype=torch.long)
    return _FC_EDGES


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_match_teams(matchinfo_path: Path) -> dict:
    """Parse matchinfo XML to get team role mapping.

    Returns:
        {team_id: "home" or "guest", ...}
        plus "home_id" and "guest_id" keys.
    """
    tree = ET.parse(matchinfo_path)
    root = tree.getroot()
    teams = {}
    for team_el in root.iter("Team"):
        tid = team_el.get("TeamId")
        role = team_el.get("Role")  # "home" or "guest"
        teams[tid] = role
        if role == "home":
            teams["home_id"] = tid
        else:
            teams["guest_id"] = tid
    return teams


def parse_positions(positions_path: Path, subsample: int = 10
                    ) -> tuple[dict[str, dict[int, tuple[float, float]]],
                               dict[str, str],
                               list[int]]:
    """Stream-parse the positions XML.

    Args:
        positions_path: Path to positions_raw_observed XML.
        subsample: Keep every Nth frame (25fps / 10 = 2.5fps default).

    Returns:
        player_positions: {person_id: {frame_num: (x, y)}}
        player_teams:     {person_id: team_id}
        frame_numbers:    sorted list of unique frame numbers (after subsample)
    """
    player_positions: dict[str, dict[int, tuple[float, float]]] = {}
    player_teams: dict[str, str] = {}
    all_frames: set[int] = set()

    current_person = None
    current_team = None

    for event, elem in ET.iterparse(str(positions_path), events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current_team = elem.get("TeamId", "")
            current_person = elem.get("PersonId", "")
            if current_team == "referee":
                current_person = None  # skip referees
            elif current_person:
                player_teams[current_person] = current_team
                if current_person not in player_positions:
                    player_positions[current_person] = {}

        elif event == "end" and elem.tag == "Frame" and current_person:
            n = int(elem.get("N", "0"))
            if n % subsample == 0:
                x = float(elem.get("X", "0"))
                y = float(elem.get("Y", "0"))
                player_positions[current_person][n] = (x, y)
                all_frames.add(n)
            elem.clear()

        elif event == "end" and elem.tag == "FrameSet":
            current_person = None
            current_team = None
            elem.clear()

    frame_numbers = sorted(all_frames)
    logger.info(f"Parsed {len(player_positions)} players, "
                f"{len(frame_numbers)} frames (subsample={subsample})")
    return player_positions, player_teams, frame_numbers


def parse_ball_events(events_path: Path) -> list[tuple[float, float, float]]:
    """Parse events XML for ball positions.

    Events use corner-based coordinates [0, 105] x [0, 68].
    We convert to center-based [-52.5, 52.5] x [-34, 34].

    Returns:
        List of (timestamp_seconds, x_center, y_center) sorted by time.
    """
    ball_positions = []
    tree = ET.parse(events_path)
    root = tree.getroot()

    for event_el in root.iter("Event"):
        x_str = event_el.get("X-Position")
        y_str = event_el.get("Y-Position")
        t_str = event_el.get("EventTime")
        if x_str and y_str and t_str:
            try:
                x = float(x_str) - PITCH_X / 2  # corner-based → center-based
                y = float(y_str) - PITCH_Y / 2
                t = datetime.fromisoformat(t_str).timestamp()
                ball_positions.append((t, x, y))
            except (ValueError, OSError):
                continue

    ball_positions.sort(key=lambda b: b[0])
    logger.info(f"Parsed {len(ball_positions)} ball events")
    return ball_positions


def _frame_to_timestamp(frame_num: int, base_frame: int,
                        base_timestamp: float, fps: float = 25.0) -> float:
    """Convert frame number to timestamp in seconds."""
    return base_timestamp + (frame_num - base_frame) / fps


def interpolate_ball(frame_numbers: list[int],
                     ball_events: list[tuple[float, float, float]],
                     base_frame: int, base_timestamp: float,
                     fps: float = 25.0) -> dict[int, tuple[float, float]]:
    """Interpolate ball position for each frame from discrete events.

    Returns {frame_num: (x, y)}.
    """
    if not ball_events:
        return {fn: (0.0, 0.0) for fn in frame_numbers}

    event_times = np.array([b[0] for b in ball_events])
    event_x = np.array([b[1] for b in ball_events])
    event_y = np.array([b[2] for b in ball_events])

    ball_pos = {}
    for fn in frame_numbers:
        t = _frame_to_timestamp(fn, base_frame, base_timestamp, fps)
        bx = float(np.interp(t, event_times, event_x))
        by = float(np.interp(t, event_times, event_y))
        ball_pos[fn] = (bx, by)

    return ball_pos


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------

def build_match_frames(
    matchinfo_path: Path,
    positions_path: Path,
    events_path: Path,
    subsample: int = 10,
) -> list[torch.Tensor]:
    """Build a list of (23, 6) frame tensors from raw DFL data for one match.

    Features per node: [x_norm, y_norm, vx, vy, team, is_ball]
    Velocity is computed from consecutive subsampled frames.

    Returns list of tensors in frame order, or empty list on failure.
    """
    # Parse team info
    teams = parse_match_teams(matchinfo_path)
    home_id = teams.get("home_id", "")

    # Parse positions (streaming, subsampled)
    player_positions, player_teams, frame_numbers = parse_positions(
        positions_path, subsample=subsample
    )
    if len(frame_numbers) < 2:
        return []

    # Determine which team_id maps to 0 (home) vs 1 (away)
    team_label = {}
    for pid, tid in player_teams.items():
        team_label[pid] = 0.0 if tid == home_id else 1.0

    # Parse ball events and interpolate
    ball_events = parse_ball_events(events_path)

    # We need a timestamp reference. Parse the first Frame timestamp from XML
    # to map frame numbers to absolute time.
    base_frame = frame_numbers[0]
    # Quick parse for first timestamp
    base_timestamp = _get_base_timestamp(positions_path, base_frame)
    ball_pos = interpolate_ball(
        frame_numbers, ball_events, base_frame, base_timestamp
    )

    # For each frame, find which players are active (have position data)
    # We sort players consistently: home by x, then away by x
    frames = []
    prev_frame_tensor = None

    for fn in frame_numbers:
        home_players = []
        away_players = []

        for pid, positions in player_positions.items():
            if fn not in positions:
                continue
            x, y = positions[fn]
            t = team_label.get(pid, 0.0)
            if t == 0.0:
                home_players.append((x, y, t))
            else:
                away_players.append((x, y, t))

        # Sort each team by x position for consistent ordering
        home_players.sort(key=lambda p: p[0])
        away_players.sort(key=lambda p: p[0])

        # Combine: home first, then away
        all_players = home_players + away_players

        # Pad or truncate to exactly 22
        while len(all_players) < NUM_PLAYERS:
            # Pad with zeros (off-pitch placeholder)
            all_players.append((0.0, 0.0, 0.0))
        all_players = all_players[:NUM_PLAYERS]

        # Build node features (no velocity yet)
        node_feats = []
        for px, py, team in all_players:
            x_norm = px / (PITCH_X / 2)
            y_norm = py / (PITCH_Y / 2)
            node_feats.append([x_norm, y_norm, 0.0, 0.0, team, 0.0])

        # Ball node
        bx, by = ball_pos.get(fn, (0.0, 0.0))
        bx_norm = bx / (PITCH_X / 2)
        by_norm = by / (PITCH_Y / 2)
        node_feats.append([bx_norm, by_norm, 0.0, 0.0, 0.0, 1.0])

        frame_tensor = torch.tensor(node_feats, dtype=torch.float)  # (23, 6)

        # Compute velocity from previous frame
        if prev_frame_tensor is not None:
            frame_tensor[:, 2] = frame_tensor[:, 0] - prev_frame_tensor[:, 0]  # vx
            frame_tensor[:, 3] = frame_tensor[:, 1] - prev_frame_tensor[:, 1]  # vy

        frames.append(frame_tensor)
        prev_frame_tensor = frame_tensor

    logger.info(f"Built {len(frames)} frames for match")
    return frames


def _get_base_timestamp(positions_path: Path, target_frame: int) -> float:
    """Get the timestamp of a specific frame number from positions XML."""
    for _, elem in ET.iterparse(str(positions_path), events=("end",)):
        if elem.tag == "Frame":
            n = int(elem.get("N", "0"))
            if n == target_frame:
                t_str = elem.get("T", "")
                try:
                    return datetime.fromisoformat(t_str).timestamp()
                except (ValueError, OSError):
                    break
            elem.clear()
        elif elem.tag == "FrameSet":
            elem.clear()
    # Fallback: approximate from first event
    return 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectorySequenceDataset(Dataset):
    """Sliding-window dataset of graph sequences for next-frame prediction.

    Each sample:
        input_frames:  (seq_len, 23, 6)
        target_frame:  (23, 6)
        edge_index:    (2, E) fully connected
        edge_attr_seq: (seq_len, E, 1) per-frame edge distances
    """

    def __init__(self, match_frames: list[list[torch.Tensor]], seq_len: int):
        self.seq_len = seq_len
        self.edge_index = _get_fc_edges()

        self.samples: list[tuple[int, int]] = []
        self.matches = match_frames

        for m_idx, frames in enumerate(match_frames):
            for start in range(len(frames) - seq_len):
                self.samples.append((m_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        m_idx, start = self.samples[idx]
        frames = self.matches[m_idx]

        input_frames = torch.stack(
            frames[start : start + self.seq_len], dim=0
        )  # (seq_len, 23, 6)
        target_frame = frames[start + self.seq_len]  # (23, 6)

        # Edge distances per input frame
        edge_attrs = []
        for t in range(self.seq_len):
            pos = input_frames[t, :, :2]  # (23, 2)
            src_pos = pos[self.edge_index[0]]
            dst_pos = pos[self.edge_index[1]]
            dist = torch.norm(src_pos - dst_pos, dim=1, keepdim=True)
            edge_attrs.append(dist)
        edge_attr_seq = torch.stack(edge_attrs, dim=0)  # (seq_len, E, 1)

        return input_frames, target_frame, self.edge_index, edge_attr_seq


def collate_fn(batch):
    """Stack batch items; edge_index is shared."""
    input_frames = torch.stack([b[0] for b in batch], dim=0)    # (B, T, 23, 6)
    target_frames = torch.stack([b[1] for b in batch], dim=0)   # (B, 23, 6)
    edge_index = batch[0][2]                                      # (2, E)
    edge_attr_seq = torch.stack([b[3] for b in batch], dim=0)   # (B, T, E, 1)
    return input_frames, target_frames, edge_index, edge_attr_seq


# ---------------------------------------------------------------------------
# Dataloader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str | Path,
    seq_len: int,
    batch_size: int,
    subsample: int = 10,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders from raw DFL data.

    Splits at the match level to avoid leakage.
    """
    from tokenizer.utils import discover_matches

    data_dir = Path(data_dir)
    matches = discover_matches(data_dir)
    logger.info(f"Discovered {len(matches)} matches")

    all_match_frames = []
    for match in matches:
        frames = build_match_frames(
            matchinfo_path=match["matchinfo_path"],
            positions_path=match["positions_path"],
            events_path=match["events_path"],
            subsample=subsample,
        )
        if len(frames) > seq_len + 1:
            all_match_frames.append(frames)

    logger.info(f"Loaded {len(all_match_frames)} matches with sufficient frames")

    # Split at match level
    n = len(all_match_frames)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_matches = [all_match_frames[i] for i in perm[:n_train]]
    val_matches = [all_match_frames[i] for i in perm[n_train:n_train + n_val]]
    test_matches = [all_match_frames[i] for i in perm[n_train + n_val:]]

    train_ds = TrajectorySequenceDataset(train_matches, seq_len)
    val_ds = TrajectorySequenceDataset(val_matches, seq_len)
    test_ds = TrajectorySequenceDataset(test_matches, seq_len)

    print(f"Dataset splits — train: {len(train_ds)}, val: {len(val_ds)}, "
          f"test: {len(test_ds)} samples (seq_len={seq_len})")

    kwargs = dict(num_workers=num_workers,
                  persistent_workers=num_workers > 0,
                  collate_fn=collate_fn)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs),
    )
