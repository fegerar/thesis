"""Per-frame matrix construction and ground-cost helpers.

Each annotated frame is turned into five histograms — role grid per team
(always 5x5) and joint zone grid per team + ball (NxN where N is the zone
grid side used at annotation time) — then stacked across all frames.
"""

import json
import os

import torch

from src.annotate.shape_graph import ROLE_MATRIX


ROLE_TO_CELL = {ROLE_MATRIX[r][c]: (r, c) for r in range(5) for c in range(5)}
ROLE_SIDE = 5


def load_frames(path):
    if os.path.isdir(path):
        json_files = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(".json")
        )
        if not json_files:
            raise ValueError(f"no JSON files found in directory: {path}")

        frames = []
        for json_path in json_files:
            with open(json_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(
                    f"expected a list of frames in {json_path}, got {type(data).__name__}"
                )
            frames.extend(data)
        return frames

    with open(path, "r") as f:
        return json.load(f)


def subsample(frames, stride, max_frames):
    out = frames[::max(1, stride)]
    if max_frames is not None:
        out = out[:max_frames]
    return out


def infer_zone_side(frames):
    """Infer the joint-zone grid side by scanning zone coordinates."""
    hi = -1
    for fr in frames:
        for p in fr.get("players", []):
            z = p.get("zone")
            if isinstance(z, list) and len(z) == 2:
                hi = max(hi, int(z[0]), int(z[1]))
    if hi < 0:
        raise ValueError("no zone coordinates found in frames")
    return hi + 1


def frame_matrices(frame, zone_side):
    role_home = torch.zeros((ROLE_SIDE, ROLE_SIDE), dtype=torch.float32)
    role_guest = torch.zeros((ROLE_SIDE, ROLE_SIDE), dtype=torch.float32)
    zone_home = torch.zeros((zone_side, zone_side), dtype=torch.float32)
    zone_guest = torch.zeros((zone_side, zone_side), dtype=torch.float32)
    zone_ball = torch.zeros((zone_side, zone_side), dtype=torch.float32)

    attack_sign = frame.get("attack_sign") or {}

    for p in frame.get("players", []):
        team = p.get("team")
        role = p.get("tactical_role")
        zone = p.get("zone")

        if role in ROLE_TO_CELL:
            rr, rc = ROLE_TO_CELL[role]
            if team == "home":
                role_home[rr, rc] += 1.0
            elif team == "guest":
                role_guest[rr, rc] += 1.0

        if isinstance(zone, list) and len(zone) == 2:
            col, row = int(zone[0]), int(zone[1])
            if 0 <= row < zone_side and 0 <= col < zone_side:
                if p.get("player_id") == "BALL":
                    zone_ball[row, col] += 1.0
                elif team == "home":
                    zone_home[row, col] += 1.0
                elif team == "guest":
                    zone_guest[row, col] += 1.0

    # Orient each team's zone so attacking is +col (right of the image):
    # defending end on the left, attacking end on the right. Ball stays in
    # pitch frame. Falls back to no flip if the annotation predates attack_sign.
    if int(attack_sign.get("home", 1)) < 0:
        zone_home = zone_home.flip(dims=[1])
    if int(attack_sign.get("guest", 1)) < 0:
        zone_guest = zone_guest.flip(dims=[1])

    return {
        "role_home": role_home, "role_guest": role_guest,
        "zone_home": zone_home, "zone_guest": zone_guest,
        "zone_ball": zone_ball,
    }


def build_all_matrices(frames, zone_side):
    mats = [frame_matrices(fr, zone_side) for fr in frames]
    if not mats:
        return {}
    return {k: torch.stack([m[k] for m in mats], dim=0) for k in mats[0]}


def grid_cost(side, device):
    """L2 cost between flattened grid cells on a `side`x`side` lattice."""
    coords = torch.stack(
        torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"),
        dim=-1,
    ).reshape(-1, 2).to(device=device, dtype=torch.float32)
    return torch.cdist(coords, coords, p=2)


def to_distribution(x, eps=1e-8):
    """Flatten trailing dims and convert to row-stochastic histograms."""
    x = x.reshape(x.shape[0], -1)
    sums = x.sum(dim=1, keepdim=True)
    return (x + eps) / (sums + eps * x.shape[1])
