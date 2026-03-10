"""
Frame to Shape Graph
converting each frame of soccer tracking data into a shape graph representation.
"""

import math
from typing import Dict, Optional, Tuple
import networkx as nx
import numpy as np
from tqdm import tqdm

from idsse.data import parse_match_info, parse_position_data
from shapegraphs.inference import infer_positions_all
from shapegraphs.utils import compute_shape_graph


def frame_to_shapegraph(
    frame_data: dict,
    match_info: dict,
    home_attacking_up_first_half: bool = True,
    game_section: str = "firstHalf",
) -> Optional[nx.Graph]:
    """
    Convert a single frame of tracking data to a shape graph (NetworkX graph).
    """
    players_data = frame_data.get("players", {})
    ball_data = frame_data.get("ball", None)

    if len(players_data) < 4:
        return None

    # Build ordered arrays
    # Filter out referees and unknown persons
    person_ids = [pid for pid in players_data.keys() if players_data[pid].get("team") in ("home", "away")]
    n = len(person_ids)
    points = np.zeros((n, 2))
    teams = []

    for i, pid in enumerate(person_ids):
        pd = players_data[pid]
        points[i] = [pd["x"], pd["y"]]
        teams.append(pd["team"])

    home_indices = [i for i, t in enumerate(teams) if t == "home"]
    away_indices = [i for i, t in enumerate(teams) if t == "away"]

    # Compute shape graph on ALL players (22 nodes)
    sg_edges = compute_shape_graph(points, alpha_threshold=math.pi / 4)

    # Determine attacking direction
    if game_section == "firstHalf":
        home_up = home_attacking_up_first_half
    else:
        home_up = not home_attacking_up_first_half

    # Infer positions
    positions = infer_positions_all(
        points, home_indices, away_indices, home_attacking_up=home_up)

    # Determine who has the ball
    has_ball_idx = None
    if ball_data is not None:
        ball_pos = np.array([ball_data["x"], ball_data["y"]])
        dists = np.linalg.norm(points - ball_pos, axis=1)
        has_ball_idx = int(np.argmin(dists))

    # Build NetworkX graph
    G = nx.Graph()
    for i, pid in enumerate(person_ids):
        pinfo = match_info["players"].get(pid, {})
        G.add_node(pid,
                   index=i,
                   x=float(points[i, 0]),
                   y=float(points[i, 1]),
                   team=teams[i],
                   inferred_role=positions.get(i, "?"),
                   original_position=pinfo.get("position", ""),
                   shirt=pinfo.get("shirt", 0),
                   name=pinfo.get("name", ""),
                   has_ball=(i == has_ball_idx)
                )

    for u, v in sg_edges:
        pid_u = person_ids[u]
        pid_v = person_ids[v]
        dist = float(np.linalg.norm(points[u] - points[v]))
        cross_team = teams[u] != teams[v]
        G.add_edge(pid_u, pid_v, distance=dist, cross_team=cross_team)

    G.graph["timestamp"] = frame_data.get("timestamp", "")
    G.graph["ball"] = ball_data

    return G


def generate_shapegraphs(
    match_info_path: str,
    position_data_path: str,
    frame_range: Optional[Tuple[int, int]] = None,
    ball_in_play_only: bool = True,
    verbose: bool = True,
) -> Dict[int, nx.Graph]:
    """
    Generate shape graphs for a range of frames from the Bassek et al. soccer tracking dataset.
    """

    if verbose:
        print("Parsing match information...")
    match_info = parse_match_info(match_info_path)
    if verbose:
        n_players = len(match_info.get("players", {}))
        print(f"  Found {n_players} players in match info")

    if verbose:
        print("Parsing position data...")
    frames = parse_position_data(position_data_path, match_info, frame_range)
    if verbose:
        print(f"  Loaded {len(frames)} frames")

    # Determine attacking direction from first frame
    # Heuristic: the team whose GK has smaller X is defending the left (negative X) goal,
    # so they are attacking the right (positive X) goal ("up").
    home_attacking_up = True  # default
    if frames and match_info.get("players"):
        frame_numbers = sorted(frames.keys())
        first_frame = frames[frame_numbers[0]]
        home_gk_x = None
        away_gk_x = None
        players_data = first_frame.get("players", {})
        for pid, pd in players_data.items():
            pinfo = match_info["players"].get(pid, {})
            if pinfo.get("position", "") == "TW":  # Torwart (Goalkeeper) in DFL data
                if pd["team"] == "home":
                    home_gk_x = pd["x"]
                elif pd["team"] == "away":
                    away_gk_x = pd["x"]

        if home_gk_x is not None and away_gk_x is not None:
            home_attacking_up = home_gk_x < away_gk_x

    results: Dict[int, nx.Graph] = {}
    frame_numbers = sorted(frames.keys())

    for fn in tqdm(frame_numbers, desc="Processing frames"):
        fd = frames[fn]

        # Skip if ball not in play
        if ball_in_play_only and fd.get("ball") is not None:
            if fd["ball"].get("status", 1) == 0:
                continue

        # Determine game section (rough heuristic based on frame number)
        # In the Bassek dataset, first half is typically frames < ~75000
        game_section = "firstHalf" if fn < 80000 else "secondHalf"

        G = frame_to_shapegraph(fd, match_info, home_attacking_up, game_section)
        if G is not None:
            G.graph["frame_number"] = fn
            results[fn] = G

    if verbose:
        print(f"Generated {len(results)} shape graphs")

    return results
