"""
Step 1: Load or generate shapegraphs per match.

Supports two modes:
  - Load from pre-computed shapegraphs.pkl (maps games to match IDs via timestamps)
  - Generate fresh per match using the shapegraphs package
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import networkx as nx

from .utils import discover_matches, parse_match_start_time

logger = logging.getLogger(__name__)


def load_shapegraphs_per_match(
    data_dir: str | Path,
    shapegraphs_pkl: str | Path | None = None,
) -> list[dict]:
    """Load shapegraphs organized by match.

    If shapegraphs_pkl is provided, loads from pickle and maps each game
    to a match ID using timestamp matching. Otherwise, generates fresh
    shapegraphs per match using the shapegraphs package.

    Returns a list of dicts with keys:
        match_id: str
        frames: dict[int, nx.Graph]  — frame_number -> "original" graph
        events_path: Path
    """
    matches = discover_matches(data_dir)

    if shapegraphs_pkl is not None:
        return _load_from_pkl(shapegraphs_pkl, matches)
    else:
        return _generate_fresh(matches)


def _load_from_pkl(pkl_path: str | Path, matches: list[dict]) -> list[dict]:
    """Load shapegraphs from pickle and map to matches via kickoff timestamps."""
    logger.info(f"Loading shapegraphs from {pkl_path}")
    with open(pkl_path, "rb") as f:
        games = pickle.load(f)

    logger.info(f"Loaded {len(games)} games from pickle")

    # Get kickoff time for each match from event XML
    match_start_times = {}
    for m in matches:
        start = parse_match_start_time(m["events_path"])
        if start is not None:
            match_start_times[m["match_id"]] = start

    # Get first timestamp from each game in the pickle
    game_start_times = []
    for i, game in enumerate(games):
        first_frame_num = min(game.keys())
        G = game[first_frame_num]["original"]
        ts_str = G.graph.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = None
        game_start_times.append((i, ts))

    # Match games to match IDs by finding closest kickoff time
    results = []
    used_games = set()

    for m in matches:
        mid = m["match_id"]
        kickoff = match_start_times.get(mid)
        if kickoff is None:
            logger.warning(f"No kickoff time for {mid}, skipping")
            continue

        best_game_idx = None
        best_delta = None
        for game_idx, game_ts in game_start_times:
            if game_idx in used_games or game_ts is None:
                continue
            delta = abs((game_ts - kickoff).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_game_idx = game_idx

        if best_game_idx is None:
            logger.warning(f"Could not match game to {mid}")
            continue

        used_games.add(best_game_idx)
        game = games[best_game_idx]

        # Extract only "original" graphs, keyed by frame number
        frames = {fn: data["original"] for fn, data in game.items()}

        logger.info(
            f"Matched {mid} to game {best_game_idx} "
            f"(delta={best_delta:.1f}s, {len(frames)} frames)"
        )
        results.append({
            "match_id": mid,
            "frames": frames,
            "events_path": m["events_path"],
        })

    return results


def _generate_fresh(matches: list[dict]) -> list[dict]:
    """Generate shapegraphs per match using the shapegraphs package."""
    from shapegraphs.readers.bassek import generate_shapegraphs_from_files

    results = []
    for m in matches:
        mid = m["match_id"]
        logger.info(f"Generating shapegraphs for {mid}...")

        game = generate_shapegraphs_from_files(
            match_info_path=str(m["matchinfo_path"]),
            position_data_path=str(m["positions_path"]),
            verbose=True,
        )

        frames = {fn: data["original"] for fn, data in game.items()}
        logger.info(f"  {mid}: {len(frames)} frames")

        results.append({
            "match_id": mid,
            "frames": frames,
            "events_path": m["events_path"],
        })

    return results
