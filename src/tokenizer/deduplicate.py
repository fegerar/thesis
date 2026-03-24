"""
Step 2: Temporal deduplication of shapegraph sequences.

Consecutive frames with identical edge sets and position labels are collapsed
to a single frame, keeping only the first frame of each constant run.
"""

import logging
from datetime import datetime

import networkx as nx

logger = logging.getLogger(__name__)


def _graph_signature(G: nx.Graph) -> tuple:
    """Compute a hashable signature for a shapegraph.

    Two shapegraphs are identical iff they have the same:
      1. Edge set (pairs of player IDs)
      2. Position label (inferred_role) for every player
    """
    # Edges: frozenset of sorted tuples (player_id pairs)
    edges = frozenset(
        tuple(sorted(e)) for e in G.edges()
    )

    # Position labels: frozenset of (player_id, inferred_role)
    labels = frozenset(
        (nid, attrs.get("inferred_role", ""))
        for nid, attrs in G.nodes(data=True)
    )

    return (edges, labels)


def deduplicate_match(
    frames: dict[int, nx.Graph],
) -> dict:
    """Deduplicate a match's shapegraph sequence.

    Args:
        frames: dict mapping frame_number -> NetworkX shapegraph

    Returns:
        dict with keys:
            frames: dict[int, nx.Graph] — deduplicated frames (frame_num -> graph)
            frame_ids: list[int] — ordered frame numbers of kept frames
            timestamps: list[str] — timestamps of kept frames
            num_raw: int
            num_deduped: int
            compression_ratio: float
    """
    sorted_frame_nums = sorted(frames.keys())
    num_raw = len(sorted_frame_nums)

    kept_frame_ids = []
    kept_timestamps = []
    kept_frames = {}
    prev_sig = None

    for fn in sorted_frame_nums:
        G = frames[fn]
        sig = _graph_signature(G)

        if sig != prev_sig:
            kept_frame_ids.append(fn)
            kept_frames[fn] = G
            ts = G.graph.get("timestamp", "")
            kept_timestamps.append(ts)
            prev_sig = sig

    num_deduped = len(kept_frame_ids)
    ratio = num_deduped / num_raw if num_raw > 0 else 0.0

    logger.info(
        "Deduplication: %d -> %d frames (%.1f%% kept)",
        num_raw, num_deduped, ratio * 100,
    )

    return {
        "frames": kept_frames,
        "frame_ids": kept_frame_ids,
        "timestamps": kept_timestamps,
        "num_raw": num_raw,
        "num_deduped": num_deduped,
        "compression_ratio": ratio,
    }
