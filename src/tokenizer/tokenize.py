"""
Step 4: Build token sequences with GOAL markers.

Flattens multi-token VQ-VAE output into a single sequence per match,
then inserts GOAL tokens at positions corresponding to goal events.
"""

import logging
from datetime import datetime

from .utils import GoalEvent

logger = logging.getLogger(__name__)


def build_token_sequence(
    frame_tokens: list[list[int]],
    frame_ids: list[int],
    timestamps: list[str],
    goals: list[GoalEvent],
    codebook_size: int,
) -> dict:
    """Build a flat token sequence for one match with GOAL markers.

    Args:
        frame_tokens: list of token lists per deduplicated frame
            (each inner list has T codebook indices for multi-token VQ-VAE)
        frame_ids: original frame numbers for each deduplicated frame
        timestamps: timestamp strings for each deduplicated frame
        goals: goal events parsed from event XML
        codebook_size: VQ-VAE codebook size K

    Returns:
        dict with keys:
            tokens: list[int] — flat token sequence with GOAL tokens
            frame_ids: list[int] — frame IDs (no entries for GOAL tokens)
            timestamps: list[float] — Unix timestamps (no entries for GOAL tokens)
            goal_positions: list[int] — indices in tokens where GOAL appears
            goal_token_id: int
    """
    goal_token_id = codebook_size  # one beyond last valid codebook index

    # Parse timestamps to datetime for comparison with goal events
    frame_datetimes = []
    for ts in timestamps:
        try:
            frame_datetimes.append(datetime.fromisoformat(ts))
        except (ValueError, TypeError):
            frame_datetimes.append(None)

    # Find which deduplicated frame each goal maps to:
    # the frame whose timestamp is closest to (and <=) the goal time
    goal_after_frame = {}  # dedup_index -> list of goals
    for goal in goals:
        best_idx = None
        best_delta = None
        for i, fdt in enumerate(frame_datetimes):
            if fdt is None:
                continue
            delta = (goal.event_time - fdt).total_seconds()
            if delta < 0:
                continue  # frame is after the goal
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = i

        if best_idx is not None:
            goal_after_frame.setdefault(best_idx, []).append(goal)
            logger.info(
                "Goal (%s) mapped to frame idx %d (delta=%.1fs)",
                goal.result, best_idx, best_delta,
            )
        else:
            logger.warning("Could not map goal at %s to any frame", goal.event_time)

    # Build flat token sequence
    tokens = []
    out_frame_ids = []
    out_timestamps = []
    goal_positions = []

    for i, (ft, fid, ts) in enumerate(zip(frame_tokens, frame_ids, timestamps)):
        if not ft:  # skip frames that failed encoding
            continue

        # Add frame tokens
        for tok in ft:
            tokens.append(tok)
            out_frame_ids.append(fid)
            try:
                out_timestamps.append(datetime.fromisoformat(ts).timestamp())
            except (ValueError, TypeError):
                out_timestamps.append(0.0)

        # Insert GOAL token(s) after this frame if goals occurred here
        if i in goal_after_frame:
            for _ in goal_after_frame[i]:
                goal_positions.append(len(tokens))
                tokens.append(goal_token_id)

    logger.info(
        "Token sequence: %d tokens (%d from frames, %d GOAL markers)",
        len(tokens),
        len(tokens) - len(goal_positions),
        len(goal_positions),
    )

    return {
        "tokens": tokens,
        "frame_ids": out_frame_ids,
        "timestamps": out_timestamps,
        "goal_positions": goal_positions,
        "goal_token_id": goal_token_id,
    }
