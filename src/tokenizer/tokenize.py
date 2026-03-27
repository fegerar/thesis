"""
Step 4: Build token sequences with SHOT markers.

Flattens multi-token VQ-VAE output into a single sequence per match,
then inserts SHOT tokens at positions corresponding to shot events.
The SHOT token acts as EOS — sequences are later split at shot positions
so the model learns P(shot | frame history).
"""

import logging
from datetime import datetime

from .utils import GoalEvent, ShotEvent

logger = logging.getLogger(__name__)


def _map_events_to_frames(
    events: list,
    frame_datetimes: list[datetime | None],
) -> dict[int, list]:
    """Map events to the nearest preceding deduplicated frame by timestamp."""
    event_after_frame = {}  # dedup_index -> list of events
    for event in events:
        best_idx = None
        best_delta = None
        for i, fdt in enumerate(frame_datetimes):
            if fdt is None:
                continue
            delta = (event.event_time - fdt).total_seconds()
            if delta < 0:
                continue  # frame is after the event
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = i

        if best_idx is not None:
            event_after_frame.setdefault(best_idx, []).append(event)
        else:
            logger.warning("Could not map event at %s to any frame", event.event_time)

    return event_after_frame


def build_token_sequence(
    frame_tokens: list[list[int]],
    frame_ids: list[int],
    timestamps: list[str],
    goals: list[GoalEvent],
    codebook_size: int,
    shots: list[ShotEvent] | None = None,
) -> dict:
    """Build a flat token sequence for one match with SHOT (or GOAL) markers.

    When shots are provided, SHOT tokens are inserted at all shot positions
    and xG values are stored alongside. When only goals are provided,
    falls back to the legacy GOAL-only behavior.

    Args:
        frame_tokens: list of token lists per deduplicated frame
        frame_ids: original frame numbers for each deduplicated frame
        timestamps: timestamp strings for each deduplicated frame
        goals: goal events parsed from event XML (legacy, used if shots is None)
        codebook_size: VQ-VAE codebook size K
        shots: shot events parsed from event XML (preferred)

    Returns:
        dict with keys:
            tokens: list[int] — flat token sequence with special tokens
            frame_ids: list[int] — frame IDs (no entries for special tokens)
            timestamps: list[float] — Unix timestamps (no entries for special tokens)
            shot_positions: list[int] — indices in tokens where SHOT appears
            shot_xg: list[float] — xG for each shot (parallel to shot_positions)
            shot_outcomes: list[str] — outcome for each shot
            goal_positions: list[int] — kept for backward compat (subset of shot_positions)
            goal_token_id: int
    """
    shot_token_id = codebook_size  # one beyond last valid codebook index

    # Parse timestamps to datetime for comparison
    frame_datetimes = []
    for ts in timestamps:
        try:
            frame_datetimes.append(datetime.fromisoformat(ts))
        except (ValueError, TypeError):
            frame_datetimes.append(None)

    # Decide which events to use
    if shots is not None:
        events = shots
        use_shots = True
    else:
        events = goals
        use_shots = False

    event_after_frame = _map_events_to_frames(events, frame_datetimes)

    for idx, evts in event_after_frame.items():
        for evt in evts:
            label = f"xG={evt.xg:.3f} {evt.outcome}" if use_shots else evt.result
            logger.info("Event mapped to frame idx %d: %s", idx, label)

    # Build flat token sequence
    tokens = []
    out_frame_ids = []
    out_timestamps = []
    shot_positions = []
    shot_xg = []
    shot_outcomes = []
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

        # Insert SHOT/GOAL token(s) after this frame if events occurred here
        if i in event_after_frame:
            for evt in event_after_frame[i]:
                pos = len(tokens)
                tokens.append(shot_token_id)
                shot_positions.append(pos)
                if use_shots:
                    shot_xg.append(evt.xg)
                    shot_outcomes.append(evt.outcome)
                    if evt.outcome == "goal":
                        goal_positions.append(pos)
                else:
                    shot_xg.append(1.0)  # legacy: goals have xG=1
                    shot_outcomes.append("goal")
                    goal_positions.append(pos)

    n_special = len(shot_positions)
    logger.info(
        "Token sequence: %d tokens (%d from frames, %d SHOT markers)",
        len(tokens), len(tokens) - n_special, n_special,
    )

    return {
        "tokens": tokens,
        "frame_ids": out_frame_ids,
        "timestamps": out_timestamps,
        "shot_positions": shot_positions,
        "shot_xg": shot_xg,
        "shot_outcomes": shot_outcomes,
        "goal_positions": goal_positions,
        "goal_token_id": shot_token_id,
    }
