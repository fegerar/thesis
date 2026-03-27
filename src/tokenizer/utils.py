"""
Shared helpers: event XML parsing, match ID extraction, logging.
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


logger = logging.getLogger(__name__)


@dataclass
class GoalEvent:
    """A goal detected from event XML."""
    event_time: datetime
    team_id: str
    player_id: str
    result: str  # e.g. "1:0"


@dataclass
class ShotEvent:
    """A shot detected from event XML."""
    event_time: datetime
    team_id: str
    player_id: str
    xg: float
    outcome: str  # "goal", "saved", "wide", "blocked", "woodwork"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def discover_matches(data_dir: str | Path) -> list[dict]:
    """Discover match files in the data directory.

    Returns a list of dicts with keys:
        match_id, matchinfo_path, events_path, positions_path
    """
    data_dir = Path(data_dir)
    files = sorted(os.listdir(data_dir))

    # Extract unique match IDs from filenames
    match_ids = set()
    for f in files:
        m = re.search(r"(DFL-MAT-\w+)", f)
        if m:
            match_ids.add(m.group(1))

    matches = []
    for mid in sorted(match_ids):
        matchinfo = [f for f in files if "matchinformation" in f and mid in f]
        events = [f for f in files if "events_raw" in f and mid in f]
        positions = [f for f in files if "positions_raw" in f and mid in f]

        if matchinfo and events and positions:
            matches.append({
                "match_id": mid,
                "matchinfo_path": data_dir / matchinfo[0],
                "events_path": data_dir / events[0],
                "positions_path": data_dir / positions[0],
            })
        else:
            logger.warning(f"Incomplete files for match {mid}, skipping")

    logger.info(f"Discovered {len(matches)} matches in {data_dir}")
    return matches


def parse_goals(events_path: str | Path) -> list[GoalEvent]:
    """Parse goal events from a DFL events XML file.

    Goals are identified by <ShotAtGoal> elements containing a <SuccessfulShot>
    child element.
    """
    goals = []
    tree = ET.parse(events_path)
    root = tree.getroot()

    for event_elem in root.iter("Event"):
        shot = event_elem.find("ShotAtGoal")
        if shot is None:
            continue

        success = shot.find("SuccessfulShot")
        if success is None:
            continue

        event_time_str = event_elem.get("EventTime", "")
        team_id = shot.get("Team", "")
        player_id = shot.get("Player", "")
        result = success.get("CurrentResult", "")

        try:
            event_time = datetime.fromisoformat(event_time_str)
        except ValueError:
            logger.warning(f"Could not parse EventTime: {event_time_str}")
            continue

        goals.append(GoalEvent(
            event_time=event_time,
            team_id=team_id,
            player_id=player_id,
            result=result,
        ))

    logger.info(f"Found {len(goals)} goals in {Path(events_path).name}")
    return goals


_SHOT_OUTCOME_MAP = {
    "SuccessfulShot": "goal",
    "SavedShot": "saved",
    "ShotWide": "wide",
    "ShotBlocked": "blocked",
    "ShotWoodWork": "woodwork",
}


def parse_shots(events_path: str | Path) -> list[ShotEvent]:
    """Parse all shot events from a DFL events XML file.

    Every <ShotAtGoal> element is captured regardless of outcome.
    The xG value is read directly from the element's xG attribute.
    """
    shots = []
    tree = ET.parse(events_path)
    root = tree.getroot()

    for event_elem in root.iter("Event"):
        shot = event_elem.find("ShotAtGoal")
        if shot is None:
            continue

        event_time_str = event_elem.get("EventTime", "")
        try:
            event_time = datetime.fromisoformat(event_time_str)
        except ValueError:
            logger.warning(f"Could not parse EventTime: {event_time_str}")
            continue

        # Determine outcome
        outcome = "unknown"
        for child_tag, outcome_name in _SHOT_OUTCOME_MAP.items():
            if shot.find(child_tag) is not None:
                outcome = outcome_name
                break

        xg_str = shot.get("xG", "0.0")
        try:
            xg = float(xg_str)
        except ValueError:
            xg = 0.0

        shots.append(ShotEvent(
            event_time=event_time,
            team_id=shot.get("Team", ""),
            player_id=shot.get("Player", ""),
            xg=xg,
            outcome=outcome,
        ))

    logger.info(f"Found {len(shots)} shots in {Path(events_path).name}")
    return shots


def parse_match_start_time(events_path: str | Path) -> datetime | None:
    """Get the timestamp of the first event (kickoff) in a match."""
    tree = ET.parse(events_path)
    root = tree.getroot()

    for event_elem in root.iter("Event"):
        if event_elem.find("KickOff") is not None:
            time_str = event_elem.get("EventTime", "")
            try:
                return datetime.fromisoformat(time_str)
            except ValueError:
                continue
    return None
