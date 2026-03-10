from typing import Optional, Tuple
import xml.etree.ElementTree as ET

def parse_match_info(path: str) -> dict:
    """
    Parse the match information XML file.

    Returns a dict with keys:
      - "home_team_id", "away_team_id"
      - "home_team_name", "away_team_name"
      - "pitch_x", "pitch_y"  (pitch dimensions in metres)
      - "players": {person_id: {"team": "home"|"away",
                                "position": str,   # e.g. "IVL", "STZ", "TW"
                                "name": str,
                                "shirt": int,
                                "starting": bool}}
    """
    tree = ET.parse(path)
    root = tree.getroot()
    
    match_info_el = root.find("MatchInformation")
    if match_info_el is not None:
        root = match_info_el

    info: dict = {"players": {}}

    gen = root.find("General")
    if gen is not None:
        info["home_team_id"] = gen.get("HomeTeamId", "")
        info["away_team_id"] = gen.get("GuestTeamId", "")
        info["home_team_name"] = gen.get("HomeTeamName", "")
        info["away_team_name"] = gen.get("GuestTeamName", "")

    env = root.find("Environment")
    if env is not None:
        info["pitch_x"] = float(env.get("PitchX", "105.0"))
        info["pitch_y"] = float(env.get("PitchY", "68.0"))
    else:
        info["pitch_x"] = 105.0
        info["pitch_y"] = 68.0

    teams_el = root.find("Teams")

    for team_el in teams_el.findall("Team"):
        role = team_el.get("Role", "").lower()
        team_label = "home" if role == "home" else "away"

        players_el = team_el.find("Players")
        if players_el is None:
            continue
        for p in players_el.findall("Player"):
            pid = p.get("PersonId", "")
            info["players"][pid] = {
                "team": team_label,
                "position": p.get("PlayingPosition", ""),
                "name": f'{p.get("FirstName", "")} {p.get("LastName", "")}',
                "shirt": int(p.get("ShirtNumber", "0")),
                "starting": p.get("Starting", "false").lower() == "true",
            }

    return info


def parse_position_data(
    path: str,
    match_info: dict,
    frame_range: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Parse the position data XML (streaming parser for memory efficiency).

    Returns a dict:
      {
        frame_number (int): {
            "players": {
                person_id: {"x": float, "y": float,
                             "team": "home"|"away",
                             "speed": float}
            },
            "ball": {"x": float, "y": float, "z": float,
                     "possession": int,   # 1=home, 2=away
                     "status": int},      # 0=inactive, 1=active
            "timestamp": str
        }
      }
    """

    frames: dict = {}

    home_tid = match_info.get("home_team_id", "")
    away_tid = match_info.get("away_team_id", "")

    context = ET.iterparse(path, events=("start", "end"))
    current_person_id = None
    current_team_id = None
    in_ball = False

    for event, elem in context:
        if event == "start" and elem.tag == "FrameSet":
            current_person_id = elem.get("PersonId", "")
            current_team_id = elem.get("TeamId", "")
            in_ball = current_person_id.upper() == "BALL" or "BALL" in elem.get("TeamId", "").upper()

        elif event == "end" and elem.tag == "Frame":
            n = int(elem.get("N", "0"))

            if frame_range is not None:
                if n < frame_range[0] or n >= frame_range[1]:
                    elem.clear()
                    continue

            x = float(elem.get("X", "0"))
            y = float(elem.get("Y", "0"))
            t = elem.get("T", "")

            if n not in frames:
                frames[n] = {"players": {}, "ball": None, "timestamp": t}

            if in_ball:
                z = float(elem.get("Z", "0"))
                bp = int(elem.get("BallPossession", "0"))
                bs = int(elem.get("BallStatus", "0"))
                frames[n]["ball"] = {
                    "x": x, "y": y, "z": z,
                    "possession": bp, "status": bs,
                }
            else:
                if current_team_id == home_tid:
                    team_label = "home"
                elif current_team_id == away_tid:
                    team_label = "away"
                else:
                    pinfo = match_info["players"].get(current_person_id, {})
                    team_label = pinfo.get("team", "unknown")

                s = float(elem.get("S", "0"))
                frames[n]["players"][current_person_id] = {
                    "x": x, "y": y, "team": team_label, "speed": s,
                }

            if not frames[n]["timestamp"]:
                frames[n]["timestamp"] = t

            elem.clear()

        elif event == "end" and elem.tag == "FrameSet":
            current_person_id = None
            current_team_id = None
            in_ball = False
            elem.clear()

    return frames
