"""Parse DFL XML tracking and emit per-frame shape-graph annotations."""

import io
import json
import os
from collections import defaultdict
from xml.etree import ElementTree as ET

import numpy as np
from tqdm import tqdm

from .shape_graph import infer_joint_zone, infer_team_roles


def find_xmls(data_dir, match_id):
    info, pos = None, None
    for name in os.listdir(data_dir):
        if match_id not in name or not name.endswith(".xml"):
            continue
        if "matchinformation" in name:
            info = os.path.join(data_dir, name)
        elif "positions_raw" in name:
            pos = os.path.join(data_dir, name)
    if info is None or pos is None:
        raise FileNotFoundError(f"match info / positions missing for {match_id}")
    return info, pos


def _read_xml_root(path):
    """Read an XML file robustly: strip a possible UTF-8 BOM and leading
    whitespace before the XML declaration (seen in some DFL dumps)."""
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    data = data.lstrip()
    return ET.fromstring(data)


def parse_match_info(path):
    """Parse the match information XML.

    Returns (players, teams) where
        players: pid -> {team, team_id, gk, shirt, name, first_name, last_name}
        teams:   role ('home' | 'guest') -> {team_id, name}
    """
    root = _read_xml_root(path)
    players = {}
    teams = {}
    for team in root.iter("Team"):
        role = team.attrib["Role"]
        tid = team.attrib["TeamId"]
        teams[role] = {"team_id": tid, "name": team.attrib.get("TeamName", role)}
        for p in team.iter("Player"):
            first = p.attrib.get("FirstName", "").strip()
            last = p.attrib.get("LastName", "").strip()
            short = p.attrib.get("Shortname", "").strip()
            players[p.attrib["PersonId"]] = {
                "team": role,
                "team_id": tid,
                "gk": p.attrib.get("PlayingPosition") == "TW",
                "shirt": int(p.attrib.get("ShirtNumber", -1)),
                "first_name": first,
                "last_name": last,
                "name": short or f"{first} {last}".strip() or p.attrib["PersonId"],
            }
    return players, teams


def iter_frames(positions_path):
    """Yield (game_section, team_id, person_id, frame_rows) tuples.

    Each frame_row: dict with N, X, Y (at least).
    """
    # iterparse rejects leading BOM/whitespace before the XML declaration;
    # feed it a cleaned in-memory stream instead.
    with open(positions_path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    raw = raw.lstrip()
    context = ET.iterparse(io.BytesIO(raw), events=("start", "end"))
    game_section = None
    team_id = None
    person_id = None
    rows = []
    for event, elem in context:
        if event == "start" and elem.tag == "FrameSet":
            game_section = elem.attrib.get("GameSection")
            team_id = elem.attrib.get("TeamId")
            person_id = elem.attrib.get("PersonId")
            rows = []
        elif event == "end" and elem.tag == "Frame":
            rows.append({
                "N": int(elem.attrib["N"]),
                "X": float(elem.attrib["X"]),
                "Y": float(elem.attrib["Y"]),
                "T": elem.attrib.get("T"),
                "M": int(elem.attrib.get("M", 1)),
                "BallStatus": elem.attrib.get("BallStatus"),
                "BallPossession": elem.attrib.get("BallPossession"),
            })
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            yield game_section, team_id, person_id, rows
            elem.clear()


def pivot_to_frames(positions_path, players):
    """Build dict: frame_N -> {"section":..., "ts":..., "points": {pid: (x,y,team_role)}}."""
    frames = {}
    for section, team_id, pid, rows in iter_frames(positions_path):
        if team_id == "referee":
            continue
        is_ball = team_id == "BALL"
        for r in rows:
            if not r["M"]:
                continue
            key = (section, r["N"])
            fr = frames.get(key)
            if fr is None:
                fr = {"section": section, "ts": r["T"], "points": {}, "ball_status": None}
                frames[key] = fr
            if is_ball:
                fr["points"]["BALL"] = (r["X"], r["Y"], None)
                fr["ball_status"] = r.get("BallStatus")
            else:
                meta = players.get(pid)
                if meta is None:
                    continue
                fr["points"][pid] = (r["X"], r["Y"], meta["team"])
    return frames


def attacking_sign(frames, team_role):
    """For each (section, team_role), infer the sign s so that attacking is +s*x.

    Uses the GK's mean x across the first N frames of each half: a GK defending
    the negative-x goal → team attacks toward +x (sign = +1), else -1.
    """
    gk_x = defaultdict(list)
    for (section, _), fr in frames.items():
        for pid, (x, _y, _role) in fr["points"].items():
            meta = team_role.get(pid)
            if meta and meta["gk"]:
                gk_x[(section, meta["team"])].append(x)
    sign = {}
    for key, xs in gk_x.items():
        mean_x = float(np.mean(xs))
        sign[key] = 1 if mean_x < 0 else -1
    return sign


def rotate_for_team(points_xy, sign):
    """Rotate raw (x, y) so the team's attacking direction is +y.

    DFL: x = pitch length, y = width. We map rotated_y = sign*x, rotated_x = sign*y
    (preserves orientation).
    """
    xy = np.asarray(points_xy, dtype=float)
    out = np.empty_like(xy)
    out[:, 0] = sign * xy[:, 1]
    out[:, 1] = sign * xy[:, 0]
    return out


def annotate_match(data_dir, match_id, out_path, frame_stride=1, max_frames=None,
                   zone_levels=5):
    info_path, pos_path = find_xmls(data_dir, match_id)
    print(f"[{match_id}] parsing match info...")
    players, _ = parse_match_info(info_path)
    print(f"[{match_id}] parsing positions XML (this is the slow part)...")
    frames = pivot_to_frames(pos_path, players)
    print(f"[{match_id}] {len(frames)} frames loaded; inferring attacking direction...")
    sign = attacking_sign(frames, players)

    frame_keys = sorted(frames.keys(), key=lambda k: (0 if k[0] == "firstHalf" else 1, k[1]))
    if frame_stride > 1:
        frame_keys = frame_keys[::frame_stride]
    if max_frames is not None:
        frame_keys = frame_keys[:max_frames]

    out = []
    for section, frame_n in tqdm(frame_keys, desc=f"[{match_id}] annotate",
                                  unit="frame", dynamic_ncols=True):
        fr = frames[(section, frame_n)]
        ids, coords, teams, is_gk = [], [], [], []
        for pid, (x, y, role) in fr["points"].items():
            ids.append(pid)
            coords.append((x, y))
            teams.append(role)
            is_gk.append(bool(pid != "BALL" and players.get(pid, {}).get("gk")))
        coords = np.array(coords, dtype=float)
        if len(coords) == 0:
            continue

        # Build the joint shape graph on outfield players + ball only; GKs
        # would otherwise pin the outer thresholds to the goal lines.
        include_mask = [not gk for gk in is_gk]
        zones = infer_joint_zone(coords, include_mask=include_mask,
                                 n_levels=zone_levels)
        roles_out = {}
        for team in ("home", "guest"):
            idx = [i for i, t in enumerate(teams) if t == team]
            if len(idx) < 5:
                for i in idx:
                    roles_out[ids[i]] = "UNKNOWN"
                continue
            rot = rotate_for_team(coords[idx], sign.get((section, team), 1))
            gk_local = None
            for local_i, orig in enumerate(idx):
                if players.get(ids[orig], {}).get("gk"):
                    gk_local = local_i; break
            roles = infer_team_roles(rot, gk_local)
            for local_i, orig in enumerate(idx):
                roles_out[ids[orig]] = roles[local_i]

        players_out = []
        for i, pid in enumerate(ids):
            xy = [float(coords[i, 0]), float(coords[i, 1])]
            if pid == "BALL":
                players_out.append({
                    "player_id": "BALL", "team": None,
                    "tactical_role": "BALL", "zone": zones[i],
                    "pitch_xy": xy,
                })
            else:
                players_out.append({
                    "player_id": pid, "team": teams[i],
                    "tactical_role": roles_out.get(pid, "UNKNOWN"),
                    "zone": zones[i], "pitch_xy": xy,
                })

        out.append({
            "frame_id": frame_n,
            "timestamp": fr["ts"],
            "phase": section,
            "attack_sign": {
                "home": int(sign.get((section, "home"), 1)),
                "guest": int(sign.get((section, "guest"), 1)),
            },
            "players": players_out,
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    return out_path, len(out)
