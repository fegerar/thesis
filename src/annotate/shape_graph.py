"""Brandes et al. 2025 shape graphs, role inference, and joint zone labeling.

Construction follows the skill spec: Delaunay triangulation, iterative removal
of the least angularly-stable interior edge (threshold alpha = pi/4), with
faces merged into polygons whose centroids replace triangle circumcenters.
"""

import math

import numpy as np
from scipy.spatial import Delaunay

ALPHA = math.pi / 4.0

ROLE_MATRIX = [
    ["LB", "LCB", "CB",  "RCB", "RB"],   # depth 0 — Defender
    ["LB", "LDM", "CDM", "RDM", "RB"],   # depth 1 — DefMid
    ["LM", "LCM", "CM",  "RCM", "RM"],   # depth 2 — Midfield
    ["LF", "LAM", "CAM", "RAM", "RF"],   # depth 3 — AttMid
    ["LF", "LCF", "CF",  "RCF", "RF"],   # depth 4 — Forward
]


def _polygon_centroid(pts):
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a2 = cross.sum()
    if abs(a2) < 1e-9:
        return pts.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (3.0 * a2)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (3.0 * a2)
    return np.array([cx, cy])


def _angle_at(p, cc1, cc2):
    v1, v2 = cc1 - p, cc2 - p
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-12:
        return math.pi
    return math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / n))))


def _edges(face):
    n = len(face)
    return [(face[i], face[(i + 1) % n]) for i in range(n)]


def _key(a, b):
    return (a, b) if a < b else (b, a)


def _merge(f1, f2, a, b):
    # Rotate each face so its boundary ends at the shared edge, then concat.
    def rot(face, x, y):
        n = len(face)
        for i in range(n):
            if face[i] == x and face[(i + 1) % n] == y:
                j = (i + 1) % n
                return face[j:] + face[:j]
        return None
    r1, r2 = rot(f1, a, b), rot(f2, b, a)
    if r1 is None or r2 is None:
        seen, out = set(), []
        for v in f1 + f2:
            if v not in seen:
                out.append(v); seen.add(v)
        return out
    return r1[:-2] + r2[:-2]


def build_shape_graph(points):
    """Return (faces, face_centroids) after iterative edge removal."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return [list(range(n))], pts.mean(axis=0, keepdims=True) if n else np.empty((0, 2))

    tri = Delaunay(pts)
    faces = {}
    for i, t in enumerate(tri.simplices):
        tp = pts[t]
        ccw = (tp[1, 0] - tp[0, 0]) * (tp[2, 1] - tp[0, 1]) \
            - (tp[2, 0] - tp[0, 0]) * (tp[1, 1] - tp[0, 1])
        faces[i] = list(t) if ccw > 0 else list(t[::-1])

    next_id = len(faces)
    edge_faces = {}
    for fid, f in faces.items():
        for a, b in _edges(f):
            edge_faces.setdefault(_key(a, b), []).append(fid)
    cents = {fid: _polygon_centroid(pts[f]) for fid, f in faces.items()}

    blacklist = set()
    while True:
        best, best_s = None, math.inf
        for e, fids in edge_faces.items():
            if e in blacklist or len(fids) != 2 or fids[0] == fids[1]:
                continue
            a, b = e
            s = _angle_at(pts[a], cents[fids[0]], cents[fids[1]])
            if s < best_s:
                best_s, best = s, e
        if best is None or best_s >= ALPHA:
            break

        fid1, fid2 = edge_faces[best]
        a, b = best
        merged = _merge(faces[fid1], faces[fid2], a, b)
        # reject degenerate merges (two faces that share multiple edges would
        # produce a figure-eight / duplicate vertices); skip and move on.
        if len(merged) < 3 or len(set(merged)) != len(merged):
            blacklist.add(best)
            continue
        new_id = next_id; next_id += 1

        for oldfid in (fid1, fid2):
            for x, y in _edges(faces[oldfid]):
                lst = edge_faces.get(_key(x, y), [])
                if oldfid in lst:
                    lst.remove(oldfid)
                if not lst:
                    edge_faces.pop(_key(x, y), None)
        faces.pop(fid1); faces.pop(fid2)
        cents.pop(fid1, None); cents.pop(fid2, None)
        faces[new_id] = merged
        cents[new_id] = _polygon_centroid(pts[merged])
        for x, y in _edges(merged):
            edge_faces.setdefault(_key(x, y), []).append(new_id)

    ordered = list(faces.values())
    centroids = np.array([cents[fid] for fid in faces]) if faces else np.empty((0, 2))
    return ordered, centroids


def _split_levels(values, faces, face_proj, node_index=None, n_levels=5):
    """Generalized shape-graph split into `n_levels` bands along one axis.

    At each pass the "inner" set shrinks: face-centroid projections of faces
    whose vertices are all currently inner give the new (lo, hi) thresholds;
    currently-inner points below lo / above hi get peeled off to the outer
    band for this pass.  Repeating (n_levels-1)//2 times yields `n_levels`
    bands (must be odd).
    """
    if n_levels < 3 or n_levels % 2 == 0:
        raise ValueError("n_levels must be an odd integer >= 3")
    n = len(values)
    center = (n_levels - 1) // 2
    lv = np.full(n, center, dtype=int)
    if n < 4 or len(face_proj) == 0:
        return lv

    local_to_vertex = np.arange(n) if node_index is None else np.asarray(node_index)
    current = np.ones(n, dtype=bool)

    for i in range(center):
        if current.sum() < 4:
            break
        inner_vertices = set(local_to_vertex[current].tolist())
        proj = [fp for f, fp in zip(faces, face_proj)
                if f and all(v in inner_vertices for v in f)]
        if not proj:
            break
        lo, hi = float(min(proj)), float(max(proj))
        below = current & (values < lo)
        above = current & (values > hi)
        lv[below] = i
        lv[above] = n_levels - 1 - i
        current &= (values >= lo) & (values <= hi)
    return lv


# Back-compat alias: the role inference is hard-wired to a 5-level split
# because ROLE_MATRIX is 5x5.
def _split5(values, faces, face_proj, node_index=None):
    return _split_levels(values, faces, face_proj, node_index, n_levels=5)


def infer_team_positions(points, gk_index):
    """Per-team role + depth/width level per player (team-attacking-upward coords).

    Returns list of dicts: {"role", "depth", "width"}.  Depth/width are ints
    in 0..4; GKs and under-filled teams get depth=width=-1.
    """
    n = len(points)
    out = [{"role": "UNKNOWN", "depth": -1, "width": -1} for _ in range(n)]
    if gk_index is not None and 0 <= gk_index < n:
        out[gk_index] = {"role": "GK", "depth": -1, "width": -1}
    outfield = [i for i in range(n) if i != gk_index]
    if len(outfield) < 5:
        return out

    sub = points[outfield]
    faces, cc = build_shape_graph(sub)
    depth = _split5(sub[:, 1], faces, cc[:, 1] if len(cc) else np.empty(0))
    width = _split5(sub[:, 0], faces, cc[:, 0] if len(cc) else np.empty(0))
    for j, orig in enumerate(outfield):
        d, w = int(depth[j]), int(width[j])
        out[orig] = {"role": ROLE_MATRIX[d][w], "depth": d, "width": w}
    return out


def infer_team_roles(points, gk_index):
    """Backward-compatible: returns only the role string per player."""
    return [p["role"] for p in infer_team_positions(points, gk_index)]


def infer_joint_zone(points, include_mask=None, n_levels=5):
    """Return [[col, row], ...] on an `n_levels` x `n_levels` grid.

    [0, 0] is the top-left cell (high y, low x).  `n_levels` must be odd and
    >= 3; higher values give a finer grid via extra shape-graph split passes.

    The shape graph (and thus the grid thresholds) is built only on points
    where `include_mask` is True — use this to exclude goalkeepers, who
    otherwise dominate the outer thresholds.  Zones are still assigned to
    every point.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return []
    if include_mask is None:
        sg_pts = pts
        sg_local_for_pts = np.arange(len(pts))
    else:
        mask = np.asarray(include_mask, dtype=bool)
        if not mask.any():
            sg_pts = pts
            sg_local_for_pts = np.arange(len(pts))
        else:
            sg_pts = pts[mask]
            # project each point onto the nearest shape-graph vertex so the
            # "inner vertices" test in _split5 can include excluded points
            # consistently (GKs stay with their nearest outfield vertex).
            idx_map = np.full(len(pts), -1)
            idx_map[mask] = np.arange(mask.sum())
            # for excluded points, pick the nearest included vertex
            excluded = np.where(~mask)[0]
            if len(excluded):
                diff = pts[excluded, None, :] - sg_pts[None, :, :]
                nearest = np.argmin((diff ** 2).sum(-1), axis=1)
                idx_map[excluded] = nearest
            sg_local_for_pts = idx_map

    faces, cc = build_shape_graph(sg_pts)
    col = _split_levels(pts[:, 0], faces,
                        cc[:, 0] if len(cc) else np.empty(0),
                        node_index=sg_local_for_pts, n_levels=n_levels)
    depth_y = _split_levels(pts[:, 1], faces,
                            cc[:, 1] if len(cc) else np.empty(0),
                            node_index=sg_local_for_pts, n_levels=n_levels)
    row = (n_levels - 1) - depth_y
    return [[int(c), int(r)] for c, r in zip(col, row)]
