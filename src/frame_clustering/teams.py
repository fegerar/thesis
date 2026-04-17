"""Team-level frame clustering: stack each frame's home + guest as separate
samples and cluster on the role + zone histograms only (no ball).

Goal: surface team-level tactical states ("build from back", "low block",
"high press", ...) without conditioning on which side of the pitch the team
defends or which match they belong to.
"""

import json
import os

import torch
from tqdm import tqdm

from .clustering import elbow_k, kmedoids
from .hellinger import pairwise_hellinger
from .matrices import (
    build_all_matrices, infer_zone_side, load_frames, subsample,
)


def _stamp(msg):
    print(f"[team-cluster] {msg}")


def _cluster_subset(tag, role, zone, indices, frames, args, device, zone_side):
    """Run Hellinger + k-medoids on a subset of team-frames and save results."""
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    role_sub = role[indices]
    zone_sub = zone[indices]
    n = len(indices)
    _stamp(f"[{tag}] {n} team-frames")

    torch.save(role_sub, os.path.join(out_dir, "role.pt"))
    torch.save(zone_sub, os.path.join(out_dir, "zone.pt"))

    sigmas = {"role": args.smooth_sigma_role, "zone": args.smooth_sigma_zone}
    dists = {}
    for key, h in (("role", role_sub), ("zone", zone_sub)):
        dists[key] = pairwise_hellinger(
            h.to(device), sigma=sigmas[key], row_batch=args.row_batch,
            desc=f"hellinger {key} ({tag})",
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    w = torch.tensor([args.weight_role, args.weight_zone], dtype=torch.float32)
    w = w / w.sum()
    for name, d in dists.items():
        _stamp(f"  [{tag}] {name}: mean={float(d.mean()):.4g} "
               f"max={float(d.max()):.4g}")
    d_role = dists["role"] / dists["role"].mean().clamp_min(1e-9)
    d_zone = dists["zone"] / dists["zone"].mean().clamp_min(1e-9)
    d_total = w[0] * d_role + w[1] * d_zone
    del dists, d_role, d_zone

    torch.save(d_total, os.path.join(out_dir, "distance_matrix.pt"))

    k_max = min(args.k_max, d_total.shape[0] - 1)
    k_min = min(args.k_min, k_max)
    ks = list(range(k_min, k_max + 1))

    _stamp(f"[{tag}] distance matrix: shape={tuple(d_total.shape)} "
           f"mean={float(d_total.mean()):.4g} std={float(d_total.std()):.4g}")

    inertias, labels_by_k = [], {}
    for k in tqdm(ks, desc=f"k-medoids ({tag})", unit="k", dynamic_ncols=True):
        labels, _medoids, inertia = kmedoids(
            d_total, k=k, max_iter=args.kmedoids_iters, seed=args.seed,
        )
        inertias.append(inertia)
        labels_by_k[k] = labels.detach().cpu().tolist()
        counts = torch.bincount(labels, minlength=k).tolist()
        _stamp(f"  [{tag}] k={k}: inertia={inertia:.4f}  sizes={counts}")

    best_k = elbow_k(ks, inertias)
    _stamp(f"[{tag}] elbow-selected k={best_k}")

    n_frames = len(frames)
    samples = []
    for local_j, global_idx in enumerate(indices):
        if global_idx < n_frames:
            team, frame_i = "home", global_idx
        else:
            team, frame_i = "guest", global_idx - n_frames
        fr = frames[frame_i]
        samples.append({
            "team": team, "frame_idx": frame_i,
            "match_id": fr.get("match_id"),
            "frame_id": fr.get("frame_id"), "phase": fr.get("phase"),
            "possession": tag,
            "cluster": int(labels_by_k[best_k][local_j]),
        })

    zone_mode = getattr(args, "zone_mode", "shape_graph")
    result = {
        "input": args.input,
        "possession": tag,
        "n_team_frames": n,
        "zone_side": zone_side,
        "device": str(device),
        "distance": {
            "kind": "smoothed_hellinger_team",
            "smooth_sigma_role": args.smooth_sigma_role,
            "smooth_sigma_zone": args.smooth_sigma_zone,
            "zone_oriented": "attacking=+col",
            "zone_mode": zone_mode,
            "weights": {"role": float(w[0]), "zone": float(w[1])},
        },
        "elbow": {"k_values": ks, "inertias": inertias, "selected_k": best_k},
        "samples": samples,
    }
    with open(os.path.join(out_dir, "team_clusters.json"), "w") as f:
        json.dump(result, f, indent=2)
    _stamp(f"[{tag}] saved to {out_dir}")


def run_team_pipeline(args):
    os.makedirs(args.output_dir, exist_ok=True)

    _stamp(f"loading {args.input}")
    frames = subsample(load_frames(args.input),
                       stride=args.stride, max_frames=args.max_frames)
    if not frames:
        raise ValueError("no frames after stride/max-frames filtering")

    zone_mode = getattr(args, "zone_mode", "shape_graph")
    if zone_mode == "pitch":
        if not args.zone_levels:
            raise ValueError("--zone-levels is required when --zone-mode pitch")
        zone_side = args.zone_levels
    else:
        zone_side = args.zone_levels or infer_zone_side(frames)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _stamp(f"device={device}  frames={len(frames)}  zone_side={zone_side}  "
           f"zone_mode={zone_mode}")

    mats = build_all_matrices(frames, zone_side, zone_mode=zone_mode)
    role = torch.cat([mats["role_home"], mats["role_guest"]], dim=0)
    zone = torch.cat([mats["zone_home"], mats["zone_guest"]], dim=0)
    n = role.shape[0]
    n_frames = len(frames)
    _stamp(f"team-frames={n} (home={mats['role_home'].shape[0]} + "
           f"guest={mats['role_guest'].shape[0]})")

    split_possession = getattr(args, "split_possession", False)

    if split_possession:
        ip_idx, oop_idx = [], []
        n_no_poss = 0
        for i, fr in enumerate(frames):
            poss = fr.get("possession")
            if poss is None:
                n_no_poss += 1
                continue
            # home team-frame is at index i, guest at n_frames + i
            if poss == "home":
                ip_idx.append(i)
                oop_idx.append(n_frames + i)
            else:
                oop_idx.append(i)
                ip_idx.append(n_frames + i)
        if n_no_poss:
            _stamp(f"skipped {n_no_poss} frames with no possession info")
        _stamp(f"in_possession={len(ip_idx)}, out_of_possession={len(oop_idx)}")

        _cluster_subset("in_possession", role, zone, ip_idx, frames,
                        args, device, zone_side)
        _cluster_subset("out_of_possession", role, zone, oop_idx, frames,
                        args, device, zone_side)
        _stamp(f"done — results in {args.output_dir}/in_possession/ and "
               f"{args.output_dir}/out_of_possession/")
        return

    # --- no possession split: original single-clustering path ---
    torch.save(role, os.path.join(args.output_dir, "role.pt"))
    torch.save(zone, os.path.join(args.output_dir, "zone.pt"))

    sigmas = {"role": args.smooth_sigma_role, "zone": args.smooth_sigma_zone}
    dists = {}
    for key, h in (("role", role), ("zone", zone)):
        dists[key] = pairwise_hellinger(
            h.to(device), sigma=sigmas[key], row_batch=args.row_batch,
            desc=f"hellinger {key}",
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    w = torch.tensor([args.weight_role, args.weight_zone], dtype=torch.float32)
    w = w / w.sum()
    for name, d in dists.items():
        _stamp(f"  {name}: mean={float(d.mean()):.4g} max={float(d.max()):.4g}")
    d_role = dists["role"] / dists["role"].mean().clamp_min(1e-9)
    d_zone = dists["zone"] / dists["zone"].mean().clamp_min(1e-9)
    d_total = w[0] * d_role + w[1] * d_zone
    del dists, d_role, d_zone

    torch.save(d_total, os.path.join(args.output_dir, "distance_matrix.pt"))

    k_max = min(args.k_max, d_total.shape[0] - 1)
    k_min = min(args.k_min, k_max)
    ks = list(range(k_min, k_max + 1))

    _stamp(
        f"distance matrix stats: shape={tuple(d_total.shape)} "
        f"min={float(d_total.min()):.4g} max={float(d_total.max()):.4g} "
        f"mean={float(d_total.mean()):.4g} std={float(d_total.std()):.4g}"
    )

    _stamp("running k-medoids elbow sweep (on CPU)")
    inertias, labels_by_k = [], {}
    for k in tqdm(ks, desc="k-medoids", unit="k", dynamic_ncols=True):
        labels, _medoids, inertia = kmedoids(
            d_total, k=k, max_iter=args.kmedoids_iters, seed=args.seed,
        )
        inertias.append(inertia)
        labels_by_k[k] = labels.detach().cpu().tolist()
        counts = torch.bincount(labels, minlength=k).tolist()
        _stamp(f"  k={k}: inertia={inertia:.4f}  cluster sizes={counts}")

    best_k = elbow_k(ks, inertias)
    _stamp(f"elbow-selected k={best_k}")

    samples = []
    for i, fr in enumerate(frames):
        samples.append({
            "team": "home", "frame_idx": i,
            "match_id": fr.get("match_id"),
            "frame_id": fr.get("frame_id"), "phase": fr.get("phase"),
            "cluster": int(labels_by_k[best_k][i]),
        })
    for i, fr in enumerate(frames):
        samples.append({
            "team": "guest", "frame_idx": i,
            "match_id": fr.get("match_id"),
            "frame_id": fr.get("frame_id"), "phase": fr.get("phase"),
            "cluster": int(labels_by_k[best_k][n_frames + i]),
        })

    result = {
        "input": args.input,
        "n_team_frames": n,
        "zone_side": zone_side,
        "device": str(device),
        "distance": {
            "kind": "smoothed_hellinger_team",
            "smooth_sigma_role": args.smooth_sigma_role,
            "smooth_sigma_zone": args.smooth_sigma_zone,
            "zone_oriented": "attacking=+col",
            "zone_mode": zone_mode,
            "weights": {"role": float(w[0]), "zone": float(w[1])},
        },
        "elbow": {"k_values": ks, "inertias": inertias, "selected_k": best_k},
        "samples": samples,
    }
    with open(os.path.join(args.output_dir, "team_clusters.json"), "w") as f:
        json.dump(result, f, indent=2)
    _stamp(f"saved tensors, distance matrix, and team_clusters.json to "
           f"{args.output_dir}")
