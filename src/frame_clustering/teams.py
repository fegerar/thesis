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


def run_team_pipeline(args):
    os.makedirs(args.output_dir, exist_ok=True)

    _stamp(f"loading {args.input}")
    frames = subsample(load_frames(args.input),
                       stride=args.stride, max_frames=args.max_frames)
    if not frames:
        raise ValueError("no frames after stride/max-frames filtering")

    zone_side = args.zone_levels or infer_zone_side(frames)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _stamp(f"device={device}  frames={len(frames)}  zone_side={zone_side}")

    mats = build_all_matrices(frames, zone_side)
    role = torch.cat([mats["role_home"], mats["role_guest"]], dim=0)
    zone = torch.cat([mats["zone_home"], mats["zone_guest"]], dim=0)
    n = role.shape[0]
    _stamp(f"team-frames={n} (home={mats['role_home'].shape[0]} + "
           f"guest={mats['role_guest'].shape[0]})")
    torch.save(role, os.path.join(args.output_dir, "role.pt"))
    torch.save(zone, os.path.join(args.output_dir, "zone.pt"))

    dists = {}
    for key, h in (("role", role), ("zone", zone)):
        dists[key] = pairwise_hellinger(
            h.to(device), sigma=args.smooth_sigma, row_batch=args.row_batch,
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

    n_home = mats["role_home"].shape[0]
    samples = []
    for i, fr in enumerate(frames):
        samples.append({
            "team": "home", "frame_idx": i,
            "frame_id": fr.get("frame_id"), "phase": fr.get("phase"),
            "cluster": int(labels_by_k[best_k][i]),
        })
    for i, fr in enumerate(frames):
        samples.append({
            "team": "guest", "frame_idx": i,
            "frame_id": fr.get("frame_id"), "phase": fr.get("phase"),
            "cluster": int(labels_by_k[best_k][n_home + i]),
        })

    result = {
        "input": args.input,
        "n_team_frames": n,
        "zone_side": zone_side,
        "device": str(device),
        "distance": {
            "kind": "smoothed_hellinger_team",
            "smooth_sigma": args.smooth_sigma,
            "weights": {"role": float(w[0]), "zone": float(w[1])},
        },
        "elbow": {"k_values": ks, "inertias": inertias, "selected_k": best_k},
        "samples": samples,
    }
    with open(os.path.join(args.output_dir, "team_clusters.json"), "w") as f:
        json.dump(result, f, indent=2)
    _stamp(f"saved tensors, distance matrix, and team_clusters.json to "
           f"{args.output_dir}")
