"""End-to-end frame clustering pipeline (smoothed Hellinger + k-medoids)."""

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
    print(f"[cluster] {msg}")


def run_pipeline(args):
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
    for key, value in mats.items():
        torch.save(value, os.path.join(args.output_dir, f"{key}.pt"))

    # per-block Hellinger distance matrices (on CPU).
    dists = {}
    for key in ("role_home", "role_guest", "zone_home", "zone_guest", "zone_ball"):
        h = mats[key].to(device)
        dists[key] = pairwise_hellinger(
            h, sigma=args.smooth_sigma, row_batch=args.row_batch,
            desc=f"hellinger {key}",
        )
        del h
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # weighted combination across blocks. Normalize each block by its mean
    # first so the weights are on a comparable scale — otherwise the sparse
    # ball histogram (mass 1) has systematically larger Hellinger values
    # than the team histograms (mass ~10) and dominates the total.
    w = torch.tensor(
        [args.weight_role, args.weight_zone, args.weight_ball],
        dtype=torch.float32,
    )
    w = w / w.sum()
    d_role = (dists["role_home"] + dists["role_guest"]) / 2.0
    d_zone = (dists["zone_home"] + dists["zone_guest"]) / 2.0
    d_ball = dists["zone_ball"]
    for name, d in (("role", d_role), ("zone", d_zone), ("ball", d_ball)):
        _stamp(f"  {name}: mean={float(d.mean()):.4g} max={float(d.max()):.4g}")
    d_role = d_role / d_role.mean().clamp_min(1e-9)
    d_zone = d_zone / d_zone.mean().clamp_min(1e-9)
    d_ball = d_ball / d_ball.mean().clamp_min(1e-9)
    d_total = w[0] * d_role + w[1] * d_zone + w[2] * d_ball
    del dists, d_role, d_zone, d_ball

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

    frame_meta = [
        {"idx": i, "frame_id": fr.get("frame_id"),
         "timestamp": fr.get("timestamp"), "phase": fr.get("phase"),
         "cluster": int(labels_by_k[best_k][i])}
        for i, fr in enumerate(frames)
    ]

    result = {
        "input": args.input,
        "n_frames": len(frames),
        "zone_side": zone_side,
        "device": str(device),
        "distance": {
            "kind": "smoothed_hellinger",
            "smooth_sigma": args.smooth_sigma,
            "weights": {"role": float(w[0]), "zone": float(w[1]),
                        "ball": float(w[2])},
        },
        "elbow": {"k_values": ks, "inertias": inertias, "selected_k": best_k},
        "frames": frame_meta,
    }
    with open(os.path.join(args.output_dir, "clusters.json"), "w") as f:
        json.dump(result, f, indent=2)
    _stamp(f"saved matrices, distance matrix, and clusters to {args.output_dir}")
