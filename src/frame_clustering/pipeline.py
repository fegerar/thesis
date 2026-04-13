"""End-to-end frame clustering pipeline."""

import json
import os

import torch
from tqdm import tqdm

from .clustering import elbow_k, kmedoids
from .matrices import (
    build_all_matrices, grid_cost, infer_zone_side, load_frames,
    subsample, to_distribution,
)
from .sinkhorn import pairwise_sinkhorn


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

    role_cost = grid_cost(5, device)
    zone_cost = grid_cost(zone_side, device)
    dists = {}
    for key, cost in (
        ("role_home", role_cost), ("role_guest", role_cost),
        ("zone_home", zone_cost), ("zone_guest", zone_cost),
        ("zone_ball", zone_cost),
    ):
        hist = to_distribution(mats[key].to(device))
        dists[key] = pairwise_sinkhorn(
            hist, cost, reg=args.reg, n_iter=args.sinkhorn_iters,
            pair_batch=args.pair_batch, desc=f"sinkhorn {key}",
        )

    d_home = (dists["role_home"] + dists["zone_home"]) / 2.0
    d_guest = (dists["role_guest"] + dists["zone_guest"]) / 2.0
    d_total = (d_home + d_guest + dists["zone_ball"]) / 3.0

    d_total_cpu = d_total.detach().cpu()
    torch.save(d_total_cpu, os.path.join(args.output_dir, "distance_matrix.pt"))

    k_max = min(args.k_max, d_total_cpu.shape[0] - 1)
    k_min = min(args.k_min, k_max)
    ks = list(range(k_min, k_max + 1))

    _stamp("running k-medoids elbow sweep")
    inertias, labels_by_k = [], {}
    d_dev = d_total_cpu.to(device)
    for k in tqdm(ks, desc="k-medoids", unit="k", dynamic_ncols=True):
        labels, _medoids, inertia = kmedoids(
            d_dev, k=k, max_iter=args.kmedoids_iters, seed=args.seed,
        )
        inertias.append(inertia)
        labels_by_k[k] = labels.detach().cpu().tolist()

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
        "sinkhorn": {"reg": args.reg, "iterations": args.sinkhorn_iters},
        "elbow": {"k_values": ks, "inertias": inertias, "selected_k": best_k},
        "frames": frame_meta,
    }
    with open(os.path.join(args.output_dir, "clusters.json"), "w") as f:
        json.dump(result, f, indent=2)
    _stamp(f"saved matrices, distance matrix, and clusters to {args.output_dir}")
