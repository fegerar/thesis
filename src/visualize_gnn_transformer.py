"""
Visualize GNN Transformer predictions as a side-by-side video.

Takes a seed sequence from the dataset, autoregressively predicts future
frames, and renders a video comparing predicted vs real trajectories.

Left panel:  predicted (seed → autoregressive rollout)
Right panel: real continuation from the data

Usage:
    python src/visualize_gnn_transformer.py \
        --checkpoint checkpoints/gnn_transformer/best-050-0.0123.ckpt \
        --config config/gnn_transformer_default.yaml \
        --output gnn_transformer_pred.mp4 \
        --seed-frames 32 \
        --gen-frames 64 \
        --seed 42
"""

import argparse

import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from gnn_transformer import GNNTransformerLightningModule
from gnn_transformer.dataset import (
    build_match_frames, _get_fc_edges, NUM_NODES, NODE_DIM,
    PITCH_X, PITCH_Y,
)
from tokenizer.utils import discover_matches


def denormalize(x_norm, y_norm):
    """Convert normalized [-1, 1] coords to pitch [0, 105] x [0, 68]."""
    x = x_norm * (PITCH_X / 2) + (PITCH_X / 2)
    y = y_norm * (PITCH_Y / 2) + (PITCH_Y / 2)
    return x, y


def draw_pitch(ax):
    ax.set_xlim(-2, PITCH_X + 2)
    ax.set_ylim(-2, PITCH_Y + 2)
    ax.set_aspect("equal")
    ax.set_facecolor("#2e8b57")

    pitch = plt.Rectangle((0, 0), PITCH_X, PITCH_Y,
                           fill=False, edgecolor="white", linewidth=1.5)
    ax.add_patch(pitch)
    ax.plot([PITCH_X / 2, PITCH_X / 2], [0, PITCH_Y], color="white", linewidth=1)
    center_circle = plt.Circle((PITCH_X / 2, PITCH_Y / 2), 9.15,
                                fill=False, edgecolor="white", linewidth=1)
    ax.add_patch(center_circle)
    for x_start in [0, PITCH_X - 16.5]:
        pa = plt.Rectangle((x_start, (PITCH_Y - 40.3) / 2), 16.5, 40.3,
                            fill=False, edgecolor="white", linewidth=1)
        ax.add_patch(pa)
    ax.set_xticks([])
    ax.set_yticks([])


def compute_edge_attrs(frame: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute edge distances for a single frame. Returns (E, 1)."""
    pos = frame[:, :2]
    src_pos = pos[edge_index[0]]
    dst_pos = pos[edge_index[1]]
    return torch.norm(src_pos - dst_pos, dim=1, keepdim=True)


@torch.no_grad()
def autoregressive_rollout(model, seed_frames: torch.Tensor,
                           edge_index: torch.Tensor,
                           n_gen: int, seq_len: int,
                           device: torch.device) -> list[torch.Tensor]:
    """Autoregressively predict n_gen frames given seed_frames.

    Args:
        model: GNNTransformer model.
        seed_frames: (S, 23, 6) tensor of seed frames (S >= seq_len).
        edge_index: (2, E) fully connected edges.
        n_gen: number of frames to generate.
        seq_len: context window size the model expects.
        device: torch device.

    Returns:
        List of n_gen predicted (23, 6) tensors.
    """
    model.eval()
    edge_index = edge_index.to(device)

    # Keep a rolling buffer of recent frames
    buffer = seed_frames.to(device)  # (S, 23, 6)

    generated = []
    for _ in range(n_gen):
        # Take last seq_len frames as context
        context = buffer[-seq_len:]  # (seq_len, 23, 6)
        input_batch = context.unsqueeze(0)  # (1, seq_len, 23, 6)

        # Compute edge distances for each frame in context
        edge_attrs = []
        for t in range(seq_len):
            edge_attrs.append(compute_edge_attrs(context[t], edge_index))
        edge_attr_seq = torch.stack(edge_attrs, dim=0).unsqueeze(0)  # (1, seq_len, E, 1)

        pred = model(input_batch, edge_index, edge_attr_seq)  # (1, 23, 6)
        pred_frame = pred.squeeze(0)  # (23, 6)

        # Preserve static features from seed (team, is_ball don't change)
        pred_frame[:, 4] = buffer[0, :, 4]  # team
        pred_frame[:, 5] = buffer[0, :, 5]  # is_ball

        # Recompute velocity relative to last frame in buffer
        pred_frame[:, 2] = pred_frame[:, 0] - buffer[-1, :, 0]
        pred_frame[:, 3] = pred_frame[:, 1] - buffer[-1, :, 1]

        generated.append(pred_frame.cpu())
        buffer = torch.cat([buffer, pred_frame.unsqueeze(0)], dim=0)

    return generated


def interpolate_frames(keyframes, interp_steps=5):
    """Linearly interpolate positions between keyframes for smooth video."""
    if len(keyframes) < 2:
        return keyframes, list(range(len(keyframes)))

    interpolated = []
    keyframe_indices = []

    for i in range(len(keyframes) - 1):
        src = keyframes[i]
        dst = keyframes[i + 1]
        keyframe_indices.append(len(interpolated))

        for s in range(interp_steps):
            t = s / interp_steps
            frame = src.clone()
            frame[:, 0] = src[:, 0] * (1 - t) + dst[:, 0] * t
            frame[:, 1] = src[:, 1] * (1 - t) + dst[:, 1] * t
            interpolated.append(frame)

    keyframe_indices.append(len(interpolated))
    interpolated.append(keyframes[-1])
    return interpolated, keyframe_indices


def render_video(seed_frames, gen_frames, real_frames, output_path,
                 fps=4, interp_steps=5):
    """Render side-by-side video: predicted vs real."""
    n_seed = len(seed_frames)

    gen_keyframes = seed_frames + gen_frames
    real_keyframes = seed_frames + real_frames

    # Trim to same length
    min_len = min(len(gen_keyframes), len(real_keyframes))
    gen_keyframes = gen_keyframes[:min_len]
    real_keyframes = real_keyframes[:min_len]

    gen_all, gen_ki = interpolate_frames(gen_keyframes, interp_steps)
    real_all, real_ki = interpolate_frames(real_keyframes, interp_steps)

    n_seed_interp = gen_ki[n_seed] if n_seed < len(gen_ki) else len(gen_all)
    effective_fps = fps * interp_steps

    fig, (ax_pred, ax_real) = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor("#1a1a2e")

    home_patch = mpatches.Patch(color="#3498db", label="Home")
    away_patch = mpatches.Patch(color="#e74c3c", label="Away")
    ball_patch = mpatches.Patch(facecolor="#f1c40f", edgecolor="white",
                                linewidth=2, label="Ball")

    def _draw_frame(ax, feats, title, color_border):
        """Draw a single frame on an axis. feats: (23, 6)."""
        ax.clear()
        draw_pitch(ax)
        ax.set_title(title, fontsize=13, fontweight="bold", color=color_border, pad=10)

        for i in range(feats.size(0)):
            x_norm, y_norm = feats[i, 0].item(), feats[i, 1].item()
            team_val = feats[i, 4].item()
            is_ball_node = feats[i, 5].item()
            x, y = denormalize(x_norm, y_norm)

            if is_ball_node > 0.5:
                # Ball node
                ax.scatter(x, y, c="#f1c40f", s=100, edgecolors="white",
                           linewidths=2, zorder=6, marker="o")
                ax.text(x, y + 1.8, "ball", fontsize=5, color="#f1c40f",
                        ha="center", va="bottom", fontweight="bold", zorder=7)
            else:
                # Player node
                is_away = team_val > 0.5
                color = "#e74c3c" if is_away else "#3498db"
                ax.scatter(x, y, c=color, s=60, edgecolors="white",
                           linewidths=1.5, zorder=5)
                ax.text(x, y + 1.5, str(i), fontsize=6, color="white",
                        ha="center", va="bottom", fontweight="bold", zorder=6)

        ax.legend(handles=[home_patch, away_patch, ball_patch],
                  loc="upper right", fontsize=8,
                  facecolor="#1a1a2e", edgecolor="white", labelcolor="white")

    def animate(frame_idx):
        is_seed = frame_idx < n_seed_interp
        key_idx = sum(1 for ki in gen_ki if ki <= frame_idx)
        total_key = len(gen_keyframes)

        if is_seed:
            phase = "SEED (real)"
            color_border = "#2ecc71"
        else:
            phase = "PREDICTED"
            color_border = "#f39c12"

        _draw_frame(ax_pred, gen_all[frame_idx],
                    f"Predicted  |  KF {key_idx}/{total_key}  |  {phase}",
                    color_border)

        real_phase = "SEED (real)" if is_seed else "REAL continuation"
        real_color = "#2ecc71" if is_seed else "#9b59b6"
        _draw_frame(ax_real, real_all[frame_idx],
                    f"Real  |  KF {key_idx}/{total_key}  |  {real_phase}",
                    real_color)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(gen_all),
        interval=1000 // effective_fps, blit=False,
    )
    anim.save(str(output_path), writer="ffmpeg", fps=effective_fps, dpi=120,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close()
    print(f"Saved video to {output_path} ({len(gen_keyframes)} keyframes, "
          f"{len(gen_all)} interpolated frames, "
          f"{n_seed} seed + {len(gen_frames)} predicted + {len(real_frames)} real)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GNN Transformer predictions"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="GNN Transformer checkpoint path")
    parser.add_argument("--config", type=str,
                        default="config/gnn_transformer_default.yaml")
    parser.add_argument("--output", type=str, default="gnn_transformer_pred.mp4")
    parser.add_argument("--seed-frames", type=int, default=32,
                        help="Number of seed context frames")
    parser.add_argument("--gen-frames", type=int, default=64,
                        help="Number of frames to predict autoregressively")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--interp-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seq_len = cfg["data"]["seq_len"]
    subsample = cfg["data"].get("subsample", 10)
    data_dir = cfg["data"]["path"]

    # Load model
    lit_model = GNNTransformerLightningModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    lit_model.eval()
    model = lit_model.model.to(device)

    # Load one match worth of frames from raw DFL data
    matches = discover_matches(data_dir)
    if not matches:
        raise ValueError(f"No matches found in {data_dir}")

    rng = torch.Generator().manual_seed(args.seed)
    match_idx = torch.randint(0, len(matches), (1,), generator=rng).item()
    match = matches[match_idx]
    print(f"Using match {match['match_id']}")

    frames = build_match_frames(
        matchinfo_path=match["matchinfo_path"],
        positions_path=match["positions_path"],
        events_path=match["events_path"],
        subsample=subsample,
    )

    total_needed = args.seed_frames + args.gen_frames
    if len(frames) < total_needed:
        raise ValueError(
            f"Match has {len(frames)} frames but need {total_needed} "
            f"({args.seed_frames} seed + {args.gen_frames} gen)"
        )

    # Pick a random starting point
    max_start = len(frames) - total_needed
    start = torch.randint(0, max(1, max_start), (1,), generator=rng).item()
    print(f"Starting at frame {start} of {len(frames)}")

    seed = [f.clone() for f in frames[start : start + args.seed_frames]]
    real = [f.clone() for f in frames[start + args.seed_frames :
                                       start + total_needed]]

    # Autoregressive rollout
    edge_index = _get_fc_edges()
    seed_tensor = torch.stack(seed, dim=0)  # (S, 23, 6)

    print(f"Generating {args.gen_frames} frames autoregressively...")
    gen = autoregressive_rollout(
        model, seed_tensor, edge_index,
        n_gen=args.gen_frames, seq_len=seq_len, device=device,
    )
    print(f"Generated {len(gen)} frames")

    # Render
    render_video(seed, gen, real, args.output,
                 fps=args.fps, interp_steps=args.interp_steps)


if __name__ == "__main__":
    main()
