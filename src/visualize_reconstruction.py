"""
Visualize original vs reconstructed shapegraphs from a trained VQ-VAE.

Usage:
    python src/visualize_reconstruction.py \
        --config config/vqvae_pos_only.yml \
        --checkpoint path/to/checkpoint.ckpt \
        --num-samples 6 \
        --output reconstruction_viz.png
"""

import argparse
from pathlib import Path

import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.loader import DataLoader

from vqvae import VQVAELightningModule
from vqvae.dataset import load_shapegraphs, ShapegraphDataset, PITCH_X, PITCH_Y


def denormalize(x_norm, y_norm):
    """Convert normalized [-1, 1] coordinates back to meters."""
    x = (x_norm + 1.0) * (PITCH_X / 2)
    y = (y_norm + 1.0) * (PITCH_Y / 2)
    return x, y


def draw_pitch(ax):
    """Draw a soccer pitch outline."""
    ax.set_xlim(-2, PITCH_X + 2)
    ax.set_ylim(-2, PITCH_Y + 2)
    ax.set_aspect("equal")
    ax.set_facecolor("#2e8b57")

    # Pitch outline
    pitch = plt.Rectangle((0, 0), PITCH_X, PITCH_Y,
                           fill=False, edgecolor="white", linewidth=1.5)
    ax.add_patch(pitch)

    # Center line and circle
    ax.plot([PITCH_X / 2, PITCH_X / 2], [0, PITCH_Y], color="white", linewidth=1)
    center_circle = plt.Circle((PITCH_X / 2, PITCH_Y / 2), 9.15,
                                fill=False, edgecolor="white", linewidth=1)
    ax.add_patch(center_circle)

    # Penalty areas
    for x_start in [0, PITCH_X - 16.5]:
        pa = plt.Rectangle((x_start, (PITCH_Y - 40.3) / 2), 16.5, 40.3,
                            fill=False, edgecolor="white", linewidth=1)
        ax.add_patch(pa)

    ax.set_xticks([])
    ax.set_yticks([])


def plot_shapegraph(ax, node_feats, title, num_real_nodes=None):
    """Plot players on a pitch.

    Args:
        ax: matplotlib axes
        node_feats: (N, 4) tensor [x_norm, y_norm, team, has_ball]
        title: plot title
        num_real_nodes: if set, only plot the first N nodes (ignore padding)
    """
    draw_pitch(ax)
    ax.set_title(title, fontsize=11, fontweight="bold", color="white")

    n = num_real_nodes if num_real_nodes is not None else node_feats.size(0)
    feats = node_feats[:n]

    for i in range(n):
        x_norm, y_norm, team_val, ball_val = feats[i].tolist()
        x, y = denormalize(x_norm, y_norm)

        # Team colors
        is_away = team_val > 0.5
        color = "#e74c3c" if is_away else "#3498db"
        edge_color = "yellow" if ball_val > 0.5 else "white"
        marker_size = 90 if ball_val > 0.5 else 60

        ax.scatter(x, y, c=color, s=marker_size, edgecolors=edge_color,
                   linewidths=1.5, zorder=5)
        ax.annotate(str(i), (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6, color="white")


def main():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE reconstructions")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--output", type=str, default="reconstruction_viz.png")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load data
    data_cfg = cfg["data"]
    all_data, node_dim = load_shapegraphs(data_cfg["path"])
    n = len(all_data)

    generator = torch.Generator().manual_seed(data_cfg["seed"])
    indices = torch.randperm(n, generator=generator).tolist()
    n_train = int(n * data_cfg["train_ratio"])
    n_val = int(n * data_cfg["val_ratio"])

    if args.split == "train":
        split_data = [all_data[i] for i in indices[:n_train]]
    elif args.split == "val":
        split_data = [all_data[i] for i in indices[n_train:n_train + n_val]]
    else:
        split_data = [all_data[i] for i in indices[n_train + n_val:]]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAELightningModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model.eval()
    model.to(device)

    # Pick random samples
    rng = torch.Generator().manual_seed(args.seed)
    sample_indices = torch.randperm(len(split_data), generator=rng)[:args.num_samples]

    # Create figure: 2 rows per sample (original, reconstructed)
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(2, n_samples, figsize=(5 * n_samples, 10))
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    fig.patch.set_facecolor("#1a1a2e")

    loader = DataLoader(
        ShapegraphDataset([split_data[i] for i in sample_indices]),
        batch_size=1, shuffle=False,
    )

    for col, batch in enumerate(loader):
        batch = batch.to(device)
        num_nodes = batch.x.size(0)

        with torch.no_grad():
            node_feats_pred, *_ = model.model(
                batch.x, batch.edge_index, batch.batch
            )

        # Original
        gt = batch.x.cpu()  # (N, 4)
        plot_shapegraph(axes[0, col], gt, f"Original (sample {col})",
                        num_real_nodes=num_nodes)

        # Reconstructed — apply sigmoid to team, softmax to ball
        pred = node_feats_pred[0].cpu()  # (22, 4)
        pred_display = pred.clone()
        pred_display[:, 2] = torch.sigmoid(pred[:, 2])      # team logit -> prob
        pred_display[:, 3] = (pred[:, 3] == pred[:, 3].max()).float()  # ball = argmax
        plot_shapegraph(axes[1, col], pred_display, f"Reconstructed (sample {col})",
                        num_real_nodes=num_nodes)

    # Legend
    home_patch = mpatches.Patch(color="#3498db", label="Home")
    away_patch = mpatches.Patch(color="#e74c3c", label="Away")
    ball_patch = mpatches.Patch(facecolor="gray", edgecolor="yellow",
                                linewidth=2, label="Ball carrier")
    fig.legend(handles=[home_patch, away_patch, ball_patch],
               loc="lower center", ncol=3, fontsize=12,
               facecolor="#1a1a2e", edgecolor="white", labelcolor="white")

    fig.suptitle("VQ-VAE Shapegraph Reconstruction", fontsize=16,
                 color="white", fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(args.output, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved visualization to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
