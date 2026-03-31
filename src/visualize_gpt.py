"""
Visualize GPT-generated shapegraph sequences as a video.

Takes a seed sequence from the dataset, generates new tokens with the GPT,
decodes each frame through the VQ-VAE decoder, and renders a video of
players moving on a pitch.

Usage:
    python src/visualize_gpt.py \
        --gpt-checkpoint checkpoints/gpt/best-000-1.2345.ckpt \
        --vqvae-checkpoint checkpoints/vqvae/best-055-0.4409.ckpt \
        --vqvae-config config/vqvae_pos_only.yml \
        --gpt-config config/gpt_default.yaml \
        --output gpt_generation.mp4 \
        --seed-frames 16 \
        --gen-frames 64 \
        --temperature 0.8 \
        --top-k 50
"""

import argparse
from pathlib import Path

import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from gpt import GPTLightningModule
from vqvae import VQVAELightningModule

PITCH_X = 105.0
PITCH_Y = 68.0
SHOT_TOKEN = 256  # codebook_size (256) + 0-indexed


def denormalize(x_norm, y_norm):
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


def decode_tokens_to_frames(token_seq, vqvae_model, tokens_per_frame, device):
    """Decode a flat token sequence into a list of (22, 4) node feature tensors.

    Skips GOAL tokens and groups remaining tokens into chunks of tokens_per_frame.
    """
    # Filter out GOAL tokens
    codebook_tokens = [t for t in token_seq if t != SHOT_TOKEN]

    # Trim to multiple of tokens_per_frame
    n_usable = (len(codebook_tokens) // tokens_per_frame) * tokens_per_frame
    codebook_tokens = codebook_tokens[:n_usable]

    if n_usable == 0:
        return []

    # Reshape into (num_frames, tokens_per_frame)
    token_tensor = torch.tensor(codebook_tokens, dtype=torch.long, device=device)
    token_tensor = token_tensor.view(-1, tokens_per_frame)

    with torch.no_grad():
        node_feats = vqvae_model.decode_from_tokens(token_tensor)  # (F, 22, 4)

    frames = []
    for i in range(node_feats.size(0)):
        feat = node_feats[i].cpu()
        # Apply sigmoid to team logit, argmax to ball
        feat_display = feat.clone()
        feat_display[:, 2] = torch.sigmoid(feat[:, 2])
        feat_display[:, 3] = (feat[:, 3] == feat[:, 3].max()).float()
        frames.append(feat_display)

    return frames


def interpolate_frames(keyframes, interp_steps=5):
    """Linearly interpolate between keyframes for smooth animation.

    Positions (x, y) are interpolated. Team and ball are held constant
    from the source keyframe until the next one.

    Returns:
        (interpolated_frames, keyframe_indices) — the list of all frames
        and which indices correspond to original keyframes.
    """
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
            # Interpolate x, y positions
            frame[:, 0] = src[:, 0] * (1 - t) + dst[:, 0] * t
            frame[:, 1] = src[:, 1] * (1 - t) + dst[:, 1] * t
            # Team and ball stay from src
            interpolated.append(frame)

    # Append last keyframe
    keyframe_indices.append(len(interpolated))
    interpolated.append(keyframes[-1])

    return interpolated, keyframe_indices


def render_video(seed_frames, gen_frames, output_path, fps=4, interp_steps=5):
    """Render seed + generated frames into a video with smooth interpolation."""
    n_seed_key = len(seed_frames)
    all_keyframes = seed_frames + gen_frames

    # Interpolate for smooth motion
    all_frames, keyframe_indices = interpolate_frames(all_keyframes, interp_steps)
    # The boundary between seed and generated in interpolated space
    n_seed_interp = keyframe_indices[n_seed_key] if n_seed_key < len(keyframe_indices) else len(all_frames)

    effective_fps = fps * interp_steps

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1a1a2e")

    home_patch = mpatches.Patch(color="#3498db", label="Home")
    away_patch = mpatches.Patch(color="#e74c3c", label="Away")
    ball_patch = mpatches.Patch(facecolor="gray", edgecolor="yellow",
                                linewidth=2, label="Ball carrier")

    def animate(frame_idx):
        ax.clear()
        draw_pitch(ax)

        feats = all_frames[frame_idx]
        is_seed = frame_idx < n_seed_interp
        phase = "SEED (real)" if is_seed else "GENERATED"
        color_border = "#f39c12" if not is_seed else "#2ecc71"

        # Show keyframe count
        key_idx = sum(1 for ki in keyframe_indices if ki <= frame_idx)
        total_key = len(all_keyframes)
        ax.set_title(
            f"Keyframe {key_idx}/{total_key}  |  {phase}",
            fontsize=13, fontweight="bold", color=color_border, pad=10,
        )

        for i in range(feats.size(0)):
            x_norm, y_norm, team_val, ball_val = feats[i].tolist()
            x, y = denormalize(x_norm, y_norm)
            is_away = team_val > 0.5
            color = "#e74c3c" if is_away else "#3498db"
            edge_color = "yellow" if ball_val > 0.5 else "white"
            marker_size = 90 if ball_val > 0.5 else 60
            ax.scatter(x, y, c=color, s=marker_size, edgecolors=edge_color,
                       linewidths=1.5, zorder=5)

        ax.legend(handles=[home_patch, away_patch, ball_patch],
                  loc="upper right", fontsize=8,
                  facecolor="#1a1a2e", edgecolor="white", labelcolor="white")

    anim = animation.FuncAnimation(
        fig, animate, frames=len(all_frames),
        interval=1000 // effective_fps, blit=False,
    )
    anim.save(str(output_path), writer="ffmpeg", fps=effective_fps, dpi=120,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close()
    print(f"Saved video to {output_path} ({len(all_keyframes)} keyframes, "
          f"{len(all_frames)} total frames with interpolation, "
          f"{n_seed_key} seed + {len(gen_frames)} generated)")


def main():
    parser = argparse.ArgumentParser(description="Visualize GPT generation as video")
    parser.add_argument("--gpt-checkpoint", type=str, required=True)
    parser.add_argument("--vqvae-checkpoint", type=str, required=True)
    parser.add_argument("--vqvae-config", type=str, default="config/vqvae_pos_only.yml")
    parser.add_argument("--gpt-config", type=str, default="config/gpt_default.yaml")
    parser.add_argument("--token-dir", type=str, default=None,
                        help="Token dir for seed sequence (default: from gpt config)")
    parser.add_argument("--output", type=str, default="gpt_generation.mp4")
    parser.add_argument("--seed-frames", type=int, default=16,
                        help="Number of decoded frames to use as seed context")
    parser.add_argument("--gen-frames", type=int, default=64,
                        help="Number of decoded frames to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Penalize recently generated tokens (1.0=off, 1.2=moderate)")
    parser.add_argument("--fps", type=int, default=4,
                        help="Keyframe rate (actual video fps = fps * interp_steps)")
    parser.add_argument("--interp-steps", type=int, default=5,
                        help="Interpolation sub-frames between each keyframe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configs
    with open(args.vqvae_config) as f:
        vqvae_cfg = yaml.safe_load(f)
    with open(args.gpt_config) as f:
        gpt_cfg = yaml.safe_load(f)

    tokens_per_frame = vqvae_cfg["model"]["encoder"].get("num_summary_tokens", 8)
    context_length = gpt_cfg["model"]["context_length"]

    # Load VQ-VAE
    vqvae_lit = VQVAELightningModule.load_from_checkpoint(
        args.vqvae_checkpoint, map_location=device
    )
    vqvae_lit.eval()
    vqvae_model = vqvae_lit.model.to(device)

    # Load GPT
    gpt_lit = GPTLightningModule.load_from_checkpoint(
        args.gpt_checkpoint, map_location=device
    )
    gpt_lit.eval()
    gpt_model = gpt_lit.model.to(device)

    # Load token data for seed sequence
    token_dir = args.token_dir or gpt_cfg["data"]["token_dir"]
    from gpt.dataset import load_tokens
    segments, info = load_tokens(token_dir)
    print(f"Loaded {info['total_tokens']} tokens from {info['num_matches']} matches "
          f"({info['num_segments']} segments)")

    # Pick a random segment long enough for the seed
    rng = torch.Generator().manual_seed(args.seed)
    seed_tokens_needed = args.seed_frames * tokens_per_frame
    valid_indices = [i for i, s in enumerate(segments) if len(s) > seed_tokens_needed]
    if not valid_indices:
        raise ValueError(
            f"No segment has enough tokens ({seed_tokens_needed}) for "
            f"{args.seed_frames} seed frames. Longest segment: "
            f"{max(len(s) for s in segments)} tokens."
        )
    seg_pick = valid_indices[torch.randint(0, len(valid_indices), (1,), generator=rng).item()]
    seg = segments[seg_pick]
    max_start = len(seg) - seed_tokens_needed - 1
    start_idx = torch.randint(0, max(1, max_start), (1,), generator=rng).item()

    # Extract seed tokens from within the segment
    seed_token_seq = seg[start_idx : start_idx + seed_tokens_needed].tolist()
    print(f"Seed: {len(seed_token_seq)} tokens from segment {seg_pick} "
          f"(len={len(seg)}), offset {start_idx} "
          f"({args.seed_frames} frames x {tokens_per_frame} tokens/frame)")

    # Decode seed frames
    seed_frames = decode_tokens_to_frames(
        seed_token_seq, vqvae_model, tokens_per_frame, device
    )

    # Generate new tokens with GPT
    gen_tokens_needed = args.gen_frames * tokens_per_frame
    # Use seed as conditioning context (crop to context_length)
    cond = torch.tensor([seed_token_seq], dtype=torch.long, device=device)
    if cond.size(1) > context_length:
        cond = cond[:, -context_length:]

    print(f"Generating {gen_tokens_needed} tokens "
          f"(temperature={args.temperature}, top_k={args.top_k})...")
    generated = gpt_model.generate(
        cond, max_new_tokens=gen_tokens_needed,
        temperature=args.temperature, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    # Extract only the newly generated tokens
    new_tokens = generated[0, cond.size(1):].cpu().tolist()

    # Decode generated frames
    gen_frames = decode_tokens_to_frames(
        new_tokens, vqvae_model, tokens_per_frame, device
    )
    print(f"Decoded {len(gen_frames)} generated frames")

    # Render video
    render_video(seed_frames, gen_frames, args.output,
                 fps=args.fps, interp_steps=args.interp_steps)


if __name__ == "__main__":
    main()
