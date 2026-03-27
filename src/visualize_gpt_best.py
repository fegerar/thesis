"""
Visualize the GPT-generated sequence that reaches a GOAL token fastest.

Uses beam search / tree expansion: at each step, expands the top-k most
probable tokens for each active beam. Tracks all beams that hit the GOAL
token and selects the shortest one. The winning sequence is then decoded
through the VQ-VAE and rendered as a smooth video.

Usage:
    python src/visualize_gpt_best.py \
        --gpt-checkpoint checkpoints/gpt/best-000-1.2345.ckpt \
        --vqvae-checkpoint checkpoints/vqvae/best-055-0.4409.ckpt \
        --output gpt_best_goal.mp4 \
        --beam-width 32 \
        --branch-k 8 \
        --max-steps 2048
"""

import argparse
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from gpt import GPTLightningModule
from vqvae import VQVAELightningModule

PITCH_X = 105.0
PITCH_Y = 68.0
GOAL_TOKEN = 256


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


@torch.no_grad()
def beam_search_to_goal(
    gpt_model,
    seed_tokens: torch.Tensor,
    beam_width: int = 32,
    branch_k: int = 8,
    max_steps: int = 2048,
    temperature: float = 0.8,
    device: str = "cpu",
):
    """Batched beam search: all active beams are forwarded in a single GPU pass per step.

    Args:
        gpt_model: ShapegraphGPT model
        seed_tokens: (1, T) seed conditioning tokens
        beam_width: max number of active beams to keep
        branch_k: number of top-k tokens to branch at each step
        max_steps: maximum generation steps before giving up
        temperature: sampling temperature for logit scaling
        device: torch device

    Returns:
        best_tokens: list of token ids for the winning sequence (seed + generated)
        goal_reached: whether GOAL was found
        stats: dict with search statistics
    """
    context_length = gpt_model.context_length
    seed = seed_tokens.to(device)  # (1, T_seed)

    # State: (B, T) tensor of active beam sequences + (B,) cumulative log-probs
    beams_seqs = seed.expand(1, -1)               # (1, T_seed)
    beams_lp   = torch.zeros(1, device=device)    # (1,)

    goal_beams = []  # (token_list, log_prob, steps_to_goal)
    total_expanded = 0

    for step in range(max_steps):
        if step % 100 == 0:
            print(f"  Step {step}/{max_steps} | "
                  f"active beams: {beams_seqs.size(0)} | "
                  f"goal paths found: {len(goal_beams)}")

        if goal_beams and step > min(gb[2] for gb in goal_beams) * 1.5:
            print(f"  Early stop: best goal at step {min(gb[2] for gb in goal_beams)}, "
                  f"current step {step}")
            break

        B = beams_seqs.size(0)

        # --- single batched forward pass for all B beams ---
        seqs_cond = beams_seqs[:, -context_length:]      # (B, T_cond)
        logits = gpt_model(seqs_cond)                    # (B, T_cond, V)
        logits = logits[:, -1, :] / temperature          # (B, V)
        log_probs = F.log_softmax(logits, dim=-1)        # (B, V)

        # top-k branches for every beam simultaneously
        top_lp, top_idx = torch.topk(log_probs, branch_k, dim=-1)  # (B, K)

        # candidate log-probs: parent cumulative + branch log-prob
        cand_lp  = beams_lp.unsqueeze(1) + top_lp       # (B, K)
        cand_lp  = cand_lp.reshape(-1)                   # (B*K,)
        cand_tok = top_idx.reshape(-1)                   # (B*K,)
        # which parent beam each candidate came from
        parent_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, branch_k).reshape(-1)

        total_expanded += B * branch_k

        # --- split into goal vs. non-goal candidates ---
        is_goal    = cand_tok == GOAL_TOKEN              # (B*K,)
        non_goal   = ~is_goal

        # collect goal-reaching sequences (CPU copy, rare)
        for i in is_goal.nonzero(as_tuple=True)[0]:
            seq = torch.cat([beams_seqs[parent_idx[i]], cand_tok[i].unsqueeze(0)], dim=0)
            goal_beams.append((seq.cpu().tolist(), cand_lp[i].item(), step + 1))

        if not non_goal.any():
            break

        # build next-step beam sequences: index parent rows, append token
        keep_parent = parent_idx[non_goal]               # (N,)
        keep_tok    = cand_tok[non_goal].unsqueeze(1)    # (N, 1)
        keep_lp     = cand_lp[non_goal]                  # (N,)

        cand_seqs = torch.cat([beams_seqs[keep_parent], keep_tok], dim=1)  # (N, T+1)

        # prune to beam_width by cumulative log-prob
        if cand_seqs.size(0) > beam_width:
            _, top = torch.topk(keep_lp, beam_width)
            beams_seqs = cand_seqs[top]
            beams_lp   = keep_lp[top]
        else:
            beams_seqs = cand_seqs
            beams_lp   = keep_lp

    stats = {
        "total_expanded": total_expanded,
        "steps_searched": step + 1,
        "goal_paths_found": len(goal_beams),
    }

    if goal_beams:
        goal_beams.sort(key=lambda x: (x[2], -x[1]))
        best_seq, best_log_prob, best_steps = goal_beams[0]
        stats["best_steps_to_goal"] = best_steps
        stats["best_log_prob"] = best_log_prob
        stats["shortest_goal_steps"] = best_steps
        stats["longest_goal_steps"] = max(gb[2] for gb in goal_beams)
        print(f"\n  Best path: {best_steps} tokens to GOAL "
              f"(log-prob: {best_log_prob:.2f})")
        print(f"  Found {len(goal_beams)} paths, range: "
              f"{best_steps}-{stats['longest_goal_steps']} steps")
        return best_seq, True, stats
    else:
        best_seq = beams_seqs[beams_lp.argmax()].cpu().tolist()
        print(f"\n  No GOAL found after {max_steps} steps. "
              f"Returning most probable beam.")
        return best_seq, False, stats


def decode_tokens_to_frames(token_seq, vqvae_model, tokens_per_frame, device):
    """Decode a flat token sequence into a list of (22, 4) node feature tensors."""
    codebook_tokens = [t for t in token_seq if t != GOAL_TOKEN]

    n_usable = (len(codebook_tokens) // tokens_per_frame) * tokens_per_frame
    codebook_tokens = codebook_tokens[:n_usable]

    if n_usable == 0:
        return []

    token_tensor = torch.tensor(codebook_tokens, dtype=torch.long, device=device)
    token_tensor = token_tensor.view(-1, tokens_per_frame)

    with torch.no_grad():
        node_feats = vqvae_model.decode_from_tokens(token_tensor)

    frames = []
    for i in range(node_feats.size(0)):
        feat = node_feats[i].cpu()
        feat_display = feat.clone()
        feat_display[:, 2] = torch.sigmoid(feat[:, 2])
        feat_display[:, 3] = (feat[:, 3] == feat[:, 3].max()).float()
        frames.append(feat_display)

    return frames


def interpolate_frames(keyframes, interp_steps=5):
    """Linearly interpolate between keyframes for smooth animation."""
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


def render_video(seed_frames, gen_frames, output_path, fps=4, interp_steps=5,
                 goal_reached=False):
    """Render seed + generated frames into a video with smooth interpolation."""
    n_seed_key = len(seed_frames)
    all_keyframes = seed_frames + gen_frames

    all_frames, keyframe_indices = interpolate_frames(all_keyframes, interp_steps)
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
        phase = "SEED (real)" if is_seed else "GENERATED (best path to GOAL)"
        color_border = "#f39c12" if not is_seed else "#2ecc71"

        key_idx = sum(1 for ki in keyframe_indices if ki <= frame_idx)
        total_key = len(all_keyframes)

        goal_str = " -> GOAL!" if goal_reached and frame_idx == len(all_frames) - 1 else ""
        ax.set_title(
            f"Keyframe {key_idx}/{total_key}  |  {phase}{goal_str}",
            fontsize=12, fontweight="bold", color=color_border, pad=10,
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
          f"{len(all_frames)} total interpolated frames)")


def main():
    parser = argparse.ArgumentParser(
        description="Beam-search GPT generation: find shortest path to GOAL"
    )
    parser.add_argument("--gpt-checkpoint", type=str, required=True)
    parser.add_argument("--vqvae-checkpoint", type=str, required=True)
    parser.add_argument("--vqvae-config", type=str, default="config/vqvae_pos_only.yml")
    parser.add_argument("--gpt-config", type=str, default="config/gpt_default.yaml")
    parser.add_argument("--token-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="gpt_best_goal.mp4")
    parser.add_argument("--seed-frames", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=32,
                        help="Max active beams to keep at each step")
    parser.add_argument("--branch-k", type=int, default=8,
                        help="Top-k tokens to branch at each step per beam")
    parser.add_argument("--max-steps", type=int, default=2048,
                        help="Max generation steps before giving up")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--interp-steps", type=int, default=5)
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

    # Load token data for seed
    token_dir = args.token_dir or gpt_cfg["data"]["token_dir"]
    from gpt.dataset import load_tokens
    all_tokens, info = load_tokens(token_dir)
    print(f"Loaded {info['total_tokens']} tokens from {info['num_matches']} matches")

    # Pick random seed
    rng = torch.Generator().manual_seed(args.seed)
    seed_tokens_needed = args.seed_frames * tokens_per_frame
    max_start = len(all_tokens) - seed_tokens_needed - 1
    start_idx = torch.randint(0, max_start, (1,), generator=rng).item()

    seed_token_seq = all_tokens[start_idx : start_idx + seed_tokens_needed].tolist()
    print(f"Seed: {len(seed_token_seq)} tokens from index {start_idx}")

    # Decode seed frames
    seed_frames = decode_tokens_to_frames(
        seed_token_seq, vqvae_model, tokens_per_frame, device
    )

    # Beam search for fastest path to GOAL
    seed_tensor = torch.tensor([seed_token_seq], dtype=torch.long, device=device)
    if seed_tensor.size(1) > context_length:
        seed_tensor = seed_tensor[:, -context_length:]

    print(f"\nStarting beam search (beam_width={args.beam_width}, "
          f"branch_k={args.branch_k}, max_steps={args.max_steps})...")
    best_seq, goal_reached, stats = beam_search_to_goal(
        gpt_model,
        seed_tensor,
        beam_width=args.beam_width,
        branch_k=args.branch_k,
        max_steps=args.max_steps,
        temperature=args.temperature,
        device=device,
    )

    print(f"\nSearch stats: {stats}")

    # Extract generated portion (after seed)
    seed_len = seed_tensor.size(1)
    gen_tokens = best_seq[seed_len:]

    # Decode generated frames
    gen_frames = decode_tokens_to_frames(
        gen_tokens, vqvae_model, tokens_per_frame, device
    )
    print(f"Decoded {len(gen_frames)} generated frames "
          f"({'GOAL reached' if goal_reached else 'no GOAL'})")

    # Render video
    render_video(seed_frames, gen_frames, args.output,
                 fps=args.fps, interp_steps=args.interp_steps,
                 goal_reached=goal_reached)


if __name__ == "__main__":
    main()
