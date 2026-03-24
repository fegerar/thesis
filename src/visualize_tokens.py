"""
Visualize and analyze token sequences produced by the tokenizer pipeline.

Shows token frequency distributions, most common n-grams, sequence statistics,
and GOAL token context — analogous to standard word-level corpus analysis.

Usage:
    python src/visualize_tokens.py --token-dir output/tokens
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np


def load_all_sequences(token_dir: Path) -> tuple[list[dict], dict]:
    """Load all per-match .pt files and the summary JSON."""
    results = []
    for pt_file in sorted(token_dir.glob("*_tokens.pt")):
        results.append(torch.load(pt_file, weights_only=False))

    summary_path = token_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return results, summary


def print_corpus_stats(results: list[dict], summary: dict):
    """Print high-level corpus statistics."""
    print("=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)

    vocab_size = summary.get("vocab_size", "?")
    codebook_size = summary.get("vqvae_codebook_size", "?")
    goal_token_id = summary.get("goal_token_id", -1)

    total_tokens = sum(len(r["tokens"]) for r in results)
    total_goals = sum(len(r.get("goal_positions", [])) for r in results)

    print(f"Matches:          {len(results)}")
    print(f"Vocabulary size:  {vocab_size} ({codebook_size} codebook + GOAL)")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Total GOAL tokens:{total_goals}")
    print()

    print(f"{'Match':<22} {'Tokens':>8} {'Raw frames':>12} {'Dedup frames':>14} {'Compress':>10} {'Goals':>6}")
    print("-" * 74)
    for r in results:
        meta = r.get("metadata", {})
        print(
            f"{r['match_id']:<22} {len(r['tokens']):>8,} "
            f"{meta.get('num_raw_frames', '?'):>12,} "
            f"{meta.get('num_deduplicated_frames', '?'):>14,} "
            f"{meta.get('compression_ratio', 0):>9.1%} "
            f"{meta.get('num_goals', 0):>6}"
        )
    print()


def plot_token_frequencies(all_tokens: list[int], goal_token_id: int, ax: plt.Axes):
    """Plot token frequency distribution (histogram + rank-frequency)."""
    # Separate codebook tokens from GOAL
    codebook_tokens = [t for t in all_tokens if t != goal_token_id]
    counts = Counter(codebook_tokens)

    token_ids = sorted(counts.keys())
    freqs = [counts[t] for t in token_ids]

    ax.bar(token_ids, freqs, width=1.0, color="steelblue", alpha=0.8)
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Frequency Distribution")


def plot_rank_frequency(all_tokens: list[int], goal_token_id: int, ax: plt.Axes):
    """Zipf-style rank-frequency plot."""
    codebook_tokens = [t for t in all_tokens if t != goal_token_id]
    counts = Counter(codebook_tokens)
    ranked = sorted(counts.values(), reverse=True)

    ax.loglog(range(1, len(ranked) + 1), ranked, "o-", markersize=3, color="coral")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title("Rank-Frequency (Zipf) Plot")
    ax.grid(True, alpha=0.3)


def print_most_common(all_tokens: list[int], goal_token_id: int, top_k: int = 20):
    """Print most and least common tokens."""
    codebook_tokens = [t for t in all_tokens if t != goal_token_id]
    counts = Counter(codebook_tokens)
    total = len(codebook_tokens)
    unique = len(counts)

    print(f"Unique codebook tokens used: {unique}")
    print()

    print(f"Top {top_k} most common tokens:")
    print(f"  {'Token':>6} {'Count':>8} {'Freq':>8}")
    print("  " + "-" * 24)
    for tok, cnt in counts.most_common(top_k):
        print(f"  {tok:>6} {cnt:>8,} {cnt/total:>7.2%}")
    print()

    print(f"Top {top_k} least common tokens:")
    print(f"  {'Token':>6} {'Count':>8} {'Freq':>8}")
    print("  " + "-" * 24)
    for tok, cnt in counts.most_common()[-top_k:]:
        print(f"  {tok:>6} {cnt:>8,} {cnt/total:>7.2%}")
    print()

    # Unused codebook entries
    max_id = max(counts.keys()) if counts else 0
    all_ids = set(range(max_id + 1))
    unused = all_ids - set(counts.keys())
    print(f"Unused codebook entries: {len(unused)} / {max_id + 1}")
    if unused and len(unused) <= 30:
        print(f"  IDs: {sorted(unused)}")
    print()


def print_ngram_analysis(all_tokens: list[int], goal_token_id: int, n: int = 2, top_k: int = 15):
    """Print most common n-grams (excluding GOAL tokens)."""
    codebook_tokens = [t for t in all_tokens if t != goal_token_id]
    ngrams = [
        tuple(codebook_tokens[i:i + n])
        for i in range(len(codebook_tokens) - n + 1)
    ]
    counts = Counter(ngrams)

    print(f"Top {top_k} most common {n}-grams:")
    print(f"  {'N-gram':<30} {'Count':>8} {'Freq':>8}")
    print("  " + "-" * 48)
    total = len(ngrams)
    for ng, cnt in counts.most_common(top_k):
        ng_str = " -> ".join(str(t) for t in ng)
        print(f"  {ng_str:<30} {cnt:>8,} {cnt/total:>7.2%}")
    print()


def plot_sequence_lengths(results: list[dict], ax: plt.Axes):
    """Plot per-match sequence lengths."""
    match_ids = [r["match_id"].replace("DFL-MAT-", "") for r in results]
    lengths = [len(r["tokens"]) for r in results]

    bars = ax.bar(range(len(match_ids)), lengths, color="seagreen", alpha=0.8)
    ax.set_xticks(range(len(match_ids)))
    ax.set_xticklabels(match_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sequence length (tokens)")
    ax.set_title("Tokens per Match")

    for bar, length in zip(bars, lengths):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{length:,}", ha="center", va="bottom", fontsize=7,
        )


def plot_goal_context(results: list[dict], goal_token_id: int, context: int = 10):
    """Show token patterns around GOAL events."""
    print("GOAL TOKEN CONTEXT")
    print("=" * 60)

    for r in results:
        tokens = r["tokens"]
        goal_positions = r.get("goal_positions", [])
        if not goal_positions:
            continue

        match_id = r["match_id"]
        print(f"\n{match_id}:")

        for gpos in goal_positions:
            start = max(0, gpos - context)
            end = min(len(tokens), gpos + context + 1)
            window = tokens[start:end]

            # Format: highlight GOAL token
            parts = []
            for i, tok in enumerate(window):
                pos_in_seq = start + i
                if pos_in_seq == gpos:
                    parts.append(f"[GOAL]")
                else:
                    parts.append(str(tok))

            print(f"  ...{' '.join(parts)}...")
    print()


def plot_token_transitions(all_tokens: list[int], goal_token_id: int, codebook_size: int, ax: plt.Axes):
    """Plot transition matrix heatmap for most common tokens."""
    codebook_tokens = [t for t in all_tokens if t != goal_token_id]

    # Focus on top-N most common tokens for readability
    counts = Counter(codebook_tokens)
    top_n = min(30, len(counts))
    top_tokens = [t for t, _ in counts.most_common(top_n)]
    tok_to_idx = {t: i for i, t in enumerate(top_tokens)}

    trans = np.zeros((top_n, top_n))
    for a, b in zip(codebook_tokens[:-1], codebook_tokens[1:]):
        if a in tok_to_idx and b in tok_to_idx:
            trans[tok_to_idx[a], tok_to_idx[b]] += 1

    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_norm = trans / row_sums

    im = ax.imshow(trans_norm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(top_n))
    ax.set_yticks(range(top_n))
    ax.set_xticklabels(top_tokens, fontsize=5, rotation=90)
    ax.set_yticklabels(top_tokens, fontsize=5)
    ax.set_xlabel("Next token")
    ax.set_ylabel("Current token")
    ax.set_title(f"Transition Probabilities (top {top_n} tokens)")
    plt.colorbar(im, ax=ax, fraction=0.046)


def main():
    parser = argparse.ArgumentParser(description="Visualize token sequences")
    parser.add_argument(
        "--token-dir", type=str, required=True,
        help="Directory with *_tokens.pt files and summary.json",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save figures (shows interactively if omitted)",
    )
    args = parser.parse_args()

    token_dir = Path(args.token_dir)
    results, summary = load_all_sequences(token_dir)

    if not results:
        print(f"No token files found in {token_dir}")
        return

    goal_token_id = summary.get("goal_token_id", -1)
    codebook_size = summary.get("vqvae_codebook_size", 256)

    # Gather all tokens across matches
    all_tokens = []
    for r in results:
        all_tokens.extend(r["tokens"])

    # === Text output ===
    print_corpus_stats(results, summary)
    print_most_common(all_tokens, goal_token_id)
    print_ngram_analysis(all_tokens, goal_token_id, n=2)
    print_ngram_analysis(all_tokens, goal_token_id, n=3, top_k=10)
    plot_goal_context(results, goal_token_id)

    # === Figures ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Token Sequence Analysis", fontsize=14, fontweight="bold")

    plot_token_frequencies(all_tokens, goal_token_id, axes[0, 0])
    plot_rank_frequency(all_tokens, goal_token_id, axes[0, 1])
    plot_sequence_lengths(results, axes[1, 0])
    plot_token_transitions(all_tokens, goal_token_id, codebook_size, axes[1, 1])

    plt.tight_layout()

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "token_analysis.png", dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_dir / 'token_analysis.png'}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
