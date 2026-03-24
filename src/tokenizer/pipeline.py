"""
End-to-end tokenization pipeline.

Orchestrates: extract -> deduplicate -> encode -> tokenize for each match.
Saves per-match .pt files and a summary JSON.

Usage:
    python -m tokenizer.pipeline \
        --data-dir data/ \
        --vqvae-checkpoint checkpoints/vqvae/best-055-0.4409.ckpt \
        --output-dir output/tokens
"""

import json
import logging
from pathlib import Path

import torch

from .utils import setup_logging, parse_goals
from .extract import load_shapegraphs_per_match
from .deduplicate import deduplicate_match
from .encode import VQVAEEncoder
from .tokenize import build_token_sequence

logger = logging.getLogger(__name__)


def run_pipeline(
    data_dir: str,
    vqvae_checkpoint: str,
    output_dir: str,
    shapegraphs_pkl: str | None = None,
    device: str = "cpu",
    batch_size: int = 512,
) -> list[dict]:
    """Run the full tokenization pipeline.

    Args:
        data_dir: path to Bassek et al. raw tracking data
        vqvae_checkpoint: path to VQ-VAE Lightning checkpoint
        output_dir: where to save token sequences
        shapegraphs_pkl: optional path to pre-computed shapegraphs.pkl
        device: torch device for VQ-VAE inference
        batch_size: batch size for VQ-VAE encoding

    Returns:
        List of result dicts (one per match).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize VQ-VAE encoder
    encoder = VQVAEEncoder(
        checkpoint_path=vqvae_checkpoint,
        device=device,
        batch_size=batch_size,
    )

    # Load shapegraphs per match
    match_data = load_shapegraphs_per_match(data_dir, shapegraphs_pkl)

    all_results = []
    summary = []

    for mdata in match_data:
        match_id = mdata["match_id"]
        frames = mdata["frames"]
        events_path = mdata["events_path"]

        logger.info("=" * 60)
        logger.info("Processing match: %s", match_id)

        # Step 2: Deduplicate
        dedup = deduplicate_match(frames)

        # Step 3: Encode through VQ-VAE
        frame_tokens = encoder.encode_graphs(
            dedup["frame_ids"], dedup["frames"]
        )

        # Step 4: Parse goals and build token sequence
        goals = parse_goals(events_path)
        seq = build_token_sequence(
            frame_tokens=frame_tokens,
            frame_ids=dedup["frame_ids"],
            timestamps=dedup["timestamps"],
            goals=goals,
            codebook_size=encoder.codebook_size,
        )

        # Build result
        result = {
            "match_id": match_id,
            "tokens": seq["tokens"],
            "frame_ids": seq["frame_ids"],
            "timestamps": seq["timestamps"],
            "goal_positions": seq["goal_positions"],
            "metadata": {
                "num_raw_frames": dedup["num_raw"],
                "num_deduplicated_frames": dedup["num_deduped"],
                "compression_ratio": dedup["compression_ratio"],
                "num_goals": len(goals),
                "vqvae_codebook_size": encoder.codebook_size,
                "goal_token_id": seq["goal_token_id"],
                "num_summary_tokens": encoder.num_summary_tokens,
            },
        }

        # Save per-match .pt file
        save_path = out_path / f"{match_id}_tokens.pt"
        torch.save(result, save_path)
        logger.info("Saved %s (%d tokens)", save_path.name, len(seq["tokens"]))

        all_results.append(result)
        summary.append({
            "match_id": match_id,
            "num_tokens": len(seq["tokens"]),
            "num_raw_frames": dedup["num_raw"],
            "num_deduplicated_frames": dedup["num_deduped"],
            "compression_ratio": round(dedup["compression_ratio"], 4),
            "num_goals": len(goals),
        })

    # Save summary JSON
    summary_info = {
        "num_matches": len(summary),
        "total_tokens": sum(s["num_tokens"] for s in summary),
        "total_raw_frames": sum(s["num_raw_frames"] for s in summary),
        "total_deduped_frames": sum(s["num_deduplicated_frames"] for s in summary),
        "vqvae_codebook_size": encoder.codebook_size,
        "goal_token_id": encoder.codebook_size,
        "vocab_size": encoder.codebook_size + 1,  # codebook + GOAL
        "matches": summary,
    }
    summary_path = out_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_info, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    return all_results
