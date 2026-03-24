"""
Tokenize shapegraphs for autoregressive modeling.

Runs the full pipeline: extract -> deduplicate -> encode -> tokenize
for each match in the dataset.

Usage:
    python src/tokenize_matches.py \
        --data-dir data/ \
        --vqvae-checkpoint checkpoints/vqvae/best-055-0.4409.ckpt \
        --output-dir output/tokens

    # With pre-computed shapegraphs:
    python src/tokenize_matches.py \
        --data-dir data/ \
        --vqvae-checkpoint checkpoints/vqvae/best-055-0.4409.ckpt \
        --output-dir output/tokens \
        --shapegraphs-pkl shapegraphs.pkl
"""

import argparse

from tokenizer.utils import setup_logging
from tokenizer.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize shapegraphs for autoregressive modeling"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to Bassek et al. raw tracking data directory",
    )
    parser.add_argument(
        "--vqvae-checkpoint", type=str, required=True,
        help="Path to VQ-VAE Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/tokens",
        help="Where to save token sequences",
    )
    parser.add_argument(
        "--shapegraphs-pkl", type=str, default=None,
        help="Path to pre-computed shapegraphs.pkl (regenerates if omitted)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device for VQ-VAE inference (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Batch size for VQ-VAE encoding",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging verbosity (DEBUG/INFO/WARNING)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    run_pipeline(
        data_dir=args.data_dir,
        vqvae_checkpoint=args.vqvae_checkpoint,
        output_dir=args.output_dir,
        shapegraphs_pkl=args.shapegraphs_pkl,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
