"""
Validate trained HQA-GAE weights from a PyTorch Lightning checkpoint.

Usage:
    python src/val_gae.py --checkpoint checkpoints/run_XXX/last.ckpt
"""
import argparse
import os
import pickle
import sys

import networkx as nx
import torch
import pytorch_lightning as pl
from tqdm import tqdm

# Make hqa_gae importable
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_HQA_LINK = os.path.join(_SRC_DIR, "hqa_gae")
if not os.path.exists(_HQA_LINK):
    os.symlink(os.path.join(_SRC_DIR, "hqa-gae"), _HQA_LINK)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from hqa_gae.models import GAE
from hqa_gae.utils.argutil import parse_yaml
from hqa_gae.utils.random import set_random_seed

# Import data utilities from your training script
# We do this so we don't have to duplicate the complex conversion logic
from train_gae import build_role_vocab, load_all_shapegraphs, ShapegraphDataModule

def main():
    parser = argparse.ArgumentParser(description="Validate HQA-GAE Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the .ckpt file (e.g., checkpoints/run_XXX/last.ckpt)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    run_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(run_dir, "config.yml")
    if not os.path.exists(config_path):
        print(f"Error: config.yml not found alongside checkpoint in {run_dir}")
        sys.exit(1)

    print(f"Loading config from {config_path}")
    cfg = parse_yaml(config_path)
    set_random_seed(cfg.train.seed)

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    print(f"Loading shapegraphs from {cfg.data.pickle_path} ...")
    try:
        with open(cfg.data.pickle_path, "rb") as f:
            all_games = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find pickle file at {cfg.data.pickle_path}")
        print("Please ensure you are running this from the correct directory.")
        sys.exit(1)

    graph_type = cfg.data.get("graph_type", "original")
    
    # We must build the exact same role vocab used during training
    role_vocab = build_role_vocab(all_games, graph_type)
    
    # Convert all graphs to PyG
    all_data = load_all_shapegraphs(all_games, cfg, role_vocab)
    del all_games

    # ------------------------------------------------------------------
    # 2. DataModule
    # ------------------------------------------------------------------
    print("\nPreparing evaluation datasets (this creates pos/neg edge splits)...")
    dm = ShapegraphDataModule(all_data, cfg)
    del all_data

    # ------------------------------------------------------------------
    # 3. Load Model from Checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading model weights from {ckpt_path} ...")
    
    # We load strictly the LightningModule state.
    # The GAE wrapper handles the inner HQA_GAE architecture magically.
    model = GAE.load_from_checkpoint(ckpt_path)
    model.eval()

    # ------------------------------------------------------------------
    # 4. Run PyTorch Lightning Validation
    # ------------------------------------------------------------------
    device = cfg.train.get("device", "cuda")
    accelerator = "gpu" if ("cuda" in device and torch.cuda.is_available()) else "cpu"
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False, # Don't log to wandb for a simple validation check
    )

    print(f"\n{'='*60}")
    print(f"  Running Validation on Checkpoint")
    print(f"{'='*60}\n")
    
    # This automatically calls model.validation_step() across the entire val_dataloader
    results = trainer.validate(model, datamodule=dm)
    
    print(f"\n{'='*60}")
    print(f"  Final Validation Results")
    print(f"{'='*60}")
    for key, value in results[0].items():
        print(f"    {key}: {value:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
