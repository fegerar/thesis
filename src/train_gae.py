"""
Train HQA-GAE on all shapegraphs from shapegraphs.pkl.

Usage:
    python src/train_gae.py --config config/shapegraph_hqa_gae.yml
"""
import argparse
import csv
import os
import pickle
import sys
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    TQDMProgressBar,
)
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make hqa_gae importable  (directory is named "hqa-gae")
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_HQA_REAL = os.path.join(_SRC_DIR, "hqa-gae")
_HQA_LINK = os.path.join(_SRC_DIR, "hqa_gae")
if not os.path.exists(_HQA_LINK):
    os.symlink(_HQA_REAL, _HQA_LINK)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from hqa_gae.utils.argutil import parse_yaml          # noqa: E402
from hqa_gae.utils.random import set_random_seed       # noqa: E402
from hqa_gae.utils.datautil import DataUtil            # noqa: E402
from hqa_gae.models import create_gae, GAE             # noqa: E402
from hqa_gae.utils.test import test_link_prediction    # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Role vocabulary builder (scans ALL graphs first)
# ═══════════════════════════════════════════════════════════════════════════

def build_role_vocab(all_games, graph_type: str) -> list[str]:
    """Scan every graph in every game to collect all unique inferred roles."""
    roles = set()
    for game in all_games:
        for frame_key in game:
            entry = game[frame_key]
            G = entry[graph_type]
            for _, d in G.nodes(data=True):
                roles.add(d.get("inferred_role", "?"))
    return sorted(roles)


# ═══════════════════════════════════════════════════════════════════════════
# Shapegraph → PyG Data conversion
# ═══════════════════════════════════════════════════════════════════════════

def shapegraph_to_pyg(G: nx.Graph, cfg, role_vocab: list[str]) -> Data:
    """
    Convert a NetworkX shapegraph to a ``torch_geometric.data.Data`` object.

    Node features are configured via ``cfg.node_features`` (team one-hot,
    has_ball binary, inferred_role one-hot, shirt normalised).
    Coordinates (x, y) are **not** included.
    """
    nodes = list(G.nodes(data=True))
    n = len(nodes)

    if n < 2:
        return None

    pid_to_idx = {pid: i for i, (pid, _) in enumerate(nodes)}

    # --- Build feature vectors ---
    feat_parts: list[np.ndarray] = []

    if cfg.node_features.use_team:
        team_feat = np.zeros((n, 2), dtype=np.float32)
        for i, (_, d) in enumerate(nodes):
            if d.get("team") == "home":
                team_feat[i, 0] = 1.0
            else:
                team_feat[i, 1] = 1.0
        feat_parts.append(team_feat)

    if cfg.node_features.use_has_ball:
        ball_feat = np.zeros((n, 1), dtype=np.float32)
        for i, (_, d) in enumerate(nodes):
            ball_feat[i, 0] = float(d.get("has_ball", False))
        feat_parts.append(ball_feat)

    if cfg.node_features.use_inferred_role:
        role2idx = {r: j for j, r in enumerate(role_vocab)}
        role_feat = np.zeros((n, len(role_vocab)), dtype=np.float32)
        for i, (_, d) in enumerate(nodes):
            r = d.get("inferred_role", "?")
            if r in role2idx:
                role_feat[i, role2idx[r]] = 1.0
        feat_parts.append(role_feat)

    if cfg.node_features.get("use_shirt", False):
        shirt_feat = np.zeros((n, 1), dtype=np.float32)
        for i, (_, d) in enumerate(nodes):
            shirt_feat[i, 0] = d.get("shirt", 0) / 99.0
        feat_parts.append(shirt_feat)

    x = np.concatenate(feat_parts, axis=1) if feat_parts else np.ones((n, 1), dtype=np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float)

    # --- Build edge_index ---
    src, dst = [], []
    for u, v in G.edges():
        ui, vi = pid_to_idx[u], pid_to_idx[v]
        src.extend([ui, vi])
        dst.extend([vi, ui])

    if len(src) == 0:
        return None

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # --- Build label tensor (y) for node classification ---
    # We use inferred_role as the classification target
    role2idx = {r: j for j, r in enumerate(role_vocab)}
    y_list = [role2idx.get(d.get("inferred_role", "?"), 0) for _, d in nodes]
    y_tensor = torch.tensor(y_list, dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    return data


def load_all_shapegraphs(all_games, cfg, role_vocab: list[str]) -> list[Data]:
    """Convert ALL shapegraphs from ALL games into PyG Data objects and move to device."""
    graph_type = cfg.data.get("graph_type", "original")
    device = torch.device(cfg.train.get("device", "cuda"))
    all_data = []

    total_frames = sum(len(game) for game in all_games)
    print(f"Converting {total_frames} shapegraphs across {len(all_games)} games ...")

    for game_idx, game in enumerate(all_games):
        for frame_key in tqdm(sorted(game.keys()),
                              desc=f"Game {game_idx+1}/{len(all_games)}",
                              leave=False):
            entry = game[frame_key]
            G = entry[graph_type]
            data = shapegraph_to_pyg(G, cfg, role_vocab)
            if data is not None:
                # Pre-load entirely into VRAM!
                data = data.to(device)
                all_data.append(data)

    print(f"Converted {len(all_data)} valid shapegraphs (skipped {total_frames - len(all_data)})")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════
# CSV Metrics Logger Callback
# ═══════════════════════════════════════════════════════════════════════════

class CSVMetricsCallback(Callback):
    """Append metrics to a CSV file at every training epoch end."""

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        self._header_written = False
        self._fieldnames: list[str] | None = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = {k: v.item() if hasattr(v, "item") else v
                   for k, v in trainer.callback_metrics.items()}
        metrics["epoch"] = trainer.current_epoch

        if not self._header_written:
            self._fieldnames = sorted(metrics.keys())
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(metrics)
            self._header_written = True
        else:
            new_keys = sorted(metrics.keys())
            if new_keys != self._fieldnames:
                self._fieldnames = new_keys
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                writer.writerow(metrics)


# ═══════════════════════════════════════════════════════════════════════════
# LightningDataModule (all graphs, train/val/test split)
# ═══════════════════════════════════════════════════════════════════════════

class ShapegraphDataModule(pl.LightningDataModule):
    """
    Splits all shapegraphs into train/val/test sets.
    Training uses full edge_index per graph. Validation/test graphs
    get edge splits for link prediction evaluation.
    """

    def __init__(self, all_data: list[Data], cfg):
        super().__init__()
        val_ratio = cfg.train.get("val_ratio", 0.05)
        test_ratio = cfg.train.get("test_ratio", 0.10)
        batch_size = cfg.train.get("batch_size", 64)
        
        # When tensors are already in CUDA memory, PyTorch strict multiprocessing
        # fails if you use num_workers > 0.
        num_workers = 0 

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Shuffle and split graphs into train / val / test
        n = len(all_data)
        indices = torch.randperm(n).tolist()
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))

        val_indices = indices[:n_val]
        test_indices = indices[n_val:n_val + n_test]
        train_indices = indices[n_val + n_test:]

        self.train_data = [all_data[i] for i in train_indices]

        # For val: create edge splits per graph for link prediction.
        # We merge validation and test graphs into one conceptual "validation set"
        # because during training, we just want to see live eval metrics.
        # PyG / Lightning gets confused with multiple val dataloaders and renames metrics.
        
        eval_graphs = [all_data[i] for i in val_indices + test_indices]
        self.val_data = self._prepare_eval_graphs(eval_graphs)

        print(f"Dataset split: {len(self.train_data)} train, "
              f"{len(self.val_data)} validation")

    @staticmethod
    def _prepare_eval_graphs(graphs: list[Data]) -> list[Data]:
        """Add pos/neg edge labels for link prediction evaluation."""
        eval_data = []
        for data in graphs:
            try:
                # We must use non-zero val_ratio because PyG's negative_sampling
                # crashes if num_neg_samples=0.
                split = DataUtil.train_test_split_edges(
                    data.edge_index, num_node=data.x.size(0),
                    val_ratio=0.05, test_ratio=0.15,
                )
                eval_graph = Data(
                    x=data.x,
                    edge_index=split["pos"]["train"],
                    pos_edge_label_index=split["pos"]["test"],
                    neg_edge_label_index=split["neg"]["test"],
                    y=getattr(data, "y", None),
                )
                eval_data.append(eval_graph)
            except Exception as e:
                # Skip graphs where edge splitting fails (too few edges)
                continue
        return eval_data

    def train_dataloader(self):
        return PyGDataLoader(
            self.train_data, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        # Return a single dataloader so Lightning logs the exact keys 'valid_AUC', 
        # instead of appending '/dataloader_idx_0' which confuses wandb.
        return PyGDataLoader(
            self.val_data, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight dataset stub for create_gae()
# ═══════════════════════════════════════════════════════════════════════════

class _DatasetStub:
    """Minimal object exposing ``num_features`` so ``create_gae`` works."""
    def __init__(self, num_features: int):
        self.num_features = num_features


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train HQA-GAE on shapegraphs")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = parse_yaml(args.config)

    # Seed
    set_random_seed(cfg.train.seed)

    # ------------------------------------------------------------------
    # 1. Load ALL shapegraphs
    # ------------------------------------------------------------------
    print(f"Loading shapegraphs from {cfg.data.pickle_path} ...")
    with open(cfg.data.pickle_path, "rb") as f:
        all_games = pickle.load(f)

    graph_type = cfg.data.get("graph_type", "original")
    print(f"Using graph type: {graph_type}")
    print(f"Found {len(all_games)} games")

    # Build role vocabulary across ALL games
    print("Building role vocabulary ...")
    role_vocab = build_role_vocab(all_games, graph_type)
    print(f"Role vocabulary ({len(role_vocab)} roles): {role_vocab}")

    # Convert all graphs to PyG
    all_data = load_all_shapegraphs(all_games, cfg, role_vocab)

    # Free memory — we no longer need the raw NetworkX graphs
    del all_games

    # ------------------------------------------------------------------
    # 2. DataModule
    # ------------------------------------------------------------------
    dm = ShapegraphDataModule(all_data, cfg)
    num_features = all_data[0].x.size(1)

    # Free the full list (DataModule holds its own refs)
    del all_data

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    dataset_stub = _DatasetStub(num_features=num_features)
    hqa_gae_model = create_gae(cfg, dataset_stub)

    lightning_model = GAE(
        model=hqa_gae_model,
        optimizer=cfg.optimizer,
        scheduler=getattr(cfg, "scheduler", None),
    )

    # ------------------------------------------------------------------
    # 4. Run name & output dirs
    # ------------------------------------------------------------------
    run_name = cfg.wandb.get("run_name", None)
    if not run_name:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = os.path.join(cfg.output.checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config alongside weights
    cfg.to_yaml(os.path.join(run_dir, "config.yml"))

    # ------------------------------------------------------------------
    # 5. Callbacks
    # ------------------------------------------------------------------
    callbacks = []

    # Checkpointing
    ckpt_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="epoch_{epoch:04d}",
        every_n_epochs=cfg.train.get("checkpoint_every", 100),
        save_top_k=-1,
        save_last=True,
    )
    callbacks.append(ckpt_callback)

    # CSV metrics
    csv_path = os.path.join(run_dir, "metrics.csv")
    callbacks.append(CSVMetricsCallback(csv_path))

    # tqdm
    callbacks.append(TQDMProgressBar(refresh_rate=1))

    # ------------------------------------------------------------------
    # 6. Logger (wandb)
    # ------------------------------------------------------------------
    logger = None
    if cfg.wandb.get("enabled", False):
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=run_name,
            save_dir=run_dir,
            log_model=False,
        )
        logger.experiment.config.update(cfg.to_dict())

    # ------------------------------------------------------------------
    # 7. Trainer
    # ------------------------------------------------------------------
    device = cfg.train.get("device", "cuda")
    accelerator = "gpu" if "cuda" in device else "cpu"

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=logger if logger else True,
        log_every_n_steps=cfg.train.get("log_interval", 10),
        enable_model_summary=True,
        check_val_every_n_epoch=cfg.train.get("val_every_n_epoch", 5),
        default_root_dir=run_dir,
    )

    # ------------------------------------------------------------------
    # 8. Train!
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Training HQA-GAE  |  run: {run_name}")
    print(f"  Checkpoints & CSV → {run_dir}")
    print(f"{'='*60}\n")

    trainer.fit(lightning_model, datamodule=dm)

    # Cleanly finish wandb to avoid socket errors on Kaggle
    if cfg.wandb.get("enabled", False):
        import wandb
        wandb.finish()

    print(f"\nTraining complete. Outputs saved to {run_dir}/")


if __name__ == "__main__":
    main()
