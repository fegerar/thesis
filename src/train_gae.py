"""
Train HQA-GAE on a single shapegraph from shapegraphs.pkl.

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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

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
from hqa_gae.utils.optim import build_optimizer        # noqa: E402
from hqa_gae.utils.random import set_random_seed       # noqa: E402
from hqa_gae.utils.datautil import DataUtil            # noqa: E402
from hqa_gae.models import create_gae, GAE             # noqa: E402
from hqa_gae.utils.test import test_link_prediction    # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Shapegraph → PyG Data conversion
# ═══════════════════════════════════════════════════════════════════════════

_ROLE_VOCAB: list[str] | None = None   # built lazily on first call


def _build_role_vocab(G: nx.Graph) -> list[str]:
    """Collect all unique inferred roles from the graph."""
    roles = sorted({d.get("inferred_role", "?") for _, d in G.nodes(data=True)})
    return roles


def shapegraph_to_pyg(G: nx.Graph, cfg) -> Data:
    """
    Convert a NetworkX shapegraph to a ``torch_geometric.data.Data`` object.

    Node features are configured via ``cfg.node_features`` (team one-hot,
    has_ball binary, inferred_role one-hot, shirt normalised).
    Coordinates (x, y) are **not** included.
    """
    global _ROLE_VOCAB

    nodes = list(G.nodes(data=True))
    n = len(nodes)
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
        if _ROLE_VOCAB is None:
            _ROLE_VOCAB = _build_role_vocab(G)
        role2idx = {r: j for j, r in enumerate(_ROLE_VOCAB)}
        role_feat = np.zeros((n, len(_ROLE_VOCAB)), dtype=np.float32)
        for i, (_, d) in enumerate(nodes):
            r = d.get("inferred_role", "?")
            if r in role2idx:
                role_feat[i, role2idx[r]] = 1.0
        feat_parts.append(role_feat)

    if cfg.node_features.get("use_shirt", False):
        shirt_feat = np.zeros((n, 1), dtype=np.float32)
        for i, (_, d) in enumerate(nodes):
            shirt_feat[i, 0] = d.get("shirt", 0) / 99.0  # normalise
        feat_parts.append(shirt_feat)

    x = np.concatenate(feat_parts, axis=1) if feat_parts else np.ones((n, 1), dtype=np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float)

    # --- Build edge_index ---
    src, dst = [], []
    for u, v in G.edges():
        ui, vi = pid_to_idx[u], pid_to_idx[v]
        src.extend([ui, vi])
        dst.extend([vi, ui])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index)
    return data


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
            # Check for new columns
            new_keys = sorted(metrics.keys())
            if new_keys != self._fieldnames:
                self._fieldnames = new_keys
                # Re-write header by re-opening; simpler: just use extrasaction
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                writer.writerow(metrics)


# ═══════════════════════════════════════════════════════════════════════════
# LightningDataModule wrapper (single graph)
# ═══════════════════════════════════════════════════════════════════════════

class ShapegraphDataModule(pl.LightningDataModule):
    """Wraps a single PyG graph for training + val/test with edge splits."""

    def __init__(self, data: Data, val_ratio: float, test_ratio: float):
        super().__init__()
        self.full_data = data

        # Split edges for link prediction evaluation
        split = DataUtil.train_test_split_edges(
            data.edge_index, num_node=data.x.size(0),
            val_ratio=val_ratio, test_ratio=test_ratio,
        )

        # Training graph uses only training edges
        self.train_data = Data(
            x=data.x,
            edge_index=split["pos"]["train"],
        )

        # Validation / test graphs carry pos + neg edge labels
        self.val_data = Data(
            x=data.x,
            edge_index=split["pos"]["train"],  # message-passing on train edges
            pos_edge_label_index=split["pos"]["val"],
            neg_edge_label_index=split["neg"]["val"],
        )
        self.test_data = Data(
            x=data.x,
            edge_index=split["pos"]["train"],
            pos_edge_label_index=split["pos"]["test"],
            neg_edge_label_index=split["neg"]["test"],
        )

    def train_dataloader(self):
        return PyGDataLoader([self.train_data], batch_size=1, shuffle=False)

    def val_dataloader(self):
        return [
            PyGDataLoader([self.val_data], batch_size=1, shuffle=False),
            PyGDataLoader([self.test_data], batch_size=1, shuffle=False),
        ]


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
    # 1. Load shapegraph
    # ------------------------------------------------------------------
    print(f"Loading shapegraphs from {cfg.data.pickle_path} ...")
    with open(cfg.data.pickle_path, "rb") as f:
        all_games = pickle.load(f)

    game = all_games[cfg.data.game_index]
    frame_keys = sorted(game.keys())
    frame_key = frame_keys[cfg.data.frame_index]
    graph_type = cfg.data.get("graph_type", "original")
    G: nx.Graph = game[frame_key][graph_type]

    print(f"Game {cfg.data.game_index}, frame {frame_key} ({graph_type}): "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # 2. Convert to PyG
    # ------------------------------------------------------------------
    data = shapegraph_to_pyg(G, cfg)
    print(f"PyG Data: x={list(data.x.shape)}, edge_index={list(data.edge_index.shape)}")

    dm = ShapegraphDataModule(
        data,
        val_ratio=cfg.train.get("val_ratio", 0.05),
        test_ratio=cfg.train.get("test_ratio", 0.10),
    )

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    dataset_stub = _DatasetStub(num_features=data.x.size(1))
    hqa_gae_model = create_gae(cfg, dataset_stub)

    lightning_model = GAE(
        model=hqa_gae_model,
        optimizer=cfg.optimizer,
        scheduler=cfg.get("scheduler", None),
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
        save_top_k=-1,      # save all
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
        check_val_every_n_epoch=cfg.train.get("log_interval", 10),
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

    print(f"\nTraining complete. Outputs saved to {run_dir}/")


if __name__ == "__main__":
    main()
