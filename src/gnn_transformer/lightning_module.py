"""
PyTorch Lightning module for GNN Transformer training.

Loss:
    - Position (x, y):  MSE on normalized coordinates
    - Velocity (vx, vy): MSE
    - Team:              BCE (binary — home/away)
    - Ball flag:         BCE (is_ball indicator, mostly for gradient flow)

The primary metric is position MSE since trajectory accuracy is the goal.
"""

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import lightning as L

from .model import GNNTransformer


class GNNTransformerLightningModule(L.LightningModule):
    def __init__(self, model_cfg: dict, loss_cfg: dict, training_cfg: dict,
                 logging_cfg: dict):
        super().__init__()
        self.save_hyperparameters()

        self.model = GNNTransformer(
            node_dim=model_cfg.get("node_dim", 6),
            hidden_dim=model_cfg["hidden_dim"],
            num_nodes=model_cfg.get("num_nodes", 23),
            edge_dim=model_cfg.get("edge_dim", 1),
            spatial_layers=model_cfg["spatial_layers"],
            spatial_heads=model_cfg["spatial_heads"],
            temporal_layers=model_cfg["temporal_layers"],
            temporal_heads=model_cfg["temporal_heads"],
            dropout=model_cfg.get("dropout", 0.1),
        )

        self.lambda_pos = loss_cfg.get("lambda_pos", 1.0)
        self.lambda_vel = loss_cfg.get("lambda_vel", 0.5)
        self.lambda_team = loss_cfg.get("lambda_team", 0.1)
        self.lambda_ball = loss_cfg.get("lambda_ball", 0.1)

        self.lr = training_cfg["learning_rate"]
        self.weight_decay = training_cfg.get("weight_decay", 1e-5)
        self.lr_scheduler = training_cfg.get("lr_scheduler", "cosine")
        self.warmup_epochs = training_cfg.get("warmup_epochs", 5)
        self.max_epochs = training_cfg["max_epochs"]

        # CSV logging
        self.csv_dir = Path(logging_cfg.get("csv_dir", "logs/gnn_transformer"))
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.csv_dir / "metrics.csv"
        self._csv_initialized = False
        self._epoch_metrics: dict[str, list[float]] = {}

    def _compute_loss(self, batch):
        input_frames, target_frames, edge_index, edge_attr_seq = batch

        # Move shared tensors to device
        edge_index = edge_index.to(self.device)

        pred = self.model(input_frames, edge_index, edge_attr_seq)
        # pred: (B, 23, 6), target_frames: (B, 23, 6)

        # Position loss (indices 0-1): MSE
        pos_pred = pred[:, :, :2]
        pos_gt = target_frames[:, :, :2]
        pos_loss = F.mse_loss(pos_pred, pos_gt)

        # Velocity loss (indices 2-3): MSE
        vel_pred = pred[:, :, 2:4]
        vel_gt = target_frames[:, :, 2:4]
        vel_loss = F.mse_loss(vel_pred, vel_gt)

        # Team loss (index 4): BCE — only for player nodes (first 22)
        team_pred = pred[:, :22, 4]
        team_gt = target_frames[:, :22, 4]
        team_loss = F.binary_cross_entropy_with_logits(team_pred, team_gt)

        # Ball flag loss (index 5): BCE — node 22 should always be 1
        ball_pred = pred[:, :, 5]
        ball_gt = target_frames[:, :, 5]
        ball_loss = F.binary_cross_entropy_with_logits(ball_pred, ball_gt)

        total_loss = (
            self.lambda_pos * pos_loss
            + self.lambda_vel * vel_loss
            + self.lambda_team * team_loss
            + self.lambda_ball * ball_loss
        )

        # Compute position error in meters for interpretability
        # Predictions are in [-1, 1] normalized coords
        pos_error_m = torch.sqrt(
            ((pos_pred - pos_gt) * torch.tensor([52.5, 34.0],
             device=pos_pred.device)).pow(2).sum(dim=-1)
        ).mean()

        metrics = {
            "loss": total_loss,
            "pos_loss": pos_loss,
            "vel_loss": vel_loss,
            "team_loss": team_loss,
            "ball_loss": ball_loss,
            "pos_error_m": pos_error_m,
        }
        return total_loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            self.log(f"train/{k}", val, on_step=True, on_epoch=False,
                     prog_bar=(k == "pos_error_m"), batch_size=batch[0].size(0))
            self._accumulate(f"train/{k}", val)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            self.log(f"val/{k}", val, on_step=False, on_epoch=True,
                     prog_bar=(k in ("loss", "pos_error_m")),
                     batch_size=batch[0].size(0))
            self._accumulate(f"val/{k}", val)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            self.log(f"test/{k}", val, on_epoch=True, batch_size=batch[0].size(0))

    def on_train_epoch_end(self):
        self._write_csv()

    def _accumulate(self, key: str, value: float):
        if key not in self._epoch_metrics:
            self._epoch_metrics[key] = []
        self._epoch_metrics[key].append(value)

    def _write_csv(self):
        if not self._epoch_metrics:
            return
        row = {"epoch": self.current_epoch}
        for key, values in self._epoch_metrics.items():
            row[key] = sum(values) / len(values)
        fieldnames = sorted(row.keys())
        if not self._csv_initialized:
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self._csv_initialized = True
        else:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
        self._epoch_metrics.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.lr_scheduler == "none":
            return optimizer

        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs - self.warmup_epochs
            )
        elif self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.5
            )
        else:
            return optimizer

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, scheduler],
                milestones=[self.warmup_epochs]
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
