"""
PyTorch Lightning module for VQ-VAE training.

Handles:
    - Forward pass + loss computation
    - WandB logging (metrics, codebook utilization, reconstructions)
    - CSV logging (epoch-level metrics)
    - Periodic checkpoint saving
"""

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import lightning as L
from torch_geometric.utils import to_dense_batch, to_dense_adj

from .model import VQVAE


class VQVAELightningModule(L.LightningModule):
    def __init__(self, model_cfg: dict, loss_cfg: dict, training_cfg: dict,
                 logging_cfg: dict, node_dim: int):
        super().__init__()
        self.save_hyperparameters()

        self.model = VQVAE(
            node_dim=node_dim,
            encoder_cfg=model_cfg["encoder"],
            quantizer_cfg=model_cfg["quantizer"],
            decoder_cfg=model_cfg["decoder"],
        )

        self.lambda_node = loss_cfg.get("lambda_node", 1.0)
        self.lambda_edge = loss_cfg.get("lambda_edge", 2.0)
        self.lambda_pos = loss_cfg.get("lambda_pos", 1.0)
        self.lambda_flag = loss_cfg.get("lambda_flag", 1.0)

        self.lr = training_cfg["learning_rate"]
        self.weight_decay = training_cfg.get("weight_decay", 1e-5)
        self.lr_scheduler = training_cfg.get("lr_scheduler", "cosine")
        self.warmup_epochs = training_cfg.get("warmup_epochs", 5)
        self.max_epochs = training_cfg["max_epochs"]

        self.num_roles = model_cfg["decoder"]["num_roles"]

        # CSV logging setup
        self.csv_dir = Path(logging_cfg.get("csv_dir", "logs/vqvae"))
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.csv_dir / "metrics.csv"
        self._csv_initialized = False

        # Track epoch-level metrics
        self._epoch_metrics: dict[str, list[float]] = {}

    def _compute_loss(self, batch):
        node_feats, adj_logits, z_e, z_q, tokens, vq_loss, utilization = (
            self.model(batch.x, batch.edge_index, batch.batch)
        )

        # Dense ground truth
        x_dense, mask = to_dense_batch(batch.x, batch.batch)  # (B, N_max, F)
        adj_gt = to_dense_adj(batch.edge_index, batch.batch)  # (B, N_max, N_max)

        B, N_max, F_dim = x_dense.shape
        N_roles = self.num_roles

        # Pad or truncate to match decoder output size (num_roles)
        if N_max < N_roles:
            pad_x = torch.zeros(B, N_roles - N_max, F_dim, device=x_dense.device)
            x_target = torch.cat([x_dense, pad_x], dim=1)
            pad_mask = torch.zeros(B, N_roles - N_max, dtype=torch.bool, device=mask.device)
            node_mask = torch.cat([mask, pad_mask], dim=1)

            pad_adj = torch.zeros(B, N_roles, N_roles, device=adj_gt.device)
            pad_adj[:, :N_max, :N_max] = adj_gt
            adj_target = pad_adj
        else:
            x_target = x_dense[:, :N_roles]
            node_mask = mask[:, :N_roles]
            adj_target = adj_gt[:, :N_roles, :N_roles]

        # Node reconstruction loss (only on valid nodes)
        # Continuous features: x, y (indices 0-1)
        pos_pred = node_feats[..., :2]
        pos_gt = x_target[..., :2]
        pos_diff = (pos_pred - pos_gt).pow(2) * node_mask.unsqueeze(-1)
        pos_loss = pos_diff.sum() / node_mask.sum().clamp(min=1) / 2

        # Binary/categorical features: team, has_ball, role_one_hot (indices 2+)
        flag_pred = node_feats[..., 2:].sigmoid()
        flag_gt = x_target[..., 2:]
        flag_diff = F.binary_cross_entropy(flag_pred, flag_gt, reduction="none")
        flag_loss = (flag_diff * node_mask.unsqueeze(-1)).sum() / node_mask.sum().clamp(min=1) / flag_diff.size(-1)

        # Edge reconstruction loss (BCE with class imbalance weighting)
        adj_pred = adj_logits.sigmoid()
        edge_weight = (1.0 - adj_target) * 0.1 + adj_target * 1.0
        edge_loss_raw = F.binary_cross_entropy(adj_pred, adj_target, weight=edge_weight, reduction="none")

        # Mask edges: only compute loss where both nodes exist
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # (B, N, N)
        edge_loss = (edge_loss_raw * edge_mask).sum() / edge_mask.sum().clamp(min=1)

        total_loss = (
            self.lambda_node * (self.lambda_pos * pos_loss + self.lambda_flag * flag_loss)
            + self.lambda_edge * edge_loss
            + vq_loss
        )

        metrics = {
            "loss": total_loss,
            "pos_loss": pos_loss,
            "flag_loss": flag_loss,
            "edge_loss": edge_loss,
            "vq_loss": vq_loss,
            "codebook_utilization": utilization,
        }
        return total_loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True,
                     prog_bar=(k == "loss"), batch_size=batch.num_graphs)
            self._accumulate(f"train/{k}", v.item() if isinstance(v, torch.Tensor) else v)

        # Restart unused codes periodically
        if self.model.quantizer.use_ema and batch_idx % 100 == 0:
            z_e = self.model.encoder(batch.x, batch.edge_index, batch.batch)
            n_restarted = self.model.quantizer.restart_unused_codes(z_e)
            if n_restarted > 0:
                self.log("train/codes_restarted", float(n_restarted),
                         batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch.num_graphs)
            self._accumulate(f"val/{k}", v.item() if isinstance(v, torch.Tensor) else v)
        return loss

    def on_validation_epoch_end(self):
        # Log val/loss to prog_bar once at epoch end
        val_loss = self.trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            self.log("val_loss", val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_epoch=True, batch_size=batch.num_graphs)
        return loss

    def on_train_epoch_end(self):
        self._write_csv()

    def _accumulate(self, key: str, value: float):
        if key not in self._epoch_metrics:
            self._epoch_metrics[key] = []
        self._epoch_metrics[key].append(value)

    def _write_csv(self):
        """Write averaged epoch metrics to CSV."""
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
