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
from torch_geometric.utils import to_dense_batch

from .model import VQVAE, ProductQuantizer


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
            bypass_vq=model_cfg.get("bypass_vq", False),
        )

        self.lambda_pos = loss_cfg.get("lambda_pos", 1.0)
        self.lambda_team = loss_cfg.get("lambda_team", 1.0)
        # Soft constraint weights — small by default, tune if needed
        self.lambda_ball = loss_cfg.get("lambda_ball", 0.1)
        self.lambda_diversity = loss_cfg.get("lambda_diversity", 0.0)

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
        node_feats, z_e, z_q, tokens, vq_loss, utilization = (
            self.model(batch.x, batch.edge_index, batch.batch)
        )

        # Dense ground truth
        x_dense, mask = to_dense_batch(batch.x, batch.batch)  # (B, N_max, F)

        B, N_max, F_dim = x_dense.shape
        N_roles = self.num_roles

        # Pad or truncate to match decoder output size (num_roles)
        if N_max < N_roles:
            pad_x = torch.zeros(B, N_roles - N_max, F_dim, device=x_dense.device)
            x_target = torch.cat([x_dense, pad_x], dim=1)
            pad_mask = torch.zeros(B, N_roles - N_max, dtype=torch.bool, device=mask.device)
            node_mask = torch.cat([mask, pad_mask], dim=1)
        else:
            x_target = x_dense[:, :N_roles]
            node_mask = mask[:, :N_roles]

        # --- Position reconstruction loss (indices 0-1), MSE ---
        pos_pred = node_feats[..., :2]
        pos_gt = x_target[..., :2]
        pos_diff = (pos_pred - pos_gt).pow(2) * node_mask.unsqueeze(-1)
        pos_loss = pos_diff.sum() / node_mask.sum().clamp(min=1) / 2

        # --- Team reconstruction loss (index 2), BCE ---
        # Mask out ball node (is_ball=1 in feature index 3) — ball has no team
        is_ball_gt = x_target[..., 3]                          # (B, N_roles)
        player_mask = node_mask & (is_ball_gt < 0.5)           # exclude ball
        team_logits = node_feats[..., 2]                       # (B, N_roles)
        team_gt = x_target[..., 2]                             # (B, N_roles)
        team_diff = F.binary_cross_entropy_with_logits(
            team_logits, team_gt, reduction="none"
        )
        team_loss = (team_diff * player_mask).sum() / player_mask.sum().clamp(min=1)

        # --- Ball identity loss (index 3), BCE ---
        ball_logits = node_feats[..., 3]                       # (B, N_roles)
        ball_diff = F.binary_cross_entropy_with_logits(
            ball_logits, is_ball_gt, reduction="none"
        )
        ball_loss = (ball_diff * node_mask).sum() / node_mask.sum().clamp(min=1)

        # --- Diversity loss: penalize cosine similarity between slots ---
        diversity_loss = torch.tensor(0.0, device=z_e.device)
        if self.lambda_diversity > 0 and z_e.dim() == 3 and z_e.size(1) > 1:
            z_e_norm = F.normalize(z_e, dim=-1)             # (B, T, D)
            cos_sim = torch.bmm(z_e_norm, z_e_norm.transpose(1, 2))  # (B, T, T)
            T_slots = z_e.size(1)
            mask = ~torch.eye(T_slots, dtype=torch.bool, device=z_e.device)
            diversity_loss = cos_sim[:, mask].mean()         # mean off-diagonal cosine sim

        # --- Total loss ---
        total_loss = (
            self.lambda_pos * pos_loss
            + self.lambda_team * team_loss
            + self.lambda_ball * ball_loss
            + vq_loss
            + self.lambda_diversity * diversity_loss
        )

        metrics = {
            "loss": total_loss,
            "pos_loss": pos_loss,
            "team_loss": team_loss,
            "ball_loss": ball_loss,
            "vq_loss": vq_loss,
            "diversity_loss": diversity_loss,
            "codebook_utilization": utilization,
        }
        return total_loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            v_scalar = v.item() if isinstance(v, torch.Tensor) else v
            self.log(f"train/{k}", v_scalar, on_step=True, on_epoch=False,
                     prog_bar=False, batch_size=batch.num_graphs)
            self._accumulate(f"train/{k}", v_scalar)

        # Restart unused codes periodically
        if self.model.quantizer.use_ema and batch_idx % 100 == 0:
            with torch.no_grad():
                z_e = self.model.encoder(batch.x, batch.edge_index, batch.batch)
                if not isinstance(self.model.quantizer, ProductQuantizer):
                    # Flatten (B, T, D) -> (B*T, D) for shared codebook restart
                    if z_e.dim() == 3:
                        z_e = z_e.reshape(-1, z_e.size(-1))
            n_restarted = self.model.quantizer.restart_unused_codes(z_e)
            if n_restarted > 0:
                self.log("train/codes_restarted", float(n_restarted),
                         batch_size=batch.num_graphs)

        # GPU memory diagnostics
        if batch_idx % 200 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self.log("debug/gpu_allocated_mb", allocated, on_step=True, on_epoch=False)
            self.log("debug/gpu_reserved_mb", reserved, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            v_scalar = v.item() if isinstance(v, torch.Tensor) else v
            self.log(f"val/{k}", v_scalar, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch.num_graphs)
            self._accumulate(f"val/{k}", v_scalar)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            v_scalar = v.item() if isinstance(v, torch.Tensor) else v
            self.log(f"test/{k}", v_scalar, on_epoch=True, batch_size=batch.num_graphs)

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