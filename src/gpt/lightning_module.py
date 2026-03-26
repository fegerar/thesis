"""
PyTorch Lightning module for ShapegraphGPT training.

Handles:
    - Next-token cross-entropy loss
    - Perplexity tracking
    - WandB + CSV logging
    - LR scheduling with warmup
"""

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import lightning as L

from .model import ShapegraphGPT


class GPTLightningModule(L.LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        training_cfg: dict,
        logging_cfg: dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ShapegraphGPT(
            vocab_size=model_cfg["vocab_size"],
            context_length=model_cfg["context_length"],
            embed_dim=model_cfg["embed_dim"],
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg.get("dropout", 0.1),
        )

        self.lr = training_cfg["learning_rate"]
        self.weight_decay = training_cfg.get("weight_decay", 1e-4)
        self.lr_scheduler_type = training_cfg.get("lr_scheduler", "cosine")
        self.warmup_epochs = training_cfg.get("warmup_epochs", 5)
        self.max_epochs = training_cfg["max_epochs"]

        # CSV logging
        self.csv_dir = Path(logging_cfg.get("csv_dir", "logs/gpt"))
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.csv_dir / "metrics.csv"
        self._csv_initialized = False
        self._epoch_metrics: dict[str, list[float]] = {}

    def forward(self, idx):
        return self.model(idx)

    def _compute_loss(self, batch):
        x, y = batch
        logits = self.model(x)  # (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        perplexity = torch.exp(loss).item()
        return loss, {"loss": loss, "perplexity": perplexity}

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            v_scalar = v.item() if isinstance(v, torch.Tensor) else v
            self.log(
                f"train/{k}", v_scalar,
                on_step=True, on_epoch=False,
                prog_bar=(k == "loss"), batch_size=batch[0].size(0),
            )
            self._accumulate(f"train/{k}", v_scalar)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            v_scalar = v.item() if isinstance(v, torch.Tensor) else v
            self.log(
                f"val/{k}", v_scalar,
                on_step=False, on_epoch=True,
                prog_bar=(k == "loss"), batch_size=batch[0].size(0),
            )
            self._accumulate(f"val/{k}", v_scalar)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, metrics = self._compute_loss(batch)
        for k, v in metrics.items():
            v_scalar = v.item() if isinstance(v, torch.Tensor) else v
            self.log(
                f"test/{k}", v_scalar,
                on_epoch=True, batch_size=batch[0].size(0),
            )

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
        # Separate weight-decay and no-decay parameter groups
        decay = set()
        no_decay = set()
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or "ln" in mn or "emb" in mn:
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)

        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, betas=(0.9, 0.95)
        )

        if self.lr_scheduler_type == "none":
            return optimizer

        if self.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs - self.warmup_epochs
            )
        else:
            return optimizer

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, scheduler],
                milestones=[self.warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
