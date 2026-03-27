"""
Train ShapegraphGPT for next-token prediction on tokenized match sequences.

Usage:
    python src/train_gpt.py --config config/gpt_default.yaml
"""

import argparse
from pathlib import Path

import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from gpt import GPTLightningModule, build_dataloaders


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train ShapegraphGPT")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Build dataloaders
    data_cfg = cfg["data"]
    train_loader, val_loader, test_loader = build_dataloaders(
        token_dir=data_cfg["token_dir"],
        context_length=cfg["model"]["context_length"],
        batch_size=cfg["training"]["batch_size"],
        train_ratio=data_cfg.get("train_ratio", 0.8),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        num_workers=data_cfg.get("num_workers", 4),
        seed=data_cfg.get("seed", 42),
        goal_oversample_ratio=cfg["training"].get("goal_oversample_ratio", 1.0),
    )

    # Build model
    model = GPTLightningModule(
        model_cfg=cfg["model"],
        training_cfg=cfg["training"],
        logging_cfg=cfg["logging"],
    )

    # Loggers
    log_cfg = cfg["logging"]
    loggers = []

    wandb_logger = WandbLogger(
        project=log_cfg["project_name"],
        name=log_cfg.get("run_name"),
        config=cfg,
        save_dir=log_cfg.get("csv_dir", "logs/gpt"),
    )
    loggers.append(wandb_logger)

    csv_logger = CSVLogger(
        save_dir=log_cfg.get("csv_dir", "logs/gpt"),
        name="csv_logs",
    )
    loggers.append(csv_logger)

    # Callbacks
    ckpt_cfg = cfg["checkpointing"]
    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor=ckpt_cfg["monitor"],
            mode=ckpt_cfg["mode"],
            save_top_k=ckpt_cfg["save_top_k"],
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch-{epoch:03d}",
            every_n_epochs=ckpt_cfg["save_every_n_epochs"],
            save_top_k=-1,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    train_cfg = cfg["training"]
    trainer = L.Trainer(
        max_epochs=train_cfg["max_epochs"],
        limit_train_batches=train_cfg.get("limit_train_batches"),
        limit_val_batches=train_cfg.get("limit_val_batches"),
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=log_cfg.get("log_every_n_steps", 50),
        deterministic=True,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=False,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
