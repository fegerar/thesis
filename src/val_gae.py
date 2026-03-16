"""
Validate a trained HQA-GAE checkpoint on shapegraph data.

Runs link prediction, node classification, and clustering evaluation.

Usage:
    python src/val_gae.py --checkpoint checkpoints/run_.../last.ckpt
    python src/val_gae.py --checkpoint checkpoints/run_.../last.ckpt --device cpu
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
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

# Reuse data helpers from training script
from train_gae import (                                 # noqa: E402
    build_role_vocab,
    shapegraph_to_pyg,
    _DatasetStub,
)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_link_prediction(model, dataloader, device):
    """Compute link prediction AUC / AP over all eval batches."""
    all_auc, all_ap = [], []

    for batch in tqdm(dataloader, desc="Link prediction"):
        batch = batch.to(device)
        emb = model.get_embedding(batch.x.clamp(0, 1), batch.edge_index)

        result = test_link_prediction(
            emb,
            pos_edges=batch.pos_edge_label_index,
            neg_edges=batch.neg_edge_label_index,
            batch_size=65536,
        )
        all_auc.append(result["AUC"])
        all_ap.append(result["AP"])

    return {
        "AUC_mean": np.mean(all_auc),
        "AUC_std": np.std(all_auc),
        "AP_mean": np.mean(all_ap),
        "AP_std": np.std(all_ap),
    }


@torch.no_grad()
def evaluate_node_classification(model, dataloader, device):
    """Linear-probe node classification on embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning

    # Collect all embeddings and labels
    all_emb, all_y = [], []
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        batch = batch.to(device)
        emb = model.get_embedding(batch.x.clamp(0, 1), batch.edge_index)
        all_emb.append(emb.cpu())
        if hasattr(batch, "y") and batch.y is not None:
            all_y.append(batch.y.cpu())

    if not all_y:
        return None

    X = torch.cat(all_emb).numpy()
    y = torch.cat(all_y).numpy()

    # 5-fold cross-validated logistic regression
    accs, f1_mics, f1_macs = [], [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto", n_jobs=-1)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])

            accs.append(accuracy_score(y[test_idx], preds))
            f1_mics.append(f1_score(y[test_idx], preds, average="micro"))
            f1_macs.append(f1_score(y[test_idx], preds, average="macro"))

    return {
        "accuracy": f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
        "f1_micro": f"{np.mean(f1_mics):.4f} ± {np.std(f1_mics):.4f}",
        "f1_macro": f"{np.mean(f1_macs):.4f} ± {np.std(f1_macs):.4f}",
    }


@torch.no_grad()
def evaluate_clustering(model, dataloader, device):
    """KMeans clustering evaluation on embeddings."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        silhouette_score,
    )

    all_emb, all_y = [], []
    for batch in tqdm(dataloader, desc="Extracting embeddings (cluster)"):
        batch = batch.to(device)
        emb = model.get_embedding(batch.x.clamp(0, 1), batch.edge_index)
        all_emb.append(emb.cpu())
        if hasattr(batch, "y") and batch.y is not None:
            all_y.append(batch.y.cpu())

    if not all_y:
        return None

    X = torch.cat(all_emb).numpy()
    y = torch.cat(all_y).numpy()
    n_clusters = int(y.max()) + 1

    nmis, aris, scs = [], [], []
    for seed in range(5):
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        y_pred = km.fit_predict(X)
        nmis.append(normalized_mutual_info_score(y, y_pred))
        aris.append(adjusted_rand_score(y, y_pred))
        scs.append(silhouette_score(X, y_pred))

    return {
        "NMI": f"{np.mean(nmis):.4f} ± {np.std(nmis):.4f}",
        "ARI": f"{np.mean(aris):.4f} ± {np.std(aris):.4f}",
        "Silhouette": f"{np.mean(scs):.4f} ± {np.std(scs):.4f}",
    }


@torch.no_grad()
def evaluate_reconstruction(model, dataloader, device):
    """Compute average reconstruction loss over batches."""
    total_x_loss, total_edge_loss, n_batches = 0.0, 0.0, 0
    for batch in tqdm(dataloader, desc="Reconstruction"):
        batch = batch.to(device)
        x = batch.x.clamp(0, 1)
        result = model(x, batch.edge_index)
        loss, loss_log = model.loss(x=x, edge_index=batch.edge_index, **result)
        total_x_loss += loss_log.get("x_rec_loss", 0.0)
        total_edge_loss += loss_log.get("edge_rec_loss", 0.0)
        n_batches += 1

    return {
        "x_rec_loss": total_x_loss / n_batches,
        "edge_rec_loss": total_edge_loss / n_batches,
        "total_rec_loss": (total_x_loss + total_edge_loss) / n_batches,
    }


@torch.no_grad()
def evaluate_codebook(model, dataloader, device):
    """Analyze codebook utilization."""
    all_indices = []
    for batch in tqdm(dataloader, desc="Codebook analysis"):
        batch = batch.to(device)
        x = batch.x.clamp(0, 1)
        _, indices = model.get_embedding(x, batch.edge_index, indices=True)
        all_indices.append(indices.cpu())

    all_indices = torch.cat(all_indices)
    unique_codes = torch.unique(all_indices)
    codebook_size = model.vector_quantizer.codebook_size

    return {
        "codebook_size": codebook_size,
        "used_codes": len(unique_codes),
        "utilization": f"{len(unique_codes) / codebook_size * 100:.1f}%",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (eval-only: all graphs get edge splits)
# ═══════════════════════════════════════════════════════════════════════════

def load_eval_data(all_games, cfg, role_vocab, device):
    """Load all shapegraphs and prepare them for evaluation."""
    graph_type = cfg.data.get("graph_type", "original")
    all_data = []

    total_frames = sum(len(game) for game in all_games)
    for game_idx, game in enumerate(all_games):
        for frame_key in tqdm(sorted(game.keys()),
                              desc=f"Game {game_idx+1}/{len(all_games)}",
                              leave=False):
            G = game[frame_key][graph_type]
            data = shapegraph_to_pyg(G, cfg, role_vocab)
            if data is not None:
                data = data.to(device)
                all_data.append(data)

    print(f"Loaded {len(all_data)} valid shapegraphs (total frames: {total_frames})")
    return all_data


def prepare_eval_graphs(graphs):
    """Add pos/neg edge splits for link prediction evaluation."""
    eval_data = []
    skipped = 0
    for data in graphs:
        try:
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
        except Exception:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} graphs (too few edges for splitting)")
    return eval_data


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Validate HQA-GAE checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: from config)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--skip-cluster", action="store_true",
                        help="Skip clustering evaluation")
    parser.add_argument("--skip-reconstruction", action="store_true",
                        help="Skip reconstruction loss evaluation")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load config from checkpoint directory
    # ------------------------------------------------------------------
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config.yml")
    if not os.path.exists(config_path):
        print(f"ERROR: config.yml not found in {ckpt_dir}")
        sys.exit(1)

    cfg = parse_yaml(config_path)
    set_random_seed(cfg.train.seed)

    device_str = args.device or cfg.train.get("device", "cuda")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print(f"Loading shapegraphs from {cfg.data.pickle_path} ...")
    with open(cfg.data.pickle_path, "rb") as f:
        all_games = pickle.load(f)

    graph_type = cfg.data.get("graph_type", "original")
    role_vocab = build_role_vocab(all_games, graph_type)
    print(f"Role vocabulary ({len(role_vocab)} roles): {role_vocab}")

    all_data = load_eval_data(all_games, cfg, role_vocab, device)
    del all_games

    num_features = all_data[0].x.size(1)

    # Prepare link-prediction splits for all graphs
    eval_data = prepare_eval_graphs(all_data)
    eval_loader = PyGDataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Also keep raw graphs (no edge splits) for reconstruction eval
    raw_loader = PyGDataLoader(all_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 3. Load model from checkpoint
    # ------------------------------------------------------------------
    dataset_stub = _DatasetStub(num_features=num_features)
    hqa_gae_model = create_gae(cfg, dataset_stub)

    lightning_model = GAE.load_from_checkpoint(
        args.checkpoint,
        model=hqa_gae_model,
        optimizer=cfg.optimizer,
        scheduler=getattr(cfg, "scheduler", None),
    )
    model = lightning_model.model
    model.to(device)
    model.eval()

    print(f"\nLoaded checkpoint: {args.checkpoint}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # 4. Run evaluations
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  HQA-GAE Validation")
    print(f"{'='*60}\n")

    # Link Prediction
    print("── Link Prediction ──")
    lp_results = evaluate_link_prediction(model, eval_loader, device)
    print(f"  AUC: {lp_results['AUC_mean']:.4f} ± {lp_results['AUC_std']:.4f}")
    print(f"  AP:  {lp_results['AP_mean']:.4f} ± {lp_results['AP_std']:.4f}")

    # Node Classification
    print("\n── Node Classification (5-fold LogReg) ──")
    nc_results = evaluate_node_classification(model, eval_loader, device)
    if nc_results:
        print(f"  Accuracy: {nc_results['accuracy']}")
        print(f"  F1-micro: {nc_results['f1_micro']}")
        print(f"  F1-macro: {nc_results['f1_macro']}")
    else:
        print("  Skipped (no labels available)")

    # Clustering
    if not args.skip_cluster:
        print("\n── Clustering (KMeans, 5 seeds) ──")
        cl_results = evaluate_clustering(model, eval_loader, device)
        if cl_results:
            print(f"  NMI:        {cl_results['NMI']}")
            print(f"  ARI:        {cl_results['ARI']}")
            print(f"  Silhouette: {cl_results['Silhouette']}")
        else:
            print("  Skipped (no labels available)")

    # Reconstruction
    if not args.skip_reconstruction:
        print("\n── Reconstruction Loss ──")
        rec_results = evaluate_reconstruction(model, raw_loader, device)
        print(f"  Node rec loss: {rec_results['x_rec_loss']:.6f}")
        print(f"  Edge rec loss: {rec_results['edge_rec_loss']:.6f}")
        print(f"  Total:         {rec_results['total_rec_loss']:.6f}")

    # Codebook utilization
    print("\n── Codebook Utilization ──")
    cb_results = evaluate_codebook(model, eval_loader, device)
    print(f"  Codebook size:  {cb_results['codebook_size']}")
    print(f"  Used codes:     {cb_results['used_codes']}")
    print(f"  Utilization:    {cb_results['utilization']}")

    print(f"\n{'='*60}")
    print("  Validation complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
