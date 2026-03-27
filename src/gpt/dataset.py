"""
Dataset for GPT training on shapegraph token sequences.

Loads per-match .pt files produced by the tokenizer pipeline and creates
fixed-length context windows for next-token prediction.
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


class TokenSequenceDataset(Dataset):
    """Sliding-window dataset over tokenized match sequences.

    Each sample is a pair (x, y) where:
        x = tokens[i : i + context_length]
        y = tokens[i + 1 : i + context_length + 1]
    """

    def __init__(self, tokens: torch.Tensor, context_length: int):
        """
        Args:
            tokens: 1-D tensor of all tokens (concatenated across matches)
            context_length: number of tokens in each training window
        """
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.context_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.context_length]
        y = self.tokens[idx + 1 : idx + self.context_length + 1]
        return x, y


def load_tokens(token_dir: str | Path) -> tuple[torch.Tensor, dict]:
    """Load all token sequences and concatenate with BOS/EOS separation.

    Each match sequence is kept contiguous — matches are separated by
    a boundary that the sliding window will naturally cross, providing
    the model with both within-match and cross-match transitions.

    Returns:
        (all_tokens, info) where all_tokens is a 1-D LongTensor and
        info contains metadata.
    """
    token_dir = Path(token_dir)
    all_tokens = []
    all_goal_positions = []  # absolute positions in the concatenated sequence
    total_matches = 0
    total_goals = 0
    offset = 0

    for pt_file in sorted(token_dir.glob("*_tokens.pt")):
        data = torch.load(pt_file, weights_only=False)
        tokens = data["tokens"]
        match_goal_positions = data.get("goal_positions", [])
        for pos in match_goal_positions:
            all_goal_positions.append(offset + pos)
        all_tokens.extend(tokens)
        total_matches += 1
        total_goals += len(match_goal_positions)
        offset += len(tokens)

    all_tokens = torch.tensor(all_tokens, dtype=torch.long)

    info = {
        "num_matches": total_matches,
        "total_tokens": len(all_tokens),
        "total_goals": total_goals,
        "goal_positions": all_goal_positions,
    }
    logger.info(
        "Loaded %d matches, %d tokens (%d goals)",
        total_matches, len(all_tokens), total_goals,
    )
    return all_tokens, info


def build_dataloaders(
    token_dir: str | Path,
    context_length: int,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    goal_oversample_ratio: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders from tokenized match data.

    Splits at the token level (not match level) since matches are
    concatenated into one long sequence.
    """
    all_tokens, info = load_tokens(token_dir)
    n = len(all_tokens)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_tokens = all_tokens[:n_train]
    val_tokens = all_tokens[n_train : n_train + n_val]
    test_tokens = all_tokens[n_train + n_val :]

    logger.info(
        "Split: train=%d, val=%d, test=%d tokens",
        len(train_tokens), len(val_tokens), len(test_tokens),
    )

    train_ds = TokenSequenceDataset(train_tokens, context_length)
    val_ds = TokenSequenceDataset(val_tokens, context_length)
    test_ds = TokenSequenceDataset(test_tokens, context_length)

    if goal_oversample_ratio > 1.0:
        train_goal_positions = [p for p in info["goal_positions"] if p < n_train]
        dataset_len = len(train_ds)
        weights = torch.ones(dataset_len)
        for g in train_goal_positions:
            # all windows [i, i+context_length) that contain position g
            lo = max(0, g - context_length + 1)
            hi = min(dataset_len - 1, g)
            weights[lo : hi + 1] = goal_oversample_ratio
        n_goal_windows = int((weights > 1.0).sum().item())
        logger.info(
            "Oversampling: %d goal-containing windows (ratio=%.1f)",
            n_goal_windows, goal_oversample_ratio,
        )
        sampler = WeightedRandomSampler(weights, num_samples=dataset_len, replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, persistent_workers=num_workers > 0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
