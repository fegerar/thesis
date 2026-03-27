"""
Dataset for GPT training on shapegraph token sequences.

Loads per-match .pt files produced by the tokenizer pipeline, splits
sequences at GOAL tokens (treating GOAL as EOS), and creates fixed-length
context windows for next-token prediction within each segment.
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)

SHOT_TOKEN = 256


class TokenSequenceDataset(Dataset):
    """Sliding-window dataset over multiple token segments.

    Each segment is treated independently — windows never cross segment
    boundaries. This ensures GOAL tokens only appear as terminal tokens.

    Each sample is a pair (x, y) where:
        x = segment[i : i + context_length]
        y = segment[i + 1 : i + context_length + 1]
    """

    def __init__(self, segments: list[torch.Tensor], context_length: int):
        self.segments = segments
        self.context_length = context_length
        # Cumulative window count before each segment for O(log N) index lookup
        self._offsets = []
        total = 0
        for seg in segments:
            self._offsets.append(total)
            total += max(0, len(seg) - context_length)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Binary search for the segment this index falls into
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        seg_i = lo
        local_i = idx - self._offsets[seg_i]
        seg = self.segments[seg_i]
        x = seg[local_i : local_i + self.context_length]
        y = seg[local_i + 1 : local_i + self.context_length + 1]
        return x, y


def load_tokens(token_dir: str | Path) -> tuple[list[torch.Tensor], dict]:
    """Load token sequences and split at SHOT tokens.

    Each match is split into segments at shot positions.  Shot-ending
    segments include the SHOT token as their last element.  The final
    segment of each match (after the last shot, or the whole match if
    no shots) contains no SHOT token.

    Returns:
        (segments, info) where segments is a list of 1-D LongTensors
        and info contains metadata including per-segment xG values.
    """
    token_dir = Path(token_dir)
    segments = []
    segment_xg = []  # xG for shot-ending segments, None for non-shot segments
    total_matches = 0
    total_shots = 0
    total_goals = 0

    for pt_file in sorted(token_dir.glob("*_tokens.pt")):
        data = torch.load(pt_file, weights_only=False)
        tokens = data["tokens"]

        # Prefer shot_positions (new format), fall back to goal_positions (legacy)
        shot_positions = sorted(data.get("shot_positions", data.get("goal_positions", [])))
        shot_xg_list = data.get("shot_xg", [1.0] * len(shot_positions))

        total_matches += 1
        total_shots += len(shot_positions)
        total_goals += len(data.get("goal_positions", []))

        start = 0
        for i, pos in enumerate(shot_positions):
            seg = torch.tensor(tokens[start : pos + 1], dtype=torch.long)
            if len(seg) > 1:
                segments.append(seg)
                segment_xg.append(shot_xg_list[i] if i < len(shot_xg_list) else 0.0)
            start = pos + 1
        # Remaining tokens after last shot
        if start < len(tokens):
            seg = torch.tensor(tokens[start:], dtype=torch.long)
            if len(seg) > 1:
                segments.append(seg)
                segment_xg.append(None)

    info = {
        "num_matches": total_matches,
        "num_segments": len(segments),
        "total_tokens": sum(len(s) for s in segments),
        "total_shots": total_shots,
        "total_goals": total_goals,
        "segment_xg": segment_xg,
    }
    logger.info(
        "Loaded %d matches → %d segments, %d tokens (%d shots, %d goals)",
        total_matches, info["num_segments"], info["total_tokens"],
        total_shots, total_goals,
    )
    return segments, info


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

    Splits at the segment level so that no segment is shared across splits.
    """
    segments, info = load_tokens(token_dir)

    # Deterministic segment-level split
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(segments), generator=rng).tolist()
    n = len(segments)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_segs = [segments[i] for i in perm[:n_train]]
    val_segs = [segments[i] for i in perm[n_train : n_train + n_val]]
    test_segs = [segments[i] for i in perm[n_train + n_val :]]

    train_goals = sum(1 for s in train_segs if s[-1].item() == SHOT_TOKEN)
    logger.info(
        "Split: %d/%d/%d segments (train has %d goal-ending)",
        len(train_segs), len(val_segs), len(test_segs), train_goals,
    )

    train_ds = TokenSequenceDataset(train_segs, context_length)
    val_ds = TokenSequenceDataset(val_segs, context_length)
    test_ds = TokenSequenceDataset(test_segs, context_length)

    if goal_oversample_ratio > 1.0 and train_goals > 0:
        dataset_len = len(train_ds)
        weights = torch.ones(dataset_len)
        for seg_i, seg in enumerate(train_segs):
            if seg[-1].item() == SHOT_TOKEN:
                seg_windows = max(0, len(seg) - context_length)
                if seg_windows == 0:
                    continue
                seg_start = train_ds._offsets[seg_i]
                # Oversample the last context_length windows (buildup to goal)
                n_boost = min(seg_windows, context_length)
                lo = seg_start + seg_windows - n_boost
                hi = seg_start + seg_windows
                weights[lo:hi] = goal_oversample_ratio
        n_boosted = int((weights > 1.0).sum().item())
        logger.info(
            "Oversampling: %d goal-buildup windows (ratio=%.1f)",
            n_boosted, goal_oversample_ratio,
        )
        sampler = WeightedRandomSampler(
            weights, num_samples=dataset_len, replacement=True,
        )
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
