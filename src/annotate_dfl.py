"""CLI: convert DFL (Bassek et al.) tracking XML into annotated per-frame JSON.

Each frame gets per-player tactical roles (Brandes et al. 2025 shape graph,
re-implemented from scratch) and a joint [col, row] field-zone coordinate
computed from a single shape graph over all players + ball.

Example:
    python -m src.annotate_dfl --data-dir data --out-dir data_annotated
    python -m src.annotate_dfl --match DFL-MAT-J03WMX --stride 25 --max-frames 500
"""

import argparse
import os

from src.annotate.dfl import annotate_match


def _match_ids(data_dir):
    ids = set()
    for name in os.listdir(data_dir):
        if name.endswith(".xml") and "DFL-MAT" in name:
            ids.add(name.rsplit("_", 1)[-1].replace(".xml", ""))
    return sorted(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="data_annotated")
    p.add_argument("--match", default=None, help="single match id (e.g. DFL-MAT-J03WMX)")
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--zone-levels", type=int, default=5,
                   help="odd integer >=3; granularity of the joint zone grid")
    args = p.parse_args()
    if args.zone_levels < 3 or args.zone_levels % 2 == 0:
        raise SystemExit("--zone-levels must be an odd integer >= 3")

    ids = [args.match] if args.match else _match_ids(args.data_dir)
    for mid in ids:
        out = os.path.join(args.out_dir, f"{mid}.json")
        path, n = annotate_match(
            args.data_dir, mid, out,
            frame_stride=args.stride, max_frames=args.max_frames,
            zone_levels=args.zone_levels,
        )
        print(f"{mid}: wrote {n} frames -> {path}")


if __name__ == "__main__":
    main()
