"""
Command-line interface for generating shape graphs from soccer tracking data.
"""

import argparse
import sys
from .frame2sg import generate_shapegraphs
from .utils import save_shapegraphs, save_shapegraphs_json

def main():
    """Main function to parse arguments and generate shape graphs."""
    parser = argparse.ArgumentParser(
        description="Generate shape graphs from Bassek et al. soccer tracking data")
    parser.add_argument("--match-info", type=str,
                        help="Path to match information XML file")
    parser.add_argument("--position-data", type=str,
                        help="Path to position data XML file")
    parser.add_argument("--event-data", type=str, default=None,
                        help="Path to event data XML file (optional)")
    parser.add_argument("--output", type=str, default="shapegraphs.pkl",
                        help="Output file path (.pkl or .jsonl)")
    parser.add_argument("--frames", type=str, default=None,
                        help="Frame range as START:END (e.g. 10000:10100)")
    parser.add_argument("--all-frames", action="store_true",
                        help="Process all frames (ignore ball status)")

    args = parser.parse_args()

    if not args.match_info or not args.position_data:
        print("Error: --match-info and --position-data are required.")
        print("Use --demo for a demonstration with synthetic data.")
        parser.print_help()
        sys.exit(1)

    frame_range = None
    if args.frames:
        parts = args.frames.split(":")
        frame_range = (int(parts[0]), int(parts[1]))

    results = generate_shapegraphs(
        match_info_path=args.match_info,
        position_data_path=args.position_data,
        frame_range=frame_range,
        ball_in_play_only=not args.all_frames,
    )

    if args.output.endswith(".jsonl") or args.output.endswith(".json"):
        save_shapegraphs_json(results, args.output)
    else:
        save_shapegraphs(results, args.output)


if __name__ == "__main__":
    main()
