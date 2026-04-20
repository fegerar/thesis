"""Generate an SRT subtitle file with per-frame cluster labels.

Load it into mpv alongside the match video:
    mpv game.mp4 --sub-file=clusters.srt

Adjust timing with z/Z keys (subtitle delay ±100ms) if needed.

Usage:
    python -m src.cluster_subtitles --clusters team_clusters/team_clusters.json --match DFL-MAT-J03WMX --annotated data_annotated --output clusters.srt --kickoff-at 32.5

    --kickoff-at: seconds into the video when the game actually kicks off
                  (e.g. 32.5 means the first DFL frame maps to 0:00:32.500
                  in the video). This accounts for pre-match footage.

    # with split-possession directory
    python -m src.cluster_subtitles \
        --clusters team_clusters_poss \
        --match DFL-MAT-J03WMX \
        --annotated data_annotated \
        --output clusters.srt \
        --kickoff-at 32.5

    # second half only (if your video is a separate file per half)
    --half second --kickoff-at 10.0
"""

import argparse
import json
import os
from datetime import datetime, timedelta


def _load_samples(clusters_path):
    samples = []
    if os.path.isdir(clusters_path):
        for tag in ("in_possession", "out_of_possession"):
            json_path = os.path.join(clusters_path, tag, "team_clusters.json")
            if not os.path.isfile(json_path):
                continue
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            for s in data["samples"]:
                s.setdefault("possession", tag)
                samples.append(s)
    else:
        with open(clusters_path, encoding="utf-8") as f:
            data = json.load(f)
        samples = data["samples"]
    return samples


def _load_annotated_frames(annotated_path, match_id):
    if os.path.isdir(annotated_path):
        target = os.path.join(annotated_path, f"{match_id}.json")
        if os.path.isfile(target):
            with open(target, encoding="utf-8") as f:
                return json.load(f)
        for name in sorted(os.listdir(annotated_path)):
            if match_id in name and name.endswith(".json"):
                with open(os.path.join(annotated_path, name),
                          encoding="utf-8") as f:
                    return json.load(f)
        raise FileNotFoundError(
            f"no annotated JSON for {match_id} in {annotated_path}")
    with open(annotated_path, encoding="utf-8") as f:
        return json.load(f)


def _parse_ts(ts_str):
    if not ts_str:
        return None
    ts_str = ts_str.replace("+00:00", "+0000").replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def _fmt_srt_time(td):
    total_ms = int(td.total_seconds() * 1000)
    if total_ms < 0:
        total_ms = 0
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _cluster_label(entry):
    if not entry:
        return "—"
    cluster, poss = entry
    label = f"c{cluster}"
    if poss and poss not in ("all", ""):
        short = "IP" if "in" in poss else "OOP"
        label += f" ({short})"
    return label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", required=True)
    p.add_argument("--annotated", required=True,
                   help="annotated JSON dir or file (needed for timestamps)")
    p.add_argument("--match", required=True)
    p.add_argument("--output", default="clusters.srt")
    p.add_argument("--kickoff-at", type=float, default=0.0,
                   help="seconds into the video where the game starts "
                        "(first DFL frame). E.g. 32.5 if there is 32.5s "
                        "of pre-match footage before kickoff.")
    p.add_argument("--half", choices=("first", "second", "both"),
                   default="both")
    p.add_argument("--second-half-kickoff-at", type=float, default=None,
                   help="if the video has both halves, seconds into the "
                        "video where the second half kicks off. If omitted "
                        "and --half=both, second half timestamps continue "
                        "from where first half left off.")
    args = p.parse_args()

    samples = _load_samples(args.clusters)
    lookup = {}
    for s in samples:
        if s.get("match_id") != args.match:
            continue
        key = (s["phase"], s["frame_id"], s["team"])
        poss = s.get("possession", "")
        lookup[key] = (s["cluster"], poss)

    ann_frames = _load_annotated_frames(args.annotated, args.match)
    if args.half != "both":
        target = "firstHalf" if args.half == "first" else "secondHalf"
        ann_frames = [f for f in ann_frames if f.get("phase") == target]

    ann_frames.sort(key=lambda f: (
        0 if f.get("phase") == "firstHalf" else 1,
        f.get("frame_id", 0),
    ))
    if not ann_frames:
        print("no annotated frames found")
        return

    timestamps = []
    for fr in ann_frames:
        timestamps.append(_parse_ts(fr.get("timestamp")))

    # find the first timestamp per half to use as time origin
    t0_first, t0_second = None, None
    for fr, ts in zip(ann_frames, timestamps):
        if ts is None:
            continue
        if fr.get("phase") == "firstHalf" and t0_first is None:
            t0_first = ts
        if fr.get("phase") == "secondHalf" and t0_second is None:
            t0_second = ts

    kickoff = timedelta(seconds=args.kickoff_at)

    srt_entries = []
    idx = 0
    prev_text = None

    for i, fr in enumerate(ann_frames):
        phase = fr.get("phase")
        fid = fr.get("frame_id")
        ts = timestamps[i]
        if ts is None:
            continue

        # compute video time for this frame
        if phase == "firstHalf" and t0_first:
            game_elapsed = ts - t0_first
            video_time = kickoff + game_elapsed
        elif phase == "secondHalf" and t0_second:
            game_elapsed = ts - t0_second
            if args.second_half_kickoff_at is not None:
                video_time = timedelta(
                    seconds=args.second_half_kickoff_at) + game_elapsed
            elif t0_first:
                # estimate: second half video position = kickoff + first half
                # duration + halftime gap. User should set
                # --second-half-kickoff-at for accuracy.
                first_half_dur = (t0_second - t0_first) if t0_first else timedelta(0)
                video_time = kickoff + first_half_dur + game_elapsed
            else:
                video_time = kickoff + game_elapsed
        else:
            continue

        h_entry = lookup.get((phase, fid, "home"))
        g_entry = lookup.get((phase, fid, "guest"))
        h_label = _cluster_label(h_entry)
        g_label = _cluster_label(g_entry)

        poss = fr.get("possession")
        ball_str = ""
        if poss:
            ball_icon = "◄" if poss == "home" else "►"
            ball_str = f"  {ball_icon} ball: {poss}"

        text = f"HOME: {h_label}    GUEST: {g_label}{ball_str}"

        # skip duplicate consecutive subtitles to keep the file compact
        if text == prev_text:
            continue
        prev_text = text

        start = video_time
        # end: find next different subtitle or add a default duration
        if i + 1 < len(ann_frames) and timestamps[i + 1] is not None:
            next_ts = timestamps[i + 1]
            if phase == "firstHalf" and t0_first:
                end = kickoff + (next_ts - t0_first)
            elif phase == "secondHalf" and t0_second:
                if args.second_half_kickoff_at is not None:
                    end = timedelta(
                        seconds=args.second_half_kickoff_at) + (next_ts - t0_second)
                elif t0_first:
                    first_half_dur = t0_second - t0_first
                    end = kickoff + first_half_dur + (next_ts - t0_second)
                else:
                    end = kickoff + (next_ts - t0_second)
            else:
                end = start + timedelta(milliseconds=500)
        else:
            end = start + timedelta(milliseconds=500)

        idx += 1
        display_text = f"{text}\n{phase} f{fid}"
        srt_entries.append(
            f"{idx}\n"
            f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}\n"
            f"{display_text}\n"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries))

    print(f"[cluster-subs] wrote {idx} subtitles to {args.output}")
    print(f"[cluster-subs] play with:")
    print(f"    mpv game.mp4 --sub-file={args.output}")
    print(f"[cluster-subs] fine-tune sync: z/Z keys in mpv (±100ms)")


if __name__ == "__main__":
    main()
