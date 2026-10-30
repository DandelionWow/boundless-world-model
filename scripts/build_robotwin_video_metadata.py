#!/usr/bin/env python3
"""Add source-video paths to precomputed RoboTwin window metadata."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing latent-window JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL with both video and latent paths.")
    parser.add_argument("--camera", default="head", help="RoboTwin observation camera name.")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-history-frames", type=int, default=9)
    return parser.parse_args()


def video_frame_count(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1 or not str(streams[0].get("nb_frames", "")).isdigit():
        raise ValueError(f"Cannot read exact video frame count: {path}")
    return int(streams[0]["nb_frames"])


def source_video_path(action_path, camera):
    if action_path.suffix != ".parquet" or action_path.parent.parent.name != "data":
        raise ValueError(f"Unexpected RoboTwin action path: {action_path}")
    episode_root = action_path.parents[2]
    return (
        episode_root / "videos" / action_path.parent.name
        / f"observation.images.{camera}" / f"{action_path.stem}.mp4"
    )


def main():
    args = parse_args()
    if args.num_frames <= args.num_history_frames or args.num_history_frames < 1:
        raise ValueError("Require 1 <= num-history-frames < num-frames")
    if args.num_frames % 4 != 1 or (args.num_history_frames - 1) % 4 != 0:
        raise ValueError("Wan windows must use 4n+1 frames and 4n+1 history frames")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise NotADirectoryError(args.output.parent)

    dataset_root = args.input.parent.parent
    future_len = args.num_frames - args.num_history_frames
    verified_episodes = {}
    output_rows = []

    with args.input.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            action_path = Path(row["action"])
            latent_path = Path(row["latents"])
            if not latent_path.is_absolute():
                latent_path = dataset_root / latent_path
            video_path = source_video_path(action_path, args.camera)
            raw_length = int(row["raw_length"])
            future_start = int(row["start_frame"])
            future_end = future_start + future_len - 1
            last_encoded_frame = 4 * ((raw_length - 1) // 4)

            if not action_path.is_absolute():
                raise ValueError(f"Line {line_number}: action path must be absolute")
            if row["task"] != action_path.parents[2].parent.name:
                raise ValueError(f"Line {line_number}: task does not match action path")
            if action_path.stem != f"episode_{int(row['episode_index']):06d}":
                raise ValueError(f"Line {line_number}: episode index does not match action path")
            if int(row["end_frame"]) != raw_length - 1:
                raise ValueError(f"Line {line_number}: end_frame does not match raw_length")
            if future_start < 1 or (future_start - 1) % 4 != 0:
                raise ValueError(f"Line {line_number}: future start is not Wan-aligned")
            if future_end > last_encoded_frame:
                raise ValueError(f"Line {line_number}: future extends beyond precomputed latents")

            if action_path not in verified_episodes:
                for path in (action_path, latent_path, video_path):
                    if not path.is_file():
                        raise FileNotFoundError(f"Line {line_number}: {path}")
                action_rows = pq.ParquetFile(action_path).metadata.num_rows
                video_frames = video_frame_count(video_path)
                if action_rows != raw_length or video_frames != raw_length:
                    raise ValueError(
                        f"Line {line_number}: expected {raw_length} frames, "
                        f"got action={action_rows}, video={video_frames}"
                    )
                verified_episodes[action_path] = (raw_length, latent_path)
            elif verified_episodes[action_path] != (raw_length, latent_path):
                raise ValueError(f"Line {line_number}: inconsistent episode metadata for {action_path}")

            if "video" in row and Path(row["video"]) != video_path:
                raise ValueError(f"Line {line_number}: existing video path differs from source")
            row["video"] = str(video_path)
            output_rows.append(row)

    if not output_rows:
        raise ValueError(f"No metadata rows found in {args.input}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent,
            prefix=f".{args.output.name}.", suffix=".tmp", delete=False,
        ) as target:
            temp_path = Path(target.name)
            for row in output_rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.link(temp_path, args.output)  # Fail if another process created the target.
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"Output: {args.output}")
    print(f"Windows: {len(output_rows)}")
    print(f"Verified episodes: {len(verified_episodes)}")


if __name__ == "__main__":
    main()
