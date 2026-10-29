#!/usr/bin/env python3
"""Precompute full-episode Wan 2.2 VAE latents for RoboTwin.

Launch one process per GPU with Accelerate. Outputs are atomic and validated,
so rerunning the command safely resumes an interrupted conversion.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_video_action.data.operators import eef_abs_to_pose14


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--vae_path", type=Path, required=True)
    parser.add_argument("--subset", default="aloha-agilex_clean_50")
    parser.add_argument("--camera", default="head")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--train_episodes_per_task", type=int, default=40)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_history_frames", type=int, default=9)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N episodes globally; 0 means all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    return parser.parse_args()


def discover_episodes(args):
    plans = []
    meta_pattern = f"*/{args.subset}/meta/episodes.jsonl"
    for meta_path in sorted(args.source_root.glob(meta_pattern)):
        task = meta_path.parent.parent.parent.name
        dataset_root = meta_path.parent.parent
        with meta_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                episode_index = int(row["episode_index"])
                filename = f"episode_{episode_index:06d}"
                video_path = dataset_root / "videos" / "chunk-000" / f"observation.images.{args.camera}" / f"{filename}.mp4"
                action_path = dataset_root / "data" / "chunk-000" / f"{filename}.parquet"
                output_path = args.output_root / "latents" / task / f"{filename}.pt"
                if not video_path.is_file() or not action_path.is_file():
                    raise FileNotFoundError(f"Missing source for {task}/{filename}: {video_path}, {action_path}")
                plans.append({
                    "task": task,
                    "episode_index": episode_index,
                    "instruction": row.get("instruction") or (row.get("tasks") or [task])[0],
                    "raw_length": int(row["length"]),
                    "video_path": video_path,
                    "action_path": action_path,
                    "output_path": output_path,
                })
    if args.limit > 0:
        plans = plans[: args.limit]
    if not plans:
        raise RuntimeError(f"No episodes found under {args.source_root} with pattern {meta_pattern}")
    return plans


def expected_latent_shape(raw_length, height, width):
    return (1, 48, (int(raw_length) + 3) // 4, int(height) // 16, int(width) // 16)


def valid_existing(path, expected_shape):
    if not path.is_file():
        return False
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        return isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == tuple(expected_shape)
    except Exception:
        return False


def load_video(path, raw_length, height, width):
    reader = imageio.get_reader(str(path))
    try:
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    if len(frames) != raw_length:
        raise ValueError(f"Frame-count mismatch at {path}: metadata={raw_length}, decoded={len(frames)}")
    array = np.stack(frames)
    if array.shape[1:] != (height, width, 3):
        raise ValueError(f"Unexpected video shape at {path}: {array.shape}, expected (T,{height},{width},3)")
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    return tensor.to(dtype=torch.bfloat16).div_(127.5).sub_(1.0).unsqueeze(0)


def convert_one(plan, vae, device, args):
    output_path = plan["output_path"]
    shape = expected_latent_shape(plan["raw_length"], args.height, args.width)
    if not args.overwrite and valid_existing(output_path, shape):
        return "skipped", shape

    video = load_video(plan["video_path"], plan["raw_length"], args.height, args.width)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        latent = vae.encode(video, device=device, tiled=args.tiled)
    latent = latent.to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(latent.shape) != shape:
        raise ValueError(f"Latent shape mismatch for {plan['video_path']}: got {tuple(latent.shape)}, expected {shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f".rank{os.environ.get('RANK', '0')}.tmp")
    torch.save(latent, temporary_path)
    os.replace(temporary_path, output_path)
    del video, latent
    torch.cuda.empty_cache()
    return "written", shape


def build_metadata(plans, args):
    meta_dir = args.output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = [], []
    future_len = args.num_frames - args.num_history_frames
    if future_len <= 0:
        raise ValueError("num_frames must be greater than num_history_frames")

    for plan in plans:
        relative_latent = plan["output_path"].relative_to(args.output_root).as_posix()
        max_future_start = plan["raw_length"] - future_len
        split_rows = train_rows if plan["episode_index"] < args.train_episodes_per_task else val_rows
        for future_start in range(1, max_future_start + 1, 4):
            split_rows.append({
                "task": plan["task"],
                "episode_index": plan["episode_index"],
                "instruction": plan["instruction"],
                "raw_length": plan["raw_length"],
                "start_frame": future_start,
                "end_frame": plan["raw_length"] - 1,
                "latents": relative_latent,
                "action": str(plan["action_path"]),
            })

    for split, rows in (("train", train_rows), ("val", val_rows)):
        path = meta_dir / f"episodes_{split}.jsonl"
        temporary_path = path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)

    train_actions = []
    for plan in plans:
        if plan["episode_index"] >= args.train_episodes_per_task:
            continue
        table = pq.read_table(plan["action_path"], columns=["eef_abs"])
        train_actions.append(eef_abs_to_pose14(np.asarray(table["eef_abs"].to_pylist(), dtype=np.float32)))
    actions = np.concatenate(train_actions, axis=0)
    stats = {
        "state_pose": {
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "p01": np.percentile(actions, 1, axis=0).tolist(),
            "p99": np.percentile(actions, 99, axis=0).tolist(),
        }
    }
    stats["eef_abs"] = stats["state_pose"]
    stats_path = meta_dir / "stat.json"
    temporary_stats = stats_path.with_suffix(".json.tmp")
    with temporary_stats.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_stats, stats_path)
    return len(train_rows), len(val_rows), actions.shape[0]


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    plans = discover_episodes(args)
    local_plans = plans[accelerator.process_index :: accelerator.num_processes]
    accelerator.print(
        f"Found {len(plans)} episodes; {accelerator.num_processes} workers; "
        f"output={args.output_root}"
    )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=str(args.vae_path))],
        tokenizer_config=None,
        redirect_common_files=False,
    )
    if pipe.vae is None:
        raise RuntimeError(f"Could not load Wan VAE from {args.vae_path}")

    for local_index, plan in enumerate(local_plans, start=1):
        started = time.monotonic()
        status, shape = convert_one(plan, pipe.vae, device, args)
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(
            f"[rank {accelerator.process_index}] {local_index}/{len(local_plans)} {status} "
            f"{plan['task']}/episode_{plan['episode_index']:06d} shape={shape} "
            f"elapsed={time.monotonic() - started:.1f}s peak={peak_gib:.1f}GiB",
            flush=True,
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        train_count, val_count, action_count = build_metadata(plans, args)
        print(
            f"Complete: episodes={len(plans)}, train_samples={train_count}, "
            f"val_samples={val_count}, stats_frames={action_count}",
            flush=True,
        )
    accelerator.wait_for_everyone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
