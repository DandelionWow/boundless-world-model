import argparse
import os
import subprocess
import sys
from pathlib import Path

from infer_ours import (
    build_infer_dataset,
    build_pipeline,
    prepare_sample_for_rollout,
    _run_autoregressive,
)
from wan_video_action.parsers import add_general_config, merge_yaml_and_args


def parse_args():
    parser = argparse.ArgumentParser("Distributed RoboTwin inference entrypoint.")
    parser = add_general_config(parser)
    parser.add_argument("--worker", action="store_true", help="Run one rank worker instead of launching all ranks.")
    parser.add_argument("--gpus", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output videos instead of skipping them.")
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def selected_sample_indices(args, dataset_len):
    stop_index = dataset_len
    if args.max_samples:
        stop_index = min(stop_index, args.start_index + args.max_samples)
    return list(range(args.start_index, stop_index))


def build_distributed_output_path(args, source_video_path, episode_index):
    dataset_name = Path(args.dataset_base_path).name
    if not source_video_path:
        return str(Path(args.output_path) / dataset_name / f"episode{episode_index}.mp4")

    source_path = Path(source_video_path)
    if source_path.is_absolute():
        try:
            relative_path = source_path.relative_to(Path(args.dataset_base_path).resolve())
        except ValueError:
            relative_path = Path(source_path.name)
    else:
        relative_path = source_path

    if relative_path.parts and relative_path.parts[0] == dataset_name:
        return str(Path(args.output_path) / relative_path)
    return str(Path(args.output_path) / dataset_name / relative_path)


def launch_workers(args):
    gpu_ids = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")

    world_size = int(args.world_size)
    if world_size == 1 and "WORLD_SIZE" not in os.environ:
        world_size = len(gpu_ids)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if world_size > len(gpu_ids):
        raise ValueError(f"world_size={world_size} exceeds GPU count={len(gpu_ids)} ({','.join(gpu_ids)})")

    base_cmd = [sys.executable, __file__] + sys.argv[1:] + ["--worker"]

    processes = []
    for rank in range(world_size):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = str(world_size)
        cmd = base_cmd + ["--rank", str(rank), "--world_size", str(world_size), "--gpus", gpu_ids[rank]]
        print(f"Starting inference rank {rank}/{world_size} on GPU {gpu_ids[rank]}")
        processes.append(subprocess.Popen(cmd, env=env))

    return_codes = [process.wait() for process in processes]
    failed = [code for code in return_codes if code != 0]
    if failed:
        raise SystemExit(max(failed))


def run_worker(args):
    if args.world_size <= 0:
        raise ValueError(f"world_size must be positive, got {args.world_size}")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"rank must be in [0, {args.world_size}), got {args.rank}")

    print("[resolved_config] model_paths:", args.model_paths)
    print("[resolved_config] model_config_path:", args.model_config_path)
    print("[resolved_config] dataset_base_path:", args.dataset_base_path)
    print("[resolved_config] dataset_metadata_path:", args.dataset_metadata_path)
    print("[resolved_config] action_stat_path:", args.action_stat_path)
    print("[resolved_config] ckpt_path:", args.ckpt_path)
    print("[resolved_config] output_path:", args.output_path)
    print("[resolved_config] profile: Wan2.2 TI2V, text off, image off, VAE fused latent, action adaln, first-frame rollout")
    print("[resolved_config] height:", args.height)
    print("[resolved_config] width:", args.width)
    print("[resolved_config] num_frames:", args.num_frames)
    print("[resolved_config] num_history_frames:", args.num_history_frames)
    print("[resolved_config] time_division_factor:", args.time_division_factor)
    print("[resolved_config] time_division_remainder:", args.time_division_remainder)
    print("[resolved_config] action_type:", args.action_type)
    print("[resolved_config] cfg_scale:", args.cfg_scale)
    print("[resolved_config] num_inference_steps:", args.num_inference_steps)
    print("[resolved_config] fps:", args.fps)
    print("[distributed] rank:", args.rank, "world_size:", args.world_size)

    os.makedirs(args.output_path, exist_ok=True)
    dataset = build_infer_dataset(args)
    sample_indices = selected_sample_indices(args, len(dataset))
    rank_indices = sample_indices[args.rank :: args.world_size]
    print(
        f"[distributed] rank={args.rank} assigned={len(rank_indices)} "
        f"selected_total={len(sample_indices)} indices={rank_indices}"
    )

    if not rank_indices:
        return

    pipe = build_pipeline(args)

    for sample_index in rank_indices:
        raw_sample = dataset.data[sample_index % len(dataset.data)]
        sample = dataset[sample_index]
        sample["source_video_path"] = raw_sample.get("video")
        sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
        sample["output_path"] = build_distributed_output_path(
            args,
            sample.get("source_video_path"),
            sample["episode_index"],
        )
        print(
            f"[sample] rank={args.rank} sample_index={sample['sample_index']} "
            f"episode_index={sample['episode_index']} range=[{sample['start_frame']},{sample['end_frame']}] "
            f"video_shape={tuple(sample['video'].shape)} action_shape={sample['raw_action_shape']}"
        )
        print(
            f"[sample_window_target] rank={args.rank} sample_index={sample['sample_index']} "
            f"episode_index={sample['episode_index']} range=[0,{sample['total_frames'] - 1}] "
            f"output={sample['output_path']}"
        )
        if os.path.exists(sample["output_path"]) and not args.overwrite:
            print(
                f"[skip] rank={args.rank} sample_index={sample_index} "
                f"episode_index={sample['episode_index']} output={sample['output_path']}"
            )
            continue

        predicted_path = _run_autoregressive(
            pipe=pipe,
            sample=sample,
            args=args,
        )

        print(
            f"[done] rank={args.rank} sample_index={sample_index} "
            f"episode_index={sample['episode_index']} output={predicted_path}"
        )


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        launch_workers(args)


if __name__ == "__main__":
    main()
