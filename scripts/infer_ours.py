import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from diffsynth.core import ModelConfig
from wan_video_action.data import LoadCobotAction, RoboTwinUnifiedDataset, create_video_operator
from wan_video_action.parsers import add_general_config, merge_yaml_and_args, prepare_runtime_config
from wan_video_action.pipelines.wan_video_action import build_wan_video_action_pipeline
from wan_video_action.utils import align_num_frames, resolve_model_path, save_video


def parse_args():
    parser = argparse.ArgumentParser("RoboTwin inference entrypoint.")
    parser = add_general_config(parser)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def _build_autoregressive_history_indices(
    num_generated_frames: int,
    history_frames: int,
) -> List[int]:
    if history_frames <= 0:
        return []
    if num_generated_frames <= 0:
        raise ValueError("Autoregressive history requires at least one generated frame.")
    if num_generated_frames < history_frames:
        return [0] * (history_frames - num_generated_frames) + list(range(num_generated_frames))
    if history_frames == 1:
        return [num_generated_frames - 1]
    return [0] + list(range(num_generated_frames - (history_frames - 1), num_generated_frames))


def _build_autoregressive_action_condition(
    action: torch.Tensor,
    history_indices: List[int],
    future_start: int,
    future_count: int,
    infer_frames: int,
) -> torch.Tensor:
    history_action = action[:, history_indices, :]
    future_action = action[:, future_start : future_start + future_count, :]
    action_condition = torch.cat([history_action, future_action], dim=1)
    current_frames = int(action_condition.shape[1])
    if current_frames < infer_frames:
        pad = action_condition[:, -1:, :].repeat(1, infer_frames - current_frames, 1)
        action_condition = torch.cat([action_condition, pad], dim=1)
    return action_condition


def _run_autoregressive(
    pipe,
    sample: Dict,
    args,
):
    sample_index = int(sample["sample_index"])
    episode_index = int(sample["episode_index"])
    total_frames = int(sample["total_frames"])
    input_video = sample["video"]
    action = sample["action"]
    output_path = sample["output_path"]

    history_frames = int(args.num_history_frames)
    chunk_size = int(args.num_frames)
    future_frames = chunk_size - history_frames
    if future_frames <= 0:
        raise ValueError(f"Invalid rollout setting: num_frames={chunk_size}, num_history_frames={history_frames}")

    generated_frames: List[torch.Tensor] = [input_video[:, :, 0].clone()]
    chunk_idx = 0
    while len(generated_frames) < total_frames:
        future_start = len(generated_frames)
        remaining_future = total_frames - future_start
        current_future = min(future_frames, remaining_future)
        requested_frames = history_frames + current_future
        infer_frames = requested_frames

        history_indices = _build_autoregressive_history_indices(
            future_start,
            history_frames,
        )
        chunk_input_video = torch.stack([generated_frames[index] for index in history_indices], dim=2)
        chunk_action = _build_autoregressive_action_condition(
            action=action,
            history_indices=history_indices,
            future_start=future_start,
            future_count=current_future,
            infer_frames=infer_frames,
        )
        print(
            f"[window] sample_index={sample_index} episode_index={episode_index} "
            f"window={chunk_idx} future_start={future_start} future_count={current_future} "
            f"infer_frames={infer_frames}"
        )

        seed = None if args.seed is None else int(args.seed) + chunk_idx
        chunk_video = pipe(
            input_video=chunk_input_video,
            action=chunk_action,
            seed=seed,
            rand_device="cpu",
            tiled=False,
            height=int(args.height),
            width=int(args.width),
            num_frames=infer_frames,
            num_history_frames=int(args.num_history_frames),
            cfg_scale=float(args.cfg_scale),
            num_inference_steps=int(args.num_inference_steps),
            use_history_condition_noise_in_inference=True,
            progress_bar_cmd=lambda iterable, *args, **kwargs: iterable,
            output_type="floatpoint",
        )

        future_video = chunk_video[:, :, history_frames:].detach().cpu()
        append_count = min(current_future, int(future_video.shape[2]))
        if append_count <= 0:
            raise RuntimeError("Autoregressive inference produced no future frames.")
        for frame_idx in range(append_count):
            generated_frames.append(future_video[:, :, frame_idx].clone())

        chunk_idx += 1

    predicted_video = torch.stack(generated_frames, dim=2)[:, :, :total_frames]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_video(
        predicted_video,
        output_path=output_path,
        fps=int(args.fps),
        quality=int(args.quality),
    )
    return output_path


def build_pipeline(args):
    runtime_config = prepare_runtime_config(args)
    model_configs = [
        ModelConfig(path=resolve_model_path(model_path))
        for model_path in runtime_config["model_paths_list"]
    ]

    print("[resolved_models] model_configs:", [config.path for config in model_configs])

    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = build_wan_video_action_pipeline(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=None,
        ckpt_path=args.ckpt_path,
        action_dim=args.action_dim,
        action_mode=args.action_mode,
    )
    pipe.use_gradient_checkpointing = False
    pipe.use_gradient_checkpointing_offload = False
    pipe.eval()
    return pipe


def build_infer_dataset(args):
    with open(args.action_stat_path, "r") as f:
        stat = json.load(f)
    action_stat = stat

    return RoboTwinUnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,
        data_file_keys=("video", "action"),
        main_data_operator=create_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=args.spatial_division_factor,
            width_division_factor=args.spatial_division_factor,
            num_frames=1,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
            resize_mode=args.resize_mode,
        ),
        special_operator_map={
            "action": LoadCobotAction(
                base_path=args.dataset_base_path,
                action_type=args.action_type,
                stat=action_stat,
                num_frames=None,
                align_num_frames=False,
                time_division_factor=args.time_division_factor,
                time_division_remainder=args.time_division_remainder,
            )
        },
    )


def build_output_path(args, source_video_path: str, episode_index: int) -> str:
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


def prepare_sample_for_rollout(sample: Dict, sample_index: int, pipe, args) -> Dict:
    raw_total_frames = int(sample["length"])
    total_frames = align_num_frames(
        raw_total_frames,
        time_division_factor=args.time_division_factor,
        time_division_remainder=args.time_division_remainder,
    )
    if total_frames <= 0:
        raise ValueError(f"Invalid aligned total_frames={total_frames} from raw length={raw_total_frames}")
    episode_index = int(sample["episode_index"])
    start_frame = int(sample.get("start_frame", 0))
    end_frame = int(sample.get("end_frame", start_frame + raw_total_frames - 1))
    raw_action = sample["action"]
    raw_action_shape = tuple(raw_action.shape) if torch.is_tensor(raw_action) else tuple(np.asarray(raw_action).shape)
    action = torch.as_tensor(raw_action, dtype=pipe.torch_dtype, device=pipe.device)

    sample["sample_index"] = sample_index
    sample["raw_total_frames"] = raw_total_frames
    sample["total_frames"] = total_frames
    sample["start_frame"] = start_frame
    sample["end_frame"] = end_frame
    sample["raw_action_shape"] = raw_action_shape
    sample["output_path"] = build_output_path(
        args,
        sample.get("source_video_path"),
        episode_index,
    )
    sample["action"] = action[:, :total_frames]
    return sample


def main():
    args = parse_args()
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

    os.makedirs(args.output_path, exist_ok=True)
    dataset = build_infer_dataset(args)

    pipe = build_pipeline(args)

    processed = 0
    for sample_index in range(args.start_index, len(dataset)):
        if args.max_samples and processed >= args.max_samples:
            break

        raw_sample = dataset.data[sample_index % len(dataset.data)]
        sample = dataset[sample_index]
        sample["source_video_path"] = raw_sample.get("video")
        sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
        print(
            f"[sample] sample_index={sample['sample_index']} episode_index={sample['episode_index']} "
            f"range=[{sample['start_frame']},{sample['end_frame']}] video_shape={tuple(sample['video'].shape)} "
            f"action_shape={sample['raw_action_shape']}"
        )
        print(
            f"[sample_window_target] sample_index={sample['sample_index']} episode_index={sample['episode_index']} "
            f"range=[0,{sample['total_frames'] - 1}] output={sample['output_path']}"
        )
        predicted_path = _run_autoregressive(
            pipe=pipe,
            sample=sample,
            args=args,
        )

        print(
            f"[done] sample_index={sample_index} episode_index={sample['episode_index']} "
            f"output={predicted_path}"
        )
        processed += 1


if __name__ == "__main__":
    main()
