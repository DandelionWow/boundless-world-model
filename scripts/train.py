import argparse
import json
import os
import sys
from pathlib import Path

import accelerate
import torch
from accelerate.utils import DataLoaderConfiguration
from diffsynth.core import ModelConfig
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    DirectDistillLoss,
    launch_data_process_task,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_video_action.data import build_train_dataset
from wan_video_action.logger import ModelLogger
from wan_video_action.loss import FlowMatchSFTLossWanAction
from wan_video_action.parsers import merge_yaml_and_args, prepare_model_config, resolve_data_keys, add_general_config
from wan_video_action.pipelines.wan_video_action import WanVideoActionPipeline
from wan_video_action.run_state import prepare_run_directory
from wan_video_action.runner import launch_training_task
from wan_video_action.utils import set_global_seed
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        ckpt_path=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        num_history_frames=1,
        history_template_sampling=0,
        action_dim=14,
        action_mode="adaln",
        args=None,
    ):
        super().__init__()

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        if args.modes["text"] == "t5":
            tokenizer_config = ModelConfig(tokenizer_path)
        else:
            tokenizer_config = None
        self.pipe = WanVideoActionPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            ckpt_path=ckpt_path,
            action_dim=action_dim,
            action_mode=action_mode,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        self._apply_action_encoder_trainable_mask(self.pipe)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLossWanAction(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLossWanAction(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.num_history_frames = num_history_frames
        self.history_template_sampling = int(history_template_sampling)
        self.use_precomputed_latents = args.modes["vae"] == "emb"

    @staticmethod
    def _apply_action_encoder_trainable_mask(pipe):
        """Train only the action-encoder branch used by the selected mode."""
        action_encoder = getattr(pipe, "action_encoder", None)
        if action_encoder is None or not any(param.requires_grad for param in action_encoder.parameters()):
            return

        action_mode = getattr(pipe, "action_injection_mode", None)
        if action_mode == "adaln":
            action_encoder.action_embedding.requires_grad_(False)
            action_encoder.action_mlp1.requires_grad_(True)
            action_encoder.action_mlp2.requires_grad_(True)
        elif action_mode in ("noise", "cross"):
            action_encoder.action_embedding.requires_grad_(True)
            action_encoder.action_mlp1.requires_grad_(False)
            action_encoder.action_mlp2.requires_grad_(False)

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"].select(dim=-3, index=0)
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"].select(dim=-3, index=data["video"].shape[-3] - 1)
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        if self.use_precomputed_latents:
            precomputed_latents = data["latents"]
            if not torch.is_tensor(precomputed_latents) or precomputed_latents.ndim not in (5, 6):
                raise TypeError(
                    "Expected latent tensor with shape (V,C,T,H,W) or (B,V,C,T,H,W), "
                    f"got {type(precomputed_latents).__name__} with shape "
                    f"{getattr(precomputed_latents, 'shape', None)}."
                )
            input_video = None
            action = data.get("action")
            if hasattr(action, "shape") and len(action.shape) >= 2:
                sample_num_frames = int(action.shape[-2])
            else:
                sample_num_frames = 1 + 4 * (int(precomputed_latents.shape[-3]) - 1)
            upsampling_factor = int(getattr(self.pipe.vae, "upsampling_factor", 8))
            height = int(precomputed_latents.shape[-2]) * upsampling_factor
            width = int(precomputed_latents.shape[-1]) * upsampling_factor
        else:
            input_video = data["video"]
            if not torch.is_tensor(input_video) or input_video.ndim not in (5, 6):
                raise TypeError(
                    "Expected raw video tensor with shape (V,C,T,H,W) or (B,V,C,T,H,W), "
                    f"got {type(input_video).__name__} with shape {getattr(input_video, 'shape', None)}."
                )
            precomputed_latents = None
            sample_num_frames = int(input_video.shape[-3])
            height = int(input_video.shape[-2])
            width = int(input_video.shape[-1])

        inputs_posi = {
            "prompt": data.get("prompt"),
            "prompt_emb": data.get("prompt_emb"),
        }
        inputs_nega = {
            "negative_prompt": data.get("negative_prompt"),
            "prompt_emb": data.get("negative_prompt_emb"),
        }
        inputs_shared = {
            "input_video": input_video,
            "precomputed_latents": precomputed_latents,
            "action": data.get("action"),
            "action_injection_mode": self.pipe.action_injection_mode,
            "height": height,
            "width": width,
            "num_frames": sample_num_frames,
            "num_history_frames": self.num_history_frames,
            "history_template_sampling": self.history_template_sampling,
            "temporal_future_start": data.get("temporal_future_start"),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="WAN Video Action training script.")
    parser = add_general_config(parser)
    return parser


def validate_compact_checkpoint_profile(parser, args):
    """Reject training variants that compact full-state resume cannot reconstruct safely."""
    expected_trainable = {"dit", "action_encoder"}
    try:
        actual_trainable = set(args.trainable)
    except TypeError:
        actual_trainable = set()
    if len(args.trainable) != len(expected_trainable) or actual_trainable != expected_trainable:
        parser.error(
            "Compact BWM resume requires training exactly ['dit', 'action_encoder']; "
            f"got {args.trainable!r}. Partial-module training is not supported because "
            "frozen BWM initialization weights are excluded from compact states."
        )

    if args.task != "sft":
        parser.error(
            "The validated compact BWM training profile only supports task='sft'; "
            f"got {args.task!r}."
        )

    unsupported_features = {
        "model_id_with_origin_paths": args.model_id_with_origin_paths,
        "fp8_models": args.fp8_models,
        "offload_models": args.offload_models,
        "lora_base_model": args.lora_base_model,
        "lora_checkpoint": args.lora_checkpoint,
        "preset_lora_path": args.preset_lora_path,
        "preset_lora_model": args.preset_lora_model,
    }
    enabled_features = {
        name: value
        for name, value in unsupported_features.items()
        if value is not None
    }
    if enabled_features:
        formatted = ", ".join(f"{name}={value!r}" for name, value in enabled_features.items())
        parser.error(
            "The compact BWM training profile has not been validated with LoRA, preset LoRA, "
            f"FP8, model offload, or extra model sources. Unsupported setting(s): {formatted}."
        )


def validate_training_configuration(parser, args):
    if args.ckpt_path and args.resume_from:
        parser.error("--ckpt_path and --resume_from are mutually exclusive; use one startup mode only.")
    if not args.ckpt_path and not args.resume_from:
        parser.error("Training requires either --ckpt_path for a fresh run or --resume_from for a full-state resume.")

    if not isinstance(args.modes, dict) or not isinstance(args.weights, (list, tuple)):
        parser.error(
            "This training config uses the legacy schema. Expected model.weights and model.modes; "
            "the old train_action_noise/train_action_adaln configs are not supported by this entry point."
        )
    # Mode-specific CLI flags override the composite YAML mapping. Apply them
    # before checking the supported matrix; prepare_model_config performs the
    # same idempotent merge later while resolving model files.
    for mode_name in args.modes:
        override = getattr(args, f"{mode_name}_mode")
        if override is not None:
            args.modes[mode_name] = override
    if not isinstance(args.trainable, (list, tuple)) or not args.trainable:
        parser.error("training.trainable must be a non-empty list.")
    validate_compact_checkpoint_profile(parser, args)

    required_paths = {
        "model_paths": args.model_paths,
        "model_config_path": args.model_config_path,
        "dataset_base_path": args.dataset_base_path,
        "dataset_metadata_path": args.dataset_metadata_path,
        "action_stat_path": args.action_stat_path,
        "output_path": args.output_path,
    }
    for name, value in required_paths.items():
        if not isinstance(value, str) or not value.strip():
            parser.error(f"--{name} is required and must be a non-empty path.")
    for name in ("model_config_path", "dataset_metadata_path", "action_stat_path"):
        value = os.path.abspath(os.path.expanduser(required_paths[name]))
        if not os.path.isfile(value):
            parser.error(f"--{name} does not exist or is not a file: {value}")
    model_root = os.path.abspath(os.path.expanduser(args.model_paths))
    dataset_root = os.path.abspath(os.path.expanduser(args.dataset_base_path))
    if not os.path.isdir(model_root):
        parser.error(f"--model_paths does not exist or is not a directory: {model_root}")
    if not os.path.isdir(dataset_root):
        parser.error(f"--dataset_base_path does not exist or is not a directory: {dataset_root}")
    if args.ckpt_path and not os.path.isfile(os.path.abspath(os.path.expanduser(args.ckpt_path))):
        parser.error(f"--ckpt_path does not exist or is not a file: {args.ckpt_path}")

    supported_modes = {
        "dit": "default",
        "text": "none",
        "image": "none",
        "action": "adaln",
    }
    mode_errors = [
        f"{name}={args.modes.get(name)!r} (required {expected!r})"
        for name, expected in supported_modes.items()
        if args.modes.get(name) != expected
    ]
    if args.modes.get("vae") not in ("raw", "emb"):
        mode_errors.append(f"vae={args.modes.get('vae')!r} (required 'raw' or 'emb')")
    if mode_errors:
        parser.error(
            "The current BWM training pipeline only supports raw/emb + AdaLN + no-text + no-image. "
            "Unsupported mode(s): " + ", ".join(mode_errors)
        )
    if args.extra_inputs:
        parser.error(
            "model.extra_inputs is not supported by the current BWM action pipeline. "
            "Image conditioning requires a dedicated pipeline unit and must not be enabled silently."
        )

    positive_integer_fields = (
        "batch_size",
        "gradient_accumulation_steps",
        "max_train_steps",
        "save_steps",
        "dataset_repeat",
        "height",
        "width",
        "num_frames",
        "num_history_frames",
        "action_dim",
    )
    for name in positive_integer_fields:
        value = getattr(args, name)
        if value is None or int(value) <= 0:
            parser.error(f"--{name} must be a positive integer, got {value!r}.")
    if args.dataset_num_workers < 0:
        parser.error(f"--dataset_num_workers must be non-negative, got {args.dataset_num_workers}.")
    if not 0 <= args.min_timestep_boundary < args.max_timestep_boundary <= 1:
        parser.error(
            "Timestep boundaries must satisfy 0 <= min_timestep_boundary < max_timestep_boundary <= 1."
        )
    if not 1 <= args.num_history_frames < args.num_frames:
        parser.error("num_history_frames must satisfy 1 <= num_history_frames < num_frames.")


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)

    validate_training_configuration(parser, args)

    set_global_seed(args.seed, deterministic=args.deterministic)
    model_config = prepare_model_config(args)
    args.resolved_model_paths = model_config["model_paths_list"]
    args = resolve_data_keys(args, stage="train")
    trainable_models = ",".join(args.trainable)

    print("[resolved_config] model_config_path:", args.model_config_path)
    print("[resolved_config] model_paths:", args.model_paths)
    print("[resolved_config] resolved_model_paths:", model_config["model_paths_list"])
    print("[resolved_config] model.weights:", args.weights)
    print("[resolved_config] dataset.data_keys:", args.data_keys)
    print("[resolved_config] training.trainable:", args.trainable)
    print("[resolved_config] model.modes:", args.modes)
    print("[resolved_config] dit_mode:", args.modes["dit"])
    print("[resolved_config] text_mode:", args.modes["text"])
    print("[resolved_config] image_mode:", args.modes["image"])
    print("[resolved_config] vae_mode:", args.modes["vae"])
    print("[resolved_config] action_mode:", args.modes["action"])
    print("[resolved_config] action_type:", args.action_type)
    print("[resolved_config] trainable_models:", trainable_models)
    print("[resolved_config] height,width,num_frames:", args.height, args.width, args.num_frames)
    print("[resolved_config] num_history_frames:", args.num_history_frames)
    print("[resolved_config] history_template_sampling:", args.history_template_sampling)
    print("[resolved_config] spatial_division_factor:", args.spatial_division_factor)
    print("[resolved_config] max_train_steps:", args.max_train_steps)
    print("[resolved_config] init_ckpt_path:", args.ckpt_path)
    print("[resolved_config] resume_from:", args.resume_from)
    print("[resolved_config] deterministic:", args.deterministic)
    loggers = [name for name in ("wandb", "swanlab") if getattr(args, f"use_{name}", False)]
    # Accelerate's DeepSpeed launcher may export the literal value "auto" from
    # its YAML config.  Accelerator() parses that environment variable before
    # reconciling the plugin and fails on int("auto").  The merged train args
    # are the source of truth and also allow CLI smoke-test overrides.
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(args.gradient_accumulation_steps)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=loggers or None,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=args.seed,
        ),
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    run_manifest = prepare_run_directory(accelerator, args, repo_root=str(REPO_ROOT))
    accelerator.print("[resolved_config] run_id:", run_manifest["run_id"])
    accelerator.print("[resolved_config] config_fingerprint:", run_manifest["config_fingerprint"])

    dataset = build_train_dataset(args)

    model = WanTrainingModule(
        model_paths=json.dumps(model_config["model_paths_list"]),
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=model_config["tokenizer_path"],
        trainable_models=trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        ckpt_path=args.ckpt_path,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_history_frames=args.num_history_frames,
        history_template_sampling=args.history_template_sampling,
        action_dim=args.action_dim,
        action_mode=args.modes["action"],
        args=args,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        run_manifest=run_manifest,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
