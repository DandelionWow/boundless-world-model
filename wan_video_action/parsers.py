import argparse
import json
import os
import sys
from omegaconf import OmegaConf


def merge_yaml_and_args(yaml_path, parser, args, cli_args=None):
    # priority: CLI args > YAML config > parser defaults
    if not yaml_path or not os.path.exists(yaml_path):
        return args

    if cli_args is None:
        cli_args = sys.argv[1:]
    explicit_cli_keys = {
        action.dest
        for action in parser._actions
        for option in action.option_strings
        if option in cli_args
    }
    yaml_dict = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True) or {}

    for section in yaml_dict.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if hasattr(args, key) and key not in explicit_cli_keys:
                    setattr(args, key, value)
    return args


def _expand_safetensors_index(path: str):
    if not str(path).endswith(".safetensors.index.json") or not os.path.isfile(path):
        return path
    with open(path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    return [os.path.join(os.path.dirname(path), shard_name) for shard_name in shard_names]


def _resolve_model_paths(model_root: str, model_config_path: str, weight_modules):
    cfg = OmegaConf.to_container(OmegaConf.load(model_config_path), resolve=True)
    module_files = cfg["modules"]
    paths = []
    for module in weight_modules:
        rel_path = module_files[module]
        path = rel_path if os.path.isabs(rel_path) else os.path.join(model_root, rel_path)
        paths.append(_expand_safetensors_index(path))
    return paths


def prepare_model_config(args):
    # cli args override yaml config, if provided
    for mode_name in args.modes.keys():
        value = getattr(args, f"{mode_name}_mode")
        if value is not None:
            args.modes[mode_name] = value

    model_paths_list = _resolve_model_paths(args.model_paths, args.model_config_path, args.weights)

    cfg = OmegaConf.to_container(OmegaConf.load(args.model_config_path), resolve=True)
    tokenizer_path = os.path.join(args.model_paths, cfg["tokenizer_subdir"])

    return {
        "model_paths_list": model_paths_list,
        "tokenizer_path": tokenizer_path,
    }


def resolve_data_keys(args, stage="train"):
    if stage == "infer" or args.modes["vae"] == "raw":
        keys = ["video"]
    else:
        keys = ["latents"]

    if args.modes["action"] != "none":
        keys.append("action")

    if args.modes["text"] == "emb":
        keys.extend(["prompt_emb", "negative_prompt_emb"])

    args.data_keys = keys
    return args


def add_dataset_base_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset_name", type=str, default="robotwin", help="[KEY] Dataset builder name.")
    group.add_argument("--dataset_base_path", type=str, default="", help="[REQUIRED] Base path of the dataset.")
    group.add_argument("--dataset_metadata_path", type=str, default=None, help="[OPTIONAL] Path to the metadata file of the dataset.")
    group.add_argument("--dataset_repeat", type=int, default=1, help="[TUNABLE] Number of times to repeat the dataset per epoch.")
    group.add_argument("--dataset_num_workers", type=int, default=0, help="[OPTIONAL] Number of workers for data loading.")
    parser.set_defaults(data_keys=None)
    return parser


def add_video_size_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("video")
    group.add_argument("--height", type=int, default=None, help="[KEY] Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--width", type=int, default=None, help="[KEY] Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--max_pixels", type=int, default=1048576, help="[OPTIONAL] Maximum number of pixels per frame, used for dynamic resolution.")
    group.add_argument("--num_frames", type=int, default=81, help="[KEY] Number of frames per video. Frames are sampled from the video prefix.")
    group.add_argument("--resize_mode", type=str, default="fit", choices=["crop", "fit"], help="[OPTIONAL] Resize behavior: crop (center crop), fit (no crop).")
    group.add_argument("--num_history_frames", type=int, default=1, help="[KEY] Number of conditioning history frames. Must satisfy 1 <= num_history_frames < num_frames.")
    group.add_argument("--time_division_factor", type=int, default=4, help="[OPTIONAL] Temporal frame divisor used to align video/action frame counts.")
    group.add_argument("--time_division_remainder", type=int, default=1, help="[OPTIONAL] Temporal frame remainder used with time_division_factor.")
    group.add_argument("--spatial_division_factor", type=int, default=32, help="[OPTIONAL] Spatial size divisor used to align frame height and width.")
    group.add_argument("--chunk_mode", type=str, default="static", choices=["static", "dynamic"], help="[OPTIONAL] Sampling mode for video chunks, static uses dataset bounds and dynamic uses random crop.")
    group.add_argument("--history_template_sampling", type=int, choices=[0, 1], default=0, help="[OPTIONAL] Enable history-template sampling: frame0 + recent history + future frames.")
    return parser


def add_model_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("model")
    group.add_argument("--model_paths", type=str, default=None, help="[REQUIRED] Paths to load models. In JSON format, comma-separated, or a single model root.")
    group.add_argument("--model_id_with_origin_paths", type=str, default=None, help="[OPTIONAL] Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    group.add_argument("--extra_inputs", type=str, default=None, help="[OPTIONAL] Additional model inputs, comma-separated.")
    group.add_argument("--fp8_models", type=str, default=None, help="[OPTIONAL] Models with FP8 precision, comma-separated.")
    group.add_argument("--offload_models", type=str, default=None, help="[OPTIONAL] Models with offload, comma-separated. Only used in splited training.")
    group.add_argument("--model_config_path", type=str, default="configs/model/wan2_1_fun_1_3b_inp.yaml", help="[KEY] Path to model config YAML.")
    group.add_argument("--initialize_model_on_cpu", action="store_true", default=False, help="[OPTIONAL] Whether to initialize models on CPU.")
    group.add_argument("--dit_mode", type=str, default=None, choices=["default", "none"], help="[OPTIONAL] Override model.modes.dit.")
    group.add_argument("--text_mode", type=str, default=None, choices=["t5", "emb", "none"], help="[OPTIONAL] Override model.modes.text.")
    group.add_argument("--vae_mode", type=str, default=None, choices=["raw", "emb", "none"], help="[OPTIONAL] Override model.modes.vae.")
    group.add_argument("--image_mode", type=str, default=None, choices=["none", "default", "flat"], help="[OPTIONAL] Override model.modes.image.")
    group.add_argument("--action_mode", type=str, default=None, choices=["noise", "adaln", "none"], help="[OPTIONAL] Override model.modes.action.")
    parser.set_defaults(weights=None, modes=None)
    return parser


def add_action_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("action")
    group.add_argument("--action_type", type=str, choices=["joint_abs", "eef_abs", "joint_delta", "eef_delta", "state_joint", "state_pose", "action_joint", "action_pose"], default="eef_delta", help='[KEY] Action/state representation: joint/eef x abs/delta, with state_* and action_* aliases.')
    group.add_argument("--action_stat_path", type=str, default=None, help="[OPTIONAL] Path to robot normalization stats (stat.json).")
    group.add_argument("--action_dim", type=int, default=14, help="[OPTIONAL] Action dimension.")
    return parser


def add_training_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("training")
    group.add_argument("--learning_rate", type=float, default=1e-4, help="[TUNABLE] Learning rate.")
    group.add_argument("--find_unused_parameters", action="store_true", default=False, help="[OPTIONAL] Whether to find unused parameters in DDP.")
    group.add_argument("--weight_decay", type=float, default=0.01, help="[TUNABLE] Weight decay.")
    group.add_argument("--task", type=str, default="sft", help="[OPTIONAL] Task type.")
    group.add_argument("--seed", type=int, default=42, help="[OPTIONAL] Random seed for python/numpy/torch.")
    group.add_argument("--deterministic", action="store_true", default=False, help="[OPTIONAL] Enable deterministic PyTorch algorithms for strict resume debugging.")
    group.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="[OPTIONAL] Mixed precision mode.")
    group.add_argument("--max_timestep_boundary", type=float, default=1.0, help="[OPTIONAL] Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    group.add_argument("--min_timestep_boundary", type=float, default=0.0, help="[OPTIONAL] Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    group.add_argument("--max_train_steps", type=int, default=None, help="[KEY] Maximum optimizer steps.")
    group.add_argument("--batch_size", type=int, default=1, help="[TUNABLE] Batch size per GPU.")
    parser.set_defaults(trainable=None)
    return parser


def add_output_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("output")
    group.add_argument("--output_path", type=str, default="./models", help="[KEY] Output save path.")
    group.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help='[OPTIONAL] Remove prefix in ckpt. (default: "pipe.dit.")')
    group.add_argument("--save_steps", type=int, default=None, help="[KEY] Optimizer-step interval for checkpoint saving.")
    group.add_argument("--resume_from", type=str, default=None, help="[OPTIONAL] Path to a full training state directory saved by accelerator (e.g., output_path/states/latest).")
    return parser


def add_lora_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("lora")
    group.add_argument("--lora_base_model", type=str, default=None, help="[OPTIONAL] Which model LoRA is added to.")
    group.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="[OPTIONAL] Which layers LoRA is added to.")
    group.add_argument("--lora_rank", type=int, default=32, help="[TUNABLE] Rank of LoRA.")
    group.add_argument("--lora_checkpoint", type=str, default=None, help="[OPTIONAL] Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    group.add_argument("--preset_lora_path", type=str, default=None, help="[OPTIONAL] Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    group.add_argument("--preset_lora_model", type=str, default=None, help="[OPTIONAL] Which model the preset LoRA is fused to.")
    return parser


def add_gradient_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("gradient")
    group.add_argument("--use_gradient_checkpointing", action="store_true", default=False, help="[KEY] Whether to use gradient checkpointing.")
    group.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=False, help="[KEY] Whether to offload gradient checkpointing to CPU memory.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="[TUNABLE] Gradient accumulation steps.")
    group.add_argument("--max_grad_norm", type=float, default=0.5, help="[OPTIONAL] Maximum gradient norm for clipping. (default: 0.5)")
    return parser


def add_tracking_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("tracking")
    group.add_argument("--use_wandb", action="store_true", default=False, help="[OPTIONAL] Enable Weights & Biases tracking.")
    group.add_argument("--use_swanlab", action="store_true", default=False, help="[OPTIONAL] Enable SwanLab tracking.")
    group.add_argument("--run_name", type=str, default=None, help="[OPTIONAL] Remote tracker run/project name. Defaults to output_path.")
    return parser


def add_infer_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("infer")
    group.add_argument("--ckpt_path", type=str, default=None, help="[OPTIONAL] Path to model checkpoint (.safetensors) used for inference.")
    group.add_argument("--cfg_scale", type=float, default=5.0, help="[OPTIONAL] CFG scale for generation.")
    group.add_argument("--num_inference_steps", type=int, default=50, help="[OPTIONAL] Number of inference steps.")
    group.add_argument("--negative_prompt", type=str, default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion", help="[OPTIONAL] Negative prompt for generation.")
    group.add_argument("--negative_prompt_emb", type=str, default=None, help="[OPTIONAL] Path to the pre-extracted negative prompt embedding.")
    group.add_argument("--quality", type=int, default=5, help="[OPTIONAL] Output video quality.")
    group.add_argument("--disable_chunk_infer", dest="chunk_infer", action="store_false", default=True, help="[OPTIONAL] Disable chunked inference with 81-frame segments.")
    group.add_argument("--fps", type=int, default=24, help="[OPTIONAL] Output video FPS.")
    group.add_argument("--disable_metrics", dest="enable_metrics", action="store_false", default=True, help="[OPTIONAL] Disable evaluation metrics.")
    group.add_argument("--start_index", type=int, default=0, help="[OPTIONAL] First metadata row index to process.")
    group.add_argument("--max_samples", type=int, default=0, help="[OPTIONAL] Maximum number of metadata rows to process. 0 means all.")
    return parser


def add_config_support(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file. CLI args override YAML config.")
    return parser


def add_general_config(parser: argparse.ArgumentParser):
    parser = add_config_support(parser)
    parser = add_dataset_base_config(parser)
    parser = add_video_size_config(parser)
    parser = add_model_config(parser)
    parser = add_action_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_tracking_config(parser)
    parser = add_infer_config(parser)
    return parser
