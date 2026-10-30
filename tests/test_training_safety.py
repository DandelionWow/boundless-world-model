import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from scripts.train import WanTrainingModule, validate_compact_checkpoint_profile
from wan_video_action.logger import ModelLogger, TrainingLogger
from wan_video_action.run_state import prepare_run_directory
from wan_video_action.runner import (
    collate_training_samples,
    ensure_global_finite,
    load_trainable_module_state,
)


class _LocalAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def reduce(value, reduction):
        if reduction not in ("max", "sum"):
            raise AssertionError(f"Unexpected reduction: {reduction}")
        return value


class _ManifestAccelerator:
    is_main_process = True
    num_processes = 1

    @staticmethod
    def print(*args, **kwargs):
        return None


class _ErrorParser:
    @staticmethod
    def error(message):
        raise ValueError(message)


class TrainingSafetyTest(unittest.TestCase):
    @staticmethod
    def _manifest_args(temporary_dir):
        paths = {}
        for name in ("metadata.jsonl", "stat.json", "model.yaml", "model.safetensors", "init.safetensors"):
            path = os.path.join(temporary_dir, name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(name + "\n")
            paths[name] = path
        return SimpleNamespace(
            dataset_name="robotwin",
            dataset_base_path=temporary_dir,
            dataset_metadata_path=paths["metadata.jsonl"],
            dataset_repeat=1,
            dataset_num_workers=0,
            data_keys=["latents", "action"],
            model_paths=temporary_dir,
            model_config_path=paths["model.yaml"],
            resolved_model_paths=[paths["model.safetensors"]],
            weights=["dit", "vae"],
            modes={"dit": "default", "text": "none", "vae": "emb", "image": "none", "action": "adaln"},
            trainable=["dit", "action_encoder"],
            task="sft",
            action_type="eef_abs",
            action_dim=14,
            action_stat_path=paths["stat.json"],
            height=480,
            width=640,
            num_frames=81,
            num_history_frames=9,
            history_template_sampling=1,
            time_division_factor=4,
            time_division_remainder=1,
            spatial_division_factor=32,
            resize_mode="crop",
            seed=42,
            batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=5e-5,
            weight_decay=0.01,
            max_grad_norm=0.5,
            mixed_precision="bf16",
            deterministic=False,
            find_unused_parameters=True,
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=False,
            min_timestep_boundary=0.0,
            max_timestep_boundary=1.0,
            output_path=os.path.join(temporary_dir, "output"),
            ckpt_path=paths["init.safetensors"],
            resume_from=None,
            allow_unsafe_resume=False,
        )

    def test_run_manifest_guards_fresh_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = self._manifest_args(temporary_dir)
            accelerator = _ManifestAccelerator()
            manifest = prepare_run_directory(accelerator, args, repo_root=temporary_dir)
            self.assertTrue(os.path.isfile(os.path.join(args.output_path, ".bwm_run.json")))
            self.assertEqual(manifest["training_identity"]["optimization"]["world_size"], 1)

            with self.assertRaisesRegex(RuntimeError, "Fresh training output directory is not empty"):
                prepare_run_directory(accelerator, args, repo_root=temporary_dir)

            state_dir = os.path.join(args.output_path, "states", "step-1")
            os.makedirs(state_dir)
            with open(os.path.join(state_dir, "COMPLETE"), "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            args.ckpt_path = None
            args.resume_from = state_dir
            resumed_manifest = prepare_run_directory(accelerator, args, repo_root=temporary_dir)
            self.assertEqual(resumed_manifest["run_id"], manifest["run_id"])

            args.learning_rate = 1e-4
            with self.assertRaisesRegex(RuntimeError, "Resume configuration does not match"):
                prepare_run_directory(accelerator, args, repo_root=temporary_dir)

    def test_compact_checkpoint_profile_rejects_unsafe_variants(self):
        parser = _ErrorParser()
        args = SimpleNamespace(
            trainable=["dit", "action_encoder"],
            task="sft",
            model_id_with_origin_paths=None,
            fp8_models=None,
            offload_models=None,
            lora_base_model=None,
            lora_checkpoint=None,
            preset_lora_path=None,
            preset_lora_model=None,
        )
        validate_compact_checkpoint_profile(parser, args)

        variants = (
            ("trainable", ["dit"], "training exactly"),
            ("trainable", ["action_encoder"], "training exactly"),
            ("task", "direct_distill", "only supports task='sft'"),
            ("lora_base_model", "dit", "lora_base_model='dit'"),
            ("lora_base_model", "", "lora_base_model=''"),
            ("lora_checkpoint", "lora.safetensors", "lora_checkpoint='lora.safetensors'"),
            ("preset_lora_path", "preset.safetensors", "preset_lora_path='preset.safetensors'"),
            ("preset_lora_model", "dit", "preset_lora_model='dit'"),
            ("fp8_models", "dit", "fp8_models='dit'"),
            ("offload_models", "vae", "offload_models='vae'"),
            ("model_id_with_origin_paths", "repo:model*.safetensors", "extra model sources"),
        )
        for field, value, message in variants:
            with self.subTest(field=field, value=value):
                original = getattr(args, field)
                setattr(args, field, value)
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        validate_compact_checkpoint_profile(parser, args)
                finally:
                    setattr(args, field, original)

    def test_collate_preserves_batch_and_removes_sample_action_axis(self):
        samples = [
            {
                "latents": torch.full((1, 48, 21, 2, 3), fill_value=index),
                "action": torch.full((1, 81, 14), fill_value=index),
            }
            for index in range(2)
        ]
        batch = collate_training_samples(samples)
        self.assertEqual(batch["latents"].shape, (2, 1, 48, 21, 2, 3))
        self.assertEqual(batch["action"].shape, (2, 81, 14))
        self.assertEqual(batch["_batch_size"], 2)
        self.assertTrue(torch.equal(batch["action"][:, 0, 0], torch.tensor([0, 1])))

    def test_extra_input_frame_selection_uses_time_axis(self):
        module = WanTrainingModule.__new__(WanTrainingModule)
        video = torch.arange(2 * 1 * 3 * 5 * 2 * 2).reshape(2, 1, 3, 5, 2, 2)
        result = module.parse_extra_inputs(
            {"video": video},
            ["input_image", "end_image"],
            {},
        )
        self.assertEqual(result["input_image"].shape, (2, 1, 3, 2, 2))
        self.assertEqual(result["end_image"].shape, (2, 1, 3, 2, 2))
        self.assertTrue(torch.equal(result["input_image"], video[:, :, :, 0]))
        self.assertTrue(torch.equal(result["end_image"], video[:, :, :, -1]))

    def test_non_finite_values_fail_fast(self):
        accelerator = _LocalAccelerator()
        ensure_global_finite(accelerator, torch.tensor([1.0, 2.0]), "loss")
        with self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
            ensure_global_finite(accelerator, torch.tensor([float("nan")]), "loss")
        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient norm"):
            ensure_global_finite(accelerator, torch.tensor([float("inf")]), "gradient norm")

    def test_compact_checkpoint_requires_every_trainable_parameter(self):
        module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        module[1].requires_grad_(False)
        compact_state = {
            name: tensor.clone()
            for name, tensor in module.state_dict().items()
            if name.startswith("0.")
        }
        load_trainable_module_state(compact_state, module)

        missing_state = dict(compact_state)
        missing_state.pop("0.bias")
        with self.assertRaisesRegex(RuntimeError, "missing_trainable"):
            load_trainable_module_state(missing_state, module)

        unexpected_state = dict(compact_state)
        unexpected_state["not_a_parameter"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            load_trainable_module_state(unexpected_state, module)

    def test_metric_reconciliation_truncates_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            logger = TrainingLogger.__new__(TrainingLogger)
            logger.metric_log_path = os.path.join(temporary_dir, "train_metrics.jsonl")
            records = [
                {"step": 1, "loss": 1.0},
                {"step": 2, "loss": 2.0},
                {"step": 2, "loss": 1.5},
                {"step": 3, "loss": 3.0},
            ]
            with open(logger.metric_log_path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            logger.reconcile_metric_log(resume_step=2)

            with open(logger.metric_log_path, "r", encoding="utf-8") as handle:
                reconciled = [json.loads(line) for line in handle]
            self.assertEqual(reconciled, [{"loss": 1.0, "step": 1}, {"loss": 1.5, "step": 2}])
            self.assertFalse(os.path.exists(logger.metric_log_path + ".tmp"))

    def test_checkpoint_identity_rejects_mismatch(self):
        manifest = {
            "run_id": "run-a",
            "config_fingerprint": "fingerprint-a",
            "training_identity": {"rng_scheme": "rank-offset-v1"},
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = ModelLogger(temporary_dir, run_manifest=manifest)
            source.num_steps = 7
            source.epoch_id = 2
            source.batch_in_epoch = 31
            saved_state = source.state_dict()

            restored = ModelLogger(temporary_dir, run_manifest=manifest)
            restored.load_state_dict(saved_state)
            self.assertEqual(restored.validate_resume_identity(), [])

            restored.run_manifest = {**manifest, "config_fingerprint": "fingerprint-b"}
            with self.assertRaisesRegex(RuntimeError, "Checkpoint identity does not match"):
                restored.validate_resume_identity()


if __name__ == "__main__":
    unittest.main()
