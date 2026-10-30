import gc
import json
import os
import re
import shutil

import torch
from accelerate.utils import DistributedType, broadcast_object_list
from diffsynth.diffusion.runner import ModelLogger as DiffSynthModelLogger


def _fsync_directory(path):
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _raise_main_process_error_on_all_ranks(accelerator, error):
    objects = [error if accelerator.is_main_process else None]
    broadcast_object_list(objects, from_process=0)
    if objects[0] is not None:
        raise RuntimeError(objects[0])


class TrainingLogger:
    def __init__(self, accelerator, output_path, args, resume_step=None):
        self.accelerator = accelerator
        self.output_path = output_path
        self.args = args
        self.metric_log_path = os.path.join(output_path, "train_metrics.jsonl")
        self.tracker_names = [name for name in ("wandb", "swanlab") if getattr(args, f"use_{name}")]

        error = None
        if accelerator.is_main_process:
            try:
                os.makedirs(output_path, exist_ok=True)
                if resume_step is not None:
                    self.reconcile_metric_log(int(resume_step))
            except Exception as exc:
                error = f"Failed to prepare training metrics: {type(exc).__name__}: {exc}"
        _raise_main_process_error_on_all_ranks(accelerator, error)
        metric_log_path = self.metric_log_path if accelerator.is_main_process else os.devnull
        self.metric_log_file = open(metric_log_path, "a", encoding="utf-8")

    def reconcile_metric_log(self, resume_step):
        """Atomically discard metrics newer than the restored checkpoint."""
        if not os.path.isfile(self.metric_log_path):
            return

        records_by_step = {}
        with open(self.metric_log_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                    step = int(record["step"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid metric record at {self.metric_log_path}:{line_number}: {exc}"
                    ) from exc
                if step <= resume_step:
                    # A previous unsafe resume may already have created duplicates.
                    # The last record is the one closest to the restored trajectory.
                    records_by_step[step] = record

        temporary_path = f"{self.metric_log_path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            for step in sorted(records_by_step):
                handle.write(json.dumps(records_by_step[step], sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, self.metric_log_path)
        _fsync_directory(os.path.dirname(self.metric_log_path))

    @property
    def run_name(self):
        return self.args.run_name or self.output_path

    def init_trackers(self):
        if not self.tracker_names:
            return

        accelerator_tracker_names = {str(tracker) for tracker in self.accelerator.log_with}
        missing_tracker_names = set(self.tracker_names) - accelerator_tracker_names
        if missing_tracker_names:
            raise ValueError(f"Accelerator is missing requested trackers: {sorted(missing_tracker_names)}")

        tracker_init_kwargs = {}
        if self.args.use_wandb:
            tracker_init_kwargs["wandb"] = {"name": self.run_name}
        if self.args.use_swanlab:
            tracker_init_kwargs["swanlab"] = {"name": self.run_name}

        self.accelerator.init_trackers(
            project_name=self.run_name,
            config=vars(self.args),
            init_kwargs=tracker_init_kwargs,
        )

    def print_resolved_config(self, items):
        for name, value in items:
            self.accelerator.print(f"[resolved_config] {name}:", value)

    def log_step(self, metrics, step):
        self.metric_log_file.write(json.dumps(metrics, sort_keys=True) + "\n")
        self.metric_log_file.flush()
        if self.accelerator.is_main_process:
            os.fsync(self.metric_log_file.fileno())

        if self.tracker_names:
            remote_metrics = {
                "train/loss": metrics["loss"],
                "train/lr": metrics["lr"],
                "train/grad_norm": metrics["grad_norm"],
                "train/epoch": metrics["epoch"],
                "train/batch_in_epoch": metrics["batch_in_epoch"],
            }
            self.accelerator.log(remote_metrics, step=step)

    def update_progress_bar(self, progress_bar, postfix):
        progress_bar.set_postfix(postfix)

    def close(self):
        self.metric_log_file.close()
        if self.tracker_names:
            self.accelerator.end_training()


class ModelLogger(DiffSynthModelLogger):
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        run_manifest=None,
    ):
        super().__init__(
            output_path,
            remove_prefix_in_ckpt=remove_prefix_in_ckpt,
            state_dict_converter=state_dict_converter,
        )
        self.epoch_id = 0
        self.batch_in_epoch = 0
        self.run_manifest = run_manifest or {}
        self.loaded_run_id = None
        self.loaded_config_fingerprint = None
        self.loaded_rng_scheme = None

    @property
    def states_root_dir(self):
        return os.path.join(self.output_path, "states")

    @property
    def latest_state_dir(self):
        return os.path.join(self.states_root_dir, "latest")

    @property
    def latest_state_metadata_path(self):
        return os.path.join(self.states_root_dir, "latest.json")

    def state_dir_for_step(self, num_steps=None):
        step = self.num_steps if num_steps is None else int(num_steps)
        return os.path.join(self.states_root_dir, f"step-{step}")

    @staticmethod
    def _atomic_write(path, payload):
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(os.path.dirname(path))

    def _install_latest_symlink(self, state_dir):
        latest_path = self.latest_state_dir
        previous_state_dir = None
        if os.path.lexists(latest_path):
            if os.path.islink(latest_path):
                previous_state_dir = os.path.realpath(latest_path)
            else:
                # One-time migration for checkpoints written by the old layout.
                legacy_path = os.path.join(self.states_root_dir, "previous-legacy")
                if os.path.lexists(legacy_path):
                    raise FileExistsError(
                        f"Cannot migrate legacy state because the destination exists: {legacy_path}"
                    )
                os.replace(latest_path, legacy_path)
                previous_state_dir = legacy_path

        temporary_link = f"{latest_path}.tmp"
        if os.path.lexists(temporary_link):
            os.unlink(temporary_link)
        os.symlink(os.path.basename(state_dir), temporary_link)
        os.replace(temporary_link, latest_path)
        return previous_state_dir

    def _prune_completed_states(self, keep=2, protected_paths=()):
        completed = []
        if not os.path.isdir(self.states_root_dir):
            return
        for name in os.listdir(self.states_root_dir):
            match = re.fullmatch(r"step-(\d+)", name)
            if match is None:
                continue
            path = os.path.join(self.states_root_dir, name)
            if os.path.isfile(os.path.join(path, "COMPLETE")):
                completed.append((int(match.group(1)), path))

        completed = sorted(completed, reverse=True)
        retained = []
        for path in protected_paths:
            if path is None:
                continue
            real_path = os.path.realpath(path)
            if os.path.isdir(real_path) and real_path not in retained:
                retained.append(real_path)
        for _, path in completed:
            real_path = os.path.realpath(path)
            if len(retained) >= int(keep):
                break
            if real_path not in retained:
                retained.append(real_path)
        for _, path in completed:
            if os.path.realpath(path) not in retained:
                shutil.rmtree(path)

        # After two checkpoints have been published with the new layout, the
        # one-time legacy fallback is superseded by two independently complete
        # versioned states and no longer needs to consume a third full copy.
        legacy_path = os.path.join(self.states_root_dir, "previous-legacy")
        if (
            len(retained) >= int(keep)
            and os.path.isdir(legacy_path)
            and os.path.realpath(legacy_path) not in retained
        ):
            shutil.rmtree(legacy_path)

    def state_dict(self):
        return {
            "num_steps": self.num_steps,
            "epoch_id": self.epoch_id,
            "batch_in_epoch": self.batch_in_epoch,
            "run_id": self.run_manifest.get("run_id"),
            "config_fingerprint": self.run_manifest.get("config_fingerprint"),
            "rng_scheme": self.run_manifest.get("training_identity", {}).get("rng_scheme"),
        }

    def load_state_dict(self, state):
        self.num_steps = state["num_steps"]
        self.epoch_id = state["epoch_id"]
        self.batch_in_epoch = state["batch_in_epoch"]
        self.loaded_run_id = state.get("run_id")
        self.loaded_config_fingerprint = state.get("config_fingerprint")
        self.loaded_rng_scheme = state.get("rng_scheme")

    def validate_resume_identity(self, allow_unsafe_resume=False):
        expected = {
            "run_id": self.run_manifest.get("run_id"),
            "config_fingerprint": self.run_manifest.get("config_fingerprint"),
            "rng_scheme": self.run_manifest.get("training_identity", {}).get("rng_scheme"),
        }
        loaded = {
            "run_id": self.loaded_run_id,
            "config_fingerprint": self.loaded_config_fingerprint,
            "rng_scheme": self.loaded_rng_scheme,
        }
        mismatches = [
            f"{name}: checkpoint={loaded[name]!r}, current={expected[name]!r}"
            for name in expected
            if loaded[name] != expected[name]
        ]
        if mismatches and not allow_unsafe_resume:
            raise RuntimeError(
                "Checkpoint identity does not match the run manifest:\n  - " + "\n  - ".join(mismatches)
            )
        return mismatches

    def set_progress(self, epoch_id, batch_in_epoch):
        self.epoch_id = epoch_id
        self.batch_in_epoch = batch_in_epoch

    def advance_step(self, epoch_id, batch_in_epoch):
        self.num_steps += 1
        self.set_progress(epoch_id, batch_in_epoch)

    def should_save(self, save_steps):
        return self.num_steps % int(save_steps) == 0

    def save_checkpoint(self, accelerator, model):
        self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        self.save_training_state(accelerator)

    def on_training_end(self, accelerator, model, save_steps):
        if self.num_steps % save_steps != 0:
            self.save_checkpoint(accelerator, model)

    @staticmethod
    def _state_dict_requires_collective(accelerator):
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            stage = int(accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"])
            return stage == 3
        return accelerator.distributed_type in (DistributedType.FSDP, DistributedType.XLA)

    def _export_main_process_state_dict(self, accelerator, model):
        unwrapped_model = accelerator.unwrap_model(model)
        # ZeRO-2 keeps a full parameter copy on every rank. Filter GPU views to
        # trainable weights before making exactly one CPU copy on the main rank.
        raw_state_dict = unwrapped_model.state_dict()
        state_dict = unwrapped_model.export_trainable_state_dict(
            raw_state_dict,
            remove_prefix=self.remove_prefix_in_ckpt,
        )
        state_dict = self.state_dict_converter(state_dict)
        return {
            name: tensor.detach().to(device="cpu", copy=True).contiguous()
            if torch.is_tensor(tensor)
            else tensor
            for name, tensor in state_dict.items()
        }

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        collective_state_dict = self._state_dict_requires_collective(accelerator)
        state_dict = accelerator.get_state_dict(model) if collective_state_dict else None
        error = None
        if accelerator.is_main_process:
            try:
                if collective_state_dict:
                    state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                        state_dict,
                        remove_prefix=self.remove_prefix_in_ckpt,
                    )
                    state_dict = self.state_dict_converter(state_dict)
                    state_dict = {
                        name: tensor.detach().to(device="cpu", copy=True).contiguous()
                        if torch.is_tensor(tensor)
                        else tensor
                        for name, tensor in state_dict.items()
                    }
                else:
                    state_dict = self._export_main_process_state_dict(accelerator, model)

                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, file_name)
                temporary_path = f"{path}.incomplete"
                if os.path.lexists(temporary_path):
                    os.unlink(temporary_path)
                accelerator.save(state_dict, temporary_path, safe_serialization=True)
                with open(temporary_path, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary_path, path)
                _fsync_directory(self.output_path)
            except Exception as exc:
                error = f"Model checkpoint save failed: {type(exc).__name__}: {exc}"
            finally:
                state_dict = None
                gc.collect()
        _raise_main_process_error_on_all_ranks(accelerator, error)

    def save_training_state(self, accelerator):
        accelerator.wait_for_everyone()
        final_state_dir = self.state_dir_for_step()
        staging_state_dir = os.path.join(
            self.states_root_dir,
            f".step-{self.num_steps}.incomplete",
        )

        error = None
        if accelerator.is_main_process:
            try:
                os.makedirs(self.states_root_dir, exist_ok=True)
                if os.path.lexists(staging_state_dir):
                    if os.path.isdir(staging_state_dir) and not os.path.islink(staging_state_dir):
                        shutil.rmtree(staging_state_dir)
                    else:
                        os.unlink(staging_state_dir)
                if os.path.lexists(final_state_dir):
                    raise FileExistsError(f"Refusing to overwrite completed training state: {final_state_dir}")
            except Exception as exc:
                error = f"Training-state staging failed: {type(exc).__name__}: {exc}"
        _raise_main_process_error_on_all_ranks(accelerator, error)
        accelerator.wait_for_everyone()

        # The existing latest pointer remains untouched until every rank has
        # successfully written the new state into the staging directory.
        # Frozen Wan/VAE parameters are reloaded from the immutable base-model
        # paths recorded in the run manifest. Saving them in every DeepSpeed
        # state adds ~3 GiB and, more importantly, makes all eight ranks
        # deserialize the same large CPU state concurrently on resume.
        accelerator.save_state(staging_state_dir, exclude_frozen_parameters=True)
        accelerator.wait_for_everyone()

        error = None
        if accelerator.is_main_process:
            try:
                completion_metadata = {
                    "num_steps": self.num_steps,
                    "epoch_id": self.epoch_id,
                    "batch_in_epoch": self.batch_in_epoch,
                    "run_id": self.run_manifest.get("run_id"),
                    "config_fingerprint": self.run_manifest.get("config_fingerprint"),
                }
                self._atomic_write(
                    os.path.join(staging_state_dir, "COMPLETE"),
                    json.dumps(completion_metadata, indent=2, sort_keys=True) + "\n",
                )
                os.replace(staging_state_dir, final_state_dir)
                _fsync_directory(self.states_root_dir)
                previous_state_dir = self._install_latest_symlink(final_state_dir)
                metadata = {
                    "num_steps": self.num_steps,
                    "epoch_id": self.epoch_id,
                    "batch_in_epoch": self.batch_in_epoch,
                    "state_dir": final_state_dir,
                    "latest_state_dir": self.latest_state_dir,
                    "previous_state_dir": previous_state_dir,
                    "run_id": self.run_manifest.get("run_id"),
                    "config_fingerprint": self.run_manifest.get("config_fingerprint"),
                }
                self._atomic_write(
                    self.latest_state_metadata_path,
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                )
                self._prune_completed_states(
                    keep=2,
                    protected_paths=(final_state_dir, previous_state_dir),
                )
            except Exception as exc:
                error = f"Training-state publication failed: {type(exc).__name__}: {exc}"
        _raise_main_process_error_on_all_ranks(accelerator, error)
        accelerator.wait_for_everyone()
