import json
import os
import shutil

from diffsynth.diffusion.runner import ModelLogger as DiffSynthModelLogger


class TrainingLogger:
    def __init__(self, accelerator, output_path, args):
        self.accelerator = accelerator
        self.output_path = output_path
        self.args = args
        self.metric_log_path = os.path.join(output_path, "train_metrics.jsonl")
        self.tracker_names = [name for name in ("wandb", "swanlab") if getattr(args, f"use_{name}")]

        if accelerator.is_main_process:
            os.makedirs(output_path, exist_ok=True)
        accelerator.wait_for_everyone()
        metric_log_path = self.metric_log_path if accelerator.is_main_process else os.devnull
        self.metric_log_file = open(metric_log_path, "a", encoding="utf-8")

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
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        super().__init__(
            output_path,
            remove_prefix_in_ckpt=remove_prefix_in_ckpt,
            state_dict_converter=state_dict_converter,
        )
        self.epoch_id = 0
        self.batch_in_epoch = 0

    @property
    def latest_state_dir(self):
        return os.path.join(self.output_path, "states", "latest")

    @property
    def latest_state_metadata_path(self):
        return os.path.join(self.output_path, "states", "latest.json")

    def state_dict(self):
        return {
            "num_steps": self.num_steps,
            "epoch_id": self.epoch_id,
            "batch_in_epoch": self.batch_in_epoch,
        }

    def load_state_dict(self, state):
        self.num_steps = state["num_steps"]
        self.epoch_id = state["epoch_id"]
        self.batch_in_epoch = state["batch_in_epoch"]

    def set_progress(self, epoch_id, batch_in_epoch):
        self.epoch_id = epoch_id
        self.batch_in_epoch = batch_in_epoch

    def on_step_end(
        self,
        accelerator,
        model,
        save_steps,
        epoch_id,
        batch_in_epoch,
    ):
        self.num_steps += 1
        self.set_progress(epoch_id, batch_in_epoch)
        if self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            self.save_training_state(accelerator)

    def on_training_end(self, accelerator, model, save_steps):
        if self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            self.save_training_state(accelerator)

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict,
                remove_prefix=self.remove_prefix_in_ckpt,
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

    def save_training_state(self, accelerator):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and os.path.isdir(self.latest_state_dir):
            shutil.rmtree(self.latest_state_dir)
        accelerator.wait_for_everyone()
        accelerator.save_state(self.latest_state_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metadata = {
                "num_steps": self.num_steps,
                "epoch_id": self.epoch_id,
                "batch_in_epoch": self.batch_in_epoch,
                "state_dir": self.latest_state_dir,
            }
            os.makedirs(os.path.dirname(self.latest_state_metadata_path), exist_ok=True)
            with open(self.latest_state_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.write("\n")
        accelerator.wait_for_everyone()
