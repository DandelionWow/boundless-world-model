import datetime
import os
import random
import time

from accelerate import skip_first_batches
from accelerate.utils import DistributedType
import numpy as np
import torch
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from wan_video_action.logger import TrainingLogger
from wan_video_action.utils import set_global_seed


def collate_training_samples(samples):
    """Collate dataset samples without conflating batch and view dimensions."""
    if not samples:
        raise ValueError("Cannot collate an empty training batch.")
    batch = default_collate(samples)
    if not isinstance(batch, dict):
        raise TypeError(f"Expected a dictionary batch, got {type(batch).__name__}.")

    # Dataset action tensors have shape (1,T,D), where the leading dimension is
    # a singleton sample dimension. default_collate produces (B,1,T,D); training
    # expects the conventional (B,T,D) layout.
    action = batch.get("action")
    if torch.is_tensor(action) and action.ndim == 4 and action.shape[1] == 1:
        batch["action"] = action[:, 0]
    batch["_batch_size"] = len(samples)
    return batch


def reduce_global_mean_loss(accelerator, local_loss_sum, local_sample_count):
    """Return the sample-weighted mean loss over every process and micro-batch."""
    if local_loss_sum is None or local_sample_count <= 0:
        raise RuntimeError("No local loss samples were accumulated before an optimizer step.")
    local_loss_stats = torch.stack(
        [
            local_loss_sum,
            local_loss_sum.new_tensor(float(local_sample_count)),
        ]
    )
    global_loss_stats = accelerator.reduce(local_loss_stats, reduction="sum")
    return (global_loss_stats[0] / global_loss_stats[1]).item()


def ensure_global_finite(accelerator, value, name):
    """Fail every rank together before an invalid optimizer update is applied."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=accelerator.device)
    local_invalid = (~torch.isfinite(value.detach())).any().to(
        device=accelerator.device,
        dtype=torch.int32,
    )
    global_invalid = accelerator.reduce(local_invalid, reduction="max")
    if int(global_invalid.item()) != 0:
        raise FloatingPointError(f"Non-finite {name} detected on at least one training rank.")


def load_trainable_module_state(src, dst):
    """Load a compact DeepSpeed module state and verify every trainable weight."""
    load_result = dst.load_state_dict(src, strict=False)
    trainable_names = {name for name, parameter in dst.named_parameters() if parameter.requires_grad}
    missing_trainable = sorted(trainable_names.intersection(load_result.missing_keys))
    unexpected = sorted(load_result.unexpected_keys)
    if missing_trainable or unexpected:
        raise RuntimeError(
            "Compact checkpoint does not match the trainable model: "
            f"missing_trainable={missing_trainable}, unexpected={unexpected}"
        )


def launch_training_task(
    accelerator,
    dataset,
    model,
    model_logger,
    args,
):
    os.makedirs(args.output_path, exist_ok=True)

    optimizer_kwargs = {}
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        optimizer_kwargs = {"foreach": False, "fused": False}
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        **optimizer_kwargs,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "collate_fn": collate_training_samples,
        "num_workers": args.dataset_num_workers,
        "pin_memory": True,
    }
    if args.dataset_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 16
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    accelerator.register_for_checkpointing(model_logger)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    start_epoch = 0
    loaded_checkpoint_rng_state = None
    if args.resume_from:
        accelerator.print(f"Resuming training from full state: {args.resume_from}")
        accelerator.load_state(args.resume_from, custom_load_fn=load_trainable_module_state)
        loaded_checkpoint_rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        identity_warnings = model_logger.validate_resume_identity(
            allow_unsafe_resume=args.allow_unsafe_resume,
        )
        for warning in identity_warnings:
            accelerator.print(f"[unsafe_resume_warning] {warning}")
        start_epoch = model_logger.epoch_id
    else:
        # Model initialization must be identical on all ranks, so the global
        # seed is used until after Accelerator prepares the model. Training
        # randomness is then rank-specific while the sampler keeps data_seed.
        process_seed = int(args.seed) + int(accelerator.process_index)
        set_global_seed(process_seed, deterministic=args.deterministic)
        accelerator.print(
            "[resolved_config] training_rng_scheme: seed + process_index "
            f"(rank0_seed={args.seed})"
        )

    training_logger = TrainingLogger(
        accelerator,
        args.output_path,
        args=args,
        resume_step=model_logger.num_steps if args.resume_from else None,
    )
    training_logger.init_trackers()
    if loaded_checkpoint_rng_state is not None:
        # Tracker initialization and log reconciliation must not perturb the
        # stochastic trajectory restored from the checkpoint.
        random.setstate(loaded_checkpoint_rng_state["python"])
        np.random.set_state(loaded_checkpoint_rng_state["numpy"])
        torch.set_rng_state(loaded_checkpoint_rng_state["torch"])
        if loaded_checkpoint_rng_state["cuda"] is not None:
            torch.cuda.set_rng_state_all(loaded_checkpoint_rng_state["cuda"])

    initial_num_steps = int(model_logger.num_steps)
    if args.max_train_steps < initial_num_steps:
        raise ValueError(
            f"max_train_steps={args.max_train_steps} is behind checkpoint step={initial_num_steps}."
        )

    epoch_id = start_epoch
    skip_batches = model_logger.batch_in_epoch if args.resume_from else 0
    if skip_batches > len(dataloader):
        raise ValueError(
            f"Checkpoint batch_in_epoch={skip_batches} exceeds current dataloader length={len(dataloader)}."
        )
    if args.resume_from and skip_batches == len(dataloader):
        epoch_id += 1
        skip_batches = 0
    resume_rng_state = None
    if skip_batches:
        resume_rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    resume_rng_restored = False
    last_step_time = time.monotonic()
    while model_logger.num_steps < args.max_train_steps:
        local_loss_sum = None
        local_sample_count = 0
        optimizer.zero_grad()
        epoch_label = (epoch_id + 1) * args.dataset_repeat - 1

        # Recreate this epoch's permutation before skipping completed batches on resume.
        dataloader.set_epoch(epoch_id)
        epoch_dataloader = skip_first_batches(dataloader, skip_batches) if skip_batches else dataloader
        progress_bar = tqdm(epoch_dataloader, disable=not accelerator.is_local_main_process)
        for local_batch_idx, data in enumerate(progress_bar):
            batch_idx = skip_batches + local_batch_idx
            if resume_rng_state is not None and not resume_rng_restored:
                random.setstate(resume_rng_state["python"])
                np.random.set_state(resume_rng_state["numpy"])
                torch.set_rng_state(resume_rng_state["torch"])
                if resume_rng_state["cuda"] is not None:
                    torch.cuda.set_rng_state_all(resume_rng_state["cuda"])
                resume_rng_restored = True
            with accelerator.accumulate(model):
                if getattr(dataset, "load_from_cache", False):
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                ensure_global_finite(accelerator, loss, "loss")
                batch_sample_count = int(data.get("_batch_size", args.batch_size))
                detached_loss = loss.detach().float()
                weighted_loss = detached_loss * batch_sample_count
                local_loss_sum = weighted_loss if local_loss_sum is None else local_loss_sum + weighted_loss
                local_sample_count += batch_sample_count
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm_value = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    ensure_global_finite(accelerator, grad_norm_value, "gradient norm")
                    grad_norm = float(grad_norm_value)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    now = time.monotonic()
                    step_time = now - last_step_time
                    last_step_time = now

                    avg_loss = reduce_global_mean_loss(
                        accelerator,
                        local_loss_sum,
                        local_sample_count,
                    )
                    model_logger.advance_step(
                        epoch_id=epoch_id,
                        batch_in_epoch=batch_idx + 1,
                    )
                    local_loss_sum = None
                    local_sample_count = 0

                    learning_rate = scheduler.get_last_lr()[0]
                    metrics = {
                        "step": model_logger.num_steps,
                        "epoch": epoch_label,
                        "batch_in_epoch": batch_idx + 1,
                        "loss": avg_loss,
                        "lr": learning_rate,
                        "grad_norm": grad_norm,
                    }
                    training_logger.log_step(metrics, step=model_logger.num_steps)

                    postfix = {
                        "step": model_logger.num_steps,
                        "loss": avg_loss,
                        "lr": learning_rate,
                        "grad_norm": grad_norm,
                    }
                    next_save_steps = args.save_steps - (model_logger.num_steps % args.save_steps)
                    if next_save_steps == 0:
                        next_save_steps = args.save_steps
                    eta_steps = next_save_steps
                    eta_label = "next_save_eta"
                    remaining_train_steps = args.max_train_steps - model_logger.num_steps
                    if remaining_train_steps < eta_steps:
                        eta_steps = remaining_train_steps
                        eta_label = "train_end_eta"
                    postfix["next_save_steps"] = next_save_steps
                    postfix[eta_label] = str(datetime.timedelta(seconds=max(0, int(step_time * eta_steps))))
                    training_logger.update_progress_bar(progress_bar, postfix)

                    # Metrics are durable before checkpoint publication. If a
                    # later save fails, resume reconciliation truncates them
                    # back to the last complete training state.
                    if model_logger.should_save(args.save_steps):
                        model_logger.save_checkpoint(accelerator, model)

                    if model_logger.num_steps >= args.max_train_steps:
                        break
        skip_batches = 0
        epoch_id += 1
    if args.save_at_train_end and model_logger.num_steps > initial_num_steps:
        model_logger.on_training_end(accelerator, model, args.save_steps)
    training_logger.close()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
