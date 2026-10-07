import datetime
import os
import random
import time

from accelerate import skip_first_batches
from accelerate.utils import DistributedType
import numpy as np
import torch
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from tqdm import tqdm

from wan_video_action.logger import TrainingLogger


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
        "shuffle": False,
        "collate_fn": lambda x: x[0],
        "num_workers": args.dataset_num_workers,
        "pin_memory": True,
    }
    if args.dataset_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 16
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    training_logger = TrainingLogger(accelerator, args.output_path, args=args)
    training_logger.init_trackers()
    accelerator.register_for_checkpointing(model_logger)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    start_epoch = 0
    if args.resume_from:
        accelerator.print(f"Resuming training from full state: {args.resume_from}")
        accelerator.load_state(args.resume_from)
        start_epoch = model_logger.epoch_id

    epoch_id = start_epoch
    skip_batches = model_logger.batch_in_epoch if args.resume_from else 0
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
        train_loss = 0.0
        loss_count = 0
        optimizer.zero_grad()
        epoch_label = (epoch_id + 1) * args.dataset_repeat - 1

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
                loss_item = loss.detach().float().item()
                train_loss += loss_item
                loss_count += 1
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    now = time.monotonic()
                    step_time = now - last_step_time
                    last_step_time = now

                    avg_loss = train_loss / loss_count
                    model_logger.on_step_end(
                        accelerator,
                        model,
                        args.save_steps,
                        epoch_id=epoch_id,
                        batch_in_epoch=batch_idx + 1,
                    )
                    train_loss = 0.0
                    loss_count = 0

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

                    if model_logger.num_steps >= args.max_train_steps:
                        break
        skip_batches = 0
        epoch_id += 1
    model_logger.on_training_end(accelerator, model, args.save_steps)
    training_logger.close()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
