import torch


def FlowMatchSFTLossWanAction(pipe, **inputs):
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    input_latents = inputs["input_latents"]
    noise = inputs["noise"]

    inputs["latents"] = pipe.scheduler.add_noise(input_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

    history_t = int(inputs.get("fused_condition_latent_frames") or 0)
    history_condition_latents = inputs.get("history_condition_latents")
    if history_t > 0:
        inputs["latents"][:, :, :history_t] = history_condition_latents[:, :, :history_t]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    if pipe.action_injection_mode == "adaln" and history_t > 0:
        noise_pred = noise_pred[:, :, history_t:]
        training_target = training_target[:, :, history_t:]

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
