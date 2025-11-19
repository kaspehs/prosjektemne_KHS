import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
import yaml

from HNN_helper import *
from ODE_pinn_helper import LrSchedule, WarmupCosineLrSchedule

def main(config: Config, config_name: str):
    data_cfg = config.data
    data_path = Path(data_cfg.file)
    data = np.load(data_path)
    t = data["a"]
    y_data = data["b"]
    F_data = data["c"]
    H_data = data["d"]

    middle_time_plot = data_cfg.middle_time_plot
    use_generated_train_series = data_cfg.use_generated_train_series
    train_series_dir = Path(data_cfg.train_series_dir)

    t, y_data, F_data, hamiltonian_data, dt = preprocess_timeseries(
        t,
        y_data,
        F_data,
        H_data,
        data_cfg,
    )

    # ===== 2. model =====
    model_cfg = config.model
    smoothing_cfg = config.smoothing

    training_cfg = config.training
    batch_size = training_cfg.batch_size
    force_reg = training_cfg.force_reg
    max_grad_norm = training_cfg.max_grad_norm
    lr = training_cfg.lr
    epochs = training_cfg.epochs
    rollout_every_epoch = training_cfg.rollout_every_epoch
    use_lr_scheduler = training_cfg.use_lr_scheduler
    scheduler_cfg = training_cfg.scheduler
    max_lr = scheduler_cfg.max_lr
    decay_rate = scheduler_cfg.decay_rate
    scheduler_warmup_steps = scheduler_cfg.warmup_steps
    decay_steps = scheduler_cfg.decay_steps

    device = torch.device("cpu")

    model_dict = asdict(model_cfg)
    arch_dict = asdict(config.architecture)
    model, derived_params = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    D = derived_params["D"]
    k = derived_params["k"]
    m_eff = derived_params["m_eff"]

    train_series_raw, eval_tensors = load_training_series(
        y_data,
        t,
        dt,
        use_generated_train_series,
        train_series_dir,
        m_eff,
        device,
        smoothing_cfg=smoothing_cfg,
    )
    eval_y_tensor, eval_vel_tensor, eval_t_tensor = eval_tensors

    train_loader, train_sequences, _ = build_dataloader_from_series(
        train_series_raw,
        m_eff=m_eff,
        batch_size=batch_size,
        device=device,
        smoothing_cfg=smoothing_cfg,
    )

    if use_generated_train_series:
        y_data_t, val_vel, t_tensor = eval_y_tensor, eval_vel_tensor, eval_t_tensor
    else:
        y_data_t, val_vel, t_tensor = train_sequences[0]

    logging_cfg = config.logging
    run_dir_root = logging_cfg.run_dir_root
    timestamp = time.strftime("%m%d-%H%M%S")
    run_dir = os.path.join(run_dir_root, f"{config_name}_{timestamp}")
    writer = SummaryWriter(log_dir=run_dir)

    y_true_norm = y_data / D
    force_data = F_data

    opt = optim.Adam(model.parameters(), lr=lr)
    #lr_scheduler = LrSchedule(max_lr, decay_rate, scheduler_warmup_steps, decay_steps)
    lr_scheduler = WarmupCosineLrSchedule(max_lr, 0.02*max_lr, scheduler_warmup_steps, decay_steps)

    for epoch in range(epochs):

        #Updates learningrate
        if use_lr_scheduler:
            for g in opt.param_groups:
                g["lr"] = lr_scheduler.get_lr(epoch)


        losses: list[float] = []
        res_losses: list[float] = []
        force_losses: list[float] = []
        grad_norms: list[float] = []
        avg_forces: list[float] = []
        res_grad_components: list[float] = []
        force_grad_components: list[float] = []

        for z_i, t_i, z_next, t_next in train_loader:
            z_i = z_i.to(device)
            t_i = t_i.to(device)
            z_next = z_next.to(device)
            t_next = t_next.to(device)

            opt.zero_grad()

            res_loss = model.res_loss(z_i, t_i, z_next, t_next)
            avg_force = model.avg_force(z_i, t_i, z_next, t_next)
            force_loss = force_reg * avg_force
            loss = res_loss + force_loss

            res_loss.backward(retain_graph=True)
            res_grad_components.append(compute_model_grad_norm(model))
            model.zero_grad(set_to_none=True)
            force_loss.backward(retain_graph=True)
            force_grad_components.append(compute_model_grad_norm(model))
            model.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            grad_norms.append(float(grad_norm.detach().cpu()))
            opt.step()

            losses.append(float(loss.detach().cpu()))
            res_losses.append(float(res_loss.detach().cpu()))
            force_losses.append(float(force_loss.detach().cpu()))
            avg_forces.append(float(avg_force.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_res_loss = float(np.mean(res_losses)) if res_losses else 0.0
        mean_force_loss = float(np.mean(force_losses)) if force_losses else 0.0
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        drag_coeff = float(torch.exp(model.log_Cd).detach().cpu())
        mean_force = float(np.mean(avg_forces)) if avg_forces else 0.0
        current_lr = float(opt.param_groups[0]["lr"]) if opt.param_groups else lr
        mean_res_grad_component = float(np.mean(res_grad_components)) if res_grad_components else 0.0
        mean_force_grad_component = float(np.mean(force_grad_components)) if force_grad_components else 0.0

        damping_ratio_value = (
            float(torch.sigmoid(model.zeta_raw).detach().cpu()) * model.max_damping_ratio
            if getattr(model, "discover_damping", True)
            else float(model.fixed_damping_ratio)
        )
        
        train_metrics = {
            "loss": mean_loss,
            "residual_loss": mean_res_loss,
            "force_loss": mean_force_loss,
            "learning_rate": current_lr,
            "damping_ratio": damping_ratio_value,
            "grad_norm": mean_grad_norm,
            "drag_coefficient": drag_coeff,
            "avg_force": mean_force,
            "grad_norm_residual_comp": mean_res_grad_component,
            "grad_norm_force_comp": mean_force_grad_component,
        }
        log_training_metrics(writer, epoch, train_metrics)
        print(
            f"Epoch {epoch}: loss={mean_loss:.4e}, res={mean_res_loss:.4e}, "
            f"force={mean_force_loss:.4e}"
        )

        if (epoch + 1) % rollout_every_epoch == 0:
            val_metrics = log_validation_epoch(
                writer,
                epoch + 1,
                model,
                y_data_t,
                val_vel,
                m_eff,
                dt,
                t,
                y_true_norm,
                y_data,
                force_data,
                D,
                k,
                device,
                middle_time_plot,
                hamiltonian_data,
            )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HNN model with YAML configuration.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("hnn_config.yml"),
        help="Path to YAML config file (default: hnn_config.yml)",
    )
    args = parser.parse_args()
    raw_cfg = load_config(args.config)
    cfg = parse_config(raw_cfg)
    config_name = Path(args.config).stem
    main(cfg, config_name)
