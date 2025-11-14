import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import yaml
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

from HNN_helper import *
from ODE_pinn_helper import LrSchedule

def main():
    # ===== 1. load data =====
    data = np.load("data.npz")
    t = data["a"]
    y_data = data["b"]
    F_data = data["c"]
    H_data = data["d"]
    steadystate = False
    reduce_time = True
    reduction_factor = 10

    if steadystate:
        mask = t > 10
        t = t[mask]
        y_data = y_data[mask]
        F_data = F_data[mask]
        H_data = H_data[mask]
    
    if reduce_time:
        t = t[::reduction_factor]
        y_data = y_data[::reduction_factor]
        F_data = F_data[::reduction_factor]
        H_data = H_data[::reduction_factor]
    
    dt = float(t[1] - t[0])
    middle_time_plot = [15, 17]

    # ===== 2. model =====
    rho = 1000.0
    D = 0.1
    Ca = 1.0
    m_a = 0.25 * np.pi * D**2 * rho * Ca
    m_eff = 16.79 + m_a
    k = 1218.0
    U = 0.65
    C = 1e-4
    q_scale = D
    p_scale = np.sqrt(k / m_eff) * m_eff * D
    include_physical_drag = False  # Set to False to let NN learn the total force directly
    learn_hamiltonian = False
    discover_damping = False  # Set to True to learn the Hamiltonian with a neural network
    use_pirate_force = False
    pirate_force_kwargs: dict[str, object] = {}
    use_fourier_features = False  # Enable Random Fourier Features before the force network
    fourier_features = 64
    fourier_sigma = 1.0
    use_feature_engineering = False  # Enable handcrafted features for force and Hamiltonian nets
    use_generated_train_series = False # Load multiple training series from disk
    train_series_dir = "Data_Gen/generated_series"  # Directory containing *.npz series
    use_savgol_smoothing = True
    savgol_window_length = 15
    savgol_polyorder = 4

    device = torch.device("cpu")

    def finite_difference_velocity_numpy(y_np: np.ndarray, dt_value: float) -> np.ndarray:
        vel_np = np.zeros_like(y_np)
        if y_np.shape[0] >= 2:
            vel_np[0] = (y_np[1] - y_np[0]) / dt_value
            vel_np[-1] = (y_np[-1] - y_np[-2]) / dt_value
        if y_np.shape[0] > 2:
            vel_np[1:-1] = (y_np[2:] - y_np[:-2]) / (2.0 * dt_value)
        return vel_np

    def compute_velocity_numpy(y_np: np.ndarray, dt_value: float) -> np.ndarray:
        if (
            use_savgol_smoothing
            and savgol_filter is not None
            and y_np.shape[0] >= 3
        ):
            window = min(savgol_window_length, y_np.shape[0])
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                polyorder = min(savgol_polyorder, window - 1)
                try:
                    vel_savgol = savgol_filter(
                        y_np,
                        window_length=window,
                        polyorder=polyorder,
                        deriv=1,
                        delta=dt_value,
                        axis=0,
                        mode="interp",
                    )
                    return np.ascontiguousarray(vel_savgol)
                except ValueError as exc:
                    print(f"Savitzky-Golay derivative failed ({exc}); using finite differences.")
        elif use_savgol_smoothing and savgol_filter is None:
            print("Savitzky-Golay derivative requested but SciPy is unavailable; using finite differences.")
        return finite_difference_velocity_numpy(y_np, dt_value)

    y_data_t = torch.from_numpy(y_data).float().to(device)
    val_vel_np = compute_velocity_numpy(y_data, dt)
    val_vel = torch.from_numpy(val_vel_np).float().to(device)
    t_tensor = torch.from_numpy(t).float().to(device)

    if use_generated_train_series:
        series_root = Path(train_series_dir)
        if not series_root.exists():
            raise FileNotFoundError(f"Training series directory '{series_root}' does not exist.")
        series_files = sorted(series_root.glob("*.npz"))
        if not series_files:
            raise FileNotFoundError(f"No '.npz' files found in training series directory '{series_root}'.")
        train_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for series_file in series_files:
            series_data = np.load(series_file)
            series_t = np.asarray(series_data["a"])
            series_y = np.asarray(series_data["b"])
            if series_t.ndim != 1 or series_y.ndim != 1:
                raise ValueError(f"Series '{series_file}' must contain 1D 'a' (time) and 'b' (displacement) arrays.")
            if series_t.shape[0] != series_y.shape[0]:
                raise ValueError(f"Series '{series_file}' has mismatched time and displacement lengths.")
            if series_t.shape[0] < 2:
                raise ValueError(f"Series '{series_file}' is too short to build training samples.")
            series_dt = float(series_t[1] - series_t[0])
            if not np.allclose(np.diff(series_t), series_dt, rtol=1e-6, atol=1e-9):
                raise ValueError(f"Series '{series_file}' time vector is not uniform.")
            if not np.isclose(series_dt, dt, rtol=1e-6, atol=1e-9):
                raise ValueError(
                    f"Series '{series_file}' uses dt={series_dt:.6f}, which differs from base dt={dt:.6f}."
                )
            vel_np = compute_velocity_numpy(series_y, dt)
            train_sequences.append(
                (
                    torch.from_numpy(series_y).float().to(device),
                    torch.from_numpy(vel_np).float().to(device),
                    torch.from_numpy(series_t).float().to(device),
                )
            )
        print(f"Loaded {len(train_sequences)} training series from '{series_root}'.")
    else:
        train_sequences = [(y_data_t, val_vel, t_tensor)]

    model = PHVIV(
        dt,
        m=m_eff,
        k=k,
        U=U,
        rho=rho,
        D=D,
        q_scale=q_scale,
        p_scale=p_scale,
        discover_damping=discover_damping,
        damping_c=C,
        include_physical_drag=include_physical_drag,
        learn_hamiltonian=learn_hamiltonian,
        use_pirate_force=use_pirate_force,
        pirate_force_kwargs=pirate_force_kwargs,
        use_fourier_features=use_fourier_features,
        fourier_features=fourier_features,
        fourier_sigma=fourier_sigma,
        use_feature_engineering=use_feature_engineering,
    ).to(device)

    run_dir = os.path.join("HNNruns", f"hnn_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=run_dir)

    batch_size = 32
    force_reg = 1e-2
    max_grad_norm = 1e4
    lr = 1e-3
    use_rollout_loss = False
    max_rollout_weight = 1e8
    rollout_steps = 64
    rollout_batch_size = 16
    warmup_epochs = 100
    ramp_epochs = 100

    use_lr_scheduler = False
    max_lr = 5e-4
    decay_rate = 0.9
    warmup_steps = 1000
    decay_steps = 1000

    hamiltonian_data = H_data

    t_tensor = torch.from_numpy(t).float().to(device)
    def combine_datasets(datasets: list[TensorDataset | ConcatDataset]) -> TensorDataset | ConcatDataset:
        if not datasets:
            raise ValueError("No datasets provided for training.")
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    dataset_list = [
        build_dataset(y_seq, vel_seq, m_eff, t_seq) for (y_seq, vel_seq, t_seq) in train_sequences
    ]
    dataset = combine_datasets(dataset_list)
    min_train_length = min(y_seq.shape[0] for (y_seq, _, _) in train_sequences)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    rollout_loader = None
    if use_rollout_loss:
        max_possible_steps = min_train_length - 1
        if max_possible_steps < 1:
            use_rollout_loss = False
        else:
            rollout_steps = min(rollout_steps, max_possible_steps)
            rollout_datasets = []
            for (y_seq, vel_seq, t_seq) in train_sequences:
                try:
                    rollout_ds = build_rollout_dataset(
                        y_seq, vel_seq, m_eff, t_seq, rollout_steps=rollout_steps
                    )
                except ValueError:
                    continue
                rollout_datasets.append(rollout_ds)
            if rollout_datasets:
                rollout_dataset = combine_datasets(rollout_datasets)
                rollout_loader = DataLoader(
                    rollout_dataset,
                    batch_size=rollout_batch_size,
                    shuffle=True,
                )
            else:
                use_rollout_loss = False

    y_true_norm = y_data / D
    force_data = F_data

    epochs = 2000
    rollout_every_epoch = 50
    opt = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = LrSchedule(max_lr, decay_rate, warmup_steps, decay_steps)

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
        rollout_losses: list[float] = []
        rollout_weight = (
            0.0
            if epoch < warmup_epochs
            else max_rollout_weight
            * min(
                1.0,
                (epoch - warmup_epochs) / max(ramp_epochs, 1),
            )
        )
        active_rollout = use_rollout_loss and rollout_loader is not None and rollout_weight > 0.0
        rollout_iter = iter(rollout_loader) if active_rollout else None

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
            rollout_loss_term = torch.zeros((), device=device)
            if active_rollout:
                try:
                    z0_batch, t_seq_batch, z_traj_batch = next(rollout_iter)
                except StopIteration:
                    rollout_iter = iter(rollout_loader)
                    z0_batch, t_seq_batch, z_traj_batch = next(rollout_iter)
                z0_batch = z0_batch.to(device)
                t_seq_batch = t_seq_batch.to(device)
                z_traj_batch = z_traj_batch.to(device)
                z_pred_seq, _ = model.rollout(z0_batch, t_seq_batch, dt)
                rollout_y_pred = z_pred_seq[..., 0]
                rollout_y_true = z_traj_batch[..., 0]
                rollout_loss_term = torch.mean((rollout_y_pred - rollout_y_true) ** 2)
                loss = loss + rollout_weight * rollout_loss_term
            loss.backward()

            grad_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            grad_norms.append(float(grad_norm.detach().cpu()))
            opt.step()

            losses.append(float(loss.detach().cpu()))
            res_losses.append(float(res_loss.detach().cpu()))
            force_losses.append(float(force_loss.detach().cpu()))
            avg_forces.append(float(avg_force.detach().cpu()))
            if active_rollout:
                rollout_losses.append(rollout_weight * float(rollout_loss_term.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_res_loss = float(np.mean(res_losses)) if res_losses else 0.0
        mean_force_loss = float(np.mean(force_losses)) if force_losses else 0.0
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        drag_coeff = float(torch.exp(model.log_Cd).detach().cpu())
        mean_force = float(np.mean(avg_forces)) if avg_forces else 0.0
        mean_rollout_loss = float(np.mean(rollout_losses)) if rollout_losses else 0.0
        current_lr = float(opt.param_groups[0]["lr"]) if opt.param_groups else lr

        writer.add_scalar("train/loss", mean_loss, epoch)
        writer.add_scalar("train/residual_loss", mean_res_loss, epoch)
        writer.add_scalar("train/force_loss", mean_force_loss, epoch)
        writer.add_scalar("train/learning_rate", current_lr, epoch)
        if active_rollout:
            writer.add_scalar("train/rollout_y_mse", mean_rollout_loss, epoch)
        damping_ratio_value = (
            float(torch.sigmoid(model.zeta_raw).detach().cpu()) * model.max_damping_ratio
            if getattr(model, "discover_damping", True)
            else float(model.fixed_damping_ratio)
        )
        writer.add_scalar("train/damping_ratio", damping_ratio_value, epoch)
        writer.add_scalar("train/grad_norm", mean_grad_norm, epoch)
        writer.add_scalar("train/drag_coefficient", drag_coeff, epoch)
        writer.add_scalar("train/avg_force", mean_force, epoch)

        if (epoch + 1) % rollout_every_epoch == 0:
            rollout = rollout_model(model, y_data_t, val_vel, m_eff, dt, t, D, k, device)
            rmse_eval = float(np.sqrt(np.mean((rollout["y_norm"] - y_true_norm) ** 2)))
            writer.add_scalar("val/rmse_y_over_D", rmse_eval, epoch + 1)
            y_pred_raw = rollout["y_norm"] * D
            disp_range_raw = float(np.ptp(y_data))
            if disp_range_raw <= 0.0:
                disp_range_raw = 1.0
            rel_rmse_disp = float(np.sqrt(np.mean((y_pred_raw - y_data) ** 2))) / disp_range_raw
            writer.add_scalar("val/rel_rmse_y", rel_rmse_disp, epoch + 1)
            force_total_pred = np.asarray(rollout["force_total"]).reshape(-1)
            force_target = np.asarray(force_data).reshape(-1)
            min_len = min(force_total_pred.shape[0], force_target.shape[0])
            if min_len > 0:
                force_rmse = float(
                    np.sqrt(np.mean((force_total_pred[:min_len] - force_target[:min_len]) ** 2))
                )
                writer.add_scalar("val/rmse_force_total", force_rmse, epoch + 1)
                force_range = float(np.ptp(force_target[:min_len]))
                if force_range <= 0.0:
                    force_range = 1.0
                writer.add_scalar("val/rel_rmse_force_total", force_rmse / force_range, epoch + 1)
                force_model_aligned = force_total_pred[:min_len]
                force_true_aligned = force_target[:min_len]
                damage_true = fatigue_damage(force_true_aligned)
                damage_model = fatigue_damage(force_model_aligned)
                damage_rel_err = relative_error(damage_model, damage_true)
                if np.isfinite(damage_rel_err):
                    writer.add_scalar(
                        "val/force_fatigue_damage_rel_error",
                        damage_rel_err,
                        epoch + 1,
                    )
            half_idx_disp = len(y_true_norm) // 2
            y_true_half = y_true_norm[half_idx_disp:]
            y_model_half = rollout["y_norm"][half_idx_disp:]
            freq_true_half = dominant_frequency(y_true_half, dt)
            freq_model_half = dominant_frequency(y_model_half, dt)
            freq_rel_err = relative_error(freq_model_half, freq_true_half)
            if np.isfinite(freq_rel_err):
                writer.add_scalar("val/frequency_rel_error_second_half", freq_rel_err, epoch + 1)

            if y_true_half.size > 0 and y_model_half.size > 0:
                max_disp_true = float(np.max(np.abs(y_true_half)))
                max_disp_model = float(np.max(np.abs(y_model_half)))
                disp_rel_err = relative_error(max_disp_model, max_disp_true)
                if np.isfinite(disp_rel_err):
                    writer.add_scalar("val/disp_amplitude_rel_error_second_half", disp_rel_err, epoch + 1)

            force_model_aligned = force_total_pred[:min_len] if min_len > 0 else force_total_pred
            force_true_aligned = force_target[:min_len] if min_len > 0 else force_target
            half_idx_force = force_true_aligned.size // 2
            force_true_half = force_true_aligned[half_idx_force:]
            force_model_half = force_model_aligned[half_idx_force:]
            if force_true_half.size > 0 and force_model_half.size > 0:
                max_force_true = float(np.max(np.abs(force_true_half)))
                max_force_model = float(np.max(np.abs(force_model_half)))
                force_amp_rel_err = relative_error(max_force_model, max_force_true)
                if np.isfinite(force_amp_rel_err):
                    writer.add_scalar(
                        "val/force_amplitude_rel_error_second_half",
                        force_amp_rel_err,
                        epoch + 1,
                    )
                spectral_rel_err = spectral_relative_error(force_true_half, force_model_half, dt)
                if np.isfinite(spectral_rel_err):
                    writer.add_scalar(
                        "val/force_spectral_rel_error_second_half",
                        spectral_rel_err,
                        epoch + 1,
                    )

            zoom_mask = create_zoom_mask(t)
            middle_mask = create_window_mask(t, middle_time_plot)
            log_displacement_plots(
                writer,
                epoch,
                t,
                y_true_norm,
                rollout["y_norm"],
                rollout["p_norm"],
                zoom_mask,
                middle_mask,
                middle_time_plot,
            )
            log_force_plots(
                writer,
                epoch,
                t,
                rollout["force_total"],
                rollout["force_drag"],
                rollout["force_model"],
                force_data,
                zoom_mask,
                middle_mask,
                middle_time_plot,
                model.include_physical_drag,
            )
            log_hamiltonian_plots(
                writer,
                epoch,
                t,
                rollout["hamiltonian_model"],
                zoom_mask,
                middle_mask,
                middle_time_plot,
                hamiltonian_data=hamiltonian_data,
            )

        rollout_str = f", rollout={mean_rollout_loss:.4e}" if active_rollout else ""
        print(
            f"Epoch {epoch}: loss={mean_loss:.4e}, res={mean_res_loss:.4e}, "
            f"force={mean_force_loss:.4e}, damping_ratio={damping_ratio_value:.4e}, "
            f"Cd={drag_coeff:.4f}{rollout_str}"
        )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
