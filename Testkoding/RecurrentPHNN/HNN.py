import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from HNN_helper import *


def main():
    # ===== 1. load data =====
    data = np.load("data.npz")
    t = data["a"]
    y_data = data["b"]
    F_data = data["c"]
    H_data = data["d"]
    steadystate = False
    
    if steadystate:
        mask = t > 10
        t = t[mask]
        y_data = y_data[mask]
        F_data = F_data[mask]
        H_data = H_data[mask]
    """
    t = t[::10]
    y_data = y_data[::10]
    F_data = F_data[::10]
    H_data = H_data[::10]
    """
    dt = float(t[1] - t[0])
    middle_time_plot = [15, 17]

    device = torch.device("cpu")
    y_data_t = torch.from_numpy(y_data).float().to(device)

    # ===== 2. model =====
    rho = 1000.0
    D = 0.1
    Ca = 1.0
    m_a = 0.25 * np.pi * D**2 * rho * Ca
    m_eff = 16.79 + m_a
    k = 1218.0
    U = 0.65
    C = 2.860085
    q_scale = D
    p_scale = np.sqrt(k / m_eff) * m_eff * D
    include_physical_drag = False  # Set to False to let NN learn the total force directly
    learn_hamiltonian = False  # Set to True to learn the Hamiltonian with a neural network

    model = PHVIV(
        dt,
        m=m_eff,
        k=k,
        U=U,
        rho=rho,
        D=D,
        q_scale=q_scale,
        p_scale=p_scale,
        discover_damping=False,
        damping_c=C,
        include_physical_drag=include_physical_drag,
        learn_hamiltonian=learn_hamiltonian,
    ).to(device)

    run_dir = os.path.join("HNNruns", f"hnn_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=run_dir)

    batch_size = 32
    sequence_length = 32
    force_reg = 1e-1
    max_grad_norm = 1e4
    lr = 1e-3

    vel = torch.zeros_like(y_data_t)
    vel[0] = (y_data_t[1] - y_data_t[0]) / dt
    vel[-1] = (y_data_t[-1] - y_data_t[-2]) / dt
    vel[1:-1] = (y_data_t[2:] - y_data_t[:-2]) / (2.0 * dt)

    hamiltonian_data = H_data


    t_tensor = torch.from_numpy(t).float().to(device)
    if sequence_length < 1:
        raise ValueError("Not enough time steps to form training sequences")
    dataset = build_sequence_dataset(y_data_t, vel, m_eff, t_tensor, sequence_length=sequence_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    y_true_norm = y_data / D
    force_data = F_data

    epochs = 2000
    rollout_every_epoch = 5
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        losses: list[float] = []
        res_losses: list[float] = []
        force_losses: list[float] = []
        grad_norms: list[float] = []
        avg_forces: list[float] = []

        for z_seq, _ in train_loader:
            z_seq = z_seq.to(device)

            opt.zero_grad()

            res_loss = model.sequence_residual_SRK4(z_seq)
            force_penalty = model.sequence_force_penalty_SRK4(z_seq)
            force_loss = force_reg * force_penalty
            loss = res_loss + force_loss
            loss.backward()

            grad_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            grad_norms.append(float(grad_norm.detach().cpu()))
            opt.step()

            losses.append(float(loss.detach().cpu()))
            res_losses.append(float(res_loss.detach().cpu()))
            force_losses.append(float(force_loss.detach().cpu()))
            avg_forces.append(float(force_penalty.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_res_loss = float(np.mean(res_losses)) if res_losses else 0.0
        mean_force_loss = float(np.mean(force_losses)) if force_losses else 0.0
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        drag_coeff = float(torch.exp(model.log_Cd).detach().cpu())
        mean_force = float(np.mean(avg_forces)) if avg_forces else 0.0

        writer.add_scalar("train/loss", mean_loss, epoch)
        writer.add_scalar("train/residual_loss", mean_res_loss, epoch)
        writer.add_scalar("train/force_loss", mean_force_loss, epoch)
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
            rollout = rollout_model_SRK4(model, y_data_t, vel, m_eff, dt, t, D, k, device)
            rmse_eval = float(np.sqrt(np.mean((rollout["y_norm"] - y_true_norm) ** 2)))
            writer.add_scalar("val/rmse_y_over_D", rmse_eval, epoch + 1)
            force_total_pred = np.asarray(rollout["force_total"]).reshape(-1)
            force_target = np.asarray(force_data).reshape(-1)
            min_len = min(force_total_pred.shape[0], force_target.shape[0])
            if min_len > 0:
                force_rmse = float(
                    np.sqrt(np.mean((force_total_pred[:min_len] - force_target[:min_len]) ** 2))
                )
                writer.add_scalar("val/rmse_force_total", force_rmse, epoch + 1)

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
                _,
                #rollout["force_drag"],
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

        print(
            f"Epoch {epoch}: loss={mean_loss:.4e}, res={mean_res_loss:.4e}, "
            f"force={mean_force_loss:.4e}, damping_ratio={damping_ratio_value:.4e}, "
            f"Cd={drag_coeff:.4f}"
        )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
