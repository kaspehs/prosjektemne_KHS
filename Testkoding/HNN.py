import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from HNN_helper import *

def main():
    # ===== 1. load data =====
    data = np.load("data.npz")
    t = data["a"]   # time
    y_data = data["b"]   # displacement
    dt = float(t[1] - t[0])

    device = torch.device("cpu")

    # convert to torch
    y_data_t = torch.from_numpy(y_data).float().to(device)

    # ===== 2. model =====
    rho = 1000.0
    D = 0.1
    Ca = 1.0
    m_a = 0.25 * np.pi * D**2 * rho * Ca
    m_eff = 16.79 + m_a
    k = 1218.0
    U = 0.65
    model = PHVIV(m=m_eff, k=k, U=U, rho=rho, D=D).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    run_dir = os.path.join(
        "HNNruns",
        f"hnn_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    writer = SummaryWriter(log_dir=run_dir)

    # how many steps to unroll for each training batch
    # you can start small to make it stable
    rollout_len_base = 200   # maximum window used for training
    batch_size = 32
    force_reg_coeff = 1e-4
    use_force_reg = True
    force_smooth_coeff = 1e-6
    use_force_smooth = True
    force_curvature_coeff = 1e-8
    use_force_curvature = True
    grad_norm_max = 1e4
    characteristic_cutoff_epoch = 300

    # estimate initial velocity from data (only once)
    # central diff at t=0 is awkward, so just forward diff:
    vel = torch.zeros_like(y_data_t)
    vel[0] = (y_data_t[1] - y_data_t[0]) / dt
    vel[-1] = (y_data_t[-1] - y_data_t[-2]) / dt
    vel[1:-1] = (y_data_t[2:] - y_data_t[:-2]) / (2.0 * dt)

    z =  torch.stack(y_data_t, vel)
    dz = (z[1:] - z[:-1]) / dt
    z_mean = (z[1:] + z[:-1]) / 2
    t_mean = (t[1:] + t[:-1]) / 2
    y_true_norm = y_data / D

    total_steps = 20000
    eval_every = 200

    for step in range(total_steps):
        opt.zero_grad()

        rollout_len = min(rollout_len_schedule(step, characteristic_cutoff_epoch), rollout_len_base)
        if rollout_len < 2:
            rollout_len = 2
        max_start = y_data_t.shape[0] - rollout_len
        if max_start <= 0:
            raise ValueError("rollout_len is longer than available data; reduce rollout_len.")
        time_offsets = torch.arange(rollout_len, device=device)

        idx = torch.randint(0, max_start, (batch_size,), device=device)
        seq_idx = idx.unsqueeze(1) + time_offsets.unsqueeze(0)

        y_true = y_data_t[seq_idx]

        y0 = y_data_t[idx]
        v0 = vel[idx]
        state = torch.stack((y0, v0), dim=1)  # (B, 2)

        preds = []
        force_accum = torch.tensor(0.0, device=device, dtype=y_data_t.dtype)
        force_series: list[torch.Tensor] = []
        collect_forces = (
            (use_force_reg and force_reg_coeff > 0.0)
            or (use_force_smooth and force_smooth_coeff > 0.0)
            or (use_force_curvature and force_curvature_coeff > 0.0)
        )
        for _ in range(rollout_len):
            if collect_forces:
                forces = model.u_theta(state)
                if use_force_reg and force_reg_coeff > 0.0:
                    force_accum = force_accum + (forces.squeeze(-1) ** 2).mean()
                if ((use_force_smooth and force_smooth_coeff > 0.0)
                        or (use_force_curvature and force_curvature_coeff > 0.0)):
                    force_series.append(forces)
            preds.append(state[:, 0])
            state = model.step_rk4(state, dt)

        # stack predictions: (rollout_len, 1)
        y_pred = torch.stack(preds, dim=1)

        data_mse = nn.MSELoss()(y_pred, y_true)
        loss = data_mse
        reg = None
        if use_force_reg and force_reg_coeff > 0.0:
            reg = force_accum / rollout_len
            loss = loss + force_reg_coeff * reg

        forces_tensor = None
        if force_series:
            forces_tensor = torch.stack(force_series, dim=1)  # (batch, rollout_len)

        smooth_reg = None
        if (use_force_smooth and force_smooth_coeff > 0.0
                and forces_tensor is not None and forces_tensor.size(1) > 1):
            force_diff = (forces_tensor[:, 1:, :] - forces_tensor[:, :-1, :])/dt
            smooth_reg = (force_diff ** 2).mean()
            loss = loss + force_smooth_coeff * smooth_reg

        curvature_reg = None
        if (use_force_curvature and force_curvature_coeff > 0.0
                and forces_tensor is not None and forces_tensor.size(1) > 2):
            second_diff = forces_tensor[:, 2:, :] - 2.0 * forces_tensor[:, 1:-1, :] + forces_tensor[:, :-2, :]
            second_diff = second_diff / dt**2
            curvature_reg = (second_diff ** 2).mean()
            loss = loss + force_curvature_coeff * curvature_reg

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_max)
        opt.step()

        writer.add_scalar("train/loss", float(loss.detach().cpu()), step)
        writer.add_scalar("train/data_mse", float(data_mse.detach().cpu()), step)
        writer.add_scalar("train/damping_ratio", float(torch.sigmoid(model.zeta_raw).detach().cpu())*model.max_damping_ratio, step)
        writer.add_scalar("train/grad_norm", float(grad_norm.detach().cpu()), step)
        writer.add_scalar("train/rollout_len", float(rollout_len), step)
        if reg is not None:
            writer.add_scalar("train/force_reg", float((reg * force_reg_coeff).detach().cpu()), step)
        if smooth_reg is not None:
            writer.add_scalar("train/force_smooth_reg", float((smooth_reg * force_smooth_coeff).detach().cpu()), step)
        if curvature_reg is not None:
            writer.add_scalar("train/force_curvature_reg", float((curvature_reg * force_curvature_coeff).detach().cpu()), step)

        if (step + 1) % eval_every == 0:
            v0_eval = vel[0]
            x_eval = torch.stack((y_data_t[0], v0_eval), dim=0).unsqueeze(0)
            traj_eval = [x_eval.squeeze(0).detach().cpu().numpy()]
            force_eval = [model.u_theta(x_eval).squeeze().detach().cpu().numpy()]
            for _ in range(len(y_data) - 1):
                x_eval = model.step_rk4(x_eval, dt)
                traj_eval.append(x_eval.squeeze(0).detach().cpu().numpy())
                force_eval.append(model.u_theta(x_eval).squeeze().detach().cpu().numpy())
            traj_eval = np.stack(traj_eval)
            y_sim_eval = traj_eval[:, 0]
            y_pred_norm = y_sim_eval / D
            force_pred_norm = np.asarray(force_eval) / (k * D)
            rmse_eval = float(np.sqrt(np.mean((y_sim_eval - y_data) ** 2)))
            writer.add_scalar("eval/full_rollout_rmse", rmse_eval, step + 1)

            fig_full, ax_full = plt.subplots(figsize=(6, 3))
            ax_full.plot(t, y_true_norm, label="y/D (true)")
            ax_full.plot(t, y_pred_norm, label="y/D (pred)")
            ax_full.set_xlabel("time")
            ax_full.set_ylabel("y/D")
            ax_full.grid(True, alpha=0.3)
            ax_full.set_title(f"Normalized rollout at step {step+1}")
            ax_force = ax_full.twinx()
            ax_force.plot(t, force_pred_norm, ":", label="F/(kD) pred", color="tab:purple")
            ax_force.set_ylabel("F/(kD)")
            lines, labels = ax_full.get_legend_handles_labels()
            lines2, labels2 = ax_force.get_legend_handles_labels()
            ax_full.legend(lines + lines2, labels + labels2, loc="upper right")
            writer.add_figure("eval/full_rollout", fig_full, step + 1)
            plt.close(fig_full)

            zoom_mask = (t - t[0]) <= 1.0
            if np.count_nonzero(zoom_mask) > 1:
                fig_zoom, ax_zoom = plt.subplots(figsize=(6, 3))
                ax_zoom.plot(t[zoom_mask], y_true_norm[zoom_mask], label="y/D (true)")
                ax_zoom.plot(t[zoom_mask], y_pred_norm[zoom_mask], label="y/D (pred)")
                ax_zoom.set_xlabel("time")
                ax_zoom.set_ylabel("y/D")
                ax_zoom.grid(True, alpha=0.3)
                ax_zoom.set_title(f"Normalized rollout (first 1s) step {step+1}")
                ax_force_zoom = ax_zoom.twinx()
                ax_force_zoom.plot(t[zoom_mask], force_pred_norm[zoom_mask], ":", color="tab:purple", label="F/(kD) pred")
                ax_force_zoom.set_ylabel("F/(kD)")
                lines, labels = ax_zoom.get_legend_handles_labels()
                lines2, labels2 = ax_force_zoom.get_legend_handles_labels()
                ax_zoom.legend(lines + lines2, labels + labels2, loc="upper right")
                writer.add_figure("eval/full_rollout_zoom", fig_zoom, step + 1)
                plt.close(fig_zoom)

        if (step + 1) % 100 == 0:
            print(f"step {step+1}: loss={loss.item():.4e}")

    # after training, run a final full rollout:
    v0 = vel[0]
    x = torch.stack((y_data_t[0], v0), dim=0).unsqueeze(0)
    traj = [x.squeeze(0).detach().cpu().numpy()]
    force_final = [model.u_theta(x).squeeze().detach().cpu().numpy()]
    for _ in range(len(y_data) - 1):
        x = model.step_rk4(x, dt)
        traj.append(x.squeeze(0).detach().cpu().numpy())
        force_final.append(model.u_theta(x).squeeze().detach().cpu().numpy())
    traj = np.stack(traj)  # (N, 2)
    y_sim = traj[:, 0]
    y_pred_norm_final = y_sim / D
    force_pred_norm_final = np.asarray(force_final) / (k * D)

    np.savez("simulated_phnn.npz", t=t, y=y_sim)
    final_rmse = float(np.sqrt(np.mean((y_sim - y_data) ** 2)))
    writer.add_scalar("eval/full_rollout_rmse", final_rmse, total_steps)
    fig_full, ax_full = plt.subplots(figsize=(6, 3))
    ax_full.plot(t, y_true_norm, label="y/D (true)")
    ax_full.plot(t, y_pred_norm_final, label="y/D (pred)")
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("y/D")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title("Final normalized rollout")
    ax_force = ax_full.twinx()
    ax_force.plot(t, force_pred_norm_final, ":", color="tab:purple", label="F/(kD) pred")
    ax_force.set_ylabel("F/(kD)")
    lines, labels = ax_full.get_legend_handles_labels()
    lines2, labels2 = ax_force.get_legend_handles_labels()
    ax_full.legend(lines + lines2, labels + labels2, loc="upper right")
    writer.add_figure("eval/full_rollout", fig_full, total_steps)
    plt.close(fig_full)

    zoom_mask = (t - t[0]) <= 1.0
    if np.count_nonzero(zoom_mask) > 1:
        fig_zoom, ax_zoom = plt.subplots(figsize=(6, 3))
        ax_zoom.plot(t[zoom_mask], y_true_norm[zoom_mask], label="y/D (true)")
        ax_zoom.plot(t[zoom_mask], y_pred_norm_final[zoom_mask], label="y/D (pred)")
        ax_zoom.set_xlabel("time")
        ax_zoom.set_ylabel("y/D")
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.set_title("Final normalized rollout (first 1s)")
        ax_force_zoom = ax_zoom.twinx()
        ax_force_zoom.plot(t[zoom_mask], force_pred_norm_final[zoom_mask], ":", color="tab:purple", label="F/(kD) pred")
        ax_force_zoom.set_ylabel("F/(kD)")
        lines, labels = ax_zoom.get_legend_handles_labels()
        lines2, labels2 = ax_force_zoom.get_legend_handles_labels()
        ax_zoom.legend(lines + lines2, labels + labels2, loc="upper right")
        writer.add_figure("eval/full_rollout_zoom", fig_zoom, total_steps)
        plt.close(fig_zoom)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
