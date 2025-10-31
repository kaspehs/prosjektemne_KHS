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
from torch.utils.data import TensorDataset, DataLoader
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
    q_scale = D
    p_scale = np.sqrt(k/m_eff)*m_eff*D
    model = PHVIV(dt, 
                  m=m_eff, 
                  k=k, U=U, 
                  rho=rho, D=D, 
                  q_scale=q_scale, 
                  p_scale = p_scale, 
                  discover_damping=False, 
                  damping_c=1e-4,
                  ).to(device)

    run_dir = os.path.join(
        "HNNruns",
        f"hnn_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    writer = SummaryWriter(log_dir=run_dir)

    # how many steps to unroll for each training batch
    # you can start small to make it stable
    batch_size = 32
    force_reg = 1e-3
    max_grad_norm = 1e1
    lr = 1e-3

    # estimate initial velocity from data (only once)
    # central diff at t=0 is awkward, so just forward diff:
    vel = torch.zeros_like(y_data_t)
    vel[0] = (y_data_t[1] - y_data_t[0]) / dt
    vel[-1] = (y_data_t[-1] - y_data_t[-2]) / dt
    vel[1:-1] = (y_data_t[2:] - y_data_t[:-2]) / (2.0 * dt)

    z = torch.stack((y_data_t, vel * m_eff), dim=1)  # (N, 2)
    t_tensor = torch.from_numpy(t).float().to(device)

    dataset = TensorDataset(
        z[:-1],
        t_tensor[:-1].unsqueeze(1),
        z[1:],
        t_tensor[1:].unsqueeze(1),
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    y_true_norm = y_data / D

    epochs = 10000
    rollout_every_epoch = 500
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        losses = []
        res_losses = []
        force_losses = []
        grad_norms = []
        for batch in train_loader:
            z_i, t_i, z_next, t_next = batch
            z_i = z_i.to(device)
            t_i = t_i.to(device)
            z_next = z_next.to(device)
            t_next = t_next.to(device)

            opt.zero_grad()

            res_loss = model.res_loss(z_i, t_i, z_next, t_next)
            force_loss = force_reg * model.force_loss(z_i, t_i, z_next, t_next)
            loss = res_loss + force_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            grad_norms.append(grad_norm)
            opt.step()

            losses.append(loss.detach().cpu().item())
            res_losses.append(res_loss.detach().cpu().item())
            force_losses.append(force_loss.detach().cpu().item())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_res_loss = float(np.mean(res_losses)) if res_losses else 0.0
        mean_force_loss = float(np.mean(force_losses)) if force_losses else 0.0
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        drag_coeff = float(torch.exp(model.log_Cd).detach().cpu())

        writer.add_scalar("train/loss", mean_loss, epoch)
        writer.add_scalar("train/residual_loss", mean_res_loss, epoch)
        writer.add_scalar("train/force_loss", mean_force_loss, epoch)
        if getattr(model, "discover_damping", True):
            damping_ratio_value = float(torch.sigmoid(model.zeta_raw).detach().cpu()) * model.max_damping_ratio
        else:
            damping_ratio_value = float(model.fixed_damping_ratio)
        writer.add_scalar("train/damping_ratio", damping_ratio_value, epoch)
        writer.add_scalar("train/grad_norm", mean_grad_norm, epoch)
        writer.add_scalar("train/drag_coefficient", drag_coeff, epoch)

        if (epoch + 1) % rollout_every_epoch == 0:
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
            rmse_eval = float(np.sqrt(np.mean((y_pred_norm - y_true_norm) ** 2)))
            writer.add_scalar("val/rmse_y_over_D", rmse_eval, epoch + 1)

            fig_full, ax_full = plt.subplots(figsize=(6, 3))
            ax_full.plot(t, y_true_norm, label="y/D (true)")
            ax_full.plot(t, y_pred_norm, label="y/D (pred)")
            ax_full.set_xlabel("time")
            ax_full.set_ylabel("y/D")
            ax_full.grid(True, alpha=0.3)
            ax_full.set_title(f"Normalized rollout at epoch {epoch+1}")
            ax_force = ax_full.twinx()
            ax_force.plot(t, force_pred_norm, ":", label="F/(kD) pred", color="tab:purple")
            ax_force.set_ylabel("F/(kD)")
            lines, labels = ax_full.get_legend_handles_labels()
            lines2, labels2 = ax_force.get_legend_handles_labels()
            ax_full.legend(lines + lines2, labels + labels2, loc="upper right")
            writer.add_figure("val/rollout_full", fig_full, epoch + 1)
            plt.close(fig_full)

            zoom_mask = (t - t[0]) <= 1.0
            if np.count_nonzero(zoom_mask) > 1:
                fig_zoom, ax_zoom = plt.subplots(figsize=(6, 3))
                ax_zoom.plot(t[zoom_mask], y_true_norm[zoom_mask], label="y/D (true)")
                ax_zoom.plot(t[zoom_mask], y_pred_norm[zoom_mask], label="y/D (pred)")
                ax_zoom.set_xlabel("time")
                ax_zoom.set_ylabel("y/D")
                ax_zoom.grid(True, alpha=0.3)
                ax_zoom.set_title(f"Normalized rollout (first 1s) epoch {epoch+1}")
                ax_force_zoom = ax_zoom.twinx()
                ax_force_zoom.plot(t[zoom_mask], force_pred_norm[zoom_mask], ":", color="tab:purple", label="F/(kD) pred")
                ax_force_zoom.set_ylabel("F/(kD)")
                lines, labels = ax_zoom.get_legend_handles_labels()
                lines2, labels2 = ax_force_zoom.get_legend_handles_labels()
                ax_zoom.legend(lines + lines2, labels + labels2, loc="upper right")
                writer.add_figure("val/rollout_zoom", fig_zoom, epoch + 1)
                plt.close(fig_zoom)
        print(
            f"Epoch {epoch}: loss={mean_loss:.4e}, res:{mean_res_loss:.4e}, "
            f"force={mean_force_loss:.4e}, damping_ratio={damping_ratio_value:.4e}, "
            f"Cd={drag_coeff:.4f}"
        )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
