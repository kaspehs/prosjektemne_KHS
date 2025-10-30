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
    rollout_len = 200   # e.g. 0.4 s if dt=0.001
    batch_size = 32
    max_start = y_data_t.shape[0] - rollout_len
    if max_start <= 0:
        raise ValueError("rollout_len is longer than available data; reduce rollout_len.")
    force_reg_coeff = 1e-4
    use_force_reg = True
    grad_norm_max = 1e4

    # estimate initial velocity from data (only once)
    # central diff at t=0 is awkward, so just forward diff:
    vel = torch.zeros_like(y_data_t)
    vel[0] = (y_data_t[1] - y_data_t[0]) / dt
    vel[-1] = (y_data_t[-1] - y_data_t[-2]) / dt
    vel[1:-1] = (y_data_t[2:] - y_data_t[:-2]) / (2.0 * dt)

    time_offsets = torch.arange(rollout_len, device=device)
    total_steps = 20000
    eval_every = 500

    for step in range(total_steps):
        opt.zero_grad()

        idx = torch.randint(0, max_start, (batch_size,), device=device)
        seq_idx = idx.unsqueeze(1) + time_offsets.unsqueeze(0)

        y_true = y_data_t[seq_idx]

        y0 = y_data_t[idx]
        v0 = vel[idx]
        state = torch.stack((y0, v0), dim=1)  # (B, 2)

        preds = []
        force_accum = torch.tensor(0.0, device=device, dtype=y_data_t.dtype)
        for _ in range(rollout_len):
            if use_force_reg and force_reg_coeff > 0.0:
                forces = model.u_theta(state)
                force_accum = force_accum + (forces.squeeze(-1) ** 2).mean()
            preds.append(state[:, 0])
            state = model.step_rk4(state, dt)

        # stack predictions: (rollout_len, 1)
        y_pred = torch.stack(preds, dim=1)

        data_mse = nn.MSELoss()(y_pred, y_true)
        loss = data_mse
        if use_force_reg and force_reg_coeff > 0.0:
            reg = force_accum / rollout_len
            loss = loss + force_reg_coeff * reg
        else:
            reg = None

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_max)
        opt.step()

        writer.add_scalar("train/loss", float(loss.detach().cpu()), step)
        writer.add_scalar("train/data_mse", float(data_mse.detach().cpu()), step)
        writer.add_scalar("train/damping_ratio", float(torch.sigmoid(model.zeta_raw).detach().cpu()), step)
        writer.add_scalar("train/grad_norm", float(grad_norm.detach().cpu()), step)
        if reg is not None:
            writer.add_scalar("train/force_reg", float(reg.detach().cpu())*force_reg_coeff, step)

        if (step + 1) % eval_every == 0:
            v0_eval = vel[0]
            x_eval = torch.stack((y_data_t[0], v0_eval), dim=0).unsqueeze(0)
            traj_eval = [x_eval.squeeze(0).detach().cpu().numpy()]
            for _ in range(len(y_data) - 1):
                x_eval = model.step_rk4(x_eval, dt)
                traj_eval.append(x_eval.squeeze(0).detach().cpu().numpy())
            traj_eval = np.stack(traj_eval)
            y_sim_eval = traj_eval[:, 0]
            rmse_eval = float(np.sqrt(np.mean((y_sim_eval - y_data) ** 2)))
            writer.add_scalar("eval/full_rollout_rmse", rmse_eval, step + 1)

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t, y_data, label="ground truth")
            ax.plot(t, y_sim_eval, label="prediction")
            ax.set_xlabel("time")
            ax.set_ylabel("y")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Full rollout at step {step+1}")
            writer.add_figure("eval/full_rollout", fig, step + 1)
            plt.close(fig)

        if (step + 1) % 100 == 0:
            print(f"step {step+1}: loss={loss.item():.4e}")

    # after training, run a final full rollout:
    v0 = vel[0]
    x = torch.stack((y_data_t[0], v0), dim=0).unsqueeze(0)
    traj = [x.squeeze(0).detach().cpu().numpy()]
    for _ in range(len(y_data) - 1):
        x = model.step_rk4(x, dt)
        traj.append(x.squeeze(0).detach().cpu().numpy())
    traj = np.stack(traj)  # (N, 2)
    y_sim = traj[:, 0]

    np.savez("simulated_phnn.npz", t=t, y=y_sim)
    final_rmse = float(np.sqrt(np.mean((y_sim - y_data) ** 2)))
    writer.add_scalar("eval/full_rollout_rmse", final_rmse, total_steps)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, y_data, label="ground truth")
    ax.plot(t, y_sim, label="prediction")
    ax.set_xlabel("time")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Final full rollout")
    writer.add_figure("eval/full_rollout", fig, total_steps)
    plt.close(fig)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
