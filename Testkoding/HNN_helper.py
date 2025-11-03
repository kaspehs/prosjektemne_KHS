import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import math
import matplotlib.pyplot as plt



class PHVIV(nn.Module):
    """
    Pseudo-/port-Hamiltonian 1-DOF oscillator with NN force.
    State x = [y, v].
    dot x = (J - R(x)) ∇H(x) + G u_theta(x)
    """
    def __init__(
        self,
        dt,
        m=16.79,
        k=1218.0,
        U=0.65,
        rho=1000.0,
        D=0.1,
        q_scale=0.1,
        p_scale=10.0,
        max_damping_ratio=0.2,
        discover_damping: bool = False,
        damping_c: float | None = None,
        include_physical_drag: bool = True,
        learn_hamiltonian: bool = False,
    ):
        super().__init__()
        self.dt = dt
        self.m = m
        self.k = k
        self.U = U
        self.rho = rho
        self.D = D
        self.max_damping_ratio = torch.tensor(max_damping_ratio)
        self.q_scale = q_scale
        self.p_scale = p_scale
        self.discover_damping = bool(discover_damping)
        self.include_physical_drag = bool(include_physical_drag)
        self.learn_hamiltonian = bool(learn_hamiltonian)

        self.nn_q_scale = q_scale
        self.nn_p_scale = p_scale
        self.q_scale = 1.0
        self.p_scale = 1.0

        # NN for instantaneous force u(x)
        self.u_net = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

        if self.learn_hamiltonian:
            self.h_net = nn.Sequential(
                nn.Linear(2, 100),
                nn.Tanh(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 1),
            )
        else:
            self.h_net = None

        # damping handling
        if self.discover_damping:
            self.zeta0 = torch.tensor(0.01)
            self.zeta_raw = nn.Parameter(torch.logit(self.zeta0/self.max_damping_ratio))
            self.register_buffer("fixed_c", torch.tensor(0.0))
            self.fixed_damping_ratio = None
        else:
            if damping_c is None:
                raise ValueError("damping_c must be provided when discover_damping is False")
            damping_c = float(damping_c)
            self.register_buffer("fixed_c", torch.tensor(damping_c, dtype=torch.float32))
            self.fixed_damping_ratio = float(damping_c / (2.0 * (self.k * self.m) ** 0.5))
            self.zeta_raw = None
        #Learable drag coefficient
        self.log_Cd = nn.Parameter(torch.log(torch.tensor(1.2)))  # start at ~1.2


        self.register_buffer("J", torch.tensor([[0.0, 1.0],
                                                [-1.0, 0.0]]))
        self.register_buffer("G", torch.tensor([[0.0],
                                                [1.0]]))

    def H(self, x):
        if not self.learn_hamiltonian:
            q = x[..., 0]
            p = x[..., 1]
            return 0.5 * self.k * q**2 + 0.5 * p**2 / self.m
        x_scaled = torch.stack(
            (x[..., 0] / self.nn_q_scale, x[..., 1] / self.nn_p_scale),
            dim=-1,
        )
        return self.h_net(x_scaled).squeeze(-1)

    def grad_H(self, x):
        if not self.learn_hamiltonian:
            q = x[..., 0]
            p = x[..., 1]
            return torch.stack((self.k * q, p / self.m), dim=-1)
        grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            H_val = self.H(x_req)
            grad = torch.autograd.grad(
                H_val.sum(),
                x_req,
                create_graph=grad_enabled,
                retain_graph=grad_enabled,
            )[0]
        return grad

    def R(self, x):
        R = torch.zeros(*x.shape[:-1], 2, 2, device=x.device, dtype=x.dtype)
        if self.discover_damping:
            zeta = torch.sigmoid(self.zeta_raw) * self.max_damping_ratio
            c_eff = 2.0 * zeta * torch.sqrt(torch.tensor(self.k * self.m, device=x.device, dtype=x.dtype))
        else:
            c_eff = self.fixed_c.to(device=x.device, dtype=x.dtype)
        R[..., 1, 1] = c_eff
        return R
    
    def drag_force(self, x):
        """
        Morison-like cross-flow drag: Fd = -0.5 * rho * D * Cd * |v| * v
        x: (..., 2)
        returns: (..., 1)
        """
        v = x[..., 1] / self.m
        U = torch.full_like(v, self.U)
        Cd = torch.exp(self.log_Cd)  # keep it positive
        rel_vel = torch.sqrt(v**2 + U**2)
        Fd = -0.5 * self.rho * self.D * Cd * torch.abs(rel_vel) * v
        return Fd.unsqueeze(-1)


    def u_theta1(self, x):
        q_scaled = x[..., 0] / self.nn_q_scale
        p_scaled = x[..., 1] / self.nn_p_scale
        x_scaled = torch.stack((q_scaled, p_scaled), dim=-1)
        return self.u_net(x_scaled) * self.k * self.D
    
    def u_theta2(self, x):
        return self.u_theta1(x) + self.drag_force(x)

    def learned_force(self, x):
        return self.u_theta1(x)
    
    def u_theta(self, x):
        return self.u_theta2(x) if self.include_physical_drag else self.u_theta1(x)
    
    def f(self, x):
        u = self.u_theta(x)
        G = self.G.to(x.device).to(x.dtype)                        # (..., 1)
        Gu = torch.einsum('ij,...j->...i', G, u)
        return Gu

    def g(self, x):
        gH = self.grad_H(x)                         # (..., 2)
        R = self.R(x)                               # (..., 2, 2)

        J = self.J.to(x.device).to(x.dtype)

        JgH = torch.einsum('ij,...j->...i', J, gH) #Just  J @ gH, with batch handling
        RgH = torch.einsum('...ij,...j->...i', R, gH)

        core = JgH - RgH

        return core + self.f(x)

    def step_euler(self, x, dt):
        return x + dt * self.f(x)

    def step_rk4(self, x, t, dt):
        k1 = self.g(x)
        k2 = self.g(x + 0.5 * dt * k1)
        k3 = self.g(x + 0.5 * dt * k2)
        k4 = self.g(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    def rollout(self, z0, t_seq, dt):
        """
        z0: (B, state_dim)    starting state from data
        t_seq: (B, K+1)       absolute times t0..tK
        returns:
        Z_pred: (B, K+1, state_dim)  predictions incl. z0
        F_hist: (B, K+1, 1)          optional, learned force per step
        """
        B = z0.shape[0]
        state_dim = z0.shape[-1]
        K = t_seq.shape[1] - 1

        Z_pred = [z0]
        F_hist = []

        z = z0
        for k in range(K):
            t = t_seq[:, k]
            z, Fk = self.rk4_step(z, t, dt)   # model.g(y,p,t)->(dzdt,F)
            Z_pred.append(z)
            #F_hist.append(Fk.unsqueeze(-1))

        Z_pred = torch.stack(Z_pred, dim=1)            # (B,K+1,D)
        #F_hist = torch.stack([torch.zeros_like(F_hist[0])] + F_hist, dim=1) if F_hist else None
        return Z_pred, F_hist

    def traj_loss(Z_pred, Z_data, w_state=(1.0, 1.0)):
        # For 1-DOF, assume z=[y,p]
        y_pred, p_pred = Z_pred[...,0], Z_pred[...,1]
        y_data, p_data = Z_data[...,0], Z_data[...,1]
        Ly = ((y_pred - y_data)**2).mean()
        Lp = ((p_pred - p_data)**2).mean()
        return w_state[0]*Ly + w_state[1]*Lp

    
    def res_loss(self, zi, ti, zin, tin):
        return self.res_loss_SRK4(zi, ti, zin, tin)
    
    def avg_force(self, zi, ti, zin, tin):
        return self.avg_force_SRK4(zi, ti, zin, tin)
    
    def res_loss_Euler(self, zi, ti, zin, tin):
        dz = (zin-zi)/self.dt
        z_mean = 0.5*(zin+zi)
        res = dz - self.g(z_mean)
        scale = torch.tensor((self.q_scale, self.p_scale), device=res.device, dtype=res.dtype)
        res_scaled = res / scale
        loss = torch.mean(torch.sum(res_scaled**2, dim=1))
        return loss

    def avg_force_Euler(self, zi, ti, zin, tin):
        z_mean = 0.5*(zin+zi)
        forces = self.f(z_mean)
        scale = torch.tensor((self.q_scale, self.p_scale), device=forces.device, dtype=forces.dtype)
        forces_scaled = forces / scale
        loss = torch.mean(torch.linalg.norm(forces_scaled, ord=1, dim=1))
        return loss
    

    def res_loss_SRK4(self, zi, ti, zin, tin):
        dt = self.dt
        # constants from the scheme
        a = 0.5
        b = math.sqrt(3.0) / 6.0

        # finite difference
        dz_fd = (zin - zi) / dt              # (B, d)

        # midpoint between zn and zn+1
        z_mid = 0.5 * (zi + zin)             # (B, d)

        # stage convex combos
        z_a_plus  = (0.5 + b) * zi + (0.5 - b) * zin   # (B, d)
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin   # (B, d)

        # stage evaluations of g
        g_a_plus  = self.g(z_a_plus)                  # (B, d)
        g_a_minus = self.g(z_a_minus)                 # (B, d)

        # two corrected midpoints
        z_corr_minus = z_mid - b * dt * g_a_plus      # (B, d)
        z_corr_plus  = z_mid + b * dt * g_a_minus     # (B, d)

        # final two g-evals
        g1 = self.g(z_corr_minus)                     # (B, d)
        g2 = self.g(z_corr_plus)                      # (B, d)

        dz_model = 0.5 * g1 + 0.5 * g2                # (B, d)

        # residual
        res = dz_fd - dz_model                        # (B, d)

        # scale like before, but for time-derivatives
        res_scale = torch.tensor(
            (self.q_scale, self.p_scale),
            device=res.device, dtype=res.dtype
        )
        res_scaled = res / res_scale

        loss = torch.mean(torch.sum(res_scaled**2, dim=1))
        return loss
    
    def avg_force_SRK4(self, zi, ti, zin, tin):
        dt = self.dt
        b = math.sqrt(3.0) / 6.0

        # same stage points as in res_loss
        z_a_plus  = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin

        # evaluate learned force at both stages
        f1 = self.f(z_a_plus)    # assume (B, 1) or (B, 2)
        f2 = self.f(z_a_minus)

        loss = 0.5 * torch.mean(torch.sum(torch.abs(f1), dim=1)) \
            + 0.5 * torch.mean(torch.sum(torch.abs(f2), dim=1))
        return loss

def log_displacement_plots(
    writer,
    epoch,
    t,
    y_true_norm,
    y_pred_norm,
    p_pred_norm,
    zoom_mask,
    middle_mask,
    middle_window,
):
    fig, (ax_full, ax_zoom, ax_middle) = plt.subplots(3, 1, figsize=(6, 9), sharex=False)

    ax_full.plot(t, y_true_norm, label="y/D (true)")
    ax_full.plot(t, y_pred_norm, label="y/D (pred)")
    ax_full.plot(t, p_pred_norm, label="p_hat (pred)", linestyle="--", color="tab:purple")
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("y/D")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Normalized rollout at epoch {epoch+1}")
    ax_full.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], y_true_norm[zoom_mask], label="y/D (true)")
    ax_zoom.plot(t[zoom_mask], y_pred_norm[zoom_mask], label="y/D (pred)")
    ax_zoom.plot(t[zoom_mask], p_pred_norm[zoom_mask], label="p_hat (pred)", linestyle="--", color="tab:purple")
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("y/D")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Normalized rollout (first 1s) epoch {epoch+1}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], y_true_norm[middle_mask], label="y/D (true)")
    ax_middle.plot(t[middle_mask], y_pred_norm[middle_mask], label="y/D (pred)")
    ax_middle.plot(
        t[middle_mask],
        p_pred_norm[middle_mask],
        label="p_hat (pred)",
        linestyle="--",
        color="tab:purple",
    )
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("y/D")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Normalized rollout ({mid_start}-{mid_end}s) epoch {epoch+1}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure("val/rollout_displacement", fig, epoch + 1)
    plt.close(fig)

def log_force_plots(
    writer,
    epoch,
    t,
    force_total,
    force_drag,
    force_model,
    force_data,
    zoom_mask,
    middle_mask,
    middle_window,
    include_physical_drag: bool,
):
    fig, (ax_full, ax_zoom, ax_middle) = plt.subplots(3, 1, figsize=(6, 9), sharex=False)
    total_label = "F_total (model + drag)" if include_physical_drag else "F_total (model)"
    model_label = "F_model (wake)" if include_physical_drag else "F_model"

    ax_full.plot(t, force_total, label=total_label, color="tab:purple")
    if include_physical_drag:
        ax_full.plot(t, force_drag, label="F_drag", color="tab:red", linestyle="--")
    ax_full.plot(t, force_model, label=model_label, color="tab:green", linestyle=":")
    ax_full.plot(t, force_data, label="F_data", color="tab:blue", alpha=0.7)
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("Force")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Force rollout at epoch {epoch+1}")
    ax_full.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], force_total[zoom_mask], label=total_label, color="tab:purple")
    if include_physical_drag:
        ax_zoom.plot(t[zoom_mask], force_drag[zoom_mask], label="F_drag", color="tab:red", linestyle="--")
    ax_zoom.plot(t[zoom_mask], force_model[zoom_mask], label=model_label, color="tab:green", linestyle=":")
    ax_zoom.plot(t[zoom_mask], force_data[zoom_mask], label="F_data", color="tab:blue", alpha=0.7)
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("Force")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Force rollout (first 1s) epoch {epoch+1}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], force_total[middle_mask], label=total_label, color="tab:purple")
    if include_physical_drag:
        ax_middle.plot(
            t[middle_mask],
            force_drag[middle_mask],
            label="F_drag",
            color="tab:red",
            linestyle="--",
        )
    ax_middle.plot(
        t[middle_mask],
        force_model[middle_mask],
        label=model_label,
        color="tab:green",
        linestyle=":",
    )
    ax_middle.plot(t[middle_mask], force_data[middle_mask], label="F_data", color="tab:blue", alpha=0.7)
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("Force")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Force rollout ({mid_start}-{mid_end}s) epoch {epoch+1}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure("val/rollout_force", fig, epoch + 1)
    plt.close(fig)

def log_hamiltonian_plots(
    writer,
    epoch,
    t,
    hamiltonian_model,
    zoom_mask,
    middle_mask,
    middle_window,
    hamiltonian_data: np.ndarray | None = None,
):
    fig, (ax_full, ax_zoom, ax_middle) = plt.subplots(3, 1, figsize=(6, 9), sharex=False)
    model_kwargs = {"color": "tab:orange", "label": "H_model"}
    data_kwargs = {"color": "tab:blue", "linestyle": "--", "alpha": 0.8, "label": "H_data"}

    h_model_rel = hamiltonian_model - (hamiltonian_model[0] if hamiltonian_model.size else 0.0)
    h_data_rel = None
    if hamiltonian_data is not None:
        h_data_rel = hamiltonian_data - (hamiltonian_data[0] if hamiltonian_data.size else 0.0)

    ax_full.plot(t, h_model_rel, **model_kwargs.copy())
    if hamiltonian_data is not None:
        ax_full.plot(t, h_data_rel, **data_kwargs.copy())
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("Hamiltonian")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Hamiltonian rollout at epoch {epoch+1}")
    ax_full.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], h_model_rel[zoom_mask], **model_kwargs.copy())
    if hamiltonian_data is not None:
        ax_zoom.plot(t[zoom_mask], h_data_rel[zoom_mask], **data_kwargs.copy())
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("Hamiltonian")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Hamiltonian (first 1s) epoch {epoch+1}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], h_model_rel[middle_mask], **model_kwargs.copy())
    if hamiltonian_data is not None:
        ax_middle.plot(t[middle_mask], h_data_rel[middle_mask], **data_kwargs.copy())
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("Hamiltonian")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Hamiltonian ({mid_start}-{mid_end}s) epoch {epoch+1}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure("val/rollout_hamiltonian", fig, epoch + 1)
    plt.close(fig)

def build_dataset(y_data_t: torch.Tensor, vel: torch.Tensor, m_eff: float, t_tensor: torch.Tensor) -> TensorDataset:
    """Construct consecutive state/time pairs for training."""
    z = torch.stack((y_data_t, vel * m_eff), dim=1)
    return TensorDataset(
        z[:-1],
        t_tensor[:-1].unsqueeze(1),
        z[1:],
        t_tensor[1:].unsqueeze(1),
    )

def build_rollout_dataset(
    y_data_t: torch.Tensor,
    vel: torch.Tensor,
    m_eff: float,
    t_tensor: torch.Tensor,
    rollout_steps: int,
) -> TensorDataset:
    """
    Build sliding-window sequences matching the inputs expected by `PHVIV.rollout`
    and the targets required by `traj_loss`.

    Each sample contains:
        - z0: initial state (y, p) at the window start
        - t_seq: absolute times for the window (length rollout_steps + 1)
        - z_traj: ground-truth state trajectory over the same window
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be at least 1")

    z = torch.stack((y_data_t, vel * m_eff), dim=1)  # (T, 2)
    window = rollout_steps + 1
    total_samples = z.shape[0]
    if total_samples < window:
        raise ValueError("Not enough samples to build rollout windows of the requested length")

    num_windows = total_samples - window + 1
    z0_list = []
    t_seq_list = []
    z_traj_list = []

    for start in range(num_windows):
        end = start + window
        z_window = z[start:end]                  # (window, 2)
        t_window = t_tensor[start:end]           # (window,)
        z0_list.append(z_window[0])
        t_seq_list.append(t_window)
        z_traj_list.append(z_window)

    z0_batch = torch.stack(z0_list, dim=0)                  # (B, 2)
    t_seq_batch = torch.stack(t_seq_list, dim=0)            # (B, window)
    z_traj_batch = torch.stack(z_traj_list, dim=0)          # (B, window, 2)

    return TensorDataset(z0_batch, t_seq_batch, z_traj_batch)

def create_zoom_mask(t: np.ndarray, window: float = 1.0) -> np.ndarray | slice:
    mask = (t - t[0]) <= window
    return mask if np.count_nonzero(mask) > 1 else slice(None)

def create_window_mask(t: np.ndarray, time_window: tuple[float, float] | list[float]) -> np.ndarray | slice:
    start, end = time_window
    mask = (t >= start) & (t <= end)
    return mask if np.count_nonzero(mask) > 1 else slice(None)

def rollout_model(model: PHVIV, y0: torch.Tensor, vel: torch.Tensor, m_eff: float,
                  dt: float, t: np.ndarray, D: float, k: float, device: torch.device) -> dict[str, np.ndarray]:
    """Roll the model forward over the full time grid and return normalised traces."""
    p0 = vel[0] * m_eff
    state = torch.stack((y0[0], p0), dim=0).unsqueeze(0).to(device)
    y_samples: list[float] = []
    p_samples: list[float] = []
    force_total: list[float] = []
    force_drag: list[float] = []
    force_model: list[float] = []
    hamiltonian_model_vals: list[float] = []
    with torch.no_grad():
        for _ in range(len(t)):
            y_samples.append(float(state[0, 0].detach().cpu()))
            p_samples.append(float(state[0, 1].detach().cpu()))
            model_force = float(model.learned_force(state).squeeze().detach().cpu())
            if model.include_physical_drag:
                drag_force = float(model.drag_force(state).squeeze().detach().cpu())
            else:
                drag_force = 0.0
            total_force = float(model.u_theta(state).squeeze().detach().cpu())
            H_val = float(model.H(state).detach().cpu())
            force_model.append(model_force)
            force_drag.append(drag_force)
            force_total.append(total_force)
            hamiltonian_model_vals.append(H_val)
            state = model.step_rk4(state, t, dt)
    y_samples = np.asarray(y_samples)
    p_samples = np.asarray(p_samples)
    y_pred_norm = y_samples / D
    p_pred_norm = (p_samples / m_eff) / (np.sqrt(k / m_eff) * D)
    force_total_arr = np.asarray(force_total)
    force_drag_arr = np.asarray(force_drag)
    force_model_arr = np.asarray(force_model)
    hamiltonian_model_arr = np.asarray(hamiltonian_model_vals)
    return {
        "y_norm": y_pred_norm,
        "p_norm": p_pred_norm,
        "force_total": force_total_arr,
        "force_drag": force_drag_arr,
        "force_model": force_model_arr,
        "hamiltonian_model": hamiltonian_model_arr,
    }
