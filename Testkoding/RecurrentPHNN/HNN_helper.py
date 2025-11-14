import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import math



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
        force_hidden_size: int = 64,
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

        self.nn_q_scale = float(q_scale)
        self.nn_p_scale = float(p_scale)

        # Recurrent module for instantaneous force u(x)
        self.force_gru = nn.GRU(
            input_size=2,
            hidden_size=force_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.force_head = nn.Sequential(
            nn.Linear(force_hidden_size, force_hidden_size),
            nn.GELU(),
            nn.Linear(force_hidden_size, 1),
        )

        if self.learn_hamiltonian:
            self.h_net = nn.Sequential(
                nn.Linear(2, 100),
                nn.Tanh(),
                nn.Linear(100, 100),
                nn.GELU(),
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

        self.register_buffer(
            "J",
            torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32),
        )
        self.register_buffer(
            "G",
            torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        )

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
    
    def g(self, x, h):
        gH = self.grad_H(x)                         # (..., 2)
        R = self.R(x)                               # (..., 2, 2)

        J = self.J.to(x.device).to(x.dtype)

        JgH = torch.einsum('ij,...j->...i', J, gH) #Just  J @ gH, with batch handling
        RgH = torch.einsum('...ij,...j->...i', R, gH)

        core = JgH - RgH

        force_model, h_next = self.force_step(x, h)
        if self.include_physical_drag:
            force_drag = self.drag_force(x)
        else:
            force_drag = torch.zeros_like(force_model)
        force_total = force_model + force_drag

        G = self.G.to(device=x.device, dtype=x.dtype)
        Gu = torch.einsum("ij,...j->...i", G, force_total)

        return core + Gu, force_model, force_total, h_next
    
    def drag_force(self, x):
        """
        Morison-like cross-flow drag: Fd = -0.5 * rho * D * Cd * |v| * v
        x: (..., 2) with x[..., 1] = momentum
        returns: (..., 1)
        """
        momentum = x[..., 1]
        v = momentum / self.m
        U = torch.full_like(v, self.U)
        Cd = torch.exp(self.log_Cd)  # keep it positive
        rel_vel = torch.sqrt(v**2 + U**2)
        Fd = -0.5 * self.rho * self.D * Cd * torch.abs(rel_vel) * v
        return Fd.unsqueeze(-1)

    def init_force_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return torch.zeros(
            1,
            batch_size,
            self.force_gru.hidden_size,
            device=device,
            dtype=dtype,
        )

    def _scaled_state(self, x: torch.Tensor) -> torch.Tensor:
        q_scaled = x[..., 0] / self.nn_q_scale
        p_scaled = x[..., 1] / self.nn_p_scale
        return torch.stack((q_scaled, p_scaled), dim=-1)

    def force_step(self, x: torch.Tensor, hidden: torch.Tensor):
        """
        Run one recurrent force step.
        Args:
            x: (..., 2) state [y, p]
            hidden: (1, B, hidden_size)
        Returns:
            force_pred: (..., 1)
            hidden_next: (1, B, hidden_size)
        """
        x_scaled = self._scaled_state(x)
        gru_in = x_scaled.unsqueeze(1)  # (..., 1, 2)
        out, hidden_next = self.force_gru(gru_in, hidden)
        force = self.force_head(out.squeeze(1)) * self.k * self.D
        return force, hidden_next

    def dynamics_step(
        self, x: torch.Tensor, hidden: torch.Tensor
    ):
        """
        Evaluate Hamiltonian dynamics plus learned force for one step.
        Returns derivative dz/dt along with updated hidden and force components.
        """
        gH = self.grad_H(x)
        R = self.R(x)
        J = self.J.to(device=x.device, dtype=x.dtype)

        JgH = torch.einsum("ij,...j->...i", J, gH)
        RgH = torch.einsum("...ij,...j->...i", R, gH)
        core = JgH - RgH

        force_model, hidden_next = self.force_step(x, hidden)
        if self.include_physical_drag:
            force_drag = self.drag_force(x)
        else:
            force_drag = torch.zeros_like(force_model)
        force_total = force_model + force_drag

        G = self.G.to(device=x.device, dtype=x.dtype)
        Gu = torch.einsum("ij,...j->...i", G, force_total)
        dz = core + Gu
        return dz, hidden_next, force_model, force_total
    
    def SRK4_dynamics_step(
        self, zi: torch.Tensor, zin: torch.Tensor, h: torch.Tensor
    ):
        """
        Evaluate Hamiltonian dynamics plus learned force for one step.
        Returns derivative dz/dt along with updated hidden and force components.
        Using SRK4 
        """
        a = 0.5
        b = math.sqrt(3.0) / 6.0
        dt = self.dt

        # midpoint between zn and zn+1
        z_mid = 0.5 * (zi + zin)             # (B, d)

        # stage convex combos
        z_a_plus  = (0.5 + b) * zi + (0.5 - b) * zin   # (B, d)
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin   # (B, d)

        # stage evaluations of g
        g_a_plus, _, _, _  = self.g(z_a_plus, h)                  # (B, d)
        g_a_minus, _, _, _= self.g(z_a_minus, h)  

        # two corrected midpoints
        z_corr_minus = z_mid - b * dt * g_a_plus      # (B, d)
        z_corr_plus  = z_mid + b * dt * g_a_minus     # (B, d)

        # final two g-evals
        g1, f1m, f1t, h1 = self.g(z_corr_minus, h)                     # (B, d)
        g2, f2m, f2t, h2 = self.g(z_corr_plus, h)
        
        dz_model = 0.5 * g1 + 0.5 * g2
        h_next = 0.5 * h1 + 0.5 * h2
        force_total = 0.5 * f1t + 0.5 * f2t
        force_model= 0.5 * f1m + 0.5 * f2m                            # (B, d)

        return dz_model, h_next, force_model, force_total

    def RK4_dynamics_step(self, z, h):
        dt = self.dt

        # k1
        k1, f1m, f1t, h1 = self.g(z, h)
        z2 = z + 0.5 * dt * k1

        # k2
        k2, f2m, f2t, h2 = self.g(z2, h)   # branch from h (don’t chain via h1)

        z3 = z + 0.5 * dt * k2

        # k3
        k3, f3m, f3t, h3 = self.g(z3, h)

        z4 = z + dt * k3

        # k4
        k4, f4m, f4t, h4 = self.g(z4, h)

        z_next = z + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        h_next = (h1 + 2*h2 + 2*h3 + h4) / 6.0

        # (optional) stage-averaged forces if you want to log them
        f_model_avg = (f1m + 2*f2m + 2*f3m + f4m) / 6.0
        f_total_avg = (f1t + 2*f2t + 2*f3t + f4t) / 6.0
        return z_next, h_next, f_model_avg, f_total_avg


    def sequence_residual(self, z_seq: torch.Tensor, warmup_steps = 20):
        """
        Compute residual loss over a batch of sequences.
        z_seq: (B, L, 2) states ordered in time.
        """
        batch, length, _ = z_seq.shape
        if length < 2:
            raise ValueError("sequence length must be at least 2 for residual computation")
        hidden = self.init_force_hidden(batch, z_seq.device, z_seq.dtype)
        res_terms: list[torch.Tensor] = []
        for idx in range(length - 1):
            zi = z_seq[:, idx]
            zin = z_seq[:, idx + 1]
            dz_fd = (zin - zi) / self.dt
            dz_model, hidden, _, _, _ = self.dynamics_step(zi, hidden)
            res = dz_fd - dz_model
            res_terms.append(torch.sum(res**2, dim=1))
        res_stack = torch.stack(res_terms, dim=1)
        res_stack = res_stack[:, warmup_steps:]

        return res_stack.mean()
    
    def sequence_residual_SRK4(self, z_seq: torch.Tensor, warmup_steps = 20):
        """
        Compute residual loss over a batch of sequences.
        z_seq: (B, L, 2) states ordered in time.
        """
        batch, length, _ = z_seq.shape
        if length < 2:
            raise ValueError("sequence length must be at least 2 for residual computation")
        hidden = self.init_force_hidden(batch, z_seq.device, z_seq.dtype)
        res_terms: list[torch.Tensor] = []
        for idx in range(length - 1):
            zi = z_seq[:, idx]
            zin = z_seq[:, idx + 1]
            dz_fd = (zin - zi) / self.dt
            dz_model, hidden, _, _ = self.SRK4_dynamics_step(zi, zin, hidden)
            res = dz_fd - dz_model
            res_terms.append(torch.sum(res**2, dim=1))
        res_stack = torch.stack(res_terms, dim=1)
        res_stack = res_stack[:, warmup_steps:]

        return res_stack.mean()

    def sequence_force_penalty(self, z_seq: torch.Tensor, warmup_steps = 20):
        """
        L1 penalty on recurrent force magnitude across a sequence.
        """
        batch, length, _ = z_seq.shape
        if length < 2:
            return torch.tensor(0.0, device=z_seq.device, dtype=z_seq.dtype)
        hidden = self.init_force_hidden(batch, z_seq.device, z_seq.dtype)
        penalties: list[torch.Tensor] = []
        for idx in range(length - 1):
            zi = z_seq[:, idx]
            _, hidden, force_model, _, _ = self.dynamics_step(zi, hidden)
            penalties.append(torch.abs(force_model.squeeze(-1)))
        penalty = torch.stack(penalties, dim=1)
        penalty = penalty[:, warmup_steps:]

        return penalty.mean()
    
    def sequence_force_penalty_SRK4(self, z_seq: torch.Tensor, warmup_steps = 20):
        """
        L1 penalty on recurrent force magnitude across a sequence.
        """
        batch, length, _ = z_seq.shape
        if length < 2:
            return torch.tensor(0.0, device=z_seq.device, dtype=z_seq.dtype)
        hidden = self.init_force_hidden(batch, z_seq.device, z_seq.dtype)
        penalties: list[torch.Tensor] = []
        for idx in range(length - 1):
            zi = z_seq[:, idx]
            zin = z_seq[:, idx + 1]
            _, hidden, force_model, force_total= self.SRK4_dynamics_step(zi, zin, hidden)
            penalties.append(torch.abs(force_model.squeeze(-1)))
        penalty = torch.stack(penalties, dim=1)
        penalty = penalty[:, warmup_steps:]

        return penalty.mean()

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
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes

    ax_full.plot(t, y_true_norm, label="y/D (true)")
    ax_full.plot(t, y_pred_norm, label="y/D (pred)")
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("y/D")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Normalized rollout at epoch {epoch+1}")
    ax_full.legend(loc="upper right")

    diff_y_norm = y_pred_norm - y_true_norm
    ax_diff.plot(t, diff_y_norm, label="Δ(y/D)", color="tab:orange")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("time")
    ax_diff.set_ylabel("Δy/D")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title(f"Difference (pred - true) epoch {epoch+1}")
    ax_diff.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], y_true_norm[zoom_mask], label="y/D (true)")
    ax_zoom.plot(t[zoom_mask], y_pred_norm[zoom_mask], label="y/D (pred)")
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("y/D")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Normalized rollout (first 1s) epoch {epoch+1}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], y_true_norm[middle_mask], label="y/D (true)")
    ax_middle.plot(t[middle_mask], y_pred_norm[middle_mask], label="y/D (pred)")
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
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes
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

    diff_force = force_total - force_data
    ax_diff.plot(t, diff_force, label="ΔF_total", color="tab:orange")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("time")
    ax_diff.set_ylabel("ΔForce")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title(f"Force difference (model - data) epoch {epoch+1}")
    ax_diff.legend(loc="upper right")

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
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes
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

    if hamiltonian_data is not None:
        diff_h = h_model_rel - h_data_rel
        ax_diff.plot(t, diff_h, label="ΔH", color="tab:purple")
    else:
        ax_diff.plot(t, np.zeros_like(t), label="ΔH (no data)", color="tab:gray", linestyle="--")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("time")
    ax_diff.set_ylabel("ΔH")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title(f"Hamiltonian difference epoch {epoch+1}")
    ax_diff.legend(loc="upper right")

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

def build_sequence_dataset(
    y_data_t: torch.Tensor,
    vel: torch.Tensor,
    m_eff: float,
    t_tensor: torch.Tensor,
    sequence_length: int,
) -> TensorDataset:
    """
    Build overlapping state sequences suitable for recurrent training.

    Each sample contains:
        - z_seq: (sequence_length + 1, 2) states [y, p] over the window
        - t_seq: (sequence_length + 1,) absolute timestamps
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    z = torch.stack((y_data_t, vel * m_eff), dim=1)  # (T, 2)
    window = sequence_length + 1
    total_steps = z.shape[0]
    if total_steps < window:
        raise ValueError("Not enough samples for the requested sequence_length")

    z_seq_list: list[torch.Tensor] = []
    t_seq_list: list[torch.Tensor] = []
    for start in range(total_steps - window + 1):
        end = start + window
        z_seq_list.append(z[start:end])
        t_seq_list.append(t_tensor[start:end])

    z_batch = torch.stack(z_seq_list, dim=0)
    t_batch = torch.stack(t_seq_list, dim=0)
    return TensorDataset(z_batch, t_batch)

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
    hidden = model.init_force_hidden(batch_size=1, device=device, dtype=state.dtype)
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
            dz, hidden, force_model_t, force_drag_t, force_total_t = model.dynamics_step(state, hidden)
            H_val = float(model.H(state).detach().cpu())

            force_model.append(float(force_model_t.squeeze().detach().cpu()))
            force_drag.append(float(force_drag_t.squeeze().detach().cpu()))
            force_total.append(float(force_total_t.squeeze().detach().cpu()))
            hamiltonian_model_vals.append(H_val)

            state = state + dt * dz
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

def rollout_model_SRK4(model: PHVIV, y0: torch.Tensor, vel: torch.Tensor, m_eff: float,
                  dt: float, t: np.ndarray, D: float, k: float, device: torch.device) -> dict[str, np.ndarray]:
    """Roll the model forward over the full time grid and return normalised traces."""
    p0 = vel[0] * m_eff
    state = torch.stack((y0[0], p0), dim=0).unsqueeze(0).to(device)
    hidden = model.init_force_hidden(batch_size=1, device=device, dtype=state.dtype)
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
            H_val = float(model.H(state).detach().cpu())
            hamiltonian_model_vals.append(H_val)
            state, hidden, force_model_t, force_total_t = model.RK4_dynamics_step(state, hidden)
            
            force_model.append(float(force_model_t.squeeze().detach().cpu()))
            force_total.append(float(force_total_t.squeeze().detach().cpu()))
            
    y_samples = np.asarray(y_samples)
    p_samples = np.asarray(p_samples)
    y_pred_norm = y_samples / D
    p_pred_norm = (p_samples / m_eff) / (np.sqrt(k / m_eff) * D)
    force_total_arr = np.asarray(force_total)
    force_model_arr = np.asarray(force_model)
    hamiltonian_model_arr = np.asarray(hamiltonian_model_vals)
    return {
        "y_norm": y_pred_norm,
        "p_norm": p_pred_norm,
        "force_total": force_total_arr,
        "force_model": force_model_arr,
        "hamiltonian_model": hamiltonian_model_arr,
    }
