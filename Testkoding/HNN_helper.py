import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math



class PHVIV(nn.Module):
    """
    Pseudo-/port-Hamiltonian 1-DOF oscillator with NN force.
    State x = [y, v].
    dot x = (J - R(x)) ∇H(x) + G u_theta(x)
    """
    def __init__(self, dt, m=16.79, k=1218.0, U=0.65, rho=1000.0, D=0.1,
                 q_scale=0.1, p_scale=10.0, max_damping_ratio=0.2,
                 discover_damping: bool = True,
                 damping_c: float | None = None):
        super().__init__()
        self.dt = dt
        self.m = m
        self.k = k
        self.U = U
        self.rho = rho
        self.D = D
        self.max_damping_ratio = max_damping_ratio
        self.q_scale = q_scale
        self.p_scale = p_scale
        self.discover_damping = bool(discover_damping)

        # NN for instantaneous force u(x)
        self.u_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # damping handling
        if self.discover_damping:
            self.zeta_raw = nn.Parameter(torch.log(torch.tensor(1e-3)))
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
        q = x[..., 0]
        p = x[..., 1]
        return 0.5 * self.k * q**2 + 0.5 * q**2 / self.m

    def grad_H(self, x):
        q = x[..., 0]
        p = x[..., 1]
        return torch.stack((self.k * q, p / self.m), dim=-1)

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
        return self.u_net(x)
    
    def u_theta2(self, x):
        q_scaled = x[..., 0] / self.q_scale
        p_scaled = x[..., 1] / self.p_scale
        x_scaled = torch.stack((q_scaled, p_scaled), dim = 1)
        return self.u_net(x_scaled) + self.drag_force(x)
    
    def u_theta(self, x):
        return self.u_theta2(x)
    
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

    def step_rk4(self, x, dt):
        k1 = self.g(x)
        k2 = self.g(x + 0.5 * dt * k1)
        k3 = self.g(x + 0.5 * dt * k2)
        k4 = self.g(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    def res_loss(self, zi, ti, zin, tin):
        return self.res_loss_SRK4(zi, ti, zin, tin)
    
    def force_loss(self, zi, ti, zin, tin):
        return self.force_loss_SRK4(zi, ti, zin, tin)
    
    def res_loss_Euler(self, zi, ti, zin, tin):
        dz = (zin-zi)/self.dt
        z_mean = 0.5*(zin+zi)
        res = dz - self.g(z_mean)
        scale = torch.tensor((self.q_scale, self.p_scale), device=res.device, dtype=res.dtype)
        res_scaled = res / scale
        loss = torch.mean(torch.sum(res_scaled**2, dim=1))
        return loss

    def force_loss_Euler(self, zi, ti, zin, tin):
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
    
    def force_loss_SRK4(self, zi, ti, zin, tin):
        dt = self.dt
        b = math.sqrt(3.0) / 6.0

        # same stage points as in res_loss
        z_a_plus  = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin

        # evaluate learned force at both stages
        f1 = self.f(z_a_plus)    # assume (B, 1) or (B, 2)
        f2 = self.f(z_a_minus)
      
        # --- alt version: force is 2D (correction to whole vector field) ---
        force_scale = torch.tensor(
            (self.q_scale, self.p_scale),
            device=f1.device, dtype=f1.dtype
        )
        f1_scaled = f1 / force_scale
        f2_scaled = f2 / force_scale
        
        loss = 0.5 * torch.mean(torch.sum(f1_scaled**2, dim=1)) \
            + 0.5 * torch.mean(torch.sum(f2_scaled**2, dim=1))
        return loss




def rollout_len_schedule(epoch, characteristic_epoch):
    if epoch < 1*characteristic_epoch:
        return 32
    elif epoch < 2*characteristic_epoch:
        return 64
    elif epoch < 3*characteristic_epoch:
        return 128
    elif epoch < 4*characteristic_epoch:
        return 256
    elif epoch < 5*characteristic_epoch:
        return 400
    else:
        return 600
