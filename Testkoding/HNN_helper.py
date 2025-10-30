import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



class PHVIV(nn.Module):
    """
    Pseudo-/port-Hamiltonian 1-DOF oscillator with NN force.
    State x = [y, v].
    dot x = (J - R(x)) ∇H(x) + G u_theta(x)
    """
    def __init__(self, m=16.79, k=1218.0, U = 0.65, rho=1000.0, D=0.1):
        super().__init__()
        self.m = m
        self.k = k
        self.U = U
        self.rho = rho
        self.D = D

        # NN for instantaneous force u(x)
        self.u_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # learnable damping
        self.zeta_raw = nn.Parameter(torch.tensor(-3.0))
        #Learable drag coefficient
        self.log_Cd = nn.Parameter(torch.log(torch.tensor(1.2)))  # start at ~1.2


        self.register_buffer("J", torch.tensor([[0.0, 1.0],
                                                [-1.0, 0.0]]))
        self.register_buffer("G", torch.tensor([[0.0],
                                                [1.0 / self.m]]))

    def H(self, x):
        y = x[..., 0]
        v = x[..., 1]
        return 0.5 * self.k * y**2 + 0.5 * self.m * v**2

    def grad_H(self, x):
        y = x[..., 0]
        v = x[..., 1]
        return torch.stack((self.k * y, self.m * v), dim=-1)

    def R(self, x):

        zeta = torch.sigmoid(self.zeta_raw)
        R = torch.zeros(*x.shape[:-1], 2, 2, device=x.device, dtype=x.dtype)
        R[..., 1, 1] = 2*zeta*torch.sqrt(torch.tensor(self.k*self.m))/self.m**2
        return R
    
    def drag_force(self, x):
        """
        Morison-like cross-flow drag: Fd = -0.5 * rho * D * Cd * |v| * v
        x: (..., 2)
        returns: (..., 1)
        """
        v = x[..., 1]
        U = torch.full_like(v, self.U)
        Cd = torch.exp(self.log_Cd)  # keep it positive
        rel_vel = torch.sqrt(v**2 + U**2)
        Fd = -0.5 * self.rho * self.D * Cd * torch.abs(rel_vel) * v
        return Fd.unsqueeze(-1)


    def u_theta(self, x):
        return self.u_net(x)
    
    def u_theta2(self, x):
        return self.u_net(x) + self.drag_force(x)

    def f(self, x):
        gH = self.grad_H(x)                         # (..., 2)
        R = self.R(x)                               # (..., 2, 2)

        J = self.J.to(x.device).to(x.dtype)
        G = self.G.to(x.device).to(x.dtype)

        JgH = torch.einsum('ij,...j->...i', J, gH) #Just  J @ gH, with batch handling
        RgH = torch.einsum('...ij,...j->...i', R, gH)

        core = JgH - RgH

        u = self.u_theta2(x)                         # (..., 1)
        Gu = torch.einsum('ij,...j->...i', G, u)    # (..., 2)

        return core + Gu

    def step_euler(self, x, dt):
        return x + dt * self.f(x)

    def step_rk4(self, x, dt):
        k1 = self.f(x)
        k2 = self.f(x + 0.5 * dt * k1)
        k3 = self.f(x + 0.5 * dt * k2)
        k4 = self.f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
