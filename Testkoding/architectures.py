import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.colors as colors

class RandomFactorizedLinear(nn.Module):
    """
    Random Weight Factorization (RWF) linear layer.

    Implements W = diag(exp(s)) @ V, where V is a trainable weight matrix
    initialized with Glorot/Xavier, and s ~ N(mu, sigma) is a trainable
    per-output scale vector. Bias is standard trainable bias.

    Recommended defaults: mu=1.0, sigma=0.1 (from RWF paper/algorithm).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 mu: float = 1.0, sigma: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = float(mu)
        self.sigma = float(sigma)

        # Trainable base weight and scale
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        self.s = nn.Parameter(torch.empty(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialization: Glorot for V, Normal for s, zeros for bias
        nn.init.xavier_uniform_(self.V)
        with torch.no_grad():
            self.s.normal_(mean=self.mu, std=self.sigma)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Efficient row scaling by exp(s): no explicit diag matrix needed
        W_eff = torch.exp(self.s).unsqueeze(1) * self.V  # (out, in)
        return F.linear(x, W_eff, self.bias)


class SimpleMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers: int = 4,
                 hidden_sizes=None, activation=torch.nn.Tanh, fourier_features: int = 16, sigma: float = 1.0,
                 use_rwf: bool = False, rwf_mu: float = 1.0, rwf_sigma: float = 0.1, dtype: torch.dtype = torch.float64,
                 factorize_output: bool = True):
        super().__init__()

        if hidden_sizes is None:
            if num_hidden_layers < 1:
                raise ValueError("num_hidden_layers must be >= 1")
            hidden_sizes = [hidden_size] * num_hidden_layers
        else:
            if isinstance(hidden_sizes, int):
                hidden_sizes = [hidden_sizes]
            if len(hidden_sizes) < 1:
                raise ValueError("hidden_sizes must have at least one layer width")
            
        self.ff = FourierFeatures(input_size, fourier_features, sigma, dtype)
        embed_dim = 2 * fourier_features
        
        self.activation = activation()
        layer_sizes = [embed_dim] + list(hidden_sizes)

        def _make_layer(in_f, out_f, is_output=False):
            if use_rwf and (factorize_output or not is_output):
                return RandomFactorizedLinear(in_f, out_f, bias=True, mu=rwf_mu, sigma=rwf_sigma)
            else:
                lin = nn.Linear(in_f, out_f)
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
                return lin

        self.hidden = nn.ModuleList([
            _make_layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(hidden_sizes))
        ])
        self.out = _make_layer(layer_sizes[-1], output_size, is_output=True)

    def forward(self, x):
        h = self.ff(x)
        for lin in self.hidden:
            h = self.activation(lin(h))
        return self.out(h)

class Sine(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SirenMLP(torch.nn.Module):
    """
    SIREN-based MLP using sine activations, suitable for PINNs and signals with
    high-frequency content. Follows recommended initialization from the SIREN paper.

    Args:
      input_size: input feature dimension
      hidden_size: width of hidden layers
      output_size: output dimension
      num_hidden_layers: number of hidden layers (>=1)
      w0: frequency scale for the first layer (typically 30.0)
      w0_hidden: frequency scale for hidden layers (typically 1.0)
      final_activation: optional activation for the output (default: None for linear)
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_hidden_layers: int = 4,
                 w0: float = 30.0,
                 w0_hidden: float = 1.0,
                 final_activation: torch.nn.Module | None = None):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        self.w0 = float(w0)
        self.w0_hidden = float(w0_hidden)
        self.final_activation = final_activation

        self.first = torch.nn.Linear(input_size, hidden_size)
        self.hidden = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)
        ])
        self.out = torch.nn.Linear(hidden_size, output_size)

        self.sine = Sine()

        self._siren_init()

    def _siren_init(self):
        # First layer init: U(-1/in, 1/in)
        in_f = self.first.in_features
        bound_first = 1.0 / in_f
        torch.nn.init.uniform_(self.first.weight, -bound_first, bound_first)
        torch.nn.init.uniform_(self.first.bias, -bound_first, bound_first)

        # Hidden layers init: U(-sqrt(6/in)/w0_hidden, sqrt(6/in)/w0_hidden)
        for lin in self.hidden:
            in_h = lin.in_features
            bound_h = (6.0 ** 0.5) / in_h / max(self.w0_hidden, 1e-8)
            torch.nn.init.uniform_(lin.weight, -bound_h, bound_h)
            torch.nn.init.uniform_(lin.bias, -bound_h, bound_h)

        # Output layer: small init to prevent large outputs initially
        torch.nn.init.uniform_(self.out.weight, -1e-4, 1e-4)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.sine(self.w0 * self.first(x))
        for lin in self.hidden:
            h = self.sine(self.w0_hidden * lin(h))
        y = self.out(h)
        return self.final_activation(y) if self.final_activation is not None else y
    



class FourierFeatures(torch.nn.Module):
    """Random Fourier feature mapping for [x,t] inputs: phi(z) = [sin(Bz), cos(Bz)]."""
    def __init__(self, in_dim: int, out_features: int, sigma: float = 1.0, dtype=torch.float64):
        super().__init__()
        B = torch.randn(out_features, in_dim, dtype=dtype) * sigma
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input matches buffer dtype/device to avoid matmul dtype errors
        x = x.to(device=self.B.device, dtype=self.B.dtype)
        z = x @ self.B.t()
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim: int, activation: torch.nn.Module, alpha_init: float = 0.0,
                 use_rwf: bool = False, rwf_mu: float = 1.0, rwf_sigma: float = 0.1):
        super().__init__()
        if use_rwf:
            self.w1 = RandomFactorizedLinear(dim, dim, bias=True, mu=rwf_mu, sigma=rwf_sigma)
            self.w2 = RandomFactorizedLinear(dim, dim, bias=True, mu=rwf_mu, sigma=rwf_sigma)
            self.w3 = RandomFactorizedLinear(dim, dim, bias=True, mu=rwf_mu, sigma=rwf_sigma)
        else:
            self.w1 = torch.nn.Linear(dim, dim)
            self.w2 = torch.nn.Linear(dim, dim)
            self.w3 = torch.nn.Linear(dim, dim)
        self.act = activation
        self.alpha = nn.Parameter(torch.tensor(alpha_init)) 
        if not use_rwf:
            torch.nn.init.xavier_uniform_(self.w1.weight); torch.nn.init.zeros_(self.w1.bias)
            torch.nn.init.xavier_uniform_(self.w2.weight); torch.nn.init.zeros_(self.w2.bias)
            torch.nn.init.xavier_uniform_(self.w3.weight); torch.nn.init.zeros_(self.w3.bias)

    def forward(self, x: torch.Tensor, U: torch.Tensor, V:torch.Tensor) -> torch.Tensor:
        f = self.act(self.w1(x))
        z = f * U + (1-f) * V
        g = self.act(self.w2(z))
        z = g * U + (1-g) * V
        h = self.act(self.w3(z))
        y = self.alpha * h + (1-self.alpha) * x
        return y

class PirateNet(torch.nn.Module):
    """
    PirateNet: A flexible residual MLP with optional Fourier features and sine activations.

    Features:
      - Optional random Fourier feature embedding of inputs (sin/cos of Gaussian projections)
      - Residual blocks with optional LayerNorm and Dropout
      - Choice of tanh or sine activations (SIREN-style first layer scaling)
      - Periodic-friendly and suitable for PINNs with higher-order derivatives

    Args:
      input_size: base input dimension (e.g., 2 for [x,t])
      output_size: outputs (e.g., 1)
      hidden_size: width of hidden state
      depth: number of residual blocks
      fourier_features: number of random Fourier features per sin/cos branch (0 disables)
      sigma: std for random feature matrix
      use_sine: use sine activation; else tanh
      w0: frequency scale for the first linear layer when use_sine
      layer_norm: enable LayerNorm in residual blocks
      dropout: dropout probability in residual blocks
      skip_every: concatenate input skip every k blocks (0 disables); concatenation is projected back to hidden_size
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int = 128,
                 depth: int = 6,
                 fourier_features: int = 128,
                 sigma: float = 10.0,
                 layer_norm: bool = False,
                 dropout: float = 0.0,
                 skip_every: int = 0,
                 dtype: torch.dtype = torch.float32,
                 use_rwf: bool = False,
                 rwf_mu: float = 1.0,
                 rwf_sigma: float = 0.1,
                 factorize_output: bool = False,
                 # Optional: make the first embedding branch sinusoidal
                 use_sine_embed: bool = False,
                 w0_embed: float = 5.0):
        super().__init__()

        self.ff = FourierFeatures(input_size, fourier_features, sigma, dtype)
        embed_dim = 2 * fourier_features
        self.use_sine_embed = bool(use_sine_embed)
        self.w0_embed = float(w0_embed)

        # First projection: map Fourier features
        if use_rwf:
            self.w1 = RandomFactorizedLinear(embed_dim, embed_dim, bias=True, mu=rwf_mu, sigma=rwf_sigma)
            self.w2 = RandomFactorizedLinear(embed_dim, embed_dim, bias=True, mu=rwf_mu, sigma=rwf_sigma)
        else:
            self.w1 = torch.nn.Linear(embed_dim, embed_dim)
            self.w2 = torch.nn.Linear(embed_dim, embed_dim)
            # If using sinusoidal first branch, follow SIREN-style init for w1; otherwise Xavier
            if self.use_sine_embed:
                in_f = self.w1.in_features
                bound = 1.0 / in_f
                torch.nn.init.uniform_(self.w1.weight, -bound, bound)
                torch.nn.init.uniform_(self.w1.bias, -bound, bound)
            else:
                torch.nn.init.xavier_uniform_(self.w1.weight); torch.nn.init.zeros_(self.w1.bias)
            torch.nn.init.xavier_uniform_(self.w2.weight); torch.nn.init.zeros_(self.w2.bias)

        # Residual stack
        self.act = torch.nn.Tanh()
        self.sine = Sine()
        self.blocks = torch.nn.ModuleList([
            ResidualBlock(embed_dim, self.act, use_rwf=use_rwf, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
            for _ in range(depth)
        ])

        # Output
        if use_rwf and factorize_output:
            self.out = RandomFactorizedLinear(embed_dim, output_size, bias=True, mu=rwf_mu, sigma=rwf_sigma)
        else:
            self.out = torch.nn.Linear(embed_dim, output_size)
            torch.nn.init.xavier_uniform_(self.out.weight); torch.nn.init.zeros_(self.out.bias)

        # Ensure all parameters/buffers use requested dtype
        self.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.ff(x)
        U_lin = self.w1(phi)
        U = self.sine(self.w0_embed * U_lin) if self.use_sine_embed else self.act(U_lin)
        V = self.act(self.w2(phi))
        # Propagate hidden state through residual blocks
        h = phi
        for blk in self.blocks:
            h = blk(h, U, V)
        return self.out(h)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden representation before the final linear layer."""
        phi = self.ff(x)
        U_lin = self.w1(phi)
        U = self.sine(self.w0_embed * U_lin) if self.use_sine_embed else self.act(U_lin)
        V = self.act(self.w2(phi))
        h = phi
        for blk in self.blocks:
            h = blk(h, U, V)
        return h

    @torch.no_grad()
    def physics_init(self,
                                   X: torch.Tensor,
                                   y: torch.Tensor,
                                   ridge: float = 1e-6,
                                   add_bias: bool = False):
        """
        Initialize the final linear layer (self.out) via least squares on
        provided (X, y) pairs (e.g., initial condition u(x, t0)).

        Solves min_{W,b} || H(X) W^T + b - y ||_2^2 with ridge regularization,
        where H(X) are the hidden features before the output layer.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        X = X.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        # Ensure y is 2D: (N, out). If given as (N,), unsqueeze to (N,1).
        if y.ndim == 1:
            y = y.unsqueeze(1)

        H = self.features(X)  # (N, hidden)
        if add_bias:
            ones = torch.ones(H.size(0), 1, device=device, dtype=dtype)
            Phi = torch.cat([H, ones], dim=1)  # (N, hidden+1)
        else:
            Phi = H

        PT_P = Phi.T @ Phi
        if ridge > 0:
            PT_P = PT_P + ridge * torch.eye(PT_P.size(0), device=device, dtype=dtype)
        PT_y = Phi.T @ y

        theta = torch.linalg.solve(PT_P, PT_y)  # (hidden[+1], out)

        if add_bias:
            W = theta[:-1, :].mT  # (out, hidden)
            b = theta[-1, :]
            self.out.weight.data.copy_(W)
            self.out.bias.data.copy_(b)
        else:
            W = theta.mT  # (out, hidden)
            self.out.weight.data.copy_(W)
            torch.nn.init.zeros_(self.out.bias)

        return theta
