from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ODEConfig:
    """Physical constants and empirical coefficients for the VIV ODE system."""

    # Physical properties
    mass_ratio: float = 2.0
    Tny: float = 1.0
    D: float = 0.2
    zeta: float = 0.002
    St: float = 0.2
    Ca: float = 1.0
    rho: float = 1025.0
    CL0: float = 0.8
    Vr: float = 5.0

    # Empirical wake coupling parameters
    coupling_coeff: float = 0.8
    eta_y: float = 0.4

    # Initial conditions (amplitude and wake value)
    A0: float = 1
    q0: float = 1

    # Time span in physical coordinates
    T_min: float = 0.0
    T_max: float = 5.0


def build_ode_setup(config: ODEConfig | None = None) -> Dict[str, Any]:
    """
    Compute derived ODE parameters, scaling factors, and helper values shared
    between training scripts.
    """
    cfg = config or ODEConfig()

    mf = cfg.rho * cfg.D**2 / (4.0 * np.pi)
    U = cfg.Vr * cfg.D / cfg.Tny
    omegany = 2.0 * np.pi / cfg.Tny
    omegaf = 2.0 * np.pi * cfg.St * cfg.Vr / cfg.Tny

    my = (cfg.mass_ratio + cfg.Ca) * mf
    cy = 2.0 * cfg.zeta * my * omegany
    ky = omegany**2 * my
    k3y = 0.0  # Duffing term disabled by default; adjust in config if needed.
    Kl = 0.5 * cfg.rho * U**2 * cfg.D * cfg.CL0

    cq = cfg.eta_y * omegaf
    kq = omegaf**2
    Kc = cfg.coupling_coeff / cfg.D

    y_scale = cfg.D * ky
    q_scale = max(cfg.q0, 1e-6) * omegaf**2

    ode_params = {
        "my": my,
        "cy": cy,
        "ky": ky,
        "k3y": k3y,
        "Kl": Kl,
        "cq": cq,
        "kq": kq,
        "Kc": Kc,
        "y_scale": y_scale,
        "q_scale": q_scale,
    }

    structural_period = 2.0 * np.pi / omegany
    vortex_period = 2.0 * np.pi / omegaf

    return {
        "config": cfg,
        "ode_params": ode_params,
        "u_ic": [cfg.A0 * cfg.D, cfg.q0, 0.0, 0.0],
        "T_MIN_CONST": cfg.T_min,
        "T_MAX_CONST": cfg.T_max,
        "structural_period": structural_period,
        "vortex_period": vortex_period,
        "omegany": omegany,
        "omegaf": omegaf,
        "U": U,
    }


def print_ode_summary(setup: Dict[str, Any]) -> None:
    """Utility helper to print characteristic periods."""
    struct = setup["structural_period"]
    vortex = setup["vortex_period"]
    print(f"Structural period: {struct}")
    print(f"Vortex-shedding period: {vortex}")
