import os as _os
# Limit CPU threads (useful if running on CPU)
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from helper_functions import *
from ODE_pinn_helper import *
from architectures import *
from ODE import build_ode_setup, print_ode_summary

# Prefer Apple GPU (MPS) if available; use float32 for stability/speed
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

#Dataset parameters
t_points = 200 #Dimentions of dataset
input_size = 1   # number of features in your data
output_size = 2 #Output size

# Plot visualization limits
Y_PLOT_LIMIT = 1.0
Q_PLOT_LIMIT = 1.0

# Load shared ODE configuration
ode_setup = build_ode_setup()
cfg = ode_setup["config"]
ODE_params = ode_setup["ode_params"]
u_ic = ode_setup["u_ic"]
T_MIN_CONST = ode_setup["T_MIN_CONST"]
T_MAX_CONST = ode_setup["T_MAX_CONST"]
D = cfg.D
print_ode_summary(ode_setup)
# TensorBoard run naming (set to a string to override timestamp folder)
LOG_RUN_NAME = None  # e.g., "pinn_exp1"; None uses timestamped default

#Architechture parameters
num_blocks = 2 #Depth of networks
hidden_size = 64  # number of hidden units
fourier_features = 64
sigma = 30.0
# Sine embedding for PirateNet's U-branch
USE_SINE_EMBED = False
W0_EMBED = 5.0
num_chunks = 8 #Time chunks for causal training
n_per_chunk = 256
use_rwf=False
rwf_mu = 1.0; rwf_sigma = 0.1
factorize_output=False

#Optimization parameters
total_steps = int(5e4)
grad_clip_max_norm = 1e5  # gradient clipping threshold (L2 norm)
causal_weight = 5.0
lambda_freq = 1000
grad_norm_alpha = 0.9
# GradNorm clamp limits (min/max lambda); configurable
LAMBDA_MIN = 0.2
LAMBDA_MAX = 5.0

#Learning rate parameters
base_lr = 1e-3
decay_rate = 0.9
decay_steps = 2000
warmup_steps = 3000

#Logging parameters
steps_per_heatmap = 500  # log heatmap figures to TensorBoard every N epochs (0 to disable)
log_every_n_steps = 100

def main():
    #Loading the data
    # Physics-driven normalization for inputs: x in [-1,1], t in [-1,1]; keep u unscaled
    T_range = float(T_MAX_CONST - T_MIN_CONST)

    # Domain in standardized coordinates and IC in physical units
    t_min = torch.tensor(-1.0, dtype=dtype, device=device)
    t_max = torch.tensor(1.0, dtype=dtype, device=device)

    # Instantiate Simple MLP on [x, t]
    model = ODEPirateNet(input_size=input_size,
                      output_size=output_size,
                      depth=num_blocks,
                      fourier_features=fourier_features,
                      sigma = sigma, 
                      dtype=dtype,
                      use_rwf=use_rwf, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma,
                      factorize_output=factorize_output,
                      use_sine_embed=USE_SINE_EMBED,
                      w0_embed=W0_EMBED).to(device=device, dtype=dtype)

    t_scale = torch.tensor(T_range/2.0,  dtype=dtype, device=device)

    # TensorBoard writer
    import os as _os, time as _time
    run_dir = _os.path.join(
        "runs",
        LOG_RUN_NAME if LOG_RUN_NAME else f"pinn_kdv_{_time.strftime('%Y%m%d-%H%M%S')}"
    )
    writer = SummaryWriter(log_dir=run_dir)

    """
    # Initialize the last layer to map features to IC across all times
    stats = model.physics_init(X_val, u0 = u0, add_bias=True, return_diagnostics=True)

    print('PirateNet Physics Initialization:')
    print("RMSE:", stats["rmse"].cpu().numpy())
    print("rel L2:", stats["rel_l2"].cpu().numpy())
    print("max |err|:", stats["max_abs"].cpu().numpy())
    print("||theta||_2:", stats["theta_l2"].cpu().numpy())
    """
    # Stepper-based training like pinn.py
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    lr_scheduler = LrSchedule(base_lr, decay_rate, warmup_steps, decay_steps)
    stepper = ODETrainingStepper(model, t_min, t_max, u_ic,
                              ODE_params,
                              t_scale,
                              optimizer, grad_clip_max_norm,
                              n_per_chunk, num_chunks, causal_weight,
                              lambda_freq, log_every_n_steps, writer,
                              lr_scheduler, grad_norm_alpha, diameter=D,
                              y_plot_limit=Y_PLOT_LIMIT, q_plot_limit=Q_PLOT_LIMIT)
    
    # Apply GradNorm clamp limits from config
    try:
        stepper.balancer.w_min = float(LAMBDA_MIN)
        stepper.balancer.w_max = float(LAMBDA_MAX)
    except Exception:
        pass
    
    #stepper.define_validation_data(torch.as_tensor(X_val, dtype=dtype), torch.as_tensor(y_val, dtype=dtype))

    for _ in range(total_steps):
        model.train()
        stepper.step()
        if (stepper.steps % steps_per_heatmap) == 0:
            stepper.log_yq_curves()

    # Close TensorBoard writer
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
