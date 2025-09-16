import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os as _os, time as _time

from helper_functions import *
from architectures import *

# Force CPU and float64 precision
device = torch.device("cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

#Dataset parameters
x_points, t_points = 100, 400 #Dimentions of dataset
input_size = 2   # number of features in your data
output_size = 1 #Output size
train_frac, test_frac = 0.6, 0.2

# Hardcoded domain constants (raw standardized coordinates)
X0_CONST, L_CONST, T_MIN_CONST, T_MAX_CONST = 0.0, 20.0, 0.0, 10.0

#Architechture parameters
num_hidden_layers = 6 #Depth of network
hidden_size = 64  # number of hidden units
batch_size = 128 #Batchsize for training

#Optimization parameters
epochs = 200
lr = 4e-5
grad_clip_max_norm = 10.0  # gradient clipping threshold (L2 norm)
early_stop_patience = 50   # epochs without val improvement before stopping
early_stop_min_delta = 1e-5  # required improvement in val data MSE
use_early_stopping = False

# LBFGS finisher configuration
use_lbfgs_finisher = False
lbfgs_max_iter = 100
lbfgs_history_size = 10
lbfgs_lr = 1.0

#Logging parameters
heatmap_log_every = 50  # log heatmap figures to TensorBoard every N epochs (0 to disable)

# TensorBoard run naming (set string to override timestamp folder)
LOG_RUN_NAME = None  # e.g., "mlp_exp1"; None uses timestamped default

def main():

    # Load dataset
    data = np.load('data_generation/data/data_kdv.npz')
    g_u = data['g_u']
    u_init = data['u']
    xt = data['xt']

    # Build loaders using helpers (single-IC forecasting split)
    train_loader, val_loader, test_loader, x_scaler, y_scaler, x_scale, t_scale, u_scale, y_min, y_max = \
        build_loaders_single_ic(u0=u_init[0], g_u0=g_u[0], xt=xt,
                                train_frac=train_frac, test_frac=test_frac, device=device,
                                batch_size_train=batch_size, batch_size_eval=256,
                                dtype=dtype, num_workers_train=0, num_workers_eval=0)

    # Model, loss, optimizer
    model = SirenMLP(input_size=input_size, hidden_size=hidden_size,
                    output_size=output_size, num_hidden_layers=num_hidden_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    run_dir = _os.path.join(
        "runs",
        LOG_RUN_NAME if LOG_RUN_NAME else f"mlp_kdv_{_time.strftime('%Y%m%d-%H%M%S')}"
    )

    writer = SummaryWriter(log_dir=run_dir)

    # Build full loader for periodic image logging
    X_full_t, y_full_t, _ = transform_with_scalers(
        xt, g_u[0].reshape(-1, 1), device,
        x_scaler, y_scaler,
        u0=None, augment_ic=False,
        dtype=dtype,
    )
    full_loader = to_loader(
        X_full_t, y_full_t,
        batch_size=256, shuffle=False,
        num_workers=0, persistent_workers=False,
        pin_memory=False, dtype=dtype,
    )

    #Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_grad_norms = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            # Compute total L2 grad norm (no clipping)
            total_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    total_sq += float(g.norm(2).item())**2
            train_grad_norms.append(total_sq**0.5)
            optimizer.step()
            train_losses.append(loss.item())

        # quick val
        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        train_mse = float(np.mean(train_losses)) if train_losses else float('nan')
        val_mse = float(np.mean(val_losses)) if val_losses else float('nan')
        print(f"Epoch {epoch+1:02d}  train MSE: {train_mse:.4f}  val MSE: {val_mse:.4f}")

        # Log scalars similar to pinn.py
        log_epoch_scalars(
            writer,
            epoch,
            train={"total": train_mse, "data": train_mse},
            val={"total": val_mse, "data": val_mse},
            weights=(1.0, 0.0, 0.0),
            lr=optimizer.param_groups[0]["lr"],
            grad_norm_mean=float(np.mean(train_grad_norms)) if train_grad_norms else None,
        )

        # Periodic heatmap logging on full dataset
        if heatmap_log_every and (epoch % heatmap_log_every == 0):
            model.eval()
            if hasattr(full_loader, 'dataset') and len(full_loader.dataset) > 0:
                y_pred_np, y_truth_np = preds_and_truth(model, full_loader, device=next(model.parameters()).device, as_numpy=True)
                fig = figure_compare_data(y_pred_np, y_truth_np, x_points, y_min, y_max, title_prefix="MLP[full]")
                writer.add_figure('qual/full_pred_truth', fig, epoch)
                import matplotlib.pyplot as _plt
                _plt.close(fig)

        # Evaluate and visualize
    test_results = evaluate_regression(model, test_loader, y_scaler)
    print(test_results)

    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
