import torch
import numpy as np
from torch import nn, autograd
from helper_functions import figure_compare_data

def d(outputs, inputs, retain_graph=True, create_graph=True):
    """First derivative helper that is safe for higher-order calls.

    If `outputs` does not require grad (can happen after vectorization or when
    a branch is constant), return a zeros-like tensor instead of calling
    autograd.grad, which would error.
    """
    if not isinstance(outputs, torch.Tensor) or not outputs.requires_grad:
        return torch.zeros_like(inputs)
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(inputs)
    return grad

def d_t(u, t):   return d(u, t)
def d_x(u, x):   return d(u, x)
def d_xx(u, x):  return d(d_x(u, x), x)
def d_xxx(u, x): return d(d_xx(u, x), x)

def pinn_kdv_loss(model,
                  # data points (used for supervised loss)
                  xb,
                  # BC points at x=0 and x=L for periodicity
                  t_bc, L, x0,
                  # IC ground truth, IC collocatino points at t=O
                  u0, x_ic, t0,
                  # scales for inputs and outputs for the PDE
                  x_scale, t_scale, u_scale,
                  # mean of u in original (unscaled) units for standardized outputs
                  u_mean,
                  # supervised targets at xb
                  yb,
                  # weights for each term
                  w_pde=1.0, w_bc=1.0, w_data=1.0, w_ic = 1.0,
                  #Number of time domain chunks
                  n_chunks: int = 16,
                  # optional separate collocation points for PDE residual
                  xb_r=None,
                  # optional conditioning features (e.g., IC vector), shape (F,) or (N,F)
                  normalize_terms: bool = True,
                  #If validation, compute all loss terms
                  validation: bool = False,
                ):
    """
    Returns total_loss, dict_of_terms. All inputs are 1D tensors of same length per group.
    L is domain length in x for periodic BCs.
    """
    # If physics weights are zero, short-circuit to pure data loss
    if w_pde == 0.0 and w_bc == 0.0 and not validation:
        # Pure data path: ensure shapes match to avoid broadcasting
        pred = model(xb).squeeze(-1)
        target = yb.squeeze(-1)
        loss_data = torch.mean((pred - target)**2)
        total = w_data * loss_data
        return total, {
            "pde": 0.0,
            "bc":  0.0,
            "data": loss_data.item(),
            "total": total.item(),
            #"ut": 0.0,
            #"nl": 0.0,
            #"uxxx": 0.0,
        }
    
    # Choose collocation set for PDE residual
    colloc = xb_r if xb_r is not None else xb
    colloc = colloc.clone().detach().requires_grad_(True)

    #Requires gradient to be created for BC and IC points
    t_bc = t_bc.clone().detach().requires_grad_(True)
    x_ic = x_ic.clone().detach().requires_grad_(True)

    # ---- PDE residual on collocation points ----
    # model expects concat [x, t] -> u(t,x)
    pred_r = model(colloc).squeeze(-1)
    u_r = pred_r

    # First derivatives wrt both inputs in one call
    grads = d(pred_r, colloc)                 # shape (N, 2)
    ux, ut = grads[:, 0], grads[:, 1]

    # Higher-order x-derivatives via repeated grad wrt xb, select x column
    grads2 = d(ux, colloc)
    uxx = grads2[:, 0]
    grads3 = d(uxx, colloc)
    uxxx = grads3[:, 0]

    # Per-term residual normalization for optimization, but also compute raw residual for logging
    eps = torch.as_tensor(1e-12, dtype=u_r.dtype, device=u_r.device)
    c2s = (6.0 * u_scale * t_scale / x_scale)  # scales u' * u'_x term
    c2mu = (6.0 * u_mean * t_scale / x_scale)  # mean term mu * u'_x
    c3 = (t_scale / (x_scale**3))

    t1 = ut
    t2_base = (u_r * ux)
    t2_mu_base = ux
    t3 = uxxx

    # Raw (unweighted) residual for monitoring
    res_raw = t1 + c2s * t2_base + c2mu * t2_mu_base + c3 * t3
    pde_raw = torch.mean(res_raw.pow(2))

    if normalize_terms:
        # Normalized residual used for the optimization
        s1 = torch.sqrt((t1.pow(2)).mean() + eps)
        s2 = torch.sqrt((t2_base.pow(2)).mean() + eps)
        s2mu = torch.sqrt((t2_mu_base.pow(2)).mean() + eps)
        s3 = torch.sqrt((t3.pow(2)).mean() + eps)

        res = (t1 / s1) + (c2s * (t2_base / s2)) + (c2mu * (t2_mu_base / s2mu)) + (c3 * (t3 / s3))
        loss_pde = torch.mean(res**2)
    else:
        # Use raw residual directly
        loss_pde = torch.mean(res_raw**2)

    # ---- Periodic boundary conditions at x=0 and x=L ----
    # u(t,0) == u(t,L), ux(t,0) == ux(t,L), uxx(t,0) == uxx(t,L) (smooth periodicity)
    x0_val = float(x0.item())
    L_val  = float(L.item())
    x0v = torch.full_like(t_bc, fill_value=x0_val)
    xLv = torch.full_like(t_bc, fill_value=L_val)

    xb0 = torch.stack([x0v, t_bc], dim=1).clone().detach().requires_grad_(True)
    xbL = torch.stack([xLv, t_bc], dim=1).clone().detach().requires_grad_(True)

    u_0 = model(xb0).squeeze(-1)
    u_L = model(xbL).squeeze(-1)

    g0 = d(u_0, xb0)
    gL = d(u_L, xbL)
    ux_0 = g0[:, 0]
    ux_L = gL[:, 0]
    g0_2 = d(ux_0, xb0)
    gL_2 = d(ux_L, xbL)
    uxx_0 = g0_2[:, 0]
    uxx_L = gL_2[:, 0]

    loss_bc = torch.mean((u_0 - u_L)**2) + \
              torch.mean((ux_0 - ux_L)**2) + \
              torch.mean((uxx_0 - uxx_L)**2)
    
    # ---- Initial condition at t=t0 with interpolation from u0(x) ----
    # Build evaluation inputs at IC time
    t0_val = float(t0.item())
    t0v = torch.full_like(x_ic, fill_value=t0_val)
    xbIC = torch.stack([x_ic, t0v], dim=1).clone().detach().requires_grad_(True)

    # Interpolate ground-truth u0 at x_ic along standardized x-grid [x0, L]
    x_grid = torch.linspace(x0, L, steps=u0.numel(), device=xbIC.device, dtype=xbIC.dtype)
    def _interp1d_sorted(xg, yg, xq, eps: float = 1e-12):
        idx = torch.searchsorted(xg, xq, right=False)
        idx = idx.clamp(min=1, max=xg.numel()-1)
        x0i = xg[idx-1]; x1i = xg[idx]
        y0i = yg[idx-1]; y1i = yg[idx]
        w = (xq - x0i) / (x1i - x0i + eps)
        return y0i + w * (y1i - y0i)
    u0_interp = _interp1d_sorted(x_grid, u0.view(-1), x_ic)

    uIC = model(xbIC).squeeze(-1)
    loss_ic = torch.mean((uIC - u0_interp)**2)
    
    # Supervised data term (match shapes to avoid broadcasting)
    pred_d = model(xb).squeeze(-1)
    target = yb.squeeze(-1)
    loss_data = torch.mean((pred_d - target)**2)

    total = w_pde*loss_pde + w_bc*loss_bc + w_ic*loss_ic + w_data*loss_data
    return total, {
    # Log the unweighted PDE residual (raw), keep weighted for optimization
    "pde": pde_raw.item(),
    "pde_w": loss_pde.item(),
    "bc":  (loss_bc.item() + loss_ic.item()),
    "pure_bc":  loss_bc.item(),
    "ic":  loss_ic.item(),
    "data": loss_data.item(),
    "total": total.item(),
    #"ut": s1.item(),
    #"nl": s2.item(),
    #"uxxx": s3.item(),
}
def pinn_kdv_loss(model,
                  # data points (used for supervised loss)
                  xb,
                  # BC points at x=0 and x=L for periodicity
                  t_bc, L, x0,
                  # IC ground truth, IC collocatino points at t=O
                  u0, x_ic, t0,
                  # scales for inputs and outputs for the PDE
                  x_scale, t_scale, u_scale,
                  # mean of u in original (unscaled) units for standardized outputs
                  u_mean,
                  # supervised targets at xb
                  yb,
                  # weights for each term
                  w_pde=1.0, w_bc=1.0, w_data=1.0, w_ic = 1.0,
                  #Number of time domain chunks
                  n_chunks: int = 16,
                  # optional separate collocation points for PDE residual
                  xb_r=None,
                  # optional conditioning features (e.g., IC vector), shape (F,) or (N,F)
                  normalize_terms: bool = True,
                  #If validation, compute all loss terms
                  validation: bool = False,
                ):
    """
    Returns total_loss, dict_of_terms. All inputs are 1D tensors of same length per group.
    L is domain length in x for periodic BCs.
    """
    # If physics weights are zero, short-circuit to pure data loss
    if w_pde == 0.0 and w_bc == 0.0 and not validation:
        # Pure data path: ensure shapes match to avoid broadcasting
        pred = model(xb).squeeze(-1)
        target = yb.squeeze(-1)
        loss_data = torch.mean((pred - target)**2)
        total = w_data * loss_data
        return total, {
            "pde": 0.0,
            "bc":  0.0,
            "data": loss_data.item(),
            "total": total.item(),
            #"ut": 0.0,
            #"nl": 0.0,
            #"uxxx": 0.0,
        }
    
    # Choose collocation set for PDE residual
    colloc = xb_r if xb_r is not None else xb
    colloc = colloc.clone().detach().requires_grad_(True)

    #Requires gradient to be created for BC and IC points
    t_bc = t_bc.clone().detach().requires_grad_(True)
    x_ic = x_ic.clone().detach().requires_grad_(True)

    # ---- PDE residual on collocation points ----
    # model expects concat [x, t] -> u(t,x)
    pred_r = model(colloc).squeeze(-1)
    u_r = pred_r

    # First derivatives wrt both inputs in one call
    grads = d(pred_r, colloc)                 # shape (N, 2)
    ux, ut = grads[:, 0], grads[:, 1]

    # Higher-order x-derivatives via repeated grad wrt xb, select x column
    grads2 = d(ux, colloc)
    uxx = grads2[:, 0]
    grads3 = d(uxx, colloc)
    uxxx = grads3[:, 0]

    # Per-term residual normalization for optimization, but also compute raw residual for logging
    eps = torch.as_tensor(1e-12, dtype=u_r.dtype, device=u_r.device)
    c2s = (6.0 * u_scale * t_scale / x_scale)  # scales u' * u'_x term
    c2mu = (6.0 * u_mean * t_scale / x_scale)  # mean term mu * u'_x
    c3 = (t_scale / (x_scale**3))

    t1 = ut
    t2_base = (u_r * ux)
    t2_mu_base = ux
    t3 = uxxx

    # Raw (unweighted) residual for monitoring
    res_raw = t1 + c2s * t2_base + c2mu * t2_mu_base + c3 * t3
    pde_raw = torch.mean(res_raw.pow(2))

    if normalize_terms:
        # Normalized residual used for the optimization
        s1 = torch.sqrt((t1.pow(2)).mean() + eps)
        s2 = torch.sqrt((t2_base.pow(2)).mean() + eps)
        s2mu = torch.sqrt((t2_mu_base.pow(2)).mean() + eps)
        s3 = torch.sqrt((t3.pow(2)).mean() + eps)

        res = (t1 / s1) + (c2s * (t2_base / s2)) + (c2mu * (t2_mu_base / s2mu)) + (c3 * (t3 / s3))
        loss_pde = torch.mean(res**2)
    else:
        # Use raw residual directly
        loss_pde = torch.mean(res_raw**2)

    # ---- Periodic boundary conditions at x=0 and x=L ----
    # u(t,0) == u(t,L), ux(t,0) == ux(t,L), uxx(t,0) == uxx(t,L) (smooth periodicity)
    x0_val = float(x0.item())
    L_val  = float(L.item())
    x0v = torch.full_like(t_bc, fill_value=x0_val)
    xLv = torch.full_like(t_bc, fill_value=L_val)

    xb0 = torch.stack([x0v, t_bc], dim=1).clone().detach().requires_grad_(True)
    xbL = torch.stack([xLv, t_bc], dim=1).clone().detach().requires_grad_(True)

    u_0 = model(xb0).squeeze(-1)
    u_L = model(xbL).squeeze(-1)

    g0 = d(u_0, xb0)
    gL = d(u_L, xbL)
    ux_0 = g0[:, 0]
    ux_L = gL[:, 0]
    g0_2 = d(ux_0, xb0)
    gL_2 = d(ux_L, xbL)
    uxx_0 = g0_2[:, 0]
    uxx_L = gL_2[:, 0]

    loss_bc = torch.mean((u_0 - u_L)**2) + \
              torch.mean((ux_0 - ux_L)**2) + \
              torch.mean((uxx_0 - uxx_L)**2)
    
    # ---- Initial condition at t=t0 with interpolation from u0(x) ----
    # Build evaluation inputs at IC time
    t0_val = float(t0.item())
    t0v = torch.full_like(x_ic, fill_value=t0_val)
    xbIC = torch.stack([x_ic, t0v], dim=1).clone().detach().requires_grad_(True)

    # Interpolate ground-truth u0 at x_ic along standardized x-grid [x0, L]
    x_grid = torch.linspace(x0, L, steps=u0.numel(), device=xbIC.device, dtype=xbIC.dtype)
    def _interp1d_sorted(xg, yg, xq, eps: float = 1e-12):
        idx = torch.searchsorted(xg, xq, right=False)
        idx = idx.clamp(min=1, max=xg.numel()-1)
        x0i = xg[idx-1]; x1i = xg[idx]
        y0i = yg[idx-1]; y1i = yg[idx]
        w = (xq - x0i) / (x1i - x0i + eps)
        return y0i + w * (y1i - y0i)
    u0_interp = _interp1d_sorted(x_grid, u0.view(-1), x_ic)

    uIC = model(xbIC).squeeze(-1)
    loss_ic = torch.mean((uIC - u0_interp)**2)
    
    # Supervised data term (match shapes to avoid broadcasting)
    pred_d = model(xb).squeeze(-1)
    target = yb.squeeze(-1)
    loss_data = torch.mean((pred_d - target)**2)

    total = w_pde*loss_pde + w_bc*loss_bc + w_ic*loss_ic + w_data*loss_data
    return total, {
    # Log the unweighted PDE residual (raw), keep weighted for optimization
    "pde": pde_raw.item(),
    "pde_w": loss_pde.item(),
    "bc":  (loss_bc.item() + loss_ic.item()),
    "pure_bc":  loss_bc.item(),
    "ic":  loss_ic.item(),
    "data": loss_data.item(),
    "total": total.item(),
    #"ut": s1.item(),
    #"nl": s2.item(),
    #"uxxx": s3.item(),
    }

def residual_kdv_loss(model,
                  #Maximum time needed for causal training and chunking, assumed t0 = 0
                  t_max,
                  #A list of spatial domain collocation points for use in each chunk
                  x_r,
                  # scales for inputs and outputs for the PDE
                  x_scale, t_scale, u_scale,
                  #Number of time domain chunks
                  n_chunks: int = 16,
                  #How much the model has to learn the early time stages before the latter
                  causal_weight: float = 1.0,
                ):
    """
    Returns total_loss, dict_of_terms. All inputs are 1D tensors of same length per group.
    L is domain length in x for periodic BCs.
    """
    
    # Vectorized evaluation over all chunks in one forward/backward pass
    x_r_det = x_r.clone().detach()  # (n_chunks, n_r)
    device = x_r_det.device
    dtype = x_r_det.dtype

    # Build per-chunk time samples uniformly within each interval [t0, t1]
    idx = torch.arange(int(n_chunks), device=device, dtype=dtype)
    t0 = (idx / float(n_chunks)) * (t_max if torch.is_tensor(t_max) else torch.tensor(float(t_max), device=device, dtype=dtype))
    t1 = ((idx + 1) / float(n_chunks)) * (t_max if torch.is_tensor(t_max) else torch.tensor(float(t_max), device=device, dtype=dtype))
    t_r = torch.rand_like(x_r_det) * (t1[:, None] - t0[:, None]) + t0[:, None]  # (n_chunks, n_r)

    # Stack inputs and run model once
    xb_all = torch.stack([x_r_det, t_r], dim=2).reshape(-1, 2).clone().detach().requires_grad_(True)
    pred_all = model(xb_all).squeeze(-1)

    # Derivatives (vectorized)
    grads = d(pred_all, xb_all)
    ux, ut = grads[:, 0], grads[:, 1]
    uxx = d(ux, xb_all)[:, 0]
    uxxx = d(uxx, xb_all)[:, 0]

    # Coefficients (scalars)
    c2s = (6.0 * u_scale * t_scale / x_scale)
    c3 = (t_scale / (x_scale**3))

    # Residual per sample, then mean per chunk
    res_raw = ut + c2s * (pred_all * ux) + c3 * uxxx
    res2 = res_raw.pow(2).view(int(n_chunks), -1).mean(dim=1)  # (n_chunks,)

    # Causal weighting applied sequentially but without extra model calls
    chunk_losses: list[torch.Tensor] = []
    residual_loss = xb_all.new_zeros(())
    wi_log = [0.0] * int(n_chunks)
    processed = 0
    for M in range(int(n_chunks)):
        prev = (torch.stack(chunk_losses).sum().detach() if chunk_losses else xb_all.new_zeros(())).detach()
        wi = torch.exp(-float(causal_weight) * prev).detach()
        if (M >= 4) and (float(wi) < 1e-3):
            break
        wi_log[M] = float(wi)
        loss_m = res2[M]
        residual_loss = residual_loss + wi * loss_m
        chunk_losses.append(loss_m.detach())
        processed += 1

    denom = float(processed if processed > 0 else 1)
    residual_loss = residual_loss / denom

    return residual_loss, {
        "residual": residual_loss.item(),
        "wi": wi_log,
        "kept": processed,
    }

def init_condition_loss(model,
        # IC ground truth, IC collocatin points at t=O
                  u0, x_ic, t0, x0, L
            ):
    
    #Ensures gradients are possible
    x_ic = x_ic.clone().detach().requires_grad_(True)

    # ---- Initial condition at t=t0 with interpolation from u0(x) ----
    # Build evaluation inputs at IC time
    t0_val = float(t0.item())
    t0v = torch.full_like(x_ic, fill_value=t0_val)
    xbIC = torch.stack([x_ic, t0v], dim=1).clone().detach().requires_grad_(True)

    # Interpolate ground-truth u0 at x_ic along standardized x-grid [x0, L]
    x_grid = torch.linspace(x0, L, steps=u0.numel(), device=xbIC.device, dtype=xbIC.dtype)
    def _interp1d_sorted(xg, yg, xq, eps: float = 1e-12):
        idx = torch.searchsorted(xg, xq, right=False)
        idx = idx.clamp(min=1, max=xg.numel()-1)
        x0i = xg[idx-1]; x1i = xg[idx]
        y0i = yg[idx-1]; y1i = yg[idx]
        w = (xq - x0i) / (x1i - x0i + eps)
        return y0i + w * (y1i - y0i)
    u0_interp = _interp1d_sorted(x_grid, u0.view(-1), x_ic)

    uIC = model(xbIC).squeeze(-1)
    ic_loss = torch.mean((uIC - u0_interp)**2)

    return ic_loss, {
        "ic": ic_loss.item()
    }

def periodic_bc_loss(model,
        t_bc, x0, L,
):
    
    t_bc = t_bc.clone().detach().requires_grad_(True)

    # ---- Periodic boundary conditions at x=0 and x=L ----
    # u(t,0) == u(t,L), ux(t,0) == ux(t,L), uxx(t,0) == uxx(t,L) (smooth periodicity)
    x0_val = float(x0.item())
    L_val  = float(L.item())
    x0v = torch.full_like(t_bc, fill_value=x0_val)
    xLv = torch.full_like(t_bc, fill_value=L_val)

    xb0 = torch.stack([x0v, t_bc], dim=1).clone().detach().requires_grad_(True)
    xbL = torch.stack([xLv, t_bc], dim=1).clone().detach().requires_grad_(True)

    u_0 = model(xb0).squeeze(-1)
    u_L = model(xbL).squeeze(-1)

    g0 = d(u_0, xb0)
    gL = d(u_L, xbL)
    ux_0 = g0[:, 0]
    ux_L = gL[:, 0]
    g0_2 = d(ux_0, xb0)
    gL_2 = d(ux_L, xbL)
    uxx_0 = g0_2[:, 0]
    uxx_L = gL_2[:, 0]

    bc_loss = torch.mean((u_0 - u_L)**2) + \
              torch.mean((ux_0 - ux_L)**2) + \
              torch.mean((uxx_0 - uxx_L)**2)
    
    return bc_loss, {
        "bc": bc_loss.item()
    }

def mse_data_loss(model, xb, yb):
    """MSE on provided data. Accepts numpy arrays and moves to model's device/dtype."""
    dev = next(model.parameters()).device
    dtp = next(model.parameters()).dtype
    if not isinstance(xb, torch.Tensor):
        xb = torch.as_tensor(xb, dtype=dtp, device=dev)
    else:
        xb = xb.to(device=dev, dtype=dtp)
    if not isinstance(yb, torch.Tensor):
        yb = torch.as_tensor(yb, dtype=dtp, device=dev)
    else:
        yb = yb.to(device=dev, dtype=dtp)

    pred = model(xb).squeeze(-1)
    target = yb.squeeze(-1)
    data_loss = torch.mean((pred - target)**2)

    return data_loss, {
        "data": data_loss.item(),
    }

def get_min_max(X_train, device, dtype=torch.float32):
    t_min = min(X_train[:, 1])
    t_max = max(X_train[:, 1])
    x0 = min(X_train[:, 0])
    L = max(X_train[:, 0])

    t_min = torch.tensor(t_min, dtype=dtype, device=device)
    t_max = torch.tensor(t_max, dtype=dtype, device=device)
    x0    = torch.tensor(x0,    dtype=dtype, device=device)
    L     = torch.tensor(L,     dtype=dtype, device=device)

    return t_min, t_max, x0, L

def old_weight_schedule(epoch,
                    warmup_epochs: int,
                    ramp_epochs: int,
                    w_data_start: float,
                    w_data_end: float,
                    w_pde_start: float,
                    w_pde_end: float,
                    w_bc_start: float,
                    w_bc_end: float):
    """
    Linear ramp schedule for loss weights.
    - Holds start values for `warmup_epochs`, then linearly interpolates over `ramp_epochs`.
    - Returns (w_data, w_pde, w_bc) for the given epoch.
    """
    if epoch < warmup_epochs:
        return w_data_start, w_pde_start, w_bc_start
    if epoch < warmup_epochs + ramp_epochs:
        a = (epoch - warmup_epochs) / float(ramp_epochs)
        wd = (1 - a) * w_data_start + a * w_data_end
        wp = (1 - a) * w_pde_start + a * w_pde_end
        wb = (1 - a) * w_bc_start + a * w_bc_end
        return wd, wp, wb
    return w_data_end, w_pde_end, w_bc_end

def lr_schedule(step: int,
                max_lr: float,
                decay_rate: float,
                warmup_steps: int,
                decay_steps: int):
    
    if step <= warmup_steps:
        return max_lr * step/warmup_steps
    return max_lr * (decay_rate)**((step-warmup_steps)/decay_steps)

class LrSchedule():
    def __init__(self,max_lr, decay_rate, warmup_steps, decay_steps):
        self.max_lr = max_lr; self.decay_rate = decay_rate; self.warmup_steps = warmup_steps; self.decay_steps = decay_steps

    def get_lr(self, step):
        if step <= self.warmup_steps:
            return self.max_lr * step/self.warmup_steps
        return self.max_lr * (self.decay_rate)**((step-self.warmup_steps)/self.decay_steps) 

def sample_collocation(n_r: int,
                       n_bc: int,
                       n_ic: int,
                       num_chunks: int,
                       x0: torch.Tensor,
                       L: torch.Tensor,
                       t_min: torch.Tensor,
                       t_max: torch.Tensor,
                       device: torch.device,
                       ):
    """
    Sample interior collocation points xb_r in [x0,L] x [t_min,t_max] and boundary-condition times t_bc.
    Returns:
      x_r: (n_chunks, n_r) tensor with random x values
      t_bc: (n_bc,) tensor of times for BC enforcement at x=0 and x=L
    """
    #Build (num_chunks, n_r) by stacking generated tensors; avoid torch.Tensor(list_of_tensors)
    x_r_list = [torch.rand(n_r, device=device, dtype=x0.dtype) * (L - x0) + x0 for _ in range(num_chunks)]
    x_r = torch.stack(x_r_list, dim=0)
    t_bc = torch.rand(n_bc, device=device, dtype=t_min.dtype) * (t_max - t_min) + t_min
    x_ic = torch.rand(n_ic, device=device, dtype=x0.dtype) * (L - x0) + x0
    return x_r, t_bc, x_ic

def train_step(model,
               xb: torch.Tensor,
               yb: torch.Tensor,
               x0: torch.Tensor,
               L: torch.Tensor,
               t_min: torch.Tensor,
               t_max: torch.Tensor,
               x_scale: torch.Tensor,
               t_scale: torch.Tensor,
               u_scale: torch.Tensor,
               u_mean: torch.Tensor,
               u0: torch.Tensor,
               w_pde: float,
               w_bc: float,
               w_data: float,
               optimizer: torch.optim.Optimizer,
               grad_clip_max_norm: float,
               n_r: int = 256,
               n_bc: int = 256,
               n_ic: int = 256,
               num_chunks: int = 16,
               causal_weight: float = 1.0):
    """One training step: sample collocation, compute loss, backprop, clip, step.

    Returns: (loss_value, terms_dict, grad_total_norm)
    """
    device = xb.device
    # Sample interior collocation points and BC times
    x_r, t_bc, x_ic = sample_collocation(
        n_r=n_r,
        n_bc=n_bc,
        n_ic=n_ic,
        num_chunks=num_chunks,
        x0=x0, L=L, t_min=t_min, t_max=t_max,
        device=device,
    )

    residual_loss, terms = residual_kdv_loss(model, t_max, x_r, x_scale, t_scale, u_scale, u_mean, num_chunks, causal_weight)
    bc_loss, terms = periodic_bc_loss(model, t_bc, x0, L)
    ic_loss, terms = init_condition_loss(model, u0, x_ic, t_min, x0, L)
    #data_loss, terms = mse_data_loss(model, xb, yb)



    optimizer.zero_grad()
    loss.backward()
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
    optimizer.step()

    try:
        gnorm = float(total_grad_norm)
    except Exception:
        gnorm = float('nan')

    return float(loss.item()), terms, gnorm

class TrainingStepper:
    def __init__(self, model, #The NN model
                 x0, L, t_min, t_max, u0, #Dataset spesific parameters
                 x_scale, t_scale, u_scale, #Scaler spesific parameters
                 optimizer, grad_clip_max_norm, #Optimizer spesific parameters
                 n_r, n_bc, n_ic, num_chunks, causal_weight, lambda_freq, #Training spesific parameters
                 log_every_n_steps, writer, x_points, lr_scheduler, alpha): #Logging spesific parameters
        self.model = model
        self.x0 = x0; self.L = L; self.t_min = t_min; self.t_max = t_max; self.u0 = u0
        self.x_scale = x_scale; self.t_scale = t_scale; self.u_scale = u_scale
        self.optimizer = optimizer; self.grad_clip_max_norm = grad_clip_max_norm
        self.n_r = n_r; self.n_bc = n_bc; self.n_ic = n_ic; self.num_chunks = num_chunks
        self.causal_weight = causal_weight; self.lamda_freq = lambda_freq
        self.log_every_n_steps = log_every_n_steps; self.writer = writer; self.x_points = x_points
        self.lr_scheduler = lr_scheduler; self.alpha = alpha

        self.device = next(model.parameters()).device
        self.steps = 0
        self.lam = {'ic': 1.0, 'bc': 1.0, 'res': 1.0}
        self.balancer = GradNormBalancer(self.model, self.lam.keys(), alpha = self.alpha)

        self.lr = 0.0

        self.xb = None; self.yb = None
        self.y_min = None; self.y_max = None

    def define_validation_data(self, xb, yb):
        self.xb = xb; self.yb = yb
        self.y_min = torch.min(self.yb); self.y_max = torch.max(self.yb)

    @torch.no_grad()
    def _validate(self):
        data_loss, terms = mse_data_loss(self.model, self.xb, self.yb)
        return data_loss

    def _log(self,
                      train: dict,
                      val_data_loss: float,
                      grad_norm_mean: float | None = None):
        self.writer.add_scalar('lr', self.lr, self.steps)
        self.writer.add_scalar('loss/train/total', train.get('total', float('nan')), self.steps)
        self.writer.add_scalar('loss/train/res', train.get('res', float('nan')), self.steps)
        self.writer.add_scalar('loss/train/ic', train.get('ic', float('nan')), self.steps)
        self.writer.add_scalar('loss/train/bc', train.get('bc', float('nan')), self.steps)
        self.writer.add_scalar('loss/val/data', val_data_loss, self.steps)

        if grad_norm_mean is not None:
            self.writer.add_scalar('grad/total_norm_mean', grad_norm_mean, self.steps)

        print(
        f"Epoch {self.steps+1:02d} "
        f"train(total={train.get('total', float('nan')):.3e}, res={train.get('res', float('nan')):.3e}, bc={train.get('bc', float('nan')):.3e}), ic={train.get('ic', float('nan')):.3e} "
        f"val(data={val_data_loss:.3e}) "
        f"[lr={self.lr:.3e}]"
            )

    def step(self):
        self.steps += 1

        #Updates learningrate
        for g in self.optimizer.param_groups:
            self.lr = self.lr_scheduler.get_lr(self.steps)
            g["lr"] = self.lr

        # Sample interior collocation points and IC, BC points
        x_r, t_bc, x_ic = sample_collocation(
            n_r=self.n_r,
            n_bc=self.n_bc,
            n_ic=self.n_ic,
            num_chunks=self.num_chunks,
            x0=self.x0, L=self.L, t_min=self.t_min, t_max=self.t_max,
            device=self.device,
        )

        #Calculate losses
        residual_loss, terms1 = residual_kdv_loss(self.model, self.t_max, x_r, self.x_scale, self.t_scale, self.u_scale, self.num_chunks, self.causal_weight)
        bc_loss, terms2 = periodic_bc_loss(self.model, t_bc, self.x0, self.L)
        ic_loss, terms3 = init_condition_loss(self.model, self.u0, x_ic, self.t_min, self.x0, self.L)
        # Freeze λ weights wrt θ
        lam_res = self.lam['res'].detach() if torch.is_tensor(self.lam['res']) else torch.tensor(float(self.lam['res']), device=self.device)
        lam_bc  = self.lam['bc' ].detach() if torch.is_tensor(self.lam['bc' ]) else torch.tensor(float(self.lam['bc' ]), device=self.device)
        lam_ic  = self.lam['ic' ].detach() if torch.is_tensor(self.lam['ic' ]) else torch.tensor(float(self.lam['ic' ]), device=self.device)
        loss = residual_loss * lam_res + bc_loss * lam_bc + ic_loss * lam_ic
        
        #Updates loss weighting every lambda_freq steps
        if (self.steps % self.lamda_freq) == 0:
            self.lam = self.balancer.update({'ic':ic_loss, 'bc': bc_loss, 'res': residual_loss})

        #Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
        self.optimizer.step()

        #Logs if it is time for it
        if (self.steps % self.log_every_n_steps) == 0:
            # Log causal weights per chunk (detached floats)
            wi = terms1.get('wi') if isinstance(terms1, dict) else None
            if wi is not None:
                try:
                    kept = int(terms1.get('kept', 0))
                except Exception:
                    kept = 0
                for m, w in enumerate(wi):
                    try:
                        self.writer.add_scalar(f'causal/w_chunk_{m:02d}', float(w), self.steps)
                    except Exception:
                        pass
                # Compact bar figure
                try:
                    import numpy as _np, matplotlib.pyplot as _plt
                    _wi = _np.asarray(wi, dtype=_np.float32)
                    fig, ax = _plt.subplots(figsize=(6, 2))
                    ax.bar(_np.arange(_wi.size), _wi, width=0.9)
                    ax.set_ylim(0.0, 1.0)
                    ax.set_xlabel('chunk')
                    ax.set_ylabel('w')
                    self.writer.add_figure('causal/weights_bar', fig, self.steps)
                    _plt.close(fig)
                except Exception:
                    pass
                # Summary counters
                try:
                    self.writer.add_scalar('causal/kept_count', kept, self.steps)
                    self.writer.add_scalar('causal/sum_w', float(sum(float(x) for x in wi)), self.steps)
                except Exception:
                    pass

            self._log({'total': loss.item(), 'res': residual_loss.item(), 'ic': ic_loss.item(), 'bc': bc_loss.item()},
                      self._validate().item(), total_grad_norm.item())

    def log_heatmap(self):
        self.model.eval()
        # Visualize predictions vs truth over the ENTIRE dataset (train+val+test)
        xb = self.xb.to(self.device)
        with torch.no_grad():
            preds = self.model(xb).detach().cpu().numpy()
        truth = self.yb.detach().cpu().numpy()
        fig = figure_compare_data(preds, truth, self.x_points, self.y_min, self.y_max, title_prefix="All[train+val+test]")
        self.writer.add_figure('qual/full_pred_truth', fig, self.steps)
        import matplotlib.pyplot as _plt
        _plt.close(fig)


class GradNormBalancer:
    """
    Maintains global weights for multiple loss terms based on gradient norms.
    Updates every `freq` steps using EMA with coefficient `alpha`.
    """
    def __init__(self, model: nn.Module, names=("ic","bc","res"), alpha=0.9, eps=1e-12, device=None):
        self.model = model
        self.names = names
        self.alpha = alpha
        self.eps = eps
        self.device = device or next(model.parameters()).device
        # start with ones
        self.lam = {n: torch.tensor(1.0, device=self.device) for n in names}

    @torch.no_grad()
    def _ema_update(self, lam_hat):
        for n in self.names:
            self.lam[n] = self.alpha * self.lam[n] + (1 - self.alpha) * lam_hat[n]

    def _grad_norm(self, loss):
        """L2 norm of d(loss)/d(theta) over all trainable parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
        sq = 0.0
        for g in grads:
            if g is not None:
                sq = sq + g.pow(2).sum()
        return torch.sqrt(sq + self.eps)

    def update(self, losses: dict):
        """
        losses: dict with keys matching `names`, values are scalar tensors.
        Returns current lambda (possibly updated this step) as a dict.
        """
        # compute gradient norms
        norms = {n: self._grad_norm(losses[n]) for n in self.names}
        total = sum(norms.values())

        lam_hat = {n: (total / (norms[n] + self.eps)).detach() for n in self.names}

        self._ema_update(lam_hat)   # λ ← α λ + (1-α) λ̂

        return {n: self.lam[n].detach() for n in self.names}

def run_lbfgs_finisher(model: torch.nn.Module,
                       train_loader,
                       x0: torch.Tensor,
                       L: torch.Tensor,
                       t_min: torch.Tensor,
                       t_max: torch.Tensor,
                       x_scale: torch.Tensor,
                       t_scale: torch.Tensor,
                       u_scale: torch.Tensor,
                       u_mean: torch.Tensor,
                       u0: torch.Tensor,
                       w_data: float,
                       w_pde: float,
                       w_bc: float,
                       n_r_factor: int = 4,
                       n_bc: int = 64,
                       n_ic: int = 64,
                       lr: float = 1.0,
                       max_iter: int = 300,
                       history_size: int = 10,
                       feat: torch.Tensor | None = None,
                       use_lhs: bool = False,
                       normalize_terms: bool = True):
    """Run an LBFGS pass over the training loader; returns list of batch losses."""
    device = next(model.parameters()).device
    model.train()
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn='strong_wolfe'
    )

    batch_losses = []
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        xb_r, t_bc, x_ic = sample_collocation(
        n_r=int(n_r_factor * xb.size(0)),
        n_bc=n_bc,
        n_ic=n_ic,
        x0=x0, L=L, t_min=t_min, t_max=t_max,
        device=device,
        use_lhs=use_lhs,
    )

        def closure():
            lbfgs.zero_grad()
            loss, _ = pinn_kdv_loss(
                model, xb, t_bc, L, x0, u0, x_ic, t_min, x_scale, t_scale, u_scale, u_mean, yb,
                w_pde, w_bc, w_data, xb_r=xb_r, feat=feat, normalize_terms=normalize_terms
            )

            loss.backward()
            return loss

        loss_val = lbfgs.step(closure)
        try:
            batch_losses.append(float(loss_val.detach().cpu()))
        except Exception:
            pass

    return batch_losses

def pirate_init(model, x_points, t_points, x0, L, t_min, u0, t_max):
        
        # Get device/dtype robustly from model parameters
        dev = next(model.parameters()).device
        dtp = next(model.parameters()).dtype

        x_grid = torch.linspace(x0, L, steps=x_points, device=dev, dtype=dtp)
        t_grid = torch.linspace(t_min, t_max, steps=t_points, device=dev, dtype=dtp)
        
        # Cartesian product (x_ic, t_grid)
        x_rep = x_grid.repeat_interleave(t_points)
        t_rep = t_grid.repeat(x_grid.numel())
        xbIC = torch.stack([x_rep, t_rep], dim=1).clone().detach().requires_grad_(True)

        # Interpolate ground-truth u0 at x_ic along standardized x-grid [x0, L]
        def _interp1d_sorted(xg, yg, xq, eps: float = 1e-12):
            idx = torch.searchsorted(xg, xq, right=False)
            idx = idx.clamp(min=1, max=xg.numel()-1)
            x0i = xg[idx-1]; x1i = xg[idx]
            y0i = yg[idx-1]; y1i = yg[idx]
            w = (xq - x0i) / (x1i - x0i + eps)
            return y0i + w * (y1i - y0i)
        u0_interp = _interp1d_sorted(x_grid, u0.to(device=dev, dtype=dtp).view(-1), x_grid)
        # Repeat u0 for each sampled time at the same x
        y_ic = u0_interp.repeat(t_points)

        model.physics_init(xbIC, y_ic)

def pirate_log_epoch_scalars(writer,
                      epoch: int,
                      train: dict,
                      val: dict,
                      weights: tuple,
                      lr: float,
                      blocks: int,
                      alphas, 
                      grad_norm_mean: float | None = None):
    w_data, w_pde, w_bc = weights
    writer.add_scalar('lr', lr, epoch)
    writer.add_scalar('weights/w_data', w_data, epoch)
    writer.add_scalar('weights/w_pde', w_pde, epoch)
    writer.add_scalar('weights/w_bc', w_bc, epoch)
    writer.add_scalar('loss/train/total', train.get('total', float('nan')), epoch)
    writer.add_scalar('loss/train/data', train.get('data', float('nan')), epoch)
    writer.add_scalar('loss/train/pde', train.get('pde', float('nan')), epoch)
    writer.add_scalar('loss/train/bc', train.get('bc', float('nan')), epoch)
    writer.add_scalar('loss/val/total', val.get('total', float('nan')), epoch)
    writer.add_scalar('loss/val/data', val.get('data', float('nan')), epoch)
    writer.add_scalar('loss/val/pde', val.get('pde', float('nan')), epoch)
    writer.add_scalar('loss/val/bc', val.get('bc', float('nan')), epoch)
    # Log per-block alpha values if provided
    if alphas is not None:
        for i in range(blocks):
            try:
                a = float(alphas[i])
            except Exception:
                # fall back if alphas is a tensor/list with different indexing
                a = float(alphas[i].detach().cpu()) if hasattr(alphas[i], 'detach') else None
            if a is not None:
                writer.add_scalar(f'blocks/alpha_{i+1}', a, epoch)
    if grad_norm_mean is not None:
        writer.add_scalar('grad/total_norm_mean', grad_norm_mean, epoch)
