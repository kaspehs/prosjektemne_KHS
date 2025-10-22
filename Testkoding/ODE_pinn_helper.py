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

def residual_viv_loss(model, device, dtype,
                  # Time range for causal chunking in standardized coords
                  t_min,t_max,
                  # scales for inputs and outputs for the PDE
                  t_scale,
                  #ODE parameters
                  ODE_params,
                  #Number of time domain chunks
                  n_chunks: int = 16,
                  n_per_chunk: int = 128,
                  #How much the model has to learn the early time stages before the latter
                  causal_weight: float = 1.0,
                ):
    """
    Returns total_loss, dict_of_terms. All inputs are 1D tensors of same length per group.
    L is domain length in x for periodic BCs.
    """
    # Build per-chunk time samples uniformly within each interval [t0, t1] over [t_min, t_max]
    idx = torch.arange(int(n_chunks), device=device, dtype=dtype)
    Lt = (t_max - t_min) if torch.is_tensor(t_max) else torch.tensor(float(t_max - t_min), device=device, dtype=dtype)
    t0 = t_min + (idx / float(n_chunks)) * Lt
    t1 = t_min + ((idx + 1) / float(n_chunks)) * Lt
    rand_unit = torch.rand(int(n_chunks), int(n_per_chunk), device=device, dtype=dtype)
    dt = (t1 - t0).unsqueeze(1)
    t_r = t0.unsqueeze(1) + rand_unit * dt
    t_r_vec = t_r.reshape(-1, 1).clone().detach().requires_grad_(True)

    # Stack inputs and run model once
    pred_all = model(t_r_vec)
    y = pred_all[:, 0:1]
    q = pred_all[:, 1:2] 

    # Derivatives (vectorized)
    y_t_hat = d(y, t_r_vec)
    y_tt_hat = d(y_t_hat, t_r_vec)
    q_t_hat = d(q, t_r_vec)
    q_tt_hat = d(q_t_hat, t_r_vec)

    yt = y_t_hat / t_scale
    ytt = y_tt_hat / (t_scale ** 2)
    qt = q_t_hat / t_scale
    qtt = q_tt_hat / (t_scale ** 2)

    #Unpacking ODE parameters for cleaner, more visual code
    my = ODE_params['my']
    cy = ODE_params['cy']
    ky = ODE_params['ky']
    k3y = ODE_params['k3y']
    Kl = ODE_params['Kl']
    cq = ODE_params['cq']
    kq = ODE_params['kq']
    Kc = ODE_params['Kc']
    y_scale = ODE_params['y_scale']
    q_scale = ODE_params['q_scale']


    y_res = my * ytt + cy * yt + ky * y + k3y * y**3# - Kl * q
    q_res = qtt + cq * (q**2-1.0) * qt + kq * q# - Kc * ytt

    #Scaling the residuals
    y_res = y_res/y_scale
    q_res = q_res/q_scale

    res = torch.stack([y_res, q_res], dim=-1)  # (n_chunks*n_per_chunk, 2)
    res2 = res.pow(2).view(n_chunks, n_per_chunk, 2).mean(dim=(1, 2))

    # Causal weighting applied sequentially but without extra model calls
    chunk_losses: list[torch.Tensor] = []
    residual_loss = t_r.new_zeros(())
    wi_log = [0.0] * int(n_chunks)
    processed = 0
    for M in range(int(n_chunks)):
        prev = (torch.stack(chunk_losses).sum().detach() if chunk_losses else t_r.new_zeros(())).detach()
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

def residual_viv_weak_loss(model, device, dtype,
                  # Time range for causal chunking in standardized coords
                  t_qp, c, J, W,
                  # scales for inputs and outputs for the PDE
                  t_scale,
                  #ODE parameters
                  ODE_params,
                  #Number of time domain chunks
                  edges, lengths,
                  n_chunks,
                  n_per_chunk: int = 128,
                  #How much the model has to learn the early time stages before the latter
                  causal_weight: float = 1.0,
                ):
  
    n_qp = n_per_chunk

    t_qp_req = t_qp.reshape(-1, 1).requires_grad_(True)
    pred = model(t_qp_req)
    y = pred[:, 0:1]; q = pred[:, 1:2]
    y_tau = d(y, t_qp_req); q_tau = d(q, t_qp_req)
    yt = y_tau / t_scale; qt = q_tau / t_scale

    # reshape back to (n_chunks, n_qp, 1)
    y_qp  = y.view(n_chunks, n_qp, 1)
    q_qp  = q.view(n_chunks, n_qp, 1)
    yt_qp = yt.view(n_chunks, n_qp, 1)
    qt_qp = qt.view(n_chunks, n_qp, 1)

    # --- build Petrov-Galerkin test functions per chunk ---
    # map qp times to local ξ in [-1,1] with current edges
    xi_qp = (t_qp - c[:, None]) / J[:, None]  # (n_chunks, n_qp)
    n_test = 3
    Ry_modes, Rq_modes = [], []

    for m in range(1, n_test+1):
        phi = torch.sin(m*torch.pi * (xi_qp + 1.0) * 0.5)               # (n_chunks, n_qp)
        dphi_dxi = (m*torch.pi*0.5) * torch.cos(m*torch.pi * (xi_qp + 1.0) * 0.5)
        dphi_dt = dphi_dxi / J[:, None]                                  # (n_chunks, n_qp)

        phi_e     = phi.unsqueeze(-1)          # (n_chunks, n_qp, 1)
        dphi_dt_e = dphi_dt.unsqueeze(-1)         # (n_chunks, n_qp, 1)

        # unpack ODE params (your dict)
        my  = ODE_params['my'];  cy  = ODE_params['cy'];  ky  = ODE_params['ky']
        k3y = ODE_params['k3y']; Kl  = ODE_params['Kl']
        cq  = ODE_params['cq'];  kq  = ODE_params['kq'];  Kc  = ODE_params['Kc']
        y_scale = ODE_params['y_scale']; q_scale = ODE_params['q_scale']

        # weak integrands (no second derivatives)
        integrand_y = (-my * dphi_dt_e * yt_qp
                       + cy * phi_e * yt_qp
                       + ky * phi_e * y_qp
                       + k3y * phi_e * (y_qp**3)
                       - Kl * phi_e * q_qp)

        integrand_q = (-dphi_dt_e * qt_qp
                       + cq * phi_e * ((q_qp**2) - 1.0) * qt_qp
                       + kq * phi_e * q_qp
                       + Kc * dphi_dt_e * yt_qp)

        # quadrature sum over qp: sum_j w_j * integrand(t_j) * J  (W already has J)
        Ry_m = (integrand_y.squeeze(-1) * W).sum(dim=1)  # (n_chunks,)
        Rq_m = (integrand_q.squeeze(-1) * W).sum(dim=1)  # (n_chunks,)

        Ry_modes.append(Ry_m / y_scale)
        Rq_modes.append(Rq_m / q_scale)

    Ry = torch.stack(Ry_modes, dim=1)   # (n_chunks, n_test)
    Rq = torch.stack(Rq_modes, dim=1)

    if causal_weight > 0.0:
        per_chunk_loss = torch.mean(Ry**2, dim = 1) + torch.mean(Rq**2, dim = 1) #Mean over square of test function errors
        # Causal weighting applied sequentially but without extra model calls
        chunk_losses: list[torch.Tensor] = []
        residual_loss = Ry.new_zeros(())
        wi_log = [0.0] * int(n_chunks)
        processed = 0
        for M in range(int(n_chunks)):
            prev = (torch.stack(chunk_losses).sum().detach() if chunk_losses else Ry.new_zeros(())).detach()
            wi = torch.exp(-float(causal_weight) * prev).detach()
            wi_log[M] = float(wi)
            loss_m = per_chunk_loss[M]
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

    else:
        residual_loss = (Ry**2 + Rq**2).mean()
        return residual_loss, {
            "residual": residual_loss.item(),
            "wi": None,
            "kept": None,
        }

def dirichlet_edges(t0, T, n_chunks, alpha=5.0,*, device, dtype, seed=None):
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None
    conc = torch.full((n_chunks,), float(alpha), device=device, dtype=dtype)
    p = torch.distributions.Dirichlet(conc).sample(generator=g)           # (n_chunks,)
    lengths = p * (T - t0)                                                # positive, sums to T-t0
    edges = torch.cat([torch.tensor([t0], device=device, dtype=dtype),
                       t0 + torch.cumsum(lengths, dim=0)])                 # (n_chunks+1,)
    return edges, lengths  # edges[0]=t0, edges[-1]=T

def dirichlet_edges_bounded(t0, T, n_chunks, alpha=5.0, min_frac=0.5, max_frac=1.5, *, device, dtype, seed=None):
    # Mean target
    mean_len = (T - t0) / n_chunks
    ell_min = min_frac * mean_len
    ell_max = max_frac * mean_len

    edges, lengths = dirichlet_edges(t0, T, n_chunks, alpha, device=device, dtype=dtype, seed=seed)

    # Project lengths to [ell_min, ell_max], then renormalize to sum to (T-t0)
    L = lengths.clone()
    L = torch.clamp(L, min=ell_min, max=ell_max)
    scale = (T - t0) / L.sum()
    L = L * scale

    edges = torch.cat([torch.tensor([t0], device=device, dtype=dtype),
                       t0 + torch.cumsum(L, dim=0)])
    return edges, L

def quadrature_from_edges(edges, xi, w):
    # edges: (n_chunks+1,) ; xi,w: Gauss-Legendre nodes/weights on [-1,1]
    t_a, t_b = edges[:-1], edges[1:]               # (n_chunks,)
    c = 0.5*(t_a + t_b)                            # centers
    J = 0.5*(t_b - t_a)                            # Jacobians (half-lengths)

    # broadcast nodes to each chunk: t = c + J*xi
    t_qp = c[:, None] + J[:, None] * xi[None, :]   # (n_chunks, n_qp)
    W = (w[None, :] * J[:, None])                  # scaled weights per chunk
    return t_a, t_b, c, J, t_qp, W

def init_condition_loss(model,
        # IC ground truth, IC collocatin points at t=O
                  u_ic, t0, t_scale
            ):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    #Ensures gradients are possible
    t0 = torch.as_tensor(t0, dtype=dtype, device=device).reshape(-1, 1)
    t0.requires_grad_(True)

    u_ic = torch.as_tensor(u_ic, dtype=dtype, device=device).reshape(1, -1)
    t_scale = torch.as_tensor(t_scale, dtype=dtype, device=device)

    #Running model and extracting y and q seperatly to compute derivatives
    yq = model(t0)                      # expect shape [N, 2]
    y, q = yq[:, [0]], yq[:, [1]]   #shapes [N, 1]

    # Derivatives (vectorized)
    y_t_hat = d(y, t0)
    q_t_hat = d(q, t0)
    yt = y_t_hat / t_scale #Shape [N, 1]
    qt = q_t_hat / t_scale #Shape [N, 1]

    pred_vec = torch.cat([y, q, yt, qt], dim=1) #Shape [1, 4]
    ic_loss = torch.mean((pred_vec - u_ic)**2) 
    return ic_loss, {"ic": float(ic_loss.detach())}

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

class LrSchedule():
    def __init__(self,max_lr, decay_rate, warmup_steps, decay_steps):
        self.max_lr = max_lr; self.decay_rate = decay_rate; self.warmup_steps = warmup_steps; self.decay_steps = decay_steps

    def get_lr(self, step):
        if step <= self.warmup_steps:
            return self.max_lr * step/self.warmup_steps
        return self.max_lr * (self.decay_rate)**((step-self.warmup_steps)/self.decay_steps) 

class ODETrainingStepper:
    def __init__(self, model, #The NN model
                t_min, t_max, u_ic, #Dataset spesific parameters
                ODE_params,
                t_scale, #Scaler spesific parameters
                optimizer, grad_clip_max_norm, #Optimizer spesific parameters
                n_per_chunk, num_chunks, causal_weight, lambda_freq, #Training spesific parameters
                log_every_n_steps, writer, lr_scheduler, alpha,
                diameter: float = 1.0,
                y_plot_limit: float = 1.0,
                q_plot_limit: float = 1.0, 
                vPINN = False): #Logging spesific parameters
        self.model = model
        self.t_min = t_min; self.t_max = t_max; self.t_scale = t_scale; self.u_ic = u_ic
        self.ODE_params = ODE_params
        self.optimizer = optimizer; self.grad_clip_max_norm = grad_clip_max_norm
        self.n_per_chunk = n_per_chunk; self.num_chunks = num_chunks
        self.causal_weight = causal_weight; self.lamda_freq = lambda_freq
        self.log_every_n_steps = log_every_n_steps; self.writer = writer
        self.lr_scheduler = lr_scheduler; self.alpha = alpha
        self.diameter = float(diameter) if diameter != 0 else 1.0
        self.y_plot_limit = float(y_plot_limit) if y_plot_limit > 0 else 1.0
        self.q_plot_limit = float(q_plot_limit) if q_plot_limit > 0 else 1.0
        self.vPINN = vPINN

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self.steps = 0
        self.lam = {'ic': 1.0, 'res': 1.0}
        self.balancer = GradNormBalancer(self.model, self.lam.keys(), alpha = self.alpha)

        self.lr = 0.0

        self.xb = None; self.yb = None
        self.y_min = None; self.y_max = None

        if vPINN:
            xi_np, w_np = torch.polynomial.legendre.leggauss(n_per_chunk)
            self.xi = torch.tensor(xi_np, device=self.device, dtype=self.dtype)
            self.w  = torch.tensor(w_np,  device=self.device, dtype=self.dtype)

    def define_validation_data(self, xb, yb):
        self.xb = xb; self.yb = yb
        self.y_min = torch.min(self.yb); self.y_max = torch.max(self.yb)

    @torch.no_grad()
    def _validate(self):
        data_loss, terms = mse_data_loss(self.model, self.xb, self.yb)
        return data_loss

    def _log(self,
                      train: dict,
                      #val_data_loss: float,
                      grad_norm_mean: float | None = None):
        self.writer.add_scalar('lr', self.lr, self.steps)
        self.writer.add_scalar('loss/train/total', train.get('total', float('nan')), self.steps)
        self.writer.add_scalar('loss/train/res', train.get('res', float('nan')), self.steps)
        self.writer.add_scalar('loss/train/ic', train.get('ic', float('nan')), self.steps)
        self.writer.add_scalar('loss/train/bc', train.get('bc', float('nan')), self.steps)
        #self.writer.add_scalar('loss/val/data', val_data_loss, self.steps)

        if grad_norm_mean is not None:
            self.writer.add_scalar('grad/total_norm_mean', grad_norm_mean, self.steps)

        # Log current lambda weights for losses (res, bc, ic)
        try:
            for k, v in self.lam.items():
                lam_f = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                self.writer.add_scalar(f'lambda/{k}', lam_f, self.steps)
        except Exception:
            pass

        # Log per-block alpha parameters if present (e.g., PirateNet residual blocks)
        try:
            blks = getattr(self.model, 'blocks', None)
            if blks is not None:
                for i, b in enumerate(blks, start=1):
                    a = getattr(b, 'alpha', None)
                    if a is not None:
                        try:
                            self.writer.add_scalar(f'blocks/alpha_{i}', float(a.detach().cpu()), self.steps)
                        except Exception:
                            pass
        except Exception:
            pass

        print(
        f"Epoch {self.steps+1:02d} "
        f"train(total={train.get('total', float('nan')):.3e}, res={train.get('res', float('nan')):.3e}, bc={train.get('bc', float('nan')):.3e}), ic={train.get('ic', float('nan')):.3e} "
        #f"val(data={val_data_loss:.3e}) "
        f"[lr={self.lr:.3e}]"
            )

    def step(self):
        self.steps += 1

        #Updates learningrate
        for g in self.optimizer.param_groups:
            self.lr = self.lr_scheduler.get_lr(self.steps)
            g["lr"] = self.lr

        if self.vPINN:
            if self.steps % 20 == 0:
                edges, _ = dirichlet_edges_bounded(self.t_min, self.t_max, self.num_chunks, device=self.device, dtype=self.dtype)
                t_a, t_b, c, J, t_qp, W = quadrature_from_edges(edges, self.xi, self.w)


        else:
            #Calculate losses
            residual_loss, terms1 = residual_viv_loss(self.model, self.device, self.dtype, self.t_min, self.t_max, self.t_scale, 
                                                    self.ODE_params,
                                                    self.num_chunks, self.n_per_chunk, self.causal_weight)
        #bc_loss, terms2 = periodic_bc_loss(self.model, t_bc, self.x0, self.L)
        bc_loss, terms2 = torch.tensor(0.0), None
        ic_loss, terms3 = init_condition_loss(self.model, self.u_ic, self.t_min, self.t_scale)
        # Freeze λ weights wrt θ
        lam_res = self.lam['res'].detach() if torch.is_tensor(self.lam['res']) else torch.tensor(float(self.lam['res']), device=self.device)
        lam_ic  = self.lam['ic' ].detach() if torch.is_tensor(self.lam['ic' ]) else torch.tensor(float(self.lam['ic' ]), device=self.device)
        loss = residual_loss * lam_res + ic_loss * lam_ic
        
        #Updates loss weighting every lambda_freq steps
        if (self.steps % self.lamda_freq) == 0:
            #self.lam = self.balancer.update({'ic':ic_loss, 'bc': bc_loss, 'res': residual_loss})
            self.lam = self.balancer.update({'ic':ic_loss, 'res': residual_loss})

        #Backpropagation
        self.optimizer.zero_grad(set_to_none=True)
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
                      total_grad_norm.item())

    def log_yq_curves(self,
                      steps: int = 512,
                      tag: str = "qual/yq_curves",
                      title: str | None = None) -> None:
        """Plot the predicted y(t) and q(t) across the standardized time range."""
        self.model.eval()

        # Build a dense time grid in the normalized domain
        t_vals = torch.linspace(self.t_min, self.t_max, steps=steps,
                                device=self.device, dtype=self.dtype)
        t_in = t_vals.unsqueeze(1)

        with torch.no_grad():
            preds = self.model(t_in).detach().cpu()

        y = preds[:, 0].numpy()
        q = preds[:, 1].numpy()
        t_np = t_vals.cpu().numpy()
        y_norm = y / self.diameter

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize

        fig, (ax_y, ax_q) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

        ax_y.plot(t_np, y_norm, label="y/D")
        ax_y.set_ylabel("y/D")
        ax_y.set_title(title or f"y/q prediction at step {self.steps}")
        ax_y.grid(True, alpha=0.3)
        ax_y.set_ylim(-self.y_plot_limit, self.y_plot_limit)

        ax_q.plot(t_np, q, label="q(t)", color="orange")
        ax_q.set_xlabel("t (normalized)")
        ax_q.set_ylabel("q")
        ax_q.grid(True, alpha=0.3)
        ax_q.set_ylim(-self.q_plot_limit, self.q_plot_limit)

        self.writer.add_figure(tag, fig, global_step=self.steps)
        plt.close(fig)

        # Phase portrait (q vs y/D)
        fig_phase, ax_phase = plt.subplots(figsize=(4, 4))
        points = np.column_stack((q, y_norm))
        segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)
        norm = Normalize(vmin=t_np[0], vmax=t_np[-1])
        lc = LineCollection(segments, cmap="viridis", norm=norm)
        lc.set_array(t_np[:-1])
        lc.set_linewidth(2.0)
        ax_phase.add_collection(lc)
        ax_phase.scatter(q[0], y_norm[0], color="blue", s=30, label="start")
        ax_phase.scatter(q[-1], y_norm[-1], color="red", s=30, label="end")
        cbar = fig_phase.colorbar(lc, ax=ax_phase, fraction=0.046, pad=0.04)
        cbar.set_label("t (normalized)")
        ax_phase.set_xlabel("q")
        ax_phase.set_ylabel("y/D")
        ax_phase.set_title("Phase plot")
        ax_phase.grid(True, alpha=0.3)
        ax_phase.set_xlim(-self.q_plot_limit, self.q_plot_limit)
        ax_phase.set_ylim(-self.y_plot_limit, self.y_plot_limit)
        self.writer.add_figure(f"{tag}_phase", fig_phase, global_step=self.steps)
        plt.close(fig_phase)

class GradNormBalancer:
    """
    Maintains global weights for multiple loss terms based on gradient norms.
    Updates every `freq` steps using EMA with coefficient `alpha`.
    """
    def __init__(self, model: nn.Module, names=("ic","bc","res"), alpha=0.9, eps=1e-8, device=None):
        self.model  = model
        self.names  = tuple(names)
        self.alpha  = alpha
        self.eps    = eps
        self.w_min  = 0.1
        self.w_max  = 10
        self.device = device or next(model.parameters()).device

        # EMA of grad norms; init to 1 for neutrality
        self.g_ema  = {n: torch.tensor(1.0, device=self.device) for n in self.names}
        # Public weights (mean ~ 1 at start)
        self.w      = {n: torch.tensor(1.0, device=self.device) for n in self.names}
        self.step   = 0
    
    def _global_grad_norm(self, loss):
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads  = torch.autograd.grad(loss, params, retain_graph=True,
                                     create_graph=False, allow_unused=True)
        sq = None
        for g in grads:
            if g is not None:
                # ignore non-finite entries (rare but safer)
                g = g.detach()
                if not torch.isfinite(g).all():
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                v = (g*g).sum()
                sq = v if sq is None else (sq + v)
        if sq is None:
            return torch.tensor(0.0, device=self.device)
        return torch.sqrt(sq + self.eps)

    @torch.no_grad()
    def update(self, losses: dict):
        # 1) measure pre-clip grad norms for each term
        g = {n: self._global_grad_norm(losses[n]) for n in self.names}

        # 2) EMA on norms
        for n in self.names:
            self.g_ema[n] = self.alpha*self.g_ema[n] + (1 - self.alpha)*torch.clamp(g[n], min=self.eps)

        # 3) inverse-proportional proposal
        inv = {n: 1.0 / torch.clamp(self.g_ema[n], min=self.eps) for n in self.names}

        # 4) normalize so mean weight = 1
        s = sum(inv.values())
        k = float(len(self.names))
        w_hat = {n: (inv[n] * (k / s)) for n in self.names}

        # 5) clamp to [w_min, w_max]
        for n in self.names:
            self.w[n] = torch.clamp(w_hat[n], self.w_min, self.w_max)
        
        return {n: self.w[n].detach() for n in self.names}

def physics_init(model, X, u0, ridge=1e-6, add_bias=True, return_diagnostics=False):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    X = torch.as_tensor(X, dtype=dtype, device=device)
    u0 = torch.as_tensor(u0, dtype=dtype, device=device)
    X = X.to(device=device, dtype=dtype)

    # Build y from u0 by repeating across time-major blocks,
    # or by 1D interpolation on x if shapes don't divide evenly.
    if u0 is None:
        raise ValueError("physics_init: either y or u0 must be provided")
    u0 = u0.to(device=device, dtype=dtype).view(-1)
    N = X.size(0)
    M = u0.numel()
    if M > 0 and (N % M) == 0:
        y = u0.repeat(N // M).unsqueeze(1)
    else:
        x_samples = X[:, 0]
        x_grid = torch.linspace(x_samples.min(), x_samples.max(), steps=M, device=device, dtype=dtype)
        idx = torch.searchsorted(x_grid, x_samples, right=False).clamp_(1, M - 1)
        xL = x_grid[idx - 1]; xR = x_grid[idx]
        yL = u0[idx - 1];     yR = u0[idx]
        w = (x_samples - xL) / (xR - xL + torch.finfo(dtype).eps)
        y = (yL + w * (yR - yL)).unsqueeze(1)

    H = model.features(X)  # (N, hidden) — should be linear features at t=0
    Phi = torch.cat([H, torch.ones(H.size(0), 1, device=device, dtype=dtype)], dim=1) if add_bias else H

    # Use normal equations with ridge (supported on MPS):
    A = Phi.T @ Phi
    if ridge and float(ridge) > 0.0:
        A = A + ridge * torch.eye(A.size(0), device=device, dtype=dtype)
    b = Phi.T @ y
    theta = torch.linalg.solve(A, b)  # (hidden[+1], out)

    # Load into head
    if add_bias:
        model.out.weight.data.copy_(theta[:-1, :].mT)
        model.out.bias.data.copy_(theta[-1, :])
    else:
        model.out.weight.data.copy_(theta.mT)
        torch.nn.init.zeros_(model.out.bias)

    if not return_diagnostics:
        return theta

    # Diagnostics
    y_hat = Phi @ theta                      # (N, out)
    err   = y_hat - y
    mse   = (err.pow(2).mean(dim=0))         # per-output
    rmse  = mse.sqrt()
    rel_l2 = (err.norm(dim=0) / (y.norm(dim=0).clamp_min(1e-12)))
    max_abs = err.abs().max(dim=0).values
    # Coeff magnitudes
    theta_l2 = theta.norm(dim=0)             # per-output
    theta_linf = theta.abs().max(dim=0).values
    # Conditioning
    # Approximate conditioning via eigenvalues of A = Phi^T Phi
    try:
        evals = torch.linalg.eigvalsh(A)
        cond = (evals.max() / evals.min().clamp_min(1e-15)).item()
    except Exception:
        cond = float('nan')

    return {
        "theta": theta,
        "rmse": rmse,                        # tensor of size (out,)
        "rel_l2": rel_l2,                    # tensor of size (out,)
        "max_abs": max_abs,                  # tensor of size (out,)
        "theta_l2": theta_l2,
        "theta_linf": theta_linf,
        "cond_Phi": cond,
        "y_hat_sample": y_hat[:10].detach(), # small peek
    }

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
