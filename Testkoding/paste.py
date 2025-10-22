# --- setup once ---
device = t_r_vec.device
dtype  = t_r_vec.dtype
t0     = float(t_r_vec.min())
T      = float(t_r_vec.max())
n_qp   = 12              # good default for smooth ODEs
n_chunks = 20
refresh_every = 10       # epochs between re-partitions
alpha_dir = 6.0          # variability of chunk sizes
min_frac, max_frac = 0.6, 1.4

# Precompute Gauss-Legendre nodes/weights on [-1,1] as tensors
xi_np, w_np = torch.polynomial.legendre.leggauss(n_qp)
xi = torch.tensor(xi_np, device=device, dtype=dtype)
w  = torch.tensor(w_np,  device=device, dtype=dtype)

# initialize edges
edges, _ = dirichlet_edges_bounded(t0, T, n_chunks, alpha_dir, min_frac, max_frac,
                                   device=device, dtype=dtype, seed=None)
t_a, t_b, c, J, t_qp, W = quadrature_from_edges(edges, xi, w)  # W already includes J

for epoch in range(num_epochs):

    # --- refresh partition occasionally ---
    if epoch % refresh_every == 0 and epoch > 0:
        edges, _ = dirichlet_edges_bounded(t0, T, n_chunks, alpha_dir, min_frac, max_frac,
                                           device=device, dtype=dtype, seed=None)
        t_a, t_b, c, J, t_qp, W = quadrature_from_edges(edges, xi, w)

    # --- vPINN residuals on current quadrature nodes ---
    # flatten qp times and require grad for autograd of first derivatives
    t_qp_req = t_qp.reshape(-1, 1).requires_grad_(True)

    pred = model(t_qp_req)              # (n_chunks*n_qp, 2)
    y = pred[:, 0:1]; q = pred[:, 1:2]

    # first derivatives in physical time (reuse your t_scale)
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

        phi_e     = phi.unsqueeze(-1)
        dphi_dt_e = dphi_dt.unsqueeze(-1)

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
        Rq_m = (integrand_q.squeeze(-1) * W).sum(dim=1)

        Ry_modes.append(Ry_m / y_scale)
        Rq_modes.append(Rq_m / q_scale)

    Ry = torch.stack(Ry_modes, dim=1)   # (n_chunks, n_test)
    Rq = torch.stack(Rq_modes, dim=1)

    loss_res = (Ry**2 + Rq**2).mean()

    # --- (optional) causal / interface loss with current edges ---
    lambda_if = 1.0
    with torch.enable_grad():
        loss_if = 0.0
        for k in range(1, n_chunks):
            y_end_prev = model(t_b[k-1].view(1,1))
            y_start_cur = model(t_a[k].view(1,1))
            loss_if = loss_if + (y_end_prev - y_start_cur).pow(2).mean()
            # extend to q, and/or first derivatives if needed
    loss = loss_res + lambda_if * loss_if + loss_IC_BC_data  # add your IC/BC/data terms

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
