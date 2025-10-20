import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_kdv(npz_path: str, realization: int = 0):
    """
    Load KdV dataset from NPZ and reconstruct a structured grid.

    Expected arrays in NPZ:
      - 'g_u': shape (R, N) or (N,) if single realization. Flattened field values.
      - 'u'  : initial condition(s), shape (R, X) or (X,)
      - 'xt' : coordinates for each sample, shape (N, 2) with columns [x, t].

    Returns:
      x: (X,) sorted unique x-values
      t: (T,) sorted unique t-values
      U: (X, T) field for the selected realization
    """
    data = np.load(npz_path)
    g_u = data['g_u']
    xt = data['xt']  # (N, 2) -> [x, t]

    # Normalize shapes to include realizations dim
    if g_u.ndim == 1:
        g_u = g_u[None, :]

    if realization < 0 or realization >= g_u.shape[0]:
        raise IndexError(f"realization index {realization} out of range [0, {g_u.shape[0]-1}]")

    # Unique sorted coordinates and inverse indices
    x_vals, ix = np.unique(xt[:, 0], return_inverse=True)
    t_vals, it = np.unique(xt[:, 1], return_inverse=True)

    X = x_vals.shape[0]
    T = t_vals.shape[0]

    # Place samples into a dense grid U[x_idx, t_idx]
    U = np.full((X, T), np.nan, dtype=float)
    u_flat = np.asarray(g_u[realization]).reshape(-1)
    for k, val in enumerate(u_flat):
        U[ix[k], it[k]] = float(val)

    # Sanity fill if any NaNs (shouldn't happen with well-formed grids)
    if np.isnan(U).any():
        # Simple nearest-neighbor fill along time then x
        # Forward fill in time
        for j in range(T):
            mask = np.isnan(U[:, j])
            if mask.any():
                # attempt to fill from previous time
                if j > 0:
                    U[mask, j] = U[mask, j-1]
        # Backward fill
        for j in range(T-2, -1, -1):
            mask = np.isnan(U[:, j])
            if mask.any():
                U[mask, j] = U[mask, j+1]
        # Along x
        for i in range(X):
            row = U[i]
            if np.isnan(row).any():
                # fill with mean of non-nan
                nn = row[~np.isnan(row)]
                U[i, np.isnan(row)] = nn.mean() if nn.size else 0.0

    return x_vals, t_vals, U


def animate_line(x, t, U, interval_ms: int = 50, ylim_pad: float = 0.05):
    """Create a matplotlib animation of u(x, t) over x, sweeping through t."""
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(float(x.min()), float(x.max()))
    umin, umax = float(U.min()), float(U.max())
    pad = (umax - umin) * ylim_pad
    ax.set_ylim(umin - pad, umax + pad)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    title = ax.set_title('t = {:.3f}')

    def init():
        line.set_data([], [])
        title.set_text('')
        return line, title

    def update(frame):
        y = U[:, frame]
        line.set_data(x, y)
        title.set_text(f't = {t[frame]:.3f}  (frame {frame+1}/{len(t)})')
        return line, title

    anim = FuncAnimation(fig, update, frames=len(t), init_func=init,
                         interval=interval_ms, blit=True)
    return fig, anim


def main():
    parser = argparse.ArgumentParser(description='Animate KdV data: plot u(x, t) over x and animate across t')
    parser.add_argument('--data', type=str, default='data_generation/data/data_kdv.npz',
                        help='Path to data_kdv.npz')
    parser.add_argument('--realization', type=int, default=0, help='Realization index to visualize')
    parser.add_argument('--interval', type=int, default=40, help='Animation frame interval in ms')
    parser.add_argument('--save', type=str, default='', help='Optional output video path (e.g., out.mp4). If empty, just show.')
    args = parser.parse_args()

    x, t, U = load_kdv(args.data, realization=args.realization)
    fig, anim = animate_line(x, t, U, interval_ms=args.interval)

    if args.save:
        out_path = args.save
        ext = os.path.splitext(out_path)[1].lower()
        try:
            if ext in ('.mp4', '.m4v'):
                anim.save(out_path, writer='ffmpeg', dpi=120)
            elif ext in ('.gif',):
                anim.save(out_path, writer='pillow', dpi=120)
            else:
                # default to mp4
                anim.save(out_path, writer='ffmpeg', dpi=120)
            print(f'Saved animation to {out_path}')
        except Exception as e:
            print(f'Could not save animation ({e}). Showing interactively instead.')
            plt.show()
    else:
        plt.show()


if __name__ == '__main__':
    main()

