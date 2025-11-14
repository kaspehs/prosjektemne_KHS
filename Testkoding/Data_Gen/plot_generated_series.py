"""
Quick visualization utility for generated TD-model series.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_series(series_dir: Path):
    files = sorted(series_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {series_dir}")
    data = []
    for f in files:
        arr = np.load(f)
        data.append(
            {
                "path": f,
                "time": np.asarray(arr["a"]),
                "disp": np.asarray(arr["b"]),
                "force": np.asarray(arr["c"]),
            }
        )
    return data


def plot_series(series_data, columns: int = 3, save_path: Path | None = None):
    rows = int(np.ceil(len(series_data) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3 * rows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, entry in zip(axes, series_data):
        disp = entry["disp"]
        force = entry["force"]
        disp_scale = np.max(np.abs(disp))
        force_scale = np.max(np.abs(force))
        disp_scale = disp_scale if disp_scale > 0 else 1.0
        force_scale = force_scale if force_scale > 0 else 1.0
        ax.plot(entry["time"], disp / disp_scale, label=f"y / {disp_scale:.2e}")
        ax.plot(entry["time"], force / force_scale, label=f"F / {force_scale:.2e}")
        ax.set_title(entry["path"].name)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("normalized units")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for ax in axes[len(series_data) :]:
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot generated TD-model time series.")
    parser.add_argument(
        "--series-dir",
        type=Path,
        default=Path(__file__).parent / "generated_series",
        help="Directory containing *.npz series files.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of subplot columns.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the combined figure instead of showing it.",
    )
    args = parser.parse_args()

    series = load_series(args.series_dir)
    plot_series(series, columns=args.columns, save_path=args.save)


if __name__ == "__main__":
    main()
