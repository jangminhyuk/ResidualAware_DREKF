#!/usr/bin/env python3
"""
plot_nonlinear.py

Load saved results from exp_ct_tracking_nonlinear (optimal_results_*_tag-omegaVar_scale_*.pkl)
and plot MSE of EKF vs DR-EKF vs omegaVar_scale with variance (std).
"""

import os
import re
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS_DIR = "./results/exp_ct_tracking_nonlinear"


# ──────────────────────────────────────────────────────────────────────
# Design tokens (shared with plot_trajectories_combined.py)
# ──────────────────────────────────────────────────────────────────────

_C = {
    "EKF":          "#C0392B",   # flat-UI red
    "DR_EKF_trace": "#2980B9",   # flat-UI blue
    "grid":         "#D5D9E0",   # cool light gray
    "panel_bg":     "#FAFAFA",   # neutral white
}


def _tint(hex_color, amount=0.58):
    """Blend hex_color toward white by `amount` (0 = original, 1 = white)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - (1 - r) * (1 - amount),
            1 - (1 - g) * (1 - amount),
            1 - (1 - b) * (1 - amount))


# ======================================================================
# Publication style setup
# ======================================================================

def _setup_style():
    """Configure matplotlib for publication-quality output (shared style)."""
    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        # Axes
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        # Ticks
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.top": True,
        "ytick.right": True,
        # Legend
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        "legend.handlelength": 1.8,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # Lines
        "lines.linewidth": 1.5,
    })


def discover_scale_tags(results_dir, dist):
    """Find all optimal_results_{dist}_tag-omegaVar_scale_{X.XX}.pkl and return sorted (scale, path)."""
    pattern = re.compile(rf"optimal_results_{re.escape(dist)}_tag-omegaVar_scale_(\d+\.\d+)\.pkl")
    out = []
    for name in os.listdir(results_dir):
        m = pattern.match(name)
        if m:
            scale = float(m.group(1))
            out.append((scale, os.path.join(results_dir, name)))
    return sorted(out, key=lambda x: x[0])


def load_summaries(results_dir, dist, scales=None):
    """
    Load optimal_results for each scale. If scales is given, use those; else discover from dir.
    Returns (scales, ekf_means, ekf_stds, dr_means, dr_stds).
    """
    if scales is None:
        scale_tags = discover_scale_tags(results_dir, dist)
        if not scale_tags:
            return [], [], [], [], []
        scales = [s for s, _ in scale_tags]
        paths = [p for _, p in scale_tags]
    else:
        paths = [
            os.path.join(results_dir, f"optimal_results_{dist}_tag-omegaVar_scale_{s:.2f}.pkl")
            for s in scales
        ]

    ekf_means, ekf_stds = [], []
    dr_means, dr_stds = [], []

    for scale, path in zip(scales, paths):
        if not os.path.exists(path):
            ekf_means.append(None)
            ekf_stds.append(None)
            dr_means.append(None)
            dr_stds.append(None)
            continue
        with open(path, "rb") as f:
            opt = pickle.load(f)
        ekf_m = opt.get("EKF", {}).get("mse_mean")
        ekf_s = opt.get("EKF", {}).get("mse_std")
        dr_m = opt.get("DR_EKF_trace", {}).get("mse_mean")
        dr_s = opt.get("DR_EKF_trace", {}).get("mse_std")
        ekf_means.append(ekf_m)
        ekf_stds.append(ekf_s)
        dr_means.append(dr_m)
        dr_stds.append(dr_s)

    return scales, ekf_means, ekf_stds, dr_means, dr_stds


def plot_mse_vs_omega_var_scale(scales, ekf_means, ekf_stds, dr_means, dr_stds, dist, out_dir=None):
    """MSE of EKF and DR-EKF vs initial omega_0 value, using omegaVar_scale from saved results."""
    out_dir = out_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    n = len(scales)
    ekf_means = [float(m) if m is not None else np.nan for m in (ekf_means or [])[:n]]
    dr_means = [float(m) if m is not None else np.nan for m in (dr_means or [])[:n]]
    # Use 1/5 of the standard deviation for visualization (tube-style bands)
    ekf_stds_use = [0.2 * float(s) if s is not None and s > 0 else 0.0 for s in (ekf_stds or [])[:n]]
    dr_stds_use = [0.2 * float(s) if s is not None and s > 0 else 0.0 for s in (dr_stds or [])[:n]]
    if len(ekf_means) < n:
        ekf_means = (ekf_means + [np.nan] * n)[:n]
    if len(dr_means) < n:
        dr_means = (dr_means + [np.nan] * n)[:n]
    if len(ekf_stds_use) < n:
        ekf_stds_use = (ekf_stds_use + [0.0] * n)[:n]
    if len(dr_stds_use) < n:
        dr_stds_use = (dr_stds_use + [0.0] * n)[:n]

    # Convert omegaVar_scale to actual initial omega_0 values (rad/s)
    base_omega0 = 0.10  # from exp_ct_tracking_nonlinear.py (x0_mean[4] = 0.10)
    omega_vals = [base_omega0 * s for s in scales]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(r"Initial angular velocity $\omega$ (rad/s)", fontsize=22)
    # Y label: no need to mention log scale explicitly
    ax.set_ylabel("Mean Square Error", fontsize=22)
    ax.set_yscale("log")
    # Use lighter, less distracting grid: major grid only
    ax.grid(True, alpha=0.25, which="major")

    has_ekf = any(not np.isnan(m) for m in ekf_means)
    has_dr = any(not np.isnan(m) for m in dr_means)

    # EKF: solid line + tube (mean ± std), no markers (deep red)
    if has_ekf:
        ekf_means_arr = np.array(ekf_means)
        ekf_stds_arr = np.array(ekf_stds_use)
        # Use a slightly darker, richer red
        ekf_color = "#b22222"  # firebrick
        ax.plot(omega_vals, ekf_means_arr, color=ekf_color, linestyle="-", linewidth=3, label="EKF")
        ax.fill_between(
            omega_vals,
            ekf_means_arr - ekf_stds_arr,
            ekf_means_arr + ekf_stds_arr,
            color=ekf_color,
            alpha=0.25,
        )

    # DR-EKF: solid line + tube (mean ± std), no markers (deep blue)
    if has_dr:
        dr_means_arr = np.array(dr_means)
        dr_stds_arr = np.array(dr_stds_use)
        # Use a deeper blue
        dr_color = "#0055aa"
        ax.plot(omega_vals, dr_means_arr, color=dr_color, linestyle="-", linewidth=3, label="DR-EKF")
        ax.fill_between(
            omega_vals,
            dr_means_arr - dr_stds_arr,
            dr_means_arr + dr_stds_arr,
            color=dr_color,
            alpha=0.25,
        )

    # Legend: inside plot, right side vertically centered, more compact spacing
    legend = ax.legend(
        fontsize=18,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
        framealpha=0.95,
        borderpad=0.6,
        labelspacing=0.2,
        handlelength=2.4,
        handletextpad=0.6,
    )
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        labelsize=18,
        top=True,
        right=True,
    )
    # Restrict x-axis limits to [0.2, 0.8] as requested
    ax.set_xlim(0.2, 0.8)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"mse_vs_omegaVar_scale_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"MSE vs omegaVar_scale plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MSE vs omegaVar_scale from exp_ct_tracking_nonlinear results")
    parser.add_argument("--dist", default="normal", type=str)
    parser.add_argument("--results_dir", default=RESULTS_DIR, type=str)
    parser.add_argument("--scales", nargs="*", type=float, default=None,
                        help="Scale values (default: auto-detect from saved files)")
    args = parser.parse_args()

    _setup_style()

    scales, ekf_means, ekf_stds, dr_means, dr_stds = load_summaries(args.results_dir, args.dist, args.scales)
    if not scales:
        print(f"No optimal_results found in {args.results_dir} for dist={args.dist}. Run exp_ct_tracking_nonlinear.py first.")
        return
    plot_mse_vs_omega_var_scale(scales, ekf_means, ekf_stds, dr_means, dr_stds, args.dist, args.results_dir)


if __name__ == "__main__":
    main()
