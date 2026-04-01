#!/usr/bin/env python3
"""
exp_ct_compare_seeds.py

Run exp_ct_tracking.py twice with different seeds (e.g. 2024 and 2026),
and then create a side-by-side trajectory subplot figure for all filters
using the same style as exp_ct_plot.plot_subplots_all_filters.

Usage example:

  python plot_trajectories.py --dist normal \
      --seed1 2024 --seed2 2026 \
      --xlim -2 48 --ylim -5 23 \
      --out ./results/exp_ct_tracking/traj_2d_EKF_subplots_compare_seeds_normal.pdf

This script:
  1) Runs exp_ct_tracking.main(...) with seed_base=seed1, loads trajectories.
  2) Runs exp_ct_tracking.main(...) with seed_base=seed2, loads trajectories.
  3) Plots EKF vs DR-EKF: by default one row (a)(b) for seed1 only; use
     --both-seeds for a 2×2 grid comparing seed1 and seed2.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

import exp_ct_tracking
import exp_ct_plot


# ──────────────────────────────────────────────────────────────────────
# Design tokens (shared with plot_trajectories_combined.py)
# ──────────────────────────────────────────────────────────────────────

_C = {
    "EKF":          "#C0392B",   # flat-UI red
    "DR_EKF_trace": "#2980B9",   # flat-UI blue
    "true":         "#1A1A1A",   # near-black
    "grid":         "#D5D9E0",   # cool light gray
    "panel_bg":     "#FAFAFA",   # neutral white
}

_DISPLAY = {
    "EKF":          "EKF",
    "DR_EKF_trace": "DR-EKF",
}


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────

def _tint(hex_color, amount=0.58):
    """Blend hex_color toward white by `amount` (0 = original, 1 = white)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - (1 - r) * (1 - amount),
            1 - (1 - g) * (1 - amount),
            1 - (1 - b) * (1 - amount))


def _annotate_endpoint(ax, x, y, marker, label, offset=(0.6, 0.6), markersize=9):
    """Draw a clean endpoint marker with an offset text label."""
    ax.plot(x, y, marker=marker, markersize=markersize,
            markeredgewidth=1.0, markeredgecolor="white",
            markerfacecolor="#1A1A1A", zorder=8, linestyle="none")
    ax.annotate(
        label,
        xy=(x, y), xytext=(x + offset[0], y + offset[1]),
        fontsize=11, fontweight="bold", color="black",
        va="center", ha="left", zorder=9,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


def _uncertainty_tube(ax, x_mean, y_mean, x_std, y_std, color, scale=0.05):
    """Draw a perpendicular-std tube around the mean trajectory."""
    std_mag = scale * 0.5 * (x_std + y_std)
    dx = np.gradient(x_mean)
    dy = np.gradient(y_mean)
    norms = np.hypot(dx, dy)
    norms[norms == 0] = 1.0
    dx /= norms
    dy /= norms
    px, py = -dy, dx
    ux, uy = x_mean + px * std_mag, y_mean + py * std_mag
    lx, ly = x_mean - px * std_mag, y_mean - py * std_mag
    tube_x = np.concatenate([ux, lx[::-1]])
    tube_y = np.concatenate([uy, ly[::-1]])
    ax.fill(tube_x, tube_y, color=_tint(color, 0.62), zorder=2, linewidth=0)
    ax.plot(ux, uy, color=_tint(color, 0.30), linewidth=0.7, zorder=3)
    ax.plot(lx, ly, color=_tint(color, 0.30), linewidth=0.7, zorder=3)


def _setup_ax(ax, xlim, ylim, xlabel=True, ylabel=True):
    """Apply shared axis limits, ticks, and labels."""
    x0, x1 = xlim if xlim else (-2.0, 50.0)
    y0, y1 = ylim if ylim else (-2.0, 26.0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(which="both", direction="in",
                   top=True, right=True,
                   width=0.3, color="black", labelcolor="black")
    ax.tick_params(which="minor", width=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color("black")
    if xlabel:
        ax.set_xlabel(r"$x$ (m)", fontsize=8)
    if ylabel:
        ax.set_ylabel(r"$y$ (m)", fontsize=8)


def _draw_filter(ax, trajectory_data, filt, show_endpoints=True):
    """Draw one filter's tube + mean trajectory onto ax. Returns True if drawn."""
    if filt not in trajectory_data:
        return False

    color     = _C.get(filt, "C0")
    mean_traj = trajectory_data[filt]["mean"]
    std_traj  = trajectory_data[filt]["std"]
    x_mean, y_mean = mean_traj[:, 0], mean_traj[:, 1]
    x_std,  y_std  = std_traj[:, 0],  std_traj[:, 1]

    _uncertainty_tube(ax, x_mean, y_mean, x_std, y_std, color=color)
    ax.plot(x_mean, y_mean, "-", color=color, linewidth=1.5,
            zorder=4, solid_capstyle="round",
            label=_DISPLAY.get(filt, filt))

    if show_endpoints:
        _annotate_endpoint(ax, x_mean[0],  y_mean[0],
                           "^", "Start", offset=( 0.8,  0.8), markersize=10)
        _annotate_endpoint(ax, x_mean[-1], y_mean[-1],
                           "*", "Goal",  offset=( 0.8, -1.2), markersize=12)
    return True


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
        # PDF/PS: use TrueType (Type 42) fonts to avoid Type 3 font errors
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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


def run_and_load_trajectories(seed_base, dist, num_sim, num_exp, T_total, num_samples, results_dir):
    """
    Run exp_ct_tracking with given seed (unless results already exist),
    then load trajectory data via exp_ct_plot helpers.
    """
    os.makedirs(results_dir, exist_ok=True)
    optimal_file = os.path.join(results_dir, f"optimal_results_{dist}.pkl")

    if not os.path.exists(optimal_file):
        print(f"\n=== Running exp_ct_tracking with seed_base={seed_base}, saving to {results_dir} ===")
        # Override results path for this run
        exp_ct_tracking.RESULTS_PATH = results_dir
        exp_ct_tracking.main(dist, num_sim, num_exp, T_total, num_samples, seed_base)
    else:
        print(f"\n=== Using existing results in {results_dir} (seed_base={seed_base}) ===")

    optimal_results, _ = exp_ct_plot.load_data(results_dir, dist)
    trajectory_data, filters_order = exp_ct_plot.extract_trajectory_data_from_saved(
        optimal_results, results_dir, dist
    )

    # Determine actual time horizon from trajectories
    if trajectory_data:
        dt = 0.2
        first_key = next(iter(trajectory_data.keys()))
        actual_num_steps = len(trajectory_data[first_key]["mean"])
        actual_time = (actual_num_steps - 1) * dt
        desired_traj, time = exp_ct_plot.generate_desired_trajectory(actual_time, dt)
    else:
        desired_traj, time = exp_ct_plot.generate_desired_trajectory(10.0)

    return trajectory_data, filters_order, desired_traj, time


def plot_side_by_side(
    traj1,
    traj2,
    filters_order,
    desired_traj,
    time,
    label1,
    label2,
    dist,
    save_path,
    xlim=None,
    ylim=None,
    both_seeds=False,
):
    """Comparison with rows = seeds, cols = EKF and DR-EKF (multipass column dropped).

    If both_seeds is False (default), only the first row (a)(b) for traj1 is drawn.
    If both_seeds is True, two rows (a)–(d) compare seed1 and seed2.

    If xlim / ylim are (min, max) tuples, all subplots use the same limits.
    If omitted, axes use matplotlib auto limits.
    """
    # Override colors: EKF = red, DR-EKF = blue
    base_colors = exp_ct_plot._colors()
    colors = {
        "EKF": "#d62728",           # red
        "DR_EKF_trace": "#1f77b4",  # blue
    }
    # Fallback to base colors for any other filters (if present)
    for k, v in base_colors.items():
        colors.setdefault(k, v)
    filter_names = exp_ct_plot._filter_names()

    # Drop multipass row for this comparison
    filters_order = [f for f in filters_order if f != "DR_EKF_trace_multipass"]

    n_filters = len(filters_order)
    if both_seeds:
        seed_data = [(traj1, label1), (traj2, label2)]
        panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]
    else:
        seed_data = [(traj1, label1)]
        panel_labels = [["(a)", "(b)"]]
    n_seeds = len(seed_data)

    fig, axes = plt.subplots(
        nrows=n_seeds,
        ncols=n_filters,
        figsize=(5 * n_filters, 4 * n_seeds),
        squeeze=False,
    )
    plt.subplots_adjust(wspace=0.08, hspace=-0.5 if n_seeds > 1 else 0.12)

    for row, (trajectory_data, seed_label) in enumerate(seed_data):
        for col, filt in enumerate(filters_order):
            ax = axes[row, col]
            # Panel label (a),(b),(c),(d) in upper-left corner
            ax.text(
                0.02,
                0.95,
                panel_labels[row][col],
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="left",
            )

            if filt not in trajectory_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            color = colors.get(filt, "C0")
            mean_traj = trajectory_data[filt]["mean"]
            std_traj = trajectory_data[filt]["std"]

            x_mean = mean_traj[:, 0]
            y_mean = mean_traj[:, 1]
            x_std, y_std = std_traj[:, 0], std_traj[:, 1]
            std_mag = 0.05 * 0.5 * (x_std + y_std)  # 1/20-std tube (average of x,y std)

            dx = np.gradient(x_mean)
            dy = np.gradient(y_mean)
            norms = np.hypot(dx, dy)
            norms[norms == 0] = 1.0
            dx /= norms
            dy /= norms
            perp_x, perp_y = -dy, dx
            upper_x = x_mean + perp_x * std_mag
            upper_y = y_mean + perp_y * std_mag
            lower_x = x_mean - perp_x * std_mag
            lower_y = y_mean - perp_y * std_mag
            tube_x = np.concatenate([upper_x, lower_x[::-1]])
            tube_y = np.concatenate([upper_y, lower_y[::-1]])

            # Std tube (same color as filter, as before); no legend entry
            ax.fill(tube_x, tube_y, color=color, alpha=0.3)
            # Filter-specific estimated trajectory label
            if filt == "EKF":
                est_label = "EKF"
            elif filt == "DR_EKF_trace":
                est_label = "DR-EKF"
            else:
                est_label = "Estimated"
            ax.plot(x_mean, y_mean, "-", color=color, linewidth=2, label=est_label)

            # Mark start and end points with black markers and larger black labels
            x_start, y_start = x_mean[0], y_mean[0]
            x_end, y_end = x_mean[-1], y_mean[-1]
            ax.plot(x_start, y_start, marker="^", markersize=7, color="k")
            ax.text(
                x_start,
                y_start,
                " Start",
                fontsize=11,
                fontweight="bold",
                va="bottom",
                ha="left",
                color="k",
            )
            # End point: star marker
            ax.plot(x_end, y_end, marker="*", markersize=8, color="k")
            ax.text(
                x_end,
                y_end,
                " End",
                fontsize=11,
                fontweight="bold",
                va="bottom",
                ha="left",
                color="k",
            )

            reference_xy = None
            if filt in trajectory_data and "true_mean" in trajectory_data[filt]:
                reference_xy = trajectory_data[filt]["true_mean"][:, :2]

            if reference_xy is not None:
                ax.plot(
                    reference_xy[:, 0],
                    reference_xy[:, 1],
                    ":",
                    color="k",
                    linewidth=1.5,
                    alpha=0.9,
                    label="True trajectory",
                )

            ax.set_aspect("equal", adjustable="box")
            
            # 
            ax.set_xlim(-2.0, 50.0)
            ax.set_ylim(-2.0, 26.0)

            
            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])
            # Ticks:
            # - x major ticks at 0,10,20,... with minor ticks splitting each 10 into 5 parts (=> 2m)
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(2))

            # More y-ticks (labels)
            ax.set_yticks([0, 5, 10, 15, 20, 25])

            # Grid: keep sparse (manual), independent of tick density
            grid_ls = "-"
            grid_lw = 0.8
            grid_alpha = 0.5
            grid_color = "0.7"
            ax.grid(False)
            for gx in (0, 10, 20, 30, 40):
                ax.axvline(gx, linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha, color=grid_color, zorder=0)
            for gy in (0, 5, 10, 15, 20, 25):
                ax.axhline(gy, linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha, color=grid_color, zorder=0)
            # Ticks on all sides, pointing inward
            ax.tick_params(direction="in", which="both", top=True, right=True)
            if row == n_seeds - 1:
                ax.set_xlabel("x (m)", fontsize=13)
            if col == 0:
                ax.set_ylabel("y (m)", fontsize=13)

    # Unified legend: True trajectory, EKF (blue), DR-EKF (red)
    # Legend order: EKF -> DR-EKF -> True trajectory
    legend_handles = [
        Line2D([0], [0], color=colors.get("EKF", "C0"), linestyle="-", linewidth=2, label="EKF"),
        Line2D([0], [0], color=colors.get("DR_EKF_trace", "C1"), linestyle="-", linewidth=2, label="DR-EKF"),
        Line2D([0], [0], color="k", linestyle=":", linewidth=1.5, label="True trajectory"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.01),
        ncol=3,  # place all entries on one line
        fontsize=11,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0.0, 1, 1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load exp_ct_tracking results and plot EKF vs DR-EKF trajectories. "
            "Default: one row (panels a,b) for seed1; use --both-seeds for two rows."
        )
    )
    parser.add_argument("--dist", default="normal", type=str)
    parser.add_argument("--num_sim", default=1, type=int)
    parser.add_argument("--num_exp", default=10, type=int)
    parser.add_argument("--T_total", default=50.0, type=float)
    parser.add_argument("--num_samples", default=20, type=int)
    parser.add_argument("--seed1", default=2024, type=int)
    parser.add_argument("--seed2", default=2019, type=int)
    parser.add_argument("--label1", default="seed 2024", type=str)
    parser.add_argument("--label2", default="seed 2019", type=str)
    parser.add_argument(
        "--results_dir_1",
        default="./results/exp_ct_tracking",
        type=str,
        help="Results directory for first seed (will be created if missing)",
    )
    parser.add_argument(
        "--results_dir_2",
        default="./results/exp_ct_tracking_seed2026",
        type=str,
        help="Results directory for second seed (will be created if missing)",
    )
    parser.add_argument(
        "--out",
        default="./results/exp_ct_tracking/traj_2d_EKF_subplots_compare_seeds_normal.pdf",
        type=str,
        help="Output PDF path",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="x-axis limits for all subplots (e.g. --xlim -2 48). Omit for auto.",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="y-axis limits for all subplots (e.g. --ylim -5 23). Omit for auto.",
    )
    parser.add_argument(
        "--both-seeds",
        action="store_true",
        help=(
            "Plot two rows (seed1 vs seed2). Default: one row with EKF and DR-EKF "
            "for seed1 only."
        ),
    )
    args = parser.parse_args()

    _setup_style()

    traj1, filters_order1, desired1, time1 = run_and_load_trajectories(
        args.seed1,
        args.dist,
        args.num_sim,
        args.num_exp,
        args.T_total,
        args.num_samples,
        args.results_dir_1,
    )
    if args.both_seeds:
        traj2, _, _, _ = run_and_load_trajectories(
            args.seed2,
            args.dist,
            args.num_sim,
            args.num_exp,
            args.T_total,
            args.num_samples,
            args.results_dir_2,
        )
    else:
        traj2 = None

    xlim = tuple(args.xlim) if args.xlim is not None else None
    ylim = tuple(args.ylim) if args.ylim is not None else None
    plot_side_by_side(
        traj1,
        traj2,
        filters_order1,
        desired1,
        time1,
        args.label1,
        args.label2,
        args.dist,
        args.out,
        xlim=xlim,
        ylim=ylim,
        both_seeds=args.both_seeds,
    )


if __name__ == "__main__":
    main()
