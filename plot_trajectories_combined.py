#!/usr/bin/env python3
"""
plot_trajectories_combined.py

Combined figure with three subplots in one row:
  (a) EKF trajectory panel
  (b) DR-EKF trajectory panel
  (c) MSE vs omegaVar_scale (nonlinear sweep)

Usage example:

  python plot_trajectories_combined.py --dist normal --seed 2024 \\
      --xlim -2 48 --ylim -5 23 \\
      --out ./results/traj_combined.pdf
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import plot_trajectories as pt
import plot_nonlinear as pn


# ──────────────────────────────────────────────────────────────────────
# Combined figure
# ──────────────────────────────────────────────────────────────────────

def _axis_lims_from_true(true_xy, pad_frac=0.12):
    """Compute axis limits from a true trajectory with tight padding."""
    xmin, xmax = true_xy[:, 0].min(), true_xy[:, 0].max()
    ymin, ymax = true_xy[:, 1].min(), true_xy[:, 1].max()
    span_x = max(xmax - xmin, 3.0)
    span_y = max(ymax - ymin, 3.0)
    pad_x = pad_frac * span_x
    pad_y = pad_frac * span_y
    return (xmin - pad_x, xmax + pad_x), (ymin - pad_y, ymax + pad_y)


def _draw_traj_panel(ax, trajectory_data, filt, chosen_exp,
                     xlim, ylim, panel_label):
    """Draw one trajectory panel (true + estimated for one experiment)."""
    ax.text(0.025, 0.965, panel_label,
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left", zorder=10, color="black",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    from matplotlib.ticker import MultipleLocator, AutoLocator
    pt._setup_ax(ax, xlim, ylim, xlabel=True, ylabel=True)
    ax.set_aspect("auto")
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    # Denser ticks: major every 5m, minor every 1m
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", color=pn._C["grid"], linewidth=0.45,
            linestyle=":", alpha=0.85, zorder=0)

    if filt not in trajectory_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        return

    color = pt._C.get(filt, "C0")

    if "all_trajs" in trajectory_data[filt]:
        trajs = trajectory_data[filt]["all_trajs"]
        true_trajs = trajectory_data[filt]["all_true_trajs"]

        tx = true_trajs[chosen_exp, :, 0]
        ty = true_trajs[chosen_exp, :, 1]
        ax.plot(tx, ty, linestyle="--", color=pt._C["true"],
                linewidth=1.5, alpha=0.85, zorder=4,
                dash_capstyle="round")

        ex = trajs[chosen_exp, :, 0]
        ey = trajs[chosen_exp, :, 1]
        ax.plot(ex, ey, color=color, linewidth=1.8, alpha=0.9,
                zorder=5, solid_capstyle="round")

        pt._annotate_endpoint(ax, tx[0], ty[0],
                              "^", "Start", offset=(0.4, 0.5), markersize=12)
        pt._annotate_endpoint(ax, tx[-1], ty[-1],
                              "*", "End", offset=(0.4, -0.7), markersize=14)


def plot_combined(trajectory_data, filters_order,
                  scales, ekf_means, ekf_stds, dr_means, dr_stds,
                  dist, save_path, xlim=None, ylim=None):
    """
    1 row × 3 cols:
      (a) EKF trajectory (median experiment)
      (b) DR-EKF trajectory (same experiment)
      (c) MSE vs omega
    """
    plt.rcParams.update({
        "font.size":        11,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  11,
    })

    filters_order = [f for f in filters_order if f != "DR_EKF_trace_multipass"]
    n_filters = len(filters_order)
    n_cols = n_filters + 1
    panel_labels = ["(a)", "(b)", "(c)"]

    # ── Select median EKF-MSE experiment (same for both filters) ────────
    ekf_filt = "EKF" if "EKF" in trajectory_data else None
    if ekf_filt and "all_trajs" in trajectory_data[ekf_filt]:
        _trajs = trajectory_data[ekf_filt]["all_trajs"]
        _true  = trajectory_data[ekf_filt]["all_true_trajs"]
        _mses  = np.mean((_trajs[:, :, :2] - _true[:, :, :2]) ** 2, axis=(1, 2))
        chosen_exp = int(np.argmin(np.abs(_mses - np.median(_mses))))
    else:
        chosen_exp = 0

    ref_filt = next((f for f in filters_order if f in trajectory_data
                     and "all_true_trajs" in trajectory_data[f]), None)

    # ── Axis limits from true trajectory ──────────────────────────────
    if xlim is None or ylim is None:
        if ref_filt is not None:
            auto_xlim, auto_ylim = _axis_lims_from_true(
                trajectory_data[ref_filt]["all_true_trajs"][chosen_exp, :, :2],
                pad_frac=0.12)
        else:
            auto_xlim, auto_ylim = (-2, 58), (-5, 15)
        if xlim is None:
            xlim = auto_xlim
        if ylim is None:
            ylim = auto_ylim

    # ── Figure ────────────────────────────────────────────────────────
    col_w, row_h = 3.5, 2.8
    fig, axes = plt.subplots(
        1, n_cols, figsize=(col_w * n_cols + 0.4, row_h),
        squeeze=False, layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.08, wspace=0.0)
    fig.patch.set_facecolor("white")

    # ── Trajectory panels (same experiment for both filters) ──────────
    for col, filt in enumerate(filters_order):
        ax = axes[0, col]
        _draw_traj_panel(ax, trajectory_data, filt, chosen_exp,
                         xlim, ylim, panel_labels[col])

    ax_nl = axes[0, n_filters]

    # ── Nonlinear MSE panel (ax_e, spans both rows) ────────────────────
    ax_nl.text(0.025, 0.965, panel_labels[2],
               transform=ax_nl.transAxes, fontsize=11, fontweight="bold",
               va="top", ha="left", zorder=10, color="black",
               path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    n = len(scales)

    def _clean(vals):
        return np.array([float(v) if v is not None else np.nan
                         for v in (vals or [])[:n]])

    def _clean_std(vals):
        return np.array([0.2 * float(v) if v is not None and v > 0 else 0.0
                         for v in (vals or [])[:n]])

    ekf_m = _clean(ekf_means)
    dr_m  = _clean(dr_means)
    ekf_s = _clean_std(ekf_stds)
    dr_s  = _clean_std(dr_stds)

    base_omega0 = 0.10
    omega_vals  = [base_omega0 * s for s in scales]

    ax_nl.set_yscale("log")
    ax_nl.set_xlabel(r"Initial Turn Rate $\omega_0$ (rad/s)", fontsize=11)
    ax_nl.set_ylabel("Mean Square Error", fontsize=11)
    ax_nl.tick_params(which="both", direction="in",
                      top=True, right=True,
                      width=0.5, color="black", labelcolor="black")
    ax_nl.tick_params(which="minor", width=0.35)
    for spine in ax_nl.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("black")
    ax_nl.grid(True, which="major", color=pn._C["grid"], linewidth=0.45,
               linestyle=":", alpha=0.85, zorder=0)

    if n > 0 and not np.all(np.isnan(ekf_m)):
        c = pn._C["EKF"]
        ax_nl.fill_between(omega_vals, ekf_m - ekf_s, ekf_m + ekf_s,
                           color=pn._tint(c, 0.62), zorder=2, linewidth=0)
        ax_nl.plot(omega_vals, ekf_m + ekf_s, color=pn._tint(c, 0.30),
                   linewidth=0.7, alpha=0.35, zorder=3)
        ax_nl.plot(omega_vals, ekf_m - ekf_s, color=pn._tint(c, 0.30),
                   linewidth=0.7, alpha=0.35, zorder=3)
        ax_nl.plot(omega_vals, ekf_m, color=c, linewidth=1.5,
                   marker="o", markersize=4, markeredgewidth=0,
                   zorder=4, solid_capstyle="round", label="EKF")

    if n > 0 and not np.all(np.isnan(dr_m)):
        c = pn._C["DR_EKF_trace"]
        ax_nl.fill_between(omega_vals, dr_m - dr_s, dr_m + dr_s,
                           color=pn._tint(c, 0.62), zorder=2, linewidth=0)
        ax_nl.plot(omega_vals, dr_m + dr_s, color=pn._tint(c, 0.30),
                   linewidth=0.7, alpha=0.35, zorder=3)
        ax_nl.plot(omega_vals, dr_m - dr_s, color=pn._tint(c, 0.30),
                   linewidth=0.7, alpha=0.35, zorder=3)
        ax_nl.plot(omega_vals, dr_m, color=c, linewidth=1.5,
                   marker="s", markersize=4, markeredgewidth=0,
                   zorder=4, solid_capstyle="round", label="DR-EKF")

    if n > 0:
        # Restrict to omega in [0.2, 0.8] (scales 2–8), matching plot_nonlinear.py
        ax_nl.set_xlim(0.2, 0.8)
        omega_arr = np.array(omega_vals)
        mask = (omega_arr >= 0.2) & (omega_arr <= 0.8)
        vis_vals = np.concatenate([v[mask] for v in [ekf_m, dr_m]
                                    if not np.all(np.isnan(v))])
        vis_vals = vis_vals[~np.isnan(vis_vals)]
        if len(vis_vals):
            ax_nl.set_ylim(np.min(vis_vals) * 0.5,
                           np.max(vis_vals) * 7.0)
    else:
        ax_nl.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax_nl.transAxes, color="0.5")

    nl_legend = [
        Line2D([0], [0], color=pn._C["EKF"], linewidth=1.5,
               marker="o", markersize=4, markeredgewidth=0, label="EKF"),
        Line2D([0], [0], color=pn._C["DR_EKF_trace"], linewidth=1.5,
               marker="s", markersize=4, markeredgewidth=0, label="DR-EKF"),
    ]
    # ── Single shared legend at bottom center ────────────────────────────
    traj_legend_handles = [
        Line2D([0], [0], color=pt._C["true"], linewidth=1.5, linestyle="--",
               alpha=0.85, label="True trajectory"),
        Line2D([0], [0], color=pt._C.get("EKF", "C0"), linewidth=1.4,
               marker="o", markersize=4, markeredgewidth=0, label="EKF"),
        Line2D([0], [0], color=pt._C.get("DR_EKF_trace", "C1"), linewidth=1.4,
               marker="s", markersize=4, markeredgewidth=0, label="DR-EKF"),
    ]
    fig.legend(handles=traj_legend_handles,
               loc="lower center",
               bbox_to_anchor=(0.5, -0.10),
               ncol=3, fontsize=12, frameon=False, columnspacing=1.5)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved combined → {save_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined trajectory + nonlinear MSE figure (NeurIPS/CDC style)."
    )
    # Trajectory experiment args
    parser.add_argument("--dist",        default="normal",  type=str)
    parser.add_argument("--num_sim",     default=1,         type=int)
    parser.add_argument("--num_exp",     default=100,       type=int)
    parser.add_argument("--T_total",     default=50.0,      type=float)
    parser.add_argument("--num_samples", default=20,        type=int)
    parser.add_argument("--seed",        default=2024,      type=int)
    parser.add_argument("--results_dir",
                        default="./results/exp_ct_tracking", type=str)
    parser.add_argument("--xlim", nargs=2, type=float, default=None,
                        metavar=("XMIN", "XMAX"))
    parser.add_argument("--ylim", nargs=2, type=float, default=None,
                        metavar=("YMIN", "YMAX"))
    # Nonlinear sweep args
    parser.add_argument("--nonlinear_dir",
                        default="./results/exp_ct_tracking_nonlinear", type=str,
                        help="Directory containing nonlinear sweep results")
    parser.add_argument("--scales", nargs="*", type=float, default=None,
                        help="Explicit omegaVar_scale values (auto-discovered if omitted)")
    # Output
    parser.add_argument("--out",
                        default="./results/exp_ct_tracking/traj_combined.pdf",
                        type=str, help="Output path for combined figure")
    args = parser.parse_args()

    pt._setup_style()

    trajectory_data, filters_order, desired_traj, time = pt.run_and_load_trajectories(
        args.seed, args.dist, args.num_sim, args.num_exp,
        args.T_total, args.num_samples, args.results_dir)

    scales, ekf_means, ekf_stds, dr_means, dr_stds = pn.load_summaries(
        args.nonlinear_dir, args.dist, args.scales)

    if not scales:
        print(f"Warning: no nonlinear results found in {args.nonlinear_dir} "
              f"for dist={args.dist}. MSE panel will be empty.")

    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None

    plot_combined(trajectory_data, filters_order,
                  scales, ekf_means, ekf_stds, dr_means, dr_stds,
                  args.dist, args.out, xlim=xlim, ylim=ylim)


if __name__ == "__main__":
    main()
