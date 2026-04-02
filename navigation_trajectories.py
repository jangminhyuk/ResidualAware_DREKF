#!/usr/bin/env python3
"""
Static matplotlib plot: pedestrian ground truth + EKF vs DR-EKF estimated robot paths.

Derived from ``navigation_canvas_uwb.py`` with identical dynamics, noise, and default CLI
parameters. Built-in defaults (see ``_DEFAULT_NAV_*``) use seeds ``2024…2073``, ``t=900…1000``,
and the purple-oracle plot path; run ``python navigation_trajectories.py`` with no args to use them.

Pedestrian GT plots **all** non-ego agents that appear in ``obs['non-ego']`` across the rollout
(typically multiple pedestrians when the scene has several). Robot paths
are the **true** simulated robot positions (one rollout per estimator; MPC + noise differ).
Time steps ``t = 8 … 20`` refer to the dataset timestep ``env._step``; scatter markers use
increasing opacity (later = darker).

Run from DR-EKF-main (or set PYTHONPATH). Requires CANVAS clone; set CANVAS_ROOT if needed.

Example:
  python navigation_trajectories.py

  python navigation_trajectories.py --dataset zara2 --predictor trajectron

  Single seed (override built-in seed list):
  python navigation_trajectories.py --seeds 2024
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- repo roots ---
_DR_ROOT = os.path.dirname(os.path.abspath(__file__))
if _DR_ROOT not in sys.path:
    sys.path.insert(0, _DR_ROOT)


def _resolve_canvas_root() -> str:
    if os.environ.get("CANVAS_ROOT"):
        return os.path.abspath(os.environ["CANVAS_ROOT"])
    for p in (
        os.path.abspath(os.path.join(_DR_ROOT, "..", "CANVAS")),
        os.path.abspath(os.path.join(_DR_ROOT, "..", "..", "CANVAS")),
    ):
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canvas")):
            return p
    return os.path.abspath(os.path.join(_DR_ROOT, "..", "CANVAS"))


CANVAS_ROOT = _resolve_canvas_root()
if os.path.isdir(CANVAS_ROOT) and CANVAS_ROOT not in sys.path:
    sys.path.insert(0, CANVAS_ROOT)

from navigation_uwb_estimator import h_uwb, make_uwb_estimator, unicycle_f  # noqa: E402

from canvas.controllers.mpc import BaseMPC  # noqa: E402
from canvas.datasets import RegisteredDatasets  # noqa: E402
from canvas.envs.env import Environment  # noqa: E402
from canvas.predictors import Predictors  # noqa: E402


# ---------------------------------------------------------------------------
# Noise + metrics
# ---------------------------------------------------------------------------


# Legacy DR default was θ_w = θ_v = 0.06; exp_ct style uses θ_eff = θ_w * √2.
_DEFAULT_THETA_EFF = 0.3

# Goal "arrival" box (m) — same for success and early stop when not ignoring goal.
GOAL_SUCCESS_TOL = 0.3

# Right-hand δ_t panel: time axis cap and ticks (align with plot_safe_navigation_uwb.py margin panels).
_NAV_FADE_DELTA_X_MAX_S = 14.0
_NAV_FADE_DELTA_X_TICKS = (4.0, 8.0, 12.0)
_NAV_FADE_DELTA_Y_TICKS = (0.0, 0.3, 0.6, 0.9)

# Default ``main()`` batch profile for ``navigation_trajectories.py``: RNG seeds, time window,
# and output path. Override any field by passing the corresponding CLI flag.
_DEFAULT_NAV_SEED_START = 2024
_DEFAULT_NAV_SEED_END = 2073  # inclusive (same as ``seq 2024 2073``)
_DEFAULT_NAV_SEEDS_STR = ",".join(
    str(i) for i in range(_DEFAULT_NAV_SEED_START, _DEFAULT_NAV_SEED_END + 1)
)
_DEFAULT_NAV_TRAJ_SEED = 2048
_DEFAULT_NAV_T_BEGIN = 900
_DEFAULT_NAV_T_END = 1000
_DEFAULT_NAV_T_SCATTER_MIN = 900
_DEFAULT_NAV_T_SCATTER_MAX = 1000
_DEFAULT_NAV_OUTPUT = os.path.join(
    "results",
    "navigation_plots",
    "navigation_trajectories_purple_robot_oracle_t900_to_1000_deltaMeanStd_trajSeed2024.png",
)


def _navigation_publication_style() -> None:
    """Matplotlib rcParams aligned with ResidualAware ``plot_safe_navigation_uwb._setup_style``."""
    import matplotlib.pyplot as plt

    _C = {"grid": "#D5D9E0", "panel_bg": "#FFFFFF"}
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10,
            "axes.labelcolor": "black",
            "axes.linewidth": 0.8,
            "axes.facecolor": _C["panel_bg"],
            "axes.edgecolor": "black",
            "axes.grid": True,
            "grid.color": _C["grid"],
            "grid.linewidth": 0.45,
            "grid.linestyle": ":",
            "grid.alpha": 0.85,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.2,
            "ytick.minor.size": 2.2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.labelcolor": "black",
            "ytick.labelcolor": "black",
            "legend.fontsize": 10,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "black",
            "legend.fancybox": True,
            "legend.handlelength": 2.2,
            "legend.handleheight": 0.85,
            "legend.borderpad": 0.55,
            "legend.labelspacing": 0.4,
            "legend.shadow": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "lines.linewidth": 1.8,
            "patch.linewidth": 0.5,
        }
    )


def derive_dr_radii_from_theta_eff(args: argparse.Namespace) -> None:
    """Set DR-EKF_trace static radii from the sole CLI knob ``theta_eff`` (exp_ct_tracking*.py).

    Always:  s = θ_eff / √2,  θ_w = θ_v = θ_x0 = s

    ``args.theta_w``, ``args.theta_v``, ``args.theta_x0`` are set for ``run_navigation`` /
    ``make_uwb_estimator``; they are not separate CLI flags.
    """
    te = float(args.theta_eff)
    s = te / np.sqrt(2.0)
    args.theta_w = float(s)
    args.theta_v = float(s)
    args.theta_x0 = float(s)


def _parse_n_floats(s: str, n: int, field: str) -> List[float]:
    """Parse 'a,b,c' or a single value repeated n times."""
    parts = [p.strip() for p in s.replace(" ", "").split(",") if p.strip()]
    if len(parts) == 1 and n > 1:
        return [float(parts[0])] * n
    if len(parts) != n:
        raise ValueError(f"{field}: need {n} values (comma-separated) or one value repeated, got {s!r}")
    return [float(p) for p in parts]


def w2_gaussian_diagonal(
    mu1: np.ndarray, d1: np.ndarray, mu2: np.ndarray, d2: np.ndarray
) -> float:
    """2-Wasserstein distance between axis-aligned Gaussians (diagonal covariances)."""
    mu1 = np.asarray(mu1).reshape(-1)
    mu2 = np.asarray(mu2).reshape(-1)
    d1 = np.maximum(np.asarray(d1).reshape(-1), 0.0)
    d2 = np.maximum(np.asarray(d2).reshape(-1), 0.0)
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    var_term = float(np.sum((np.sqrt(d1) - np.sqrt(d2)) ** 2))
    return float(np.sqrt(mean_term + var_term))


def sample_process_noise(Sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """w_t in R^3, shape (3,1)."""
    S = np.asarray(Sigma, dtype=float)
    nx = S.shape[0]
    w = rng.multivariate_normal(np.zeros(nx), S)
    return w.reshape(nx, 1)


def sample_measurement_noise(Sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """v_t in R^4, shape (4,1)."""
    S = np.asarray(Sigma, dtype=float)
    ny = S.shape[0]
    v = rng.multivariate_normal(np.zeros(ny), S)
    return v.reshape(ny, 1)


def ego_dict_from_vec(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(3)
    return {"position_x": float(x[0]), "position_y": float(x[1]), "orientation_z": float(x[2])}


def save_navigation_video(
    video_frames: List[Dict[str, Any]],
    *,
    dataset_name: str,
    predictor_name: str,
    estimator_name: str,
    output_dir: str,
    t_begin: int,
    init_x: float,
    init_y: float,
    goal_x: float,
    goal_y: float,
    seed: int,
) -> Optional[str]:
    """Save navigation animation (same structure as CANVAS custom_control_task_evaluation)."""
    if not video_frames:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    est_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", estimator_name.replace(" ", "_"))
    video_filename = (
        f"{dataset_name}_{predictor_name}_{est_tag}_t{t_begin}_"
        f"init{init_x:.1f}_{init_y:.1f}_goal{goal_x:.1f}_{goal_y:.1f}_seed{seed}.mp4"
    )
    video_path = os.path.join(output_dir, video_filename)

    all_positions: List[np.ndarray] = []
    for frame_data in video_frames:
        all_positions.append(frame_data["robot_pos"])
        est_pos = frame_data.get("estimated_pos")
        if est_pos is not None:
            all_positions.append(np.asarray(est_pos, dtype=float).reshape(2))
        est_traj = frame_data.get("estimated_trajectory")
        if est_traj is not None and len(est_traj) > 0:
            all_positions.extend(np.asarray(est_traj, dtype=float))
        if frame_data["non_ego_agents"]:
            for agent_pos in frame_data["non_ego_agents"].values():
                if agent_pos is not None:
                    all_positions.append(agent_pos)
        if frame_data["prediction"]:
            for pred_traj in frame_data["prediction"].values():
                if len(pred_traj) > 0:
                    all_positions.extend(pred_traj)
        if frame_data["ground_truth"]:
            for gt_traj in frame_data["ground_truth"].values():
                if len(gt_traj) > 0:
                    all_positions.extend(gt_traj)

    if len(all_positions) == 0:
        return None
    ap = np.array(all_positions)
    x_min, x_max = float(ap[:, 0].min()), float(ap[:, 0].max())
    y_min, y_max = float(ap[:, 1].min()), float(ap[:, 1].max())
    pad = max(x_max - x_min, y_max - y_min) * 0.15
    xlim = (x_min - pad, x_max + pad)
    ylim = (y_min - pad, y_max + pad)

    fig, ax = plt.subplots(figsize=(14, 10))

    def animate(frame_idx: int) -> None:
        ax.clear()
        frame_data = video_frames[frame_idx]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{dataset_name.upper()} — {predictor_name.upper()} — {estimator_name.upper()}\n"
            f"Frame: {frame_data['frame']}/{len(video_frames) - 1}",
            fontsize=14,
            fontweight="bold",
        )
        goal = frame_data["goal"]
        ax.scatter(
            goal[0],
            goal[1],
            s=400,
            marker="*",
            color="green",
            edgecolors="darkgreen",
            linewidths=3,
            zorder=15,
            label="Goal",
        )
        robot_traj = frame_data["robot_trajectory"]
        if len(robot_traj) > 1:
            ax.plot(
                robot_traj[:, 0],
                robot_traj[:, 1],
                "b-",
                linewidth=3,
                alpha=0.7,
                label="True path",
                zorder=5,
            )
        robot_pos = frame_data["robot_pos"]
        ax.scatter(
            robot_pos[0],
            robot_pos[1],
            s=250,
            marker="o",
            color="blue",
            edgecolors="darkblue",
            linewidths=2,
            zorder=12,
            label="True pose",
        )
        est_traj = frame_data.get("estimated_trajectory")
        if est_traj is not None and len(est_traj) > 0:
            et = np.asarray(est_traj, dtype=float)
            if len(et) > 1:
                ax.plot(
                    et[:, 0],
                    et[:, 1],
                    color="magenta",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.85,
                    label="Estimated path (filter)",
                    zorder=4,
                )
            est_pos = frame_data.get("estimated_pos")
            if est_pos is not None:
                ep = np.asarray(est_pos, dtype=float).reshape(2)
                ax.scatter(
                    ep[0],
                    ep[1],
                    s=220,
                    marker="X",
                    color="magenta",
                    edgecolors="darkmagenta",
                    linewidths=2,
                    zorder=11,
                    label="Estimated pose",
                )
        all_agent_ids = set()
        if frame_data["non_ego_agents"]:
            all_agent_ids.update(frame_data["non_ego_agents"].keys())
        if frame_data["prediction"]:
            all_agent_ids.update(frame_data["prediction"].keys())
        if frame_data["ground_truth"]:
            all_agent_ids.update(frame_data["ground_truth"].keys())
        first_agent = next(iter(all_agent_ids), None)
        for agent_id in all_agent_ids:
            if frame_data["non_ego_agents"] and agent_id in frame_data["non_ego_agents"]:
                agent_pos = frame_data["non_ego_agents"][agent_id]
                if agent_pos is not None:
                    ax.scatter(
                        agent_pos[0],
                        agent_pos[1],
                        s=120,
                        marker="s",
                        color="orange",
                        edgecolors="darkorange",
                        linewidths=2,
                        zorder=9,
                        alpha=0.8,
                    )
            if frame_data["ground_truth"] and agent_id in frame_data["ground_truth"]:
                gt = frame_data["ground_truth"][agent_id]
                if len(gt) > 0:
                    ax.plot(
                        gt[:, 0],
                        gt[:, 1],
                        "g-",
                        linewidth=2.5,
                        alpha=0.8,
                        zorder=7,
                        label="GT" if agent_id == first_agent else "",
                    )
            if frame_data["prediction"] and agent_id in frame_data["prediction"]:
                pred = frame_data["prediction"][agent_id]
                if len(pred) > 0:
                    ax.plot(
                        pred[:, 0],
                        pred[:, 1],
                        "r--",
                        linewidth=2.5,
                        alpha=0.8,
                        zorder=6,
                        label="Prediction" if agent_id == first_agent else "",
                    )
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(video_frames), interval=200, blit=False, repeat=True
    )
    try:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=5, metadata=dict(artist="CANVAS navigation UWB"), bitrate=1800)
        anim.save(video_path, writer=writer)
        plt.close(fig)
        return video_path
    except Exception:
        plt.close(fig)
        raise


def save_compare_true_trajectory_plot(
    *,
    dataset_name: str,
    predictor_name: str,
    output_dir: str,
    t_begin: int,
    init_x: float,
    init_y: float,
    goal_x: float,
    goal_y: float,
    seed: int,
    traj_ekf_runs: List[np.ndarray],
    traj_ekf_oracle_runs: List[np.ndarray],
    traj_dr_runs: List[np.ndarray],
) -> Optional[str]:
    """Save one static plot with all true robot paths for EKF / EKF-oracle / DR-EKF."""
    traj_ekf_runs = [np.asarray(t, dtype=float) for t in traj_ekf_runs if np.asarray(t).size > 0]
    traj_ekf_oracle_runs = [
        np.asarray(t, dtype=float) for t in traj_ekf_oracle_runs if np.asarray(t).size > 0
    ]
    traj_dr_runs = [np.asarray(t, dtype=float) for t in traj_dr_runs if np.asarray(t).size > 0]
    if not traj_ekf_runs or not traj_ekf_oracle_runs or not traj_dr_runs:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    png_name = (
        f"{dataset_name}_{predictor_name}_compare_true_paths_t{t_begin}_"
        f"init{init_x:.1f}_{init_y:.1f}_goal{goal_x:.1f}_{goal_y:.1f}_seed{seed}.png"
    )
    plot_path = os.path.join(output_dir, png_name)

    all_xy = np.vstack(traj_ekf_runs + traj_ekf_oracle_runs + traj_dr_runs)
    x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    pad = max(x_max - x_min, y_max - y_min, 1e-6) * 0.15

    fig, ax = plt.subplots(figsize=(11, 8))
    for i, traj in enumerate(traj_ekf_runs):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color="red",
            linewidth=1.6,
            alpha=0.35,
            label="EKF (nominal)" if i == 0 else None,
        )
    for i, traj in enumerate(traj_ekf_oracle_runs):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color="green",
            linewidth=1.6,
            alpha=0.35,
            label="EKF (true)" if i == 0 else None,
        )
    for i, traj in enumerate(traj_dr_runs):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color="blue",
            linewidth=1.6,
            alpha=0.35,
            label="DR-EKF" if i == 0 else None,
        )
    ax.scatter([goal_x], [goal_y], s=180, marker="*", color="black", label="Goal", zorder=5)
    ax.set_title(
        (
            f"True robot paths ({dataset_name}/{predictor_name}, seeds {seed}"
            f"…{seed + len(traj_ekf_runs) - 1})"
        ),
        fontsize=14,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _stack_margin_mean_std(runs: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Stack delta_margin_series across runs → per-time mean and std (length = min across runs)."""
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return None, None
    series = [np.asarray(r.get("delta_margin_series", []), dtype=float) for r in ok]
    if not series or min(len(s) for s in series) == 0:
        return None, None
    min_len = min(len(s) for s in series)
    arr = np.stack([s[:min_len] for s in series], axis=0)
    return np.mean(arr, axis=0), np.std(arr, axis=0, ddof=0)


def save_compare_delta_margin_plot(
    *,
    dataset_name: str,
    predictor_name: str,
    output_dir: str,
    t_begin: int,
    init_x: float,
    init_y: float,
    goal_x: float,
    goal_y: float,
    seed_start: int,
    n_runs: int,
    k_sigma: float,
    dt: float,
    ekf_runs: List[Dict[str, Any]],
    ekf_oracle_runs: List[Dict[str, Any]],
    dr_runs: List[Dict[str, Any]],
) -> Optional[str]:
    """Single axes: Δ_t mean vs time for all three estimators; each band = ±1 std across seeds."""
    m_e, s_e = _stack_margin_mean_std(ekf_runs)
    m_o, s_o = _stack_margin_mean_std(ekf_oracle_runs)
    m_d, s_d = _stack_margin_mean_std(dr_runs)
    if m_e is None or m_o is None or m_d is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    png_name = (
        f"{dataset_name}_{predictor_name}_compare_delta_margin_t{t_begin}_"
        f"init{init_x:.1f}_{init_y:.1f}_goal{goal_x:.1f}_{goal_y:.1f}_seed{seed_start}.png"
    )
    plot_path = os.path.join(output_dir, png_name)

    n = min(len(m_e), len(m_o), len(m_d))
    m_e, s_e = m_e[:n], s_e[:n]
    m_o, s_o = m_o[:n], s_o[:n]
    m_d, s_d = m_d[:n], s_d[:n]
    t = np.arange(n, dtype=float) * float(dt)
    seed_end = seed_start + n_runs - 1

    fig, ax = plt.subplots(figsize=(11, 6))
    for mean, std, color, label in (
        (m_e, s_e, "red", "EKF (nominal)"),
        (m_o, s_o, "green", "EKF (true)"),
        (m_d, s_d, "blue", "DR-EKF"),
    ):
        lo = np.maximum(mean - std, 0.0)
        hi = mean + std
        ax.fill_between(t, lo, hi, color=color, alpha=0.22)
        ax.plot(t, mean, color=color, linewidth=2.0, label=label)

    ax.set_xlabel(f"Time (s)   (seeds {seed_start}…{seed_end}, n={n_runs})")
    ax.set_ylabel(r"$\Delta_t = k\sqrt{\mathrm{tr}(P_{xy})}$")
    ax.set_title(
        f"Safety margin vs time  ({dataset_name}/{predictor_name},  k={k_sigma})\n"
        "Shaded: ±1 std across seeds",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_compare_violin_metrics_plot(
    *,
    dataset_name: str,
    predictor_name: str,
    output_dir: str,
    t_begin: int,
    init_x: float,
    init_y: float,
    goal_x: float,
    goal_y: float,
    seed_start: int,
    n_runs: int,
    ekf_runs: List[Dict[str, Any]],
    ekf_oracle_runs: List[Dict[str, Any]],
    dr_runs: List[Dict[str, Any]],
) -> Optional[str]:
    """Violin plots for scenario collision rate (binary per run) and total cost (three estimators)."""
    def collect_metric(runs: List[Dict[str, Any]], key: str) -> List[float]:
        vals: List[float] = []
        for r in runs:
            if "error" in r:
                continue
            v = r.get(key, None)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                vals.append(fv)
        return vals

    cr_e = collect_metric(ekf_runs, "collision_rate")
    cr_o = collect_metric(ekf_oracle_runs, "collision_rate")
    cr_d = collect_metric(dr_runs, "collision_rate")
    tc_e = collect_metric(ekf_runs, "total_cost")
    tc_o = collect_metric(ekf_oracle_runs, "total_cost")
    tc_d = collect_metric(dr_runs, "total_cost")
    if not (cr_e and cr_o and cr_d and tc_e and tc_o and tc_d):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    png_name = (
        f"{dataset_name}_{predictor_name}_compare_violin_metrics_t{t_begin}_"
        f"init{init_x:.1f}_{init_y:.1f}_goal{goal_x:.1f}_{goal_y:.1f}_seed{seed_start}.png"
    )
    plot_path = os.path.join(output_dir, png_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["EKF", "EKF-true", "DR-EKF"]
    colors = ["red", "green", "blue"]

    data_left = [np.asarray(cr_e) * 100.0, np.asarray(cr_o) * 100.0, np.asarray(cr_d) * 100.0]
    parts = axes[0].violinplot(data_left, showmeans=True, showmedians=False, showextrema=True)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[i])
        b.set_alpha(0.28)
        b.set_edgecolor(colors[i])
    axes[0].set_xticks([1, 2, 3], labels)
    axes[0].set_ylabel("Scenario collision rate (%)")
    axes[0].set_title("Scenario collision (1 if any collision)")
    axes[0].grid(True, axis="y", alpha=0.3)

    data_right = [np.asarray(tc_e), np.asarray(tc_o), np.asarray(tc_d)]
    parts = axes[1].violinplot(data_right, showmeans=True, showmedians=False, showextrema=True)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[i])
        b.set_alpha(0.28)
        b.set_edgecolor(colors[i])
    axes[1].set_xticks([1, 2, 3], labels)
    axes[1].set_ylabel("Total cost")
    axes[1].set_title("Total cost")
    axes[1].grid(True, axis="y", alpha=0.3)

    seed_end = seed_start + n_runs - 1
    fig.suptitle(
        f"Violin comparison ({dataset_name}/{predictor_name}, seeds {seed_start}…{seed_end})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def advance_env_with_true_pose(env: Environment, x_true: np.ndarray) -> Tuple[Any, bool, bool, dict]:
    """Advance dataset timestep and buffers; set ego pose to x_true (no extra kinematics)."""
    xt = np.asarray(x_true, dtype=float).reshape(3)
    env._x = float(xt[0])
    env._y = float(xt[1])
    env._th = float(xt[2])
    env._step += 1
    env._update_buffers()
    info = env._get_side_info()
    terminated = bool(info["goal_reached"])
    truncated = env._step > env._final_step
    return env._get_obs(), terminated, truncated, info


def run_navigation(
    dataset_name: str,
    predictor_name: str,
    estimator_name: str,
    init_robot_state: dict,
    goal_pos: np.ndarray,
    t_begin: int,
    t_end: int,
    prediction_horizon: int,
    history_len: int,
    seed: int,
    beacons: np.ndarray,
    sigma_px_true: float,
    sigma_py_true: float,
    sigma_theta_true: float,
    sigma_r_true: str,
    sigma_theta_meas_true: float,
    nom_q_var_mult: List[float],
    nom_r_var_mult: List[float],
    k_sigma: float,
    theta_w: float,
    theta_v: float,
    theta_x0: Optional[float],
    L_f: float,
    L_h: float,
    eta_scale: float,
    theta_eff_cap: Optional[float],
    verbose: bool,
    ignore_goal: bool = False,
    theta_eff_cli: Optional[float] = None,
    record_video: bool = False,
) -> Dict[str, Any]:
    est_key = estimator_name.lower()
    use_oracle_nominal = est_key in ("ekf-oracle", "ekf_oracle", "oracle-ekf", "oracle_ekf")
    estimator_backend_name = "ekf" if use_oracle_nominal else estimator_name

    rng = np.random.default_rng(seed)
    dataset = RegisteredDatasets[dataset_name]
    temp_dir = tempfile.mkdtemp(prefix="canvas_nav_uwb_")

    env = Environment(
        dataset=dataset,
        init_robot_state=init_robot_state,
        goal_pos=goal_pos,
        t_begin=t_begin,
        t_end=t_end,
        history_len=history_len,
        prediction_horizon=prediction_horizon,
        path_to_frames="",
        path_to_save=temp_dir,
    )

    # Predictor (unchanged: only non-ego)
    try:
        if predictor_name == "koopyadapt":
            predictor = Predictors(
                chosen_predictor="koopyadapt",
                prediction_len=prediction_horizon,
                history_len=history_len,
                dt=env.dt,
                dataset=dataset_name,
                device="cpu",
                koopyadapt_eta=0.0,
                koopyadapt_radius=1.0,
                koopyadapt_num_sectors=12,
            )
        elif predictor_name == "trajectron":
            predictor = Predictors(
                chosen_predictor="trajectron",
                prediction_len=prediction_horizon,
                history_len=history_len,
                dt=env.dt,
                dataset=dataset_name,
                device="cpu",
                random_seed=seed,
            )
        else:
            predictor = Predictors(
                chosen_predictor=predictor_name,
                prediction_len=prediction_horizon,
                history_len=history_len,
                dt=env.dt,
                dataset=dataset_name,
                device="cpu",
            )
    except Exception as e:
        return {"error": f"predictor_init: {e}"}

    dt = float(env.dt)
    ROBOT_RAD = 0.4
    d_min_base = ROBOT_RAD + 0.1 / np.sqrt(2.0)
    # Collision metric (static + pedestrian): closer than this counts as collision.
    collision_threshold_m = 1.0

    controller: Any = BaseMPC(
        prediction_horizon=prediction_horizon,
        dt=dt,
        goal=env.goal,
        d_min=d_min_base,
        geometry=env.geometry,
        use_ipopt=False,
    )

    # True covariances (fixed first); nominal = per-axis variance multipliers * true
    try:
        sig_r_list = _parse_n_floats(sigma_r_true, 3, "sigma_r_true")
    except ValueError as e:
        return {"error": str(e)}
    if len(nom_q_var_mult) != 3:
        return {"error": f"nom_q_var_mult must have length 3, got {nom_q_var_mult}"}
    if len(nom_r_var_mult) != 4:
        return {"error": f"nom_r_var_mult must have length 4, got {nom_r_var_mult}"}

    # Explicit diagonal model:
    #   Q_true = diag([sigma_px_true^2, sigma_py_true^2, sigma_theta_true^2])
    #   R_true = diag([sigma_r1_true^2, sigma_r2_true^2, sigma_r3_true^2, sigma_heading_true^2])
    #   Q_nom  = diag(q_var_mult * diag(Q_true)),  R_nom = diag(r_var_mult * diag(R_true))
    process_std_true = np.array([sigma_px_true, sigma_py_true, sigma_theta_true], dtype=float)
    meas_std_true = np.array(
        [sig_r_list[0], sig_r_list[1], sig_r_list[2], sigma_theta_meas_true],
        dtype=float,
    )
    q_true_var = process_std_true**2
    r_true_var = meas_std_true**2
    q_nom_var_mult = np.asarray(nom_q_var_mult, dtype=float)
    r_nom_var_mult = np.asarray(nom_r_var_mult, dtype=float)
    if use_oracle_nominal:
        q_nom_var = q_true_var.copy()
        r_nom_var = r_true_var.copy()
    else:
        q_nom_var = q_nom_var_mult * q_true_var
        r_nom_var = r_nom_var_mult * r_true_var

    Q_true = np.diag(q_true_var).astype(float)
    R_true = np.diag(r_true_var).astype(float)
    Q_nom = np.diag(q_nom_var).astype(float)
    R_nom = np.diag(r_nom_var).astype(float)

    w2_Q = w2_gaussian_diagonal(np.zeros(3), np.diag(Q_nom), np.zeros(3), np.diag(Q_true))
    w2_R = w2_gaussian_diagonal(np.zeros(4), np.diag(R_nom), np.zeros(4), np.diag(R_true))

    x0 = np.array(
        [
            init_robot_state["position_x"],
            init_robot_state["position_y"],
            init_robot_state["orientation_z"],
        ],
        dtype=float,
    ).reshape(3, 1)

    nominal_x0_mean = x0.copy()
    nominal_x0_cov = np.diag(np.diag(Q_nom)).astype(float)
    true_x0_mean = x0.copy()
    true_x0_cov = np.diag(np.diag(Q_true)).astype(float)

    est = make_uwb_estimator(
        estimator_backend_name,
        dt,
        beacons,
        nominal_x0_mean,
        nominal_x0_cov,
        Q_nom,
        R_nom,
        true_x0_mean,
        true_x0_cov,
        Q_true,
        R_true,
        theta_w=theta_w,
        theta_v=theta_v,
        theta_x0=theta_x0,
        L_f=L_f,
        L_h=L_h,
        eta_scale=eta_scale,
        theta_eff_cap=theta_eff_cap,
    )

    obs, _ = env.reset()
    x_true = np.array(
        [env._x, env._y, env._th], dtype=float
    ).reshape(3, 1)

    y0 = h_uwb(x_true, beacons) + sample_measurement_noise(R_true, rng)
    try:
        x_hat = est._initial_update(est.nominal_x0_mean, y0)
    except Exception as e:
        return {
            "error": f"estimator_initial_update: {e}",
            "w2_process_nom_vs_true": w2_Q,
            "w2_meas_nom_vs_true": w2_R,
        }

    theta_eff_initial = float("nan")
    if estimator_backend_name.lower() in (
        "dr-ekf",
        "dr_ekf",
        "drekf",
        "dr-ekf-trace",
        "dr_ekf_trace",
    ):
        theta_eff_initial = float(getattr(est, "_last_theta_eps_effective", np.nan))

    mse_list: List[float] = []
    theta_eff_list: List[float] = []
    d_eff_list: List[float] = []
    delta_margin_list: List[float] = []
    collision_count = 0
    first_ped_collision_index: Optional[int] = None
    first_ped_collision_agent_id: Optional[int] = None
    ped_collision_agent_ids: set = set()
    total_steps = 0
    total_cost = 0.0

    truncated = False
    frame = 0
    terminated = False
    video_frames: List[Dict[str, Any]] = []
    robot_traj_vid: List[List[float]] = []
    est_traj_vid: List[List[float]] = []
    # One dict per control step: agent_id -> [x, y] for everyone in ``obs['non-ego']`` that step
    ped_per_step_list: List[Dict[int, List[float]]] = []

    def _push_video_frame(x_hat_vec: np.ndarray) -> None:
        xh = np.asarray(x_hat_vec, dtype=float).reshape(3)
        est_traj_vid.append([float(xh[0]), float(xh[1])])
        if not record_video:
            return
        ground_truth: Dict[str, np.ndarray] = {}
        try:
            ground_truth = {
                k: v.copy()
                for k, v in env._dataset.get_future(
                    timestep=env._step,
                    future_length=prediction_horizon,
                    history_length=history_len,
                ).items()
            }
        except Exception:
            ground_truth = {}
        pred_copy = prediction_res.copy() if prediction_res else None
        video_frames.append(
            {
                "robot_pos": np.array([env._x, env._y], dtype=float),
                "robot_trajectory": np.array(robot_traj_vid, dtype=float),
                "estimated_pos": np.array([xh[0], xh[1]], dtype=float),
                "estimated_trajectory": np.array(est_traj_vid, dtype=float),
                "non_ego_agents": {
                    k: v[-1] if len(v) > 0 else None
                    for k, v in obs.get("non-ego", {}).items()
                },
                "prediction": pred_copy,
                "ground_truth": ground_truth,
                "goal": env.goal.copy(),
                "frame": frame,
            }
        )

    while not truncated:
        obs_ctrl = dict(obs)
        obs_ctrl["ego"] = ego_dict_from_vec(x_hat)

        try:
            prediction_res = predictor(obs_ctrl["non-ego"])
        except Exception as e:
            if verbose:
                print(f"predictor error at {frame}: {e}")
            break

        P_pos = est._P[:2, :2].copy() if est._P is not None else np.eye(2) * 1e-4
        trace_P = float(np.trace(P_pos))
        delta_t = k_sigma * np.sqrt(max(trace_P, 0.0))
        d_eff = d_min_base + delta_t
        d_eff_list.append(d_eff)
        delta_margin_list.append(delta_t)

        controller._d_min = d_eff

        try:
            u, _ = controller(obs_ctrl, prediction_res)
        except Exception as e:
            if verbose:
                print(f"controller error at {frame}: {e}")
            break

        u = np.asarray(u, dtype=float).reshape(2, 1)
        w = sample_process_noise(Q_true, rng)
        x_true = unicycle_f(x_true, u, dt) + w

        obs, terminated, truncated, _ = advance_env_with_true_pose(env, x_true)
        robot_traj_vid.append([float(env._x), float(env._y)])

        step_ped: Dict[int, List[float]] = {}
        for aid, hist in (obs.get("non-ego") or {}).items():
            aid_i = int(aid)
            if len(hist) > 0:
                p = np.asarray(hist[-1][:2], dtype=float)
                if np.isfinite(p).all():
                    step_ped[aid_i] = [float(p[0]), float(p[1])]
        ped_per_step_list.append(step_ped)

        ego_gt = {"position_x": env._x, "position_y": env._y, "orientation_z": env._th}
        cur = np.array([ego_gt["position_x"], ego_gt["position_y"]], dtype=float)
        goal = env.goal
        reached_goal = bool(
            abs(env._x - float(goal[0])) < GOAL_SUCCESS_TOL
            and abs(env._y - float(goal[1])) < GOAL_SUCCESS_TOL
        )
        intermediate_cost = float(np.sum((cur - goal) ** 2))
        collision_cost = 0.0
        step_collision = False
        if hasattr(env, "geometry"):
            d_static = env.geometry.distance_from(points=cur.reshape(1, -1))[0]
            if d_static <= collision_threshold_m:
                collision_cost += 1e3
                step_collision = True
        ped_collision_this_step = False
        if not step_collision and obs.get("non-ego"):
            for aid, hist in obs["non-ego"].items():
                if len(hist) > 0:
                    ap = hist[-1]
                    if np.linalg.norm(cur - ap) <= collision_threshold_m:
                        collision_cost += 1e3
                        step_collision = True
                        ped_collision_this_step = True
                        try:
                            ped_collision_agent_ids.add(int(aid))
                        except Exception:
                            pass
                        if first_ped_collision_agent_id is None:
                            try:
                                first_ped_collision_agent_id = int(aid)
                            except Exception:
                                first_ped_collision_agent_id = None
                        break
        if step_collision:
            collision_count += 1
            if ped_collision_this_step and first_ped_collision_index is None:
                first_ped_collision_index = len(robot_traj_vid) - 1
        total_steps += 1
        total_cost += intermediate_cost + collision_cost

        if (terminated or reached_goal) and not ignore_goal:
            _push_video_frame(x_hat)
            break

        y = h_uwb(x_true, beacons) + sample_measurement_noise(R_true, rng)
        t_idx = frame + 1
        try:
            x_hat = est.update_step(x_hat, y, t_idx, u)
        except Exception as e:
            if verbose:
                print(f"estimator update failed at {t_idx}: {e}")
            break

        mse_list.append(float(np.sum((x_hat.reshape(3) - x_true.reshape(3)) ** 2)))
        if estimator_backend_name.lower() in (
            "dr-ekf",
            "dr_ekf",
            "drekf",
            "dr-ekf-trace",
            "dr_ekf_trace",
        ):
            teff = getattr(est, "_last_theta_eps_effective", np.nan)
            theta_eff_list.append(float(teff))

        _push_video_frame(x_hat)
        frame += 1
        if frame > 2000:
            break

    if len(mse_list) > 0:
        final_pos = np.array([env._x, env._y], dtype=float)
        total_cost += 10.0 * float(np.sum((final_pos - env.goal) ** 2))

    g = env.goal
    success_goal = bool(
        abs(env._x - g[0]) < GOAL_SUCCESS_TOL and abs(env._y - g[1]) < GOAL_SUCCESS_TOL
    )

    # Per-scenario binary: 1 if any collision step, else 0 (mean over runs = fraction of scenarios with collision).
    collision_rate = 1.0 if collision_count > 0 else 0.0
    collision_step_rate = (
        collision_count / total_steps if total_steps > 0 else 0.0
    )
    mse_mean = float(np.mean(mse_list)) if mse_list else float("nan")

    all_ped_ids: List[int] = []
    seen_pid: set = set()
    for d in ped_per_step_list:
        for k in d.keys():
            if k not in seen_pid:
                seen_pid.add(k)
                all_ped_ids.append(int(k))
    all_ped_ids.sort()
    ped_gt_by_id: Dict[int, List[List[float]]] = {}
    for aid in all_ped_ids:
        row: List[List[float]] = []
        for d in ped_per_step_list:
            row.append(d.get(aid, [float("nan"), float("nan")]))
        ped_gt_by_id[aid] = row
    legacy_ped_traj: List[List[float]] = []
    if all_ped_ids:
        legacy_ped_traj = ped_gt_by_id[all_ped_ids[0]]

    out: Dict[str, Any] = {
        "w2_process_nom_vs_true": w2_Q,
        "w2_meas_nom_vs_true": w2_R,
        "q_true_variance_diag": np.diag(Q_true).tolist(),
        "q_nom_variance_diag": np.diag(Q_nom).tolist(),
        "r_true_variance_diag": np.diag(R_true).tolist(),
        "r_nom_variance_diag": np.diag(R_nom).tolist(),
        "theta_eff_cli": theta_eff_cli,
        "dr_static_ambiguity": {
            "theta_w": float(theta_w),
            "theta_v": float(theta_v),
            "theta_x0": float(theta_x0) if theta_x0 is not None else float(theta_w),
        },
        "mse_mean": mse_mean,
        "mse_per_step": mse_list,
        "collision_rate": collision_rate,
        "collision_step_rate": collision_step_rate,
        "collision_count": collision_count,
        "first_ped_collision_index": first_ped_collision_index,
        "first_ped_collision_agent_id": first_ped_collision_agent_id,
        "ped_collision_agent_ids": sorted([int(x) for x in ped_collision_agent_ids]),
        "total_steps": total_steps,
        "total_cost": total_cost,
        "success": success_goal,
        "ignore_goal": ignore_goal,
        "frames": frame,
        "d_eff_series": d_eff_list,
        "delta_margin_series": delta_margin_list,
        "theta_eff_series": theta_eff_list,
        "theta_eff_initial": theta_eff_initial,
        "theta_eff_mean": float(np.nanmean(theta_eff_list)) if theta_eff_list else float("nan"),
        "theta_eff_max": float(np.nanmax(theta_eff_list)) if theta_eff_list else float("nan"),
        "robot_true_trajectory": np.asarray(robot_traj_vid, dtype=float).tolist(),
        "estimated_trajectory": np.asarray(est_traj_vid, dtype=float).tolist(),
        "pedestrian_gt_trajectory": legacy_ped_traj,
        "pedestrian_gt_trajectories_by_id": {str(k): v for k, v in ped_gt_by_id.items()},
        "pedestrian_agent_ids": all_ped_ids,
        "pedestrian_agent_id": int(all_ped_ids[0]) if all_ped_ids else None,
        "t_begin": int(t_begin),
        "dt": dt,
        "video_frames": video_frames if record_video else None,
    }
    return out


def _index_for_dataset_timestep(t_dataset: int, t_begin: int) -> int:
    """Index into per-step arrays where point ``i`` corresponds to ``env._step == t_begin + 1 + i``."""
    return int(t_dataset - t_begin - 1)


def _format_float_for_key(x: float) -> str:
    return f"{float(x):.3f}".replace("-", "m").replace(".", "p")


def _seed_cache_path(args: argparse.Namespace, seed: int, cache_dir: str) -> str:
    init_parts = [float(x) for x in str(args.init).split(",")]
    goal_parts = [float(x) for x in str(args.goal).split(",")]
    tag = (
        f"{args.dataset}_{args.predictor}_seed{int(seed)}"
        f"_tb{int(args.t_begin)}_te{int(args.t_end)}"
        f"_init{_format_float_for_key(init_parts[0])}_{_format_float_for_key(init_parts[1])}"
        f"_goal{_format_float_for_key(goal_parts[0])}_{_format_float_for_key(goal_parts[1])}"
    )
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", tag)
    return os.path.join(cache_dir, f"{tag}.json")


def draw_navigation_trajectories_fade_on_ax(
    ax: Any,
    *,
    dataset_name: str,
    predictor_name: str,
    pedestrian_xy_by_id: Dict[int, np.ndarray],
    robot_runs: List[
        Tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Optional[int],
            Optional[int],
            Optional[int],
        ]
    ],
    t_begin: int,
    t_scatter_min: int = 8,
    t_scatter_max: int = 20,
    alpha_min: float = 0.15,
    alpha_max: float = 1.0,
    start_xy: Optional[Tuple[float, float]] = None,
    goal_xy: Optional[Tuple[float, float]] = None,
    legend_bbox_y: float = -0.14,
) -> None:
    """Draw the fade trajectory map on an existing axes (used standalone or in a composite figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as mpe
    import matplotlib.pyplot as plt
    import os
    from matplotlib.collections import LineCollection
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, MultipleLocator

    # Optional background image (from CANVAS dataset spec). If the asset is missing locally,
    # we silently continue without a background.
    try:
        from canvas.datasets.dataset_loader import get_dataset_spec, _load_background_image  # type: ignore

        spec = get_dataset_spec(str(dataset_name))
        bg_path = getattr(getattr(spec, "bg", None), "path", None)
        bg_extent = getattr(getattr(spec, "bg", None), "extent", None)
        bg_rotate90 = bool(getattr(getattr(spec, "bg", None), "rotate90", False))
        # Keep background subtle so trajectories stand out.
        bg_alpha = min(float(getattr(getattr(spec, "bg", None), "alpha", 0.6)), 0.35)
        bg_extent_used = None
        bg_path_try = None
        if bg_path is not None:
            p0 = str(bg_path)
            if os.path.exists(p0):
                bg_path_try = p0
            else:
                # Common local asset mismatch: spec points to .jpg but repository has .png (or vice versa).
                stem, _ext = os.path.splitext(p0)
                for alt_ext in (".png", ".jpg", ".jpeg"):
                    p_alt = stem + alt_ext
                    if os.path.exists(p_alt):
                        bg_path_try = p_alt
                        break

        if bg_path_try is not None and bg_extent is not None:
            img = _load_background_image(bg_path_try, rotate90=bg_rotate90)
            bg_extent_used = tuple(bg_extent)
            ax.imshow(
                img,
                extent=bg_extent,
                origin="upper",
                alpha=bg_alpha,
                zorder=0,
            )
    except Exception:
        pass

    # Keep typography aligned with ``plot_safe_navigation_uwb.py``.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Use custom trajectory colormaps aligned with the delta-panel line colors.
    cmap_ekf = mcolors.LinearSegmentedColormap.from_list(
        "ekf_custom", ["#fcbba1", "#de2d26"]
    )
    cmap_oracle = mcolors.LinearSegmentedColormap.from_list(
        "oracle_custom", ["#c7e9c0", "#31a354"]
    )
    cmap_dr = mcolors.LinearSegmentedColormap.from_list(
        "dr_custom", ["#c6dbef", "#3182bd"]
    )
    t_clip = (0.30, 0.95)

    def _tcol(cmap, t: float):
        u = float(np.clip(t, 0.0, 1.0))
        return cmap(t_clip[0] + u * (t_clip[1] - t_clip[0]))

    def _make_lc(
        x: np.ndarray,
        y: np.ndarray,
        cmap,
        lw: float,
        alpha: float,
        zorder: int,
    ) -> Optional[LineCollection]:
        if len(x) < 2:
            return None
        tn = np.linspace(0, 1, len(x))
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = [_tcol(cmap, float(ti)) for ti in tn[:-1]]
        lc = LineCollection(
            segs,
            colors=cols,
            linewidths=lw,
            alpha=alpha,
            zorder=zorder,
            capstyle="round",
            joinstyle="round",
            antialiaseds=True,
        )
        return lc

    def _add_ped_arrows(
        ax_,
        x,
        y,
        color: str,
        *,
        every: int = 8,
        zorder: int = 7,
        alpha: float = 1.0,
    ) -> None:
        """Direction arrows on the path (tail and head stay on the same tangent segment)."""
        rgba = mcolors.to_rgba(color, alpha)
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        for i in range(every, len(x) - every, every):
            dx = x[i + 1] - x[i - 1]
            dy = y[i + 1] - y[i - 1]
            ax_.annotate(
                "",
                xy=(x[i] + 0.3 * dx, y[i] + 0.3 * dy),
                xytext=(x[i], y[i]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=rgba,
                    lw=1.4,
                    mutation_scale=10,
                ),
                zorder=zorder,
            )

    def _pedestrian_tube_polygon(
        x: np.ndarray, y: np.ndarray, half_width: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Closed polygon: offset polyline left/right of center path (flow tube)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = int(x.size)
        if n < 2:
            return np.array([]), np.array([])
        tx = np.zeros(n)
        ty = np.zeros(n)
        for i in range(n):
            if i == 0:
                dx = x[1] - x[0]
                dy = y[1] - y[0]
            elif i == n - 1:
                dx = x[-1] - x[-2]
                dy = y[-1] - y[-2]
            else:
                dx = x[i + 1] - x[i - 1]
                dy = y[i + 1] - y[i - 1]
            norm = float(np.hypot(dx, dy)) + 1e-12
            tx[i] = dx / norm
            ty[i] = dy / norm
        nx = -ty
        ny = tx
        xl = x + nx * half_width
        yl = y + ny * half_width
        xr = x - nx * half_width
        yr = y - ny * half_width
        px = np.concatenate([xl, xr[::-1]])
        py = np.concatenate([yl, yr[::-1]])
        return px, py

    def _single_ped_flow_band_xy(
        xy_paths: List[np.ndarray], half_width: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One band enclosing all pedestrian tubes: union of per-path tubes → convex hull."""
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
        except ImportError:
            Polygon = None  # type: ignore
            unary_union = None  # type: ignore

        polys: List[Any] = []
        boundary_pts: List[np.ndarray] = []
        for xy_f in xy_paths:
            if xy_f.shape[0] < 2:
                continue
            tx, ty = _pedestrian_tube_polygon(xy_f[:, 0], xy_f[:, 1], half_width)
            if tx.size < 4:
                continue
            boundary_pts.append(np.column_stack([tx, ty]))
            if Polygon is not None:
                p = Polygon(np.column_stack([tx, ty]))
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty:
                    polys.append(p)

        if Polygon is not None and unary_union is not None and polys:
            u = unary_union(polys)
            envelope = u.convex_hull
            if envelope.is_empty:
                return np.array([]), np.array([])
            xs, ys = envelope.exterior.xy
            return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

        # Fallback: convex hull of all tube boundary vertices (no shapely)
        if not boundary_pts:
            return np.array([]), np.array([])
        pts = np.vstack(boundary_pts)
        if pts.shape[0] < 3:
            return np.array([]), np.array([])
        try:
            from scipy.spatial import ConvexHull

            h = ConvexHull(pts)
            b = pts[h.vertices]
            # scipy order; close polygon for fill
            return b[:, 0], b[:, 1]
        except Exception:
            return np.array([]), np.array([])

    if not robot_runs:
        raise ValueError(
            "robot_runs must list at least one (seed, ekf_path, ekf_oracle_path, dr_path)."
        )

    c_ped = "#9467bd"  # unified purple for all pedestrian GT paths
    ped_tube_half_w = 0.26  # m (total width 2 * half_w); flow corridor
    ped_tube_fill_alpha = 0.2
    ped_tube_edge_alpha = 0.35
    ped_tube_z = 0.5

    all_xy: List[np.ndarray] = []
    for _sd, e_ekf, e_or, e_dr, _ekf_ci, _or_ci, _dr_ci in robot_runs:
        all_xy.append(np.asarray(e_ekf, dtype=float))
        all_xy.append(np.asarray(e_or, dtype=float))
        all_xy.append(np.asarray(e_dr, dtype=float))
    for _pid, arr in sorted(pedestrian_xy_by_id.items()):
        all_xy.append(np.asarray(arr, dtype=float))
    finite_pts = []
    for a in all_xy:
        if a.size == 0:
            continue
        m = np.isfinite(a).all(axis=1)
        if np.any(m):
            finite_pts.append(a[m])
    if start_xy is not None:
        finite_pts.append(np.asarray(start_xy, dtype=float).reshape(1, 2))
    if goal_xy is not None:
        finite_pts.append(np.asarray(goal_xy, dtype=float).reshape(1, 2))
    if not finite_pts:
        raise ValueError("No finite trajectory points to plot.")
    _ = np.vstack(finite_pts)

    runs_sorted = sorted(robot_runs, key=lambda t: int(t[0]))

    for j, (_sd, e_ekf, e_or, e_dr, ekf_ci, or_ci, dr_ci) in enumerate(runs_sorted):
        e_ekf_a = np.asarray(e_ekf, dtype=float)
        e_or_a = np.asarray(e_or, dtype=float)
        e_dr_a = np.asarray(e_dr, dtype=float)
        for xy_full, cmap, ci in (
            (e_ekf_a, cmap_ekf, ekf_ci),
            (e_or_a, cmap_oracle, or_ci),
            (e_dr_a, cmap_dr, dr_ci),
        ):
            if xy_full.size == 0:
                continue
            plot_end = len(xy_full)
            if ci is not None:
                plot_end = min(plot_end, int(ci) + 1)
            if plot_end <= 0:
                continue
            xy_vis = xy_full[:plot_end]
            m = np.isfinite(xy_vis).all(axis=1)
            xy_f = xy_vis[m]
            if xy_f.shape[0] < 2:
                continue
            lc = _make_lc(
                xy_f[:, 0],
                xy_f[:, 1],
                cmap,
                lw=2.4,
                alpha=0.9,
                zorder=4,
            )
            if lc is not None:
                ax.add_collection(lc)

    c_collision = "black"
    ped_ids_sorted = sorted(pedestrian_xy_by_id.keys())
    ped_lw = 1.4
    ped_alpha = 0.78
    ped_xy_paths: List[np.ndarray] = []
    for pid in ped_ids_sorted:
        xy_p = np.asarray(pedestrian_xy_by_id[pid], dtype=float)
        m = np.isfinite(xy_p).all(axis=1)
        xy_f = xy_p[m]
        if xy_f.shape[0] >= 2:
            ped_xy_paths.append(xy_f)

    for xy_f in ped_xy_paths:
        ax.plot(
            xy_f[:, 0],
            xy_f[:, 1],
            color=c_ped,
            lw=ped_lw,
            linestyle="-",
            alpha=ped_alpha,
            zorder=1,
        )
        _add_ped_arrows(
            ax,
            xy_f[:, 0],
            xy_f[:, 1],
            c_ped,
            every=8,
            zorder=1,
            alpha=ped_alpha,
        )

    for _sd, e_ekf, e_or, e_dr, ekf_ci, or_ci, dr_ci in runs_sorted:
        e_ekf_a = np.asarray(e_ekf, dtype=float)
        e_or_a = np.asarray(e_or, dtype=float)
        e_dr_a = np.asarray(e_dr, dtype=float)
        if (
            ekf_ci is not None
            and 0 <= int(ekf_ci) < len(e_ekf_a)
            and np.isfinite(e_ekf_a[int(ekf_ci)]).all()
        ):
            cp = e_ekf_a[int(ekf_ci)]
            ax.scatter(
                cp[0],
                cp[1],
                marker="x",
                s=110,
                c=c_collision,
                linewidths=1.8,
                zorder=9,
            )
        if (
            or_ci is not None
            and 0 <= int(or_ci) < len(e_or_a)
            and np.isfinite(e_or_a[int(or_ci)]).all()
        ):
            cp = e_or_a[int(or_ci)]
            ax.scatter(
                cp[0],
                cp[1],
                marker="x",
                s=110,
                c=c_collision,
                linewidths=1.8,
                zorder=9,
            )
        if (
            dr_ci is not None
            and 0 <= int(dr_ci) < len(e_dr_a)
            and np.isfinite(e_dr_a[int(dr_ci)]).all()
        ):
            cp = e_dr_a[int(dr_ci)]
            ax.scatter(
                cp[0],
                cp[1],
                marker="x",
                s=110,
                c=c_collision,
                linewidths=1.8,
                zorder=9,
            )

    if start_xy is not None:
        sx, sy = float(start_xy[0]), float(start_xy[1])
        (ln_s,) = ax.plot(
            sx,
            sy,
            "^",
            color="k",
            markersize=7.0,
            zorder=12,
        )
        ln_s.set_path_effects([mpe.withStroke(linewidth=2.5, foreground="white")])
        ax.text(
            sx - 0.92,
            sy - 0.55,
            "Start",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=13,
            path_effects=[mpe.withStroke(linewidth=2.5, foreground="white")],
        )
    if goal_xy is not None:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        (ln_g,) = ax.plot(
            gx,
            gy,
            "*",
            color="k",
            markersize=9.0,
            zorder=12,
        )
        ln_g.set_path_effects([mpe.withStroke(linewidth=2.5, foreground="white")])
        ax.text(
            gx,
            gy - 0.55,
            "Goal",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=13,
            path_effects=[mpe.withStroke(linewidth=2.5, foreground="white")],
        )

    dataset_key = str(dataset_name).lower()
    axis_limits_by_dataset: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
        # Hotel has a different world-coordinate range than Zara; keep explicit limits for readability.
        "hotel": ((-3.25, 6.35), (-10.31, 4.31)),
        "univ": ((-0.174686040989, 15.4369843957), (-0.222192273533, 13.8542013734)),
    }
    if dataset_key in axis_limits_by_dataset:
        xlim_cfg, ylim_cfg = axis_limits_by_dataset[dataset_key]
        ax.set_xlim(*xlim_cfg)
        ax.set_ylim(*ylim_cfg)
    elif "bg_extent_used" in locals() and bg_extent_used is not None:
        ax.set_xlim(float(bg_extent_used[0]), float(bg_extent_used[1]))
        ax.set_ylim(float(bg_extent_used[2]), float(bg_extent_used[3]))
    else:
        # Default Zara-style corridor (match ``plot_safe_navigation_uwb`` map tick density).
        ax.set_xlim(1.0, float(_NAV_FADE_DELTA_X_MAX_S))
        ax.set_ylim(3.0, 8.0)
    ax.set_xlabel(r"x (m)", fontsize=11)
    ax.set_ylabel(r"y (m)", fontsize=11)
    ax.set_aspect("equal")
    # Map panel: no grid (trajectory readability), ticks like nav_traj_margin_combined.
    ax.grid(False, which="major")
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    if dataset_key not in axis_limits_by_dataset and not (
        "bg_extent_used" in locals() and bg_extent_used is not None
    ):
        ax.xaxis.set_major_locator(FixedLocator([2.0, 5.0, 8.0, 11.0, 14.0]))
        ax.yaxis.set_major_locator(MultipleLocator(2.0))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(3.0))
        ax.yaxis.set_major_locator(MultipleLocator(2.0))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.0,
        labelsize=10,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3.5,
        width=0.6,
    )

    class _LegendGradientHandle:
        def __init__(self, cmap, label: str):
            self.cmap = cmap
            self._label = label

        def get_label(self) -> str:
            return self._label

    class _LegendGradientLineHandler(HandlerBase):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            nseg = 18
            xs = np.linspace(xdescent, xdescent + width, nseg + 1)
            y = ydescent + 0.5 * height
            segs = np.stack(
                [np.column_stack([xs[:-1], np.full(nseg, y)]), np.column_stack([xs[1:], np.full(nseg, y)])],
                axis=1,
            )
            ts = np.linspace(0.0, 1.0, nseg)
            cols = [_tcol(orig_handle.cmap, float(t)) for t in ts]
            lc = LineCollection(segs, colors=cols, linewidths=2.6, alpha=0.95, transform=trans)
            return [lc]

    def _legend_gradient_handle(cmap, label: str):
        return _LegendGradientHandle(cmap, label)

    legend_elements: List[Line2D] = []
    if ped_ids_sorted:
        legend_elements.append(
            Line2D([0], [0], color=c_ped, lw=ped_lw, alpha=ped_alpha, label="Pedestrians"),
        )
    legend_elements.extend(
        [
            _legend_gradient_handle(cmap_ekf, "EKF (nominal)"),
            _legend_gradient_handle(cmap_oracle, "EKF (true)"),
            _legend_gradient_handle(cmap_dr, "DR-EKF"),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="x",
                color=c_collision,
                markersize=8,
                markeredgewidth=1.8,
                label="Collision",
            ),
        ]
    )
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(1.4, legend_bbox_y),
        ncol=max(1, len(legend_elements)),
        frameon=False,
        handlelength=1.6,
        columnspacing=0.75,
        fontsize=11,
        handler_map={_LegendGradientHandle: _LegendGradientLineHandler()},
    )


def save_navigation_trajectories_fade_plot(
    *,
    dataset_name: str,
    predictor_name: str,
    pedestrian_xy_by_id: Dict[int, np.ndarray],
    robot_runs: List[
        Tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Optional[int],
            Optional[int],
            Optional[int],
        ]
    ],
    ekf_runs: List[Dict[str, Any]],
    ekf_oracle_runs: List[Dict[str, Any]],
    dr_runs: List[Dict[str, Any]],
    t_begin: int,
    output_path: str,
    t_scatter_min: int = 8,
    t_scatter_max: int = 20,
    alpha_min: float = 0.15,
    alpha_max: float = 1.0,
    start_xy: Optional[Tuple[float, float]] = None,
    goal_xy: Optional[Tuple[float, float]] = None,
) -> str:
    """One figure: left δ_t mean±std, right trajectories (writes ``output_path``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as mpe
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, MultipleLocator

    _navigation_publication_style()

    fig, (ax, ax_delta) = plt.subplots(
        1, 2, figsize=(14.0, 4.6), gridspec_kw={"width_ratios": [1.25, 0.75]}
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(right=0.98, bottom=0.26, wspace=-0.16)

    # Right panel: delta_t = kappa_sigma * sqrt(tr(P_pos)) with mean ± std shading
    # (same role as ``plot_safe_navigation_uwb`` margin time-series).
    m_e, s_e = _stack_margin_mean_std(ekf_runs)
    m_o, s_o = _stack_margin_mean_std(ekf_oracle_runs)
    m_d, s_d = _stack_margin_mean_std(dr_runs)
    if m_e is not None and m_o is not None and m_d is not None:
        ax_delta.set_facecolor("#FFFFFF")
        n = min(len(m_e), len(m_o), len(m_d))
        m_e, s_e = m_e[:n], s_e[:n]
        m_o, s_o = m_o[:n], s_o[:n]
        m_d, s_d = m_d[:n], s_d[:n]
        dt = 1.0
        for rr in (ekf_runs + ekf_oracle_runs + dr_runs):
            dt0 = float(rr.get("dt", 0.0) or 0.0)
            if dt0 > 0:
                dt = dt0
                break
        t = np.arange(n, dtype=float) * dt
        c_ekf = "#de2d26"
        c_or = "#31a354"
        c_dr = "#3182bd"
        hi_stack = []
        for mean, std, color, label in (
            (m_e, s_e, c_ekf, "EKF (nominal)"),
            (m_o, s_o, c_or, "EKF (true)"),
            (m_d, s_d, c_dr, "DR-EKF"),
        ):
            lo = np.maximum(mean - std, 0.0)
            hi = mean + std
            hi_stack.append(np.nanmax(hi))
            ax_delta.fill_between(t, lo, hi, color=color, alpha=0.20)
            ax_delta.plot(t, mean, color=color, linewidth=1.8, label=label, zorder=4)
        ymax_data = float(np.nanmax(hi_stack)) if hi_stack else 0.0
        ax_delta.set_xlabel("Time (s)", fontsize=11)
        ax_delta.set_ylabel(r"Safety margin $\delta_t$ (m)", fontsize=11)
        ax_delta.set_xlim(0.0, float(_NAV_FADE_DELTA_X_MAX_S))
        ax_delta.xaxis.set_major_locator(FixedLocator(list(_NAV_FADE_DELTA_X_TICKS)))
        ax_delta.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax_delta.yaxis.set_major_locator(FixedLocator(list(_NAV_FADE_DELTA_Y_TICKS)))
        ax_delta.set_ylim(0.0, max(0.95, ymax_data * 1.08))
        ax_delta.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax_delta.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=10,
            length=6,
            width=1.0,
        )
        ax_delta.tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3.5,
            width=0.6,
        )
        # Grid only on major ticks (x = 4,8,12 s; y = 0,0.3,0.6,0.9 m); no minor grid.
        ax_delta.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.85, color="#D5D9E0")
        ax_delta.grid(False, which="minor")
    else:
        ax_delta.text(0.5, 0.5, "No delta-margin data", ha="center", va="center", fontsize=10)
        ax_delta.set_axis_off()

    draw_navigation_trajectories_fade_on_ax(
        ax,
        dataset_name=dataset_name,
        predictor_name=predictor_name,
        pedestrian_xy_by_id=pedestrian_xy_by_id,
        robot_runs=robot_runs,
        t_begin=t_begin,
        t_scatter_min=t_scatter_min,
        t_scatter_max=t_scatter_max,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        start_xy=start_xy,
        goal_xy=goal_xy,
        legend_bbox_y=-0.16,
    )
    ax.text(
        0.025,
        0.965,
        "(a)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        zorder=20,
        color="black",
        path_effects=[mpe.withStroke(linewidth=2.5, foreground="white")],
    )
    ax_delta.text(
        0.025,
        0.965,
        "(b)",
        transform=ax_delta.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        zorder=20,
        color="black",
        path_effects=[mpe.withStroke(linewidth=2.5, foreground="white")],
    )
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_output_path = os.path.splitext(output_path)[0] + ".pdf"
    if os.path.abspath(pdf_output_path) != os.path.abspath(output_path):
        fig.savefig(pdf_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _series_mean(series: Optional[List[float]]) -> float:
    if not series:
        return float("nan")
    return float(np.mean(series))


def _series_tail_mean(series: Optional[List[float]], n: int = 5) -> float:
    if not series:
        return float("nan")
    tail = series[-n:]
    return float(np.mean(tail))


def print_run_header(
    dataset: str,
    predictor: str,
    seed: int,
    k_sigma: float,
    *,
    compare: bool = False,
    n_runs: int = 1,
) -> None:
    w = 78
    print("=" * w)
    mode = "EKF vs EKF-oracle vs DR-EKF" if compare else "single estimator"
    if n_runs > 1:
        seed_part = f"seeds {seed}…{seed + n_runs - 1} ({n_runs} runs)"
    else:
        seed_part = f"seed {seed}"
    print(
        f"  CANVAS UWB navigation ({mode})   "
        f"dataset={dataset}  predictor={predictor}  controller=mpc  {seed_part}"
    )
    print(f"  estimation margin: on (k={k_sigma})")
    print("=" * w)


def print_comparison_table(
    ekf: Dict[str, Any], ekf_oracle: Dict[str, Any], dr: Dict[str, Any]
) -> None:
    """Side-by-side EKF / EKF-oracle / DR-EKF (fixed-width, no JSON)."""
    w_lab = 34
    w_c = 17

    def row(label: str, a: str, b: str, c: str) -> None:
        print(f"  {label:<{w_lab}}  {a:^{w_c}}  {b:^{w_c}}  {c:^{w_c}}")

    print("-" * 96)
    print(f"  {'Metric':<{w_lab}}  {'EKF':^{w_c}}  {'EKF-oracle':^{w_c}}  {'DR-EKF':^{w_c}}")
    print("-" * 96)
    row(
        "W₂ process (nom vs true)",
        f"{ekf['w2_process_nom_vs_true']:.6f}",
        f"{ekf_oracle['w2_process_nom_vs_true']:.6f}",
        f"{dr['w2_process_nom_vs_true']:.6f}",
    )
    row(
        "W₂ measurement (nom vs true)",
        f"{ekf['w2_meas_nom_vs_true']:.6f}",
        f"{ekf_oracle['w2_meas_nom_vs_true']:.6f}",
        f"{dr['w2_meas_nom_vs_true']:.6f}",
    )
    print("-" * 96)
    row(
        "MSE mean ‖x̂ − x‖²",
        f"{ekf['mse_mean']:.6f}",
        f"{ekf_oracle['mse_mean']:.6f}",
        f"{dr['mse_mean']:.6f}",
    )
    row(
        "Scenario collision rate",
        f"{100.0 * float(ekf['collision_rate']):.2f}%",
        f"{100.0 * float(ekf_oracle['collision_rate']):.2f}%",
        f"{100.0 * float(dr['collision_rate']):.2f}%",
    )
    row(
        "Collision steps / total steps",
        f"{ekf['collision_count']}/{ekf['total_steps']}",
        f"{ekf_oracle['collision_count']}/{ekf_oracle['total_steps']}",
        f"{dr['collision_count']}/{dr['total_steps']}",
    )
    row(
        "Total cost",
        f"{ekf['total_cost']:.2f}",
        f"{ekf_oracle['total_cost']:.2f}",
        f"{dr['total_cost']:.2f}",
    )
    row(
        "Goal success",
        "yes" if ekf["success"] else "no",
        "yes" if ekf_oracle["success"] else "no",
        "yes" if dr["success"] else "no",
    )
    row("Frames", f"{ekf['frames']}", f"{ekf_oracle['frames']}", f"{dr['frames']}")
    print("-" * 96)
    dr_t0 = dr.get("theta_eff_initial", float("nan"))
    t0_str = f"{dr_t0:.6f}" if not np.isnan(dr_t0) else "—"
    row("θ_eff (initial)", "—", "—", t0_str)
    dr_m, dr_x = dr.get("theta_eff_mean"), dr.get("theta_eff_max")
    if dr_m is not None and not np.isnan(dr_m):
        row("θ_eff mean / max", "—", "—", f"{dr_m:.4f} / {dr_x:.4f}")
    else:
        row("θ_eff mean / max", "—", "—", "—")
    row(
        "Δ margin mean (k√tr P)",
        f"{_series_mean(ekf.get('delta_margin_series')):.6f}",
        f"{_series_mean(ekf_oracle.get('delta_margin_series')):.6f}",
        f"{_series_mean(dr.get('delta_margin_series')):.6f}",
    )
    row(
        "d_eff mean (last 5)",
        f"{_series_tail_mean(ekf.get('d_eff_series'), 5):.6f}",
        f"{_series_tail_mean(ekf_oracle.get('d_eff_series'), 5):.6f}",
        f"{_series_tail_mean(dr.get('d_eff_series'), 5):.6f}",
    )
    print("=" * 96)
    print(
        "  Note: Frames can differ when goal is reached at different times (different "
        "u → different x_true → early stop). Same RNG seed does not fix this. "
        "Use --ignore-goal for equal-length runs until t_end."
    )


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.std(a))


def print_aggregate_comparison_table(
    ekf_runs: List[Dict[str, Any]],
    ekf_oracle_runs: List[Dict[str, Any]],
    dr_runs: List[Dict[str, Any]],
    seed_start: int,
    n_runs: int,
) -> None:
    """Mean ± std over multiple seeds (EKF / EKF-oracle / DR-EKF)."""
    w_lab = 34
    w_c = 18

    def row(label: str, a: str, b: str, c: str) -> None:
        print(f"  {label:<{w_lab}}  {a:^{w_c}}  {b:^{w_c}}  {c:^{w_c}}")

    def collect(key: str, runs: List[Dict[str, Any]]) -> List[float]:
        out = []
        for r in runs:
            if "error" in r:
                continue
            v = r.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
                if np.isfinite(fv):
                    out.append(fv)
            except (TypeError, ValueError):
                pass
        return out

    print("-" * 96)
    print(
        f"  Aggregate over {n_runs} runs  |  seeds {seed_start} … {seed_start + n_runs - 1}  "
        "(mean ± std)"
    )
    print(f"  {'Metric':<{w_lab}}  {'EKF':^{w_c}}  {'EKF-oracle':^{w_c}}  {'DR-EKF':^{w_c}}")
    print("-" * 96)

    for key, label in (
        ("w2_process_nom_vs_true", "W₂ process (nom vs true)"),
        ("w2_meas_nom_vs_true", "W₂ measurement (nom vs true)"),
    ):
        ve, se = _mean_std(collect(key, ekf_runs))
        vo, so = _mean_std(collect(key, ekf_oracle_runs))
        vd, sd = _mean_std(collect(key, dr_runs))
        row(label, f"{ve:.6f}±{se:.6f}", f"{vo:.6f}±{so:.6f}", f"{vd:.6f}±{sd:.6f}")
    print("-" * 96)

    km_e, ks_e = _mean_std(collect("mse_mean", ekf_runs))
    km_o, ks_o = _mean_std(collect("mse_mean", ekf_oracle_runs))
    km_d, ks_d = _mean_std(collect("mse_mean", dr_runs))
    row(
        "MSE mean ‖x̂ − x‖²",
        f"{km_e:.6f}±{ks_e:.6f}",
        f"{km_o:.6f}±{ks_o:.6f}",
        f"{km_d:.6f}±{ks_d:.6f}",
    )

    cr_e, crs_e = _mean_std(collect("collision_rate", ekf_runs))
    cr_o, crs_o = _mean_std(collect("collision_rate", ekf_oracle_runs))
    cr_d, crs_d = _mean_std(collect("collision_rate", dr_runs))
    row(
        "Scenario collision rate (mean)",
        f"{100.0 * cr_e:.2f}%±{100.0 * crs_e:.2f}%",
        f"{100.0 * cr_o:.2f}%±{100.0 * crs_o:.2f}%",
        f"{100.0 * cr_d:.2f}%±{100.0 * crs_d:.2f}%",
    )

    tc_e, tcs_e = _mean_std(collect("total_cost", ekf_runs))
    tc_o, tcs_o = _mean_std(collect("total_cost", ekf_oracle_runs))
    tc_d, tcs_d = _mean_std(collect("total_cost", dr_runs))
    row(
        "Total cost",
        f"{tc_e:.2f}±{tcs_e:.2f}",
        f"{tc_o:.2f}±{tcs_o:.2f}",
        f"{tc_d:.2f}±{tcs_d:.2f}",
    )

    succ_e = [1.0 if r.get("success") else 0.0 for r in ekf_runs if "error" not in r]
    succ_o = [1.0 if r.get("success") else 0.0 for r in ekf_oracle_runs if "error" not in r]
    succ_d = [1.0 if r.get("success") else 0.0 for r in dr_runs if "error" not in r]
    se_m, se_s = _mean_std(succ_e)
    so_m, so_s = _mean_std(succ_o)
    sd_m, sd_s = _mean_std(succ_d)
    row(
        "Success rate",
        f"{100.0 * se_m:.1f}%±{100.0 * se_s:.1f}%",
        f"{100.0 * so_m:.1f}%±{100.0 * so_s:.1f}%",
        f"{100.0 * sd_m:.1f}%±{100.0 * sd_s:.1f}%",
    )

    fr_e, frs_e = _mean_std(collect("frames", ekf_runs))
    fr_o, frs_o = _mean_std(collect("frames", ekf_oracle_runs))
    fr_d, frs_d = _mean_std(collect("frames", dr_runs))
    row("Frames", f"{fr_e:.2f}±{frs_e:.2f}", f"{fr_o:.2f}±{frs_o:.2f}", f"{fr_d:.2f}±{frs_d:.2f}")
    print("-" * 96)

    te_i = collect("theta_eff_initial", dr_runs)
    tm = collect("theta_eff_mean", dr_runs)
    tx = collect("theta_eff_max", dr_runs)
    m_i, s_i = _mean_std(te_i)
    m_m, s_m = _mean_std(tm)
    m_x, s_x = _mean_std(tx)
    row("θ_eff (initial)", "—", "—", f"{m_i:.6f}±{s_i:.6f}" if te_i else "—")
    row("θ_eff mean / max", "—", "—", f"{m_m:.4f}±{s_m:.4f} / {m_x:.4f}±{s_x:.4f}" if tm else "—")

    dm_e = [_series_mean(r.get("delta_margin_series")) for r in ekf_runs if "error" not in r]
    dm_o = [_series_mean(r.get("delta_margin_series")) for r in ekf_oracle_runs if "error" not in r]
    dm_d = [_series_mean(r.get("delta_margin_series")) for r in dr_runs if "error" not in r]
    de_e = [_series_tail_mean(r.get("d_eff_series"), 5) for r in ekf_runs if "error" not in r]
    de_o = [_series_tail_mean(r.get("d_eff_series"), 5) for r in ekf_oracle_runs if "error" not in r]
    de_d = [_series_tail_mean(r.get("d_eff_series"), 5) for r in dr_runs if "error" not in r]
    dme, dse = _mean_std(dm_e)
    dmo, dso = _mean_std(dm_o)
    dmd, dsd = _mean_std(dm_d)
    dee, des = _mean_std(de_e)
    deo, dos = _mean_std(de_o)
    ded, dds = _mean_std(de_d)
    row(
        "Δ margin mean (k√tr P)",
        f"{dme:.6f}±{dse:.6f}",
        f"{dmo:.6f}±{dso:.6f}",
        f"{dmd:.6f}±{dsd:.6f}",
    )
    row(
        "d_eff mean (last 5)",
        f"{dee:.6f}±{des:.6f}",
        f"{deo:.6f}±{dos:.6f}",
        f"{ded:.6f}±{dds:.6f}",
    )
    print("=" * 96)


def print_aggregate_single_table(
    runs: List[Dict[str, Any]],
    estimator_label: str,
    seed_start: int,
    n_runs: int,
) -> None:
    w = 78
    ok = [r for r in runs if "error" not in r]
    print("-" * w)
    print(
        f"  {estimator_label}  |  {n_runs} runs, seeds {seed_start} … {seed_start + n_runs - 1}  "
        "(mean ± std)"
    )
    print("-" * w)

    def c(key: str) -> List[float]:
        out = []
        for r in ok:
            v = r.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
                if np.isfinite(fv):
                    out.append(fv)
            except (TypeError, ValueError):
                pass
        return out

    m, s = _mean_std(c("mse_mean"))
    print(f"  MSE mean: {m:.6f} ± {s:.6f}")
    m, s = _mean_std(c("collision_rate"))
    print(f"  Scenario collision rate: {100.0 * m:.2f}% ± {100.0 * s:.2f}%")
    m, s = _mean_std(c("total_cost"))
    print(f"  Total cost: {m:.4f} ± {s:.4f}")
    succ = [1.0 if r.get("success") else 0.0 for r in ok]
    m, s = _mean_std(succ)
    print(f"  Success rate: {100.0 * m:.1f}% ± {100.0 * s:.1f}%")
    m, s = _mean_std(c("frames"))
    print(f"  Frames: {m:.2f} ± {s:.2f}")
    if estimator_label == "DR-EKF":
        m, s = _mean_std(c("theta_eff_mean"))
        print(f"  θ_eff mean (online): {m:.6f} ± {s:.6f}")
    print("=" * w)


def print_single_summary(estimator_label: str, res: Dict[str, Any]) -> None:
    """Readable block for one estimator (no raw JSON)."""
    w = 78
    print("-" * w)
    print(f"  Estimator: {estimator_label}")
    print("-" * w)
    print(
        f"  W₂ process {res['w2_process_nom_vs_true']:.6f}   "
        f"W₂ meas {res['w2_meas_nom_vs_true']:.6f}"
    )
    print(f"  MSE mean: {res['mse_mean']:.6f}")
    print(
        f"  Scenario collision rate: {100.0 * float(res['collision_rate']):.0f}%  "
        f"(collision steps {res['collision_count']}/{res['total_steps']}; "
        f"step-rate {100.0 * float(res.get('collision_step_rate', 0.0)):.2f}%)"
    )
    print(
        f"  Total cost: {res['total_cost']:.4f}   "
        f"success={'yes' if res['success'] else 'no'}   frames={res['frames']}"
    )
    if not np.isnan(res.get("theta_eff_mean", np.nan)):
        print(
            f"  θ_eff: init={res.get('theta_eff_initial', float('nan')):.6f}  "
            f"mean={res['theta_eff_mean']:.6f}  max={res['theta_eff_max']:.6f}"
        )
    else:
        print("  θ_eff: N/A (EKF)")
    dm = res.get("delta_margin_series") or []
    de = res.get("d_eff_series") or []
    if dm:
        print(
            f"  margin: mean Δ={float(np.mean(dm)):.6f}   "
            f"d_eff mean (last 5)={_series_tail_mean(de, 5):.6f}"
        )
    print("=" * w)


def _kwargs_from_args(
    args: argparse.Namespace,
    estimator_name: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    parts = [float(x) for x in args.init.split(",")]
    init_robot_state = {
        "position_x": parts[0],
        "position_y": parts[1],
        "orientation_z": float(parts[2]) if len(parts) > 2 else 0.0,
    }
    if len(parts) > 2 and abs(parts[2]) > 2 * np.pi:
        init_robot_state["orientation_z"] = np.deg2rad(parts[2])
    gp = [float(x) for x in args.goal.split(",")]
    goal_pos = np.array(gp[:2], dtype=float)
    bv = [float(x) for x in args.beacons.split(",")]
    beacons = np.array(bv[:6], dtype=float).reshape(3, 2)
    return dict(
        dataset_name=args.dataset,
        predictor_name=args.predictor,
        estimator_name=estimator_name,
        init_robot_state=init_robot_state,
        goal_pos=goal_pos,
        t_begin=args.t_begin,
        t_end=args.t_end,
        prediction_horizon=args.prediction_horizon,
        history_len=args.history_len,
        seed=int(seed) if seed is not None else int(args.seed),
        beacons=beacons,
        sigma_px_true=args.sigma_px_true,
        sigma_py_true=args.sigma_py_true,
        sigma_theta_true=args.sigma_theta_true,
        sigma_r_true=args.sigma_r_true,
        sigma_theta_meas_true=args.sigma_theta_meas_true,
        nom_q_var_mult=_parse_n_floats(args.nom_q_var_mult, 3, "nom_q_var_mult"),
        nom_r_var_mult=_parse_n_floats(args.nom_r_var_mult, 4, "nom_r_var_mult"),
        k_sigma=args.k_sigma,
        theta_w=args.theta_w,
        theta_v=args.theta_v,
        theta_x0=args.theta_x0,
        L_f=args.L_f,
        L_h=args.L_h,
        eta_scale=args.eta_scale,
        theta_eff_cap=args.theta_eff_cap,
        verbose=args.verbose,
        ignore_goal=args.ignore_goal,
        theta_eff_cli=args.theta_eff,
    )


def build_fade_plot_data(
    args: argparse.Namespace,
    seeds_list: List[int],
) -> Tuple[
    Dict[int, np.ndarray],
    List[
        Tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Optional[int],
            Optional[int],
            Optional[int],
        ]
    ],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    int,
    Tuple[float, float],
    Tuple[float, float],
]:
    """Run/load EKF + EKF-oracle (true) + DR-EKF per seed and assemble pedestrian GT + robot runs."""
    dset = RegisteredDatasets[args.dataset]
    os.makedirs(args.cache_dir, exist_ok=True)
    init_parts = [float(x) for x in args.init.split(",")]
    goal_parts = [float(x) for x in args.goal.split(",")]
    start_xy = (float(init_parts[0]), float(init_parts[1]))
    goal_xy = (float(goal_parts[0]), float(goal_parts[1]))

    robot_runs: List[
        Tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Optional[int],
            Optional[int],
            Optional[int],
        ]
    ] = []
    ekf_runs_all: List[Dict[str, Any]] = []
    ekf_oracle_runs_all: List[Dict[str, Any]] = []
    dr_runs_all: List[Dict[str, Any]] = []
    ped_by_id: Dict[int, np.ndarray] = {}
    t_begin = int(args.t_begin)

    for run_seed in seeds_list:
        setattr(args, "seed", int(run_seed))
        w = 78
        print("=" * w)
        print(
            f"  Trajectory fade plot (EKF vs EKF(true) vs DR-EKF)   dataset={args.dataset}  "
            f"predictor={args.predictor}  controller=mpc  seed={run_seed}"
        )
        print(f"  estimation margin: on (k={args.k_sigma})")
        print("=" * w)
        cache_path = _seed_cache_path(args, int(run_seed), args.cache_dir)
        loaded_from_cache = False
        r_ekf: Dict[str, Any]
        r_or: Dict[str, Any]
        r_dr: Dict[str, Any]
        if (not args.recompute) and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                r_ekf = payload["ekf"]
                r_or = payload.get("ekf_oracle") or payload.get("ekf-oracle") or payload.get("oracle") or {}
                r_dr = payload["dr"]
                loaded_from_cache = True
                print(f"  Loaded cache → {cache_path}")
            except Exception as e:
                print(f"  Cache read failed (recomputing): {e}")
                loaded_from_cache = False
        if not loaded_from_cache:
            kw_base = _kwargs_from_args(args, "ekf", seed=int(run_seed))
            r_ekf = run_navigation(
                **{**kw_base, "estimator_name": "ekf", "record_video": False}
            )
            r_or = run_navigation(
                **{**kw_base, "estimator_name": "ekf-oracle", "record_video": False}
            )
            r_dr = run_navigation(
                **{**kw_base, "estimator_name": "dr-ekf", "record_video": False}
            )
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"ekf": r_ekf, "ekf_oracle": r_or, "dr": r_dr}, f)
                print(f"  Saved cache → {cache_path}")
            except Exception as e:
                print(f"  Cache write failed: {e}")
        if loaded_from_cache and (not r_or):
            kw_base = _kwargs_from_args(args, "ekf", seed=int(run_seed))
            r_or = run_navigation(
                **{**kw_base, "estimator_name": "ekf-oracle", "record_video": False}
            )
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"ekf": r_ekf, "ekf_oracle": r_or, "dr": r_dr}, f)
                print(f"  Updated cache (added oracle) → {cache_path}")
            except Exception:
                pass

        for tag, r in (("EKF", r_ekf), ("EKF(true)", r_or), ("DR-EKF", r_dr)):
            if "error" in r:
                print(f"ERROR [seed {run_seed}][{tag}]:", r["error"])
                sys.exit(1)
        ekf_runs_all.append(r_ekf)
        ekf_oracle_runs_all.append(r_or)
        dr_runs_all.append(r_dr)

        robot_ekf = np.asarray(r_ekf.get("robot_true_trajectory", []), dtype=float)
        robot_or = np.asarray(r_or.get("robot_true_trajectory", []), dtype=float)
        robot_dr = np.asarray(r_dr.get("robot_true_trajectory", []), dtype=float)
        t_begin = int(r_ekf.get("t_begin", args.t_begin))

        if not robot_runs:
            raw_multi = r_ekf.get("pedestrian_gt_trajectories_by_id")
            if raw_multi:
                for ks, v in raw_multi.items():
                    ped_by_id[int(ks)] = np.asarray(v, dtype=float).reshape(-1, 2)
            else:
                ped_legacy = np.asarray(r_ekf.get("pedestrian_gt_trajectory", []), dtype=float)
                aid0 = r_ekf.get("pedestrian_agent_id")
                if ped_legacy.size and aid0 is not None:
                    ped_by_id[int(aid0)] = ped_legacy.reshape(-1, 2)

            for aid in list(ped_by_id.keys()):
                arr = np.asarray(ped_by_id[aid], dtype=float).reshape(-1, 2)
                for t_ds in range(args.t_scatter_min, args.t_scatter_max + 1):
                    idx = _index_for_dataset_timestep(t_ds, t_begin)
                    if idx < 0:
                        continue
                    if idx < len(arr) and np.isfinite(arr[idx]).all():
                        continue
                    try:
                        row = np.asarray(dset._data[int(t_ds), int(aid), :2], dtype=float)
                    except Exception:
                        continue
                    if not np.isfinite(row).all():
                        continue
                    if idx >= len(arr):
                        arr = np.vstack([arr, np.full((idx + 1 - len(arr), 2), np.nan, dtype=float)])
                    arr[idx] = row
                ped_by_id[aid] = arr

            try:
                scene_at_fade = dset.get_scene(int(args.t_scatter_min), int(args.history_len))
                ids_at_fade = {int(k) for k in scene_at_fade.keys()}
                if ids_at_fade:
                    ped_by_id = {k: v for k, v in ped_by_id.items() if k in ids_at_fade}
            except Exception:
                pass

            # Ensure any pedestrian involved in a collision is included in the plot, even if it
            # was not present at t_scatter_min (common reason for "collision x-mark but no purple path").
            coll_ids: set = set()
            for rr in (r_ekf, r_or, r_dr):
                try:
                    coll_ids.update(int(x) for x in (rr.get("ped_collision_agent_ids") or []))
                except Exception:
                    pass
                try:
                    cid0 = rr.get("first_ped_collision_agent_id")
                    if cid0 is not None:
                        coll_ids.add(int(cid0))
                except Exception:
                    pass
            if coll_ids:
                missing = sorted([pid for pid in coll_ids if pid not in ped_by_id])
                if missing:
                    for pid in missing:
                        try:
                            arr = np.asarray(dset._data[:, int(pid), :2], dtype=float)
                            ped_by_id[int(pid)] = arr.reshape(-1, 2)
                        except Exception:
                            continue
                print(f"  Ped collision agent ids (union): {sorted(coll_ids)}")

            if ped_by_id:
                print(f"  Pedestrian agent ids (GT, at t={args.t_scatter_min}): {sorted(ped_by_id.keys())}")

        need_idx = _index_for_dataset_timestep(args.t_scatter_max, t_begin)
        n_need = need_idx + 1
        if len(robot_ekf) < n_need or len(robot_or) < n_need or len(robot_dr) < n_need:
            print(
                f"  Warning [seed {run_seed}]: rollout shorter than t={args.t_scatter_max} "
                f"(need index {need_idx}, have EKF len={len(robot_ekf)}, OR len={len(robot_or)}, DR len={len(robot_dr)}). "
                "Scatter will omit missing times; try --ignore-goal.",
                file=sys.stderr,
            )

        ekf_ped_ci = r_ekf.get("first_ped_collision_index")
        or_ped_ci = r_or.get("first_ped_collision_index")
        dr_ped_ci = r_dr.get("first_ped_collision_index")
        robot_runs.append(
            (
                int(run_seed),
                robot_ekf,
                robot_or,
                robot_dr,
                int(ekf_ped_ci) if ekf_ped_ci is not None else None,
                int(or_ped_ci) if or_ped_ci is not None else None,
                int(dr_ped_ci) if dr_ped_ci is not None else None,
            )
        )

        print("-" * 96)
        print(f"  {'Metric':<34}  {'EKF':^17}  {'EKF(true)':^17}  {'DR-EKF':^17}")
        print("-" * 96)
        print(
            f"  {'MSE mean ‖x̂ − x‖²':<34}  {r_ekf['mse_mean']:^17.6f}  {r_or['mse_mean']:^17.6f}  {r_dr['mse_mean']:^17.6f}"
        )
        print(
            f"  {'Frames':<34}  {r_ekf['frames']:^17}  {r_or['frames']:^17}  {r_dr['frames']:^17}"
        )
        print("=" * 96)

    return (
        ped_by_id,
        robot_runs,
        ekf_runs_all,
        ekf_oracle_runs_all,
        dr_runs_all,
        t_begin,
        start_xy,
        goal_xy,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Static fade plot: pedestrian GT + EKF vs DR-EKF "
        f"(defaults: seeds {_DEFAULT_NAV_SEED_START}…{_DEFAULT_NAV_SEED_END}, "
        f"t_begin/t_end/t_scatter={_DEFAULT_NAV_T_BEGIN}…{_DEFAULT_NAV_T_END})"
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=_DEFAULT_NAV_SEEDS_STR,
        help="Comma-separated RNG seeds (all drawn on the same figure; linestyle cycles per seed)",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join("results", "navigation_cache"),
        help="Directory for per-seed cached simulation results (EKF/DR-EKF)",
    )
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cache and recompute simulations, then overwrite cache files",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=_DEFAULT_NAV_OUTPUT,
        help="Output PNG path (single figure)",
    )
    ap.add_argument(
        "--traj-seed",
        type=int,
        default=_DEFAULT_NAV_TRAJ_SEED,
        help="Seed to show on the right trajectory panel. Default from built-in profile; use 0 to fall back to first --seeds value.",
    )
    ap.add_argument(
        "--t-scatter-min",
        type=int,
        default=_DEFAULT_NAV_T_SCATTER_MIN,
        help="Dataset timestep (env._step) start for fade scatter (inclusive)",
    )
    ap.add_argument(
        "--t-scatter-max",
        type=int,
        default=_DEFAULT_NAV_T_SCATTER_MAX,
        help="Dataset timestep (env._step) end for fade scatter (inclusive)",
    )
    ap.add_argument(
        "--alpha-min",
        type=float,
        default=0.15,
        help="Scatter alpha at t-scatter-min (later times use higher alpha)",
    )
    ap.add_argument(
        "--alpha-max",
        type=float,
        default=1.0,
        help="Scatter alpha at t-scatter-max",
    )
    ap.add_argument("--dataset", type=str, default="zara2")
    ap.add_argument("--predictor", type=str, default="trajectron")
    ap.add_argument(
        "--ignore-goal",
        action="store_true",
        help="Do not stop when the goal is reached; run until the dataset horizon (t_end). "
        "Useful so the rollout reaches dataset timesteps up to --t-scatter-max.",
    )
    ap.add_argument("--t-begin", type=int, default=_DEFAULT_NAV_T_BEGIN)
    ap.add_argument("--t-end", type=int, default=_DEFAULT_NAV_T_END)
    ap.add_argument("--prediction-horizon", type=int, default=12)
    ap.add_argument("--history-len", type=int, default=8)
    ap.add_argument(
        "--init",
        type=str,
        default="2.5,6.0,0.0",
        help="x,y,theta_deg (theta in degrees, converted to rad)",
    )
    ap.add_argument("--goal", type=str, default="12.5,6.0")
    ap.add_argument(
        "--beacons",
        type=str,
        default="1.0,1.0,14.0,1.0,7.5,9.0",
        help="six numbers: bx1,by1,bx2,by2,bx3,by3",
    )
    
    _DEFAULT_SIGMA_PX_TRUE = 0.1
    _DEFAULT_SIGMA_PY_TRUE = 0.1
    _DEFAULT_SIGMA_THETA_TRUE_DEG = 5.0
    _DEFAULT_SIGMA_R_TRUE = "0.12"
    _DEFAULT_SIGMA_THETA_MEAS_TRUE_DEG = 2.0
    _DEFAULT_NOM_Q_VAR_MULT = "0.5,0.5,0.3"
    _DEFAULT_NOM_R_VAR_MULT = "0.5,0.5,0.5,0.3"
    ap.add_argument(
        "--sigma-px-true",
        type=float,
        default=_DEFAULT_SIGMA_PX_TRUE,
        help="True process noise std on p_x (m); Q_true_11 = sigma^2",
    )
    ap.add_argument(
        "--sigma-py-true",
        type=float,
        default=_DEFAULT_SIGMA_PY_TRUE,
        help="True process noise std on p_y (m)",
    )
    ap.add_argument(
        "--sigma-theta-true",
        type=float,
        default=np.deg2rad(_DEFAULT_SIGMA_THETA_TRUE_DEG),
        help="True process noise std on theta (rad)",
    )
    ap.add_argument(
        "--sigma-r-true",
        type=str,
        default=_DEFAULT_SIGMA_R_TRUE,
        help="True UWB range std (m): one value (all 3 beacons) or three comma-separated",
    )
    ap.add_argument(
        "--sigma-theta-meas-true",
        type=float,
        default=np.deg2rad(_DEFAULT_SIGMA_THETA_MEAS_TRUE_DEG),
        help="True heading measurement std (rad)",
    )
    ap.add_argument(
        "--nom-q-var-mult",
        type=str,
        default=_DEFAULT_NOM_Q_VAR_MULT,
        help="Per-axis variance multipliers: Q_nom_ii = mult_i * Q_true_ii (3 values: px, py, theta)",
    )
    ap.add_argument(
        "--nom-r-var-mult",
        type=str,
        default=_DEFAULT_NOM_R_VAR_MULT,
        help="Per-sensor variance multipliers: R_nom_jj = mult_j * R_true_jj (4: 3 ranges + heading)",
    )
    ap.add_argument("--k-sigma", type=float, default=1)
    ap.add_argument(
        "--theta-eff",
        type=float,
        default=_DEFAULT_THETA_EFF,
        help=(
            "Only DR-EKF static ambiguity knob (exp_ct_tracking*.py): "
            f"θ_w=θ_v=θ_x0=θ_eff/√2. Default { _DEFAULT_THETA_EFF:.6f} gives θ_w=θ_v=0.06. "
            "Online θ_eff^eff from the trace-tube is reported separately."
        ),
    )
    ap.add_argument("--L-f", type=float, default=0.5)
    ap.add_argument("--L-h", type=float, default=0.5)
    ap.add_argument("--eta-scale", type=float, default=1.0)
    ap.add_argument("--theta-eff-cap", type=float, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    derive_dr_radii_from_theta_eff(args)

    try:
        seeds_list = [int(x.strip()) for x in str(args.seeds).replace(" ", "").split(",") if x.strip()]
    except ValueError:
        seeds_list = []
    if not seeds_list:
        print("--seeds must list one or more integers", file=sys.stderr)
        sys.exit(2)

    if args.t_scatter_max < args.t_scatter_min:
        print("--t-scatter-max must be >= --t-scatter-min", file=sys.stderr)
        sys.exit(2)

    if not os.path.isdir(CANVAS_ROOT) or not os.path.isdir(os.path.join(CANVAS_ROOT, "canvas")):
        print(
            f"CANVAS not found at {CANVAS_ROOT}. Set CANVAS_ROOT.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"CANVAS_ROOT={CANVAS_ROOT}")
    print(f"  RNG seeds: {', '.join(str(s) for s in seeds_list)}")
    s = float(args.theta_w)
    print(
        "  DR-EKF static radii (θ_eff only → exp_ct_tracking*.py): "
        f"θ_eff={args.theta_eff:.6f}  →  θ_w=θ_v=θ_x0={s:.6f}  "
        "(online θ_eff^eff in table / JSON is separate)\n"
    )
    q_mult = _parse_n_floats(args.nom_q_var_mult, 3, "nom_q_var_mult")
    r_mult = _parse_n_floats(args.nom_r_var_mult, 4, "nom_r_var_mult")
    r_std = _parse_n_floats(args.sigma_r_true, 3, "sigma_r_true")
    print("Noise model (explicit diagonal form):")
    print(
        "  Q_true = diag(["
        f"{args.sigma_px_true:.6f}^2, {args.sigma_py_true:.6f}^2, {args.sigma_theta_true:.6f}^2])"
    )
    print(
        "  Q_nom  = diag(["
        f"{q_mult[0]:.6f}, {q_mult[1]:.6f}, {q_mult[2]:.6f}] * diag(Q_true))"
    )
    print(
        "  R_true = diag(["
        f"{r_std[0]:.6f}^2, {r_std[1]:.6f}^2, {r_std[2]:.6f}^2, {args.sigma_theta_meas_true:.6f}^2])"
    )
    print(
        "  R_nom  = diag(["
        f"{r_mult[0]:.6f}, {r_mult[1]:.6f}, {r_mult[2]:.6f}, {r_mult[3]:.6f}] * diag(R_true))\n"
    )

    (
        ped_by_id,
        robot_runs,
        ekf_runs_all,
        ekf_oracle_runs_all,
        dr_runs_all,
        t_begin,
        start_xy,
        goal_xy,
    ) = build_fade_plot_data(args, seeds_list)

    if int(args.traj_seed) == 0:
        traj_seed = int(seeds_list[0])
    else:
        traj_seed = int(args.traj_seed)
    robot_runs_plot = [rr for rr in robot_runs if int(rr[0]) == traj_seed]
    if not robot_runs_plot:
        print(
            f"Warning: --traj-seed {traj_seed} not found in --seeds; falling back to first seed {int(seeds_list[0])}.",
            file=sys.stderr,
        )
        robot_runs_plot = [rr for rr in robot_runs if int(rr[0]) == int(seeds_list[0])]

    out_path = args.output
    save_navigation_trajectories_fade_plot(
        dataset_name=args.dataset,
        predictor_name=args.predictor,
        pedestrian_xy_by_id=ped_by_id,
        robot_runs=robot_runs_plot,
        ekf_runs=ekf_runs_all,
        ekf_oracle_runs=ekf_oracle_runs_all,
        dr_runs=dr_runs_all,
        t_begin=t_begin,
        output_path=out_path,
        t_scatter_min=args.t_scatter_min,
        t_scatter_max=args.t_scatter_max,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )
    print(f"  Saved combined trajectory plot → {out_path}")


if __name__ == "__main__":
    main()
