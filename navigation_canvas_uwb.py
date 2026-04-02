#!/usr/bin/env python3
"""
CANVAS crowd navigation with UWB (3 ranges + heading) measurements and EKF / DR-EKF.

True plant: x_{t+1} = f(x_t,u_t) + w_t (same unicycle as CANVAS).
Controller sees only posterior ego estimate (not ground-truth pose).

Noise: **true** covariances are set from ``--sigma-*-true`` (Gaussian only). **Nominal** (filter)
covariances are ``Q_nom_ii = nom_q_var_mult[i] * Q_true_ii`` and ``R_nom_jj = nom_r_var_mult[j] * R_true_jj``.

DR-EKF: optional ``--theta-eff`` is a **single knob**: sets static
``θ_w = θ_v = θ_x0 = θ_eff / √2`` (as in ``exp_ct_tracking*.py``). The filter still reports
**online** ``θ_eff^eff`` (trace-tube) — not the same number as this CLI ``θ_eff``.

MPC (``canvas.controllers.mpc.BaseMPC``, ``use_ipopt=False`` in this script — discrete candidate paths, not IPOPT):

- **Horizon / time**: ``prediction_horizon`` from CLI (default 12); ``dt = env.dt`` (e.g. 0.4 s on ETH/UCY / zara2).
- **Safety margin** ``d_min`` (passed into scoring each step): base ``ROBOT_RAD + 0.1/√2`` m with ``ROBOT_RAD = 0.4``, plus trace inflation
  ``k_sigma * √(tr(P_pos[:2,:2]))`` (``--k-sigma``, default 1) from the active filter covariance; ``controller._d_min`` is updated every step.
- **Cost** (among candidate rollouts): sum of stage ‖(x,y)−goal‖² over horizon; control
  ``2e-3·Σv² + 0.15·Σω²``; smoothness ``2e-3·Σ(Δv)² + 0.25·Σ(Δω)²``; terminal ``10·‖(x_N,y_N)−goal‖²``.
- **Hard constraints**: any rollout with ``min(distance to static geometry, distance to predicted agents) ≤ d_min``
  at a horizon step is rejected (infinite cost). If no rollout is safe, command ``(v, ω) = (0, 0)``.
- **Input sampling** (``generate_paths``, ``n_skip=4``): ``v ∈ {-0.8,-0.4,-0.2,0,0.2,0.4,0.8}`` m/s,
  ``ω ∈ {-0.4, 0, 0.4}`` rad/s, held piecewise-constant for 4 steps per choice (Cartesian product over decision epochs).
- **Plant in rollout**: unicycle with the same ``dt`` as the env.

If ``use_ipopt=True`` (not default here), see ``canvas/controllers/optim_solver.py``: control box
``v∈[-0.8,0.8]``, ``ω∈[-0.4,0.4]``; smoothed static-obstacle inequalities; dynamic clearance
``‖p−p_pred‖ ≥ r_robot + r_agent`` with ``r_robot=0.4``, ``r_agent=0.1/√2``; state box
``(x,y)`` from geometry bounds and ``|θ| ≤ 4π``; objective mixes goal tracking (weight 10 on last stage) and ``1e-5‖u‖²``.

Run from DR-EKF-main (or set PYTHONPATH). Requires CANVAS clone; set CANVAS_ROOT if needed.

Example:
  python navigation_canvas_uwb.py --compare --dataset zara2 \\
    --predictor trajectron --seed 42

  python navigation_canvas_uwb.py --estimator ekf --dataset zara2

  Aggregate 10 seeds and save MP4s for all seeds:
  python navigation_canvas_uwb.py --compare --n-runs 10 --seed 44 \\
    --save-video results/navigation_videos
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
    d_min_base = ROBOT_RAD + 0.1/np.sqrt(2.0)
    # Collision metric (static + pedestrian): closer than this counts as collision.
    collision_threshold_m = d_min_base

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
    total_steps = 0
    total_cost = 0.0

    truncated = False
    frame = 0
    terminated = False
    video_frames: List[Dict[str, Any]] = []
    robot_traj_vid: List[List[float]] = []
    est_traj_vid: List[List[float]] = []

    def _push_video_frame(x_hat_vec: np.ndarray) -> None:
        if not record_video:
            return
        xh = np.asarray(x_hat_vec, dtype=float).reshape(3)
        est_traj_vid.append([float(xh[0]), float(xh[1])])
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
        if not step_collision and obs.get("non-ego"):
            for _, hist in obs["non-ego"].items():
                if len(hist) > 0:
                    ap = hist[-1]
                    if np.linalg.norm(cur - ap) <= collision_threshold_m:
                        collision_cost += 1e3
                        step_collision = True
                        break
        if step_collision:
            collision_count += 1
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
        "dt": dt,
        "video_frames": video_frames if record_video else None,
    }
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser(description="CANVAS navigation with UWB + EKF/DR-EKF")
    ap.add_argument("--dataset", type=str, default="zara2")
    ap.add_argument("--predictor", type=str, default="trajectron")
    ap.add_argument(
        "--estimator",
        type=str,
        choices=("ekf", "dr-ekf", "ekf-oracle"),
        default="ekf",
        help="Ignored when --compare is set (runs both).",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Run EKF and DR-EKF with the same settings; print a side-by-side table.",
    )
    ap.add_argument(
        "--ignore-goal",
        action="store_true",
        help="Do not stop when the goal is reached; run until the dataset horizon (t_end). "
        "Then EKF vs DR-EKF usually have the same number of frames (fair length for --compare).",
    )
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Repeat with seeds seed, seed+1, … seed+n-runs-1; print mean ± std of metrics.",
    )
    ap.add_argument("--t-begin", type=int, default=0)
    ap.add_argument("--t-end", type=int, default=70)
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
    ap.add_argument(
        "--save-trace-json",
        type=str,
        default=None,
        help="Save per-step series (d_eff, delta_margin, mse, theta_eff) to this JSON file",
    )
    ap.add_argument(
        "--save-video",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory for MP4 navigation videos (needs ffmpeg). "
            "With --n-runs>1, videos are saved for every seed."
        ),
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    derive_dr_radii_from_theta_eff(args)
    if args.n_runs < 1:
        print("--n-runs must be >= 1", file=sys.stderr)
        sys.exit(2)

    if not os.path.isdir(CANVAS_ROOT) or not os.path.isdir(os.path.join(CANVAS_ROOT, "canvas")):
        print(
            f"CANVAS not found at {CANVAS_ROOT}. Set CANVAS_ROOT.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"CANVAS_ROOT={CANVAS_ROOT}")
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

    def _video_out_paths() -> Tuple[str, float, float, float, float]:
        parts = [float(x) for x in args.init.split(",")]
        ix, iy = float(parts[0]), float(parts[1])
        gp = [float(x) for x in args.goal.split(",")]
        return args.save_video or "", ix, iy, float(gp[0]), float(gp[1])

    def save_trace(path: str, res: Dict[str, Any], tag: str) -> None:
        payload = {
            "estimator": tag,
            "theta_eff_cli": res.get("theta_eff_cli"),
            "dr_static_ambiguity": res.get("dr_static_ambiguity"),
            "theta_eff_initial": res.get("theta_eff_initial"),
            "d_eff_series": [float(x) for x in res.get("d_eff_series", [])],
            "delta_margin_series": [float(x) for x in res.get("delta_margin_series", [])],
            "mse_per_step": [float(x) for x in res.get("mse_per_step", [])],
            "theta_eff_series": [float(x) for x in res.get("theta_eff_series", [])],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved trace → {path}")

    if args.compare:
        print_run_header(
            args.dataset,
            args.predictor,
            args.seed,
            args.k_sigma,
            compare=True,
            n_runs=args.n_runs,
        )
        ekf_runs: List[Dict[str, Any]] = []
        ekf_oracle_runs: List[Dict[str, Any]] = []
        dr_runs: List[Dict[str, Any]] = []
        for i in range(args.n_runs):
            sd = args.seed + i
            kw = _kwargs_from_args(args, "ekf", seed=sd)
            record = bool(args.save_video)
            r_ekf = run_navigation(**{**kw, "estimator_name": "ekf", "record_video": record})
            r_ekf_oracle = run_navigation(
                **{**kw, "estimator_name": "ekf-oracle", "record_video": record}
            )
            r_dr = run_navigation(**{**kw, "estimator_name": "dr-ekf", "record_video": record})
            ekf_runs.append(r_ekf)
            ekf_oracle_runs.append(r_ekf_oracle)
            dr_runs.append(r_dr)
            if args.verbose:
                print(
                    f"  [seed {sd}]  EKF mse={r_ekf.get('mse_mean', 'err')!s}  "
                    f"EKF-oracle mse={r_ekf_oracle.get('mse_mean', 'err')!s}  "
                    f"DR mse={r_dr.get('mse_mean', 'err')!s}"
                )

        err_ekf = [i for i, r in enumerate(ekf_runs) if "error" in r]
        err_ekf_oracle = [i for i, r in enumerate(ekf_oracle_runs) if "error" in r]
        err_dr = [i for i, r in enumerate(dr_runs) if "error" in r]
        if err_ekf:
            print("ERROR [EKF] runs:", err_ekf, ekf_runs[err_ekf[0]].get("error"))
            sys.exit(1)
        if err_ekf_oracle:
            print(
                "ERROR [EKF-oracle] runs:",
                err_ekf_oracle,
                ekf_oracle_runs[err_ekf_oracle[0]].get("error"),
            )
            sys.exit(1)
        if err_dr:
            print("ERROR [DR-EKF] runs:", err_dr, dr_runs[err_dr[0]].get("error"))
            sys.exit(1)

        if args.n_runs > 1:
            print_aggregate_comparison_table(
                ekf_runs, ekf_oracle_runs, dr_runs, args.seed, args.n_runs
            )
        else:
            print_comparison_table(ekf_runs[0], ekf_oracle_runs[0], dr_runs[0])
        if args.save_trace_json:
            if args.n_runs > 1:
                print(
                    "  (Skipping --save-trace-json: use --n-runs 1 to save one trace, "
                    "or run again with a single seed.)",
                )
            else:
                base = args.save_trace_json
                if base.lower().endswith(".json"):
                    stem, _ = os.path.splitext(base)
                    p_ekf, p_or, p_dr = (
                        f"{stem}_ekf.json",
                        f"{stem}_ekf_oracle.json",
                        f"{stem}_dr.json",
                    )
                else:
                    p_ekf, p_or, p_dr = (
                        f"{base}_ekf.json",
                        f"{base}_ekf_oracle.json",
                        f"{base}_dr.json",
                    )
                save_trace(p_ekf, ekf_runs[0], "ekf")
                save_trace(p_or, ekf_oracle_runs[0], "ekf-oracle")
                save_trace(p_dr, dr_runs[0], "dr-ekf")
        if args.save_video:
            out_dir, ix, iy, gx, gy = _video_out_paths()
            for i in range(args.n_runs):
                sd = args.seed + i
                for tag, res in (
                    ("ekf", ekf_runs[i]),
                    ("ekf-oracle", ekf_oracle_runs[i]),
                    ("dr-ekf", dr_runs[i]),
                ):
                    vf = res.get("video_frames")
                    if vf:
                        p = save_navigation_video(
                            vf,
                            dataset_name=args.dataset,
                            predictor_name=args.predictor,
                            estimator_name=tag,
                            output_dir=out_dir,
                            t_begin=args.t_begin,
                            init_x=ix,
                            init_y=iy,
                            goal_x=gx,
                            goal_y=gy,
                            seed=sd,
                        )
                        if p:
                            print(f"  Saved video ({tag}, seed {sd}) → {p}")
        # Always save one 3-way true-path plot using all compare runs.
        ix, iy = [float(x) for x in args.init.split(",")[:2]]
        gx, gy = [float(x) for x in args.goal.split(",")[:2]]
        plot_dir = os.path.join("results", "navigation_plots")
        p_plot = save_compare_true_trajectory_plot(
            dataset_name=args.dataset,
            predictor_name=args.predictor,
            output_dir=plot_dir,
            t_begin=args.t_begin,
            init_x=ix,
            init_y=iy,
            goal_x=gx,
            goal_y=gy,
            seed=args.seed,
            traj_ekf_runs=[
                np.asarray(r.get("robot_true_trajectory", []), dtype=float)
                for r in ekf_runs
                if "error" not in r
            ],
            traj_ekf_oracle_runs=[
                np.asarray(r.get("robot_true_trajectory", []), dtype=float)
                for r in ekf_oracle_runs
                if "error" not in r
            ],
            traj_dr_runs=[
                np.asarray(r.get("robot_true_trajectory", []), dtype=float)
                for r in dr_runs
                if "error" not in r
            ],
        )
        if p_plot:
            print(f"  Saved compare true-path plot → {p_plot}")
        dt_plot = float(ekf_runs[0].get("dt", 0.0) or 0.0)
        if dt_plot > 0.0:
            p_dm = save_compare_delta_margin_plot(
                dataset_name=args.dataset,
                predictor_name=args.predictor,
                output_dir=plot_dir,
                t_begin=args.t_begin,
                init_x=ix,
                init_y=iy,
                goal_x=gx,
                goal_y=gy,
                seed_start=args.seed,
                n_runs=args.n_runs,
                k_sigma=float(args.k_sigma),
                dt=dt_plot,
                ekf_runs=ekf_runs,
                ekf_oracle_runs=ekf_oracle_runs,
                dr_runs=dr_runs,
            )
            if p_dm:
                print(f"  Saved compare Δ_t plot → {p_dm}")
        p_violin = save_compare_violin_metrics_plot(
            dataset_name=args.dataset,
            predictor_name=args.predictor,
            output_dir=plot_dir,
            t_begin=args.t_begin,
            init_x=ix,
            init_y=iy,
            goal_x=gx,
            goal_y=gy,
            seed_start=args.seed,
            n_runs=args.n_runs,
            ekf_runs=ekf_runs,
            ekf_oracle_runs=ekf_oracle_runs,
            dr_runs=dr_runs,
        )
        if p_violin:
            print(f"  Saved compare violin plot → {p_violin}")
        return

    # --- single estimator ---
    print_run_header(
        args.dataset,
        args.predictor,
        args.seed,
        args.k_sigma,
        compare=False,
        n_runs=args.n_runs,
    )
    runs: List[Dict[str, Any]] = []
    for i in range(args.n_runs):
        sd = args.seed + i
        record = bool(args.save_video)
        res = run_navigation(
            **{**_kwargs_from_args(args, args.estimator, seed=sd), "record_video": record}
        )
        runs.append(res)
        if args.verbose:
            print(f"  [seed {sd}]  mse={res.get('mse_mean', res.get('error'))!s}")

    if any("error" in r for r in runs):
        err_i = next(i for i, r in enumerate(runs) if "error" in r)
        print("ERROR:", runs[err_i]["error"])
        if "w2_process_nom_vs_true" in runs[err_i]:
            print(
                "  W₂ (nom vs true) process:",
                runs[err_i]["w2_process_nom_vs_true"],
                "meas:",
                runs[err_i]["w2_meas_nom_vs_true"],
            )
        sys.exit(1)

    est_label_map = {"ekf": "EKF", "dr-ekf": "DR-EKF", "ekf-oracle": "EKF-oracle(Q=Q_true,R=R_true)"}
    est_label = est_label_map.get(args.estimator, args.estimator)
    if args.n_runs > 1:
        print_aggregate_single_table(runs, est_label, args.seed, args.n_runs)
    else:
        print_single_summary(est_label, runs[0])
    if args.save_trace_json:
        if args.n_runs > 1:
            print(
                "  (Skipping --save-trace-json with n-runs>1; use --n-runs 1 to save.)"
            )
        else:
            save_trace(args.save_trace_json, runs[0], args.estimator)

    if args.save_video:
        out_dir, ix, iy, gx, gy = _video_out_paths()
        for i, res in enumerate(runs):
            sd = args.seed + i
            vf = res.get("video_frames")
            if vf:
                p = save_navigation_video(
                    vf,
                    dataset_name=args.dataset,
                    predictor_name=args.predictor,
                    estimator_name=args.estimator,
                    output_dir=out_dir,
                    t_begin=args.t_begin,
                    init_x=ix,
                    init_y=iy,
                    goal_x=gx,
                    goal_y=gy,
                    seed=sd,
                )
                if p:
                    print(f"  Saved video (seed {sd}) → {p}")


if __name__ == "__main__":
    main()
