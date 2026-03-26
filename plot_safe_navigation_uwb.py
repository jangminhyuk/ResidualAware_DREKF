#!/usr/bin/env python3
"""
plot_safe_navigation_uwb.py — Publication-quality figures for UWB beacon
self-localization safe navigation experiment.

Figures
-------
1. nav_main_4panel.pdf       — hero: trajectories, margins, dist-to-goal, bars
2. nav_trajectories.pdf      — side-by-side per-filter trajectory panels
3. margin_time_series.pdf    — safety margin vs time (DOP-correlated)
4. outcome_bar.pdf           — collision / safe-reach / timeout stacked bars
5. clearance_hist.pdf        — minimum clearance distributions
6. covariance_and_theta.pdf  — Tr(P), position cov, heading cov, theta_eff
7. pdop_heatmap.pdf          — Position DOP spatial map with beacons & obstacle
8. dop_vs_margin.pdf         — DOP–margin correlation twin time-series
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────────
# Design tokens  (keep in sync with plot_nonlinear.py / plot_trajectories.py)
# ──────────────────────────────────────────────────────────────────────

_C = {
    "grid":         "#D5D9E0",   # cool light gray
    "panel_bg":     "#FAFAFA",   # neutral white
}


# ──────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────

def _setup_style():
    """Configure matplotlib for NeurIPS / CDC publication quality."""
    plt.rcParams.update({
        # --- Font ---
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Helvetica",
                                 "Arial", "DejaVu Sans"],
        "mathtext.fontset":     "stixsans",
        "font.size":            10,

        # --- Axes ---
        "axes.labelsize":       10,
        "axes.labelcolor":      "black",
        "axes.linewidth":       0.8,
        "axes.facecolor":       _C["panel_bg"],
        "axes.edgecolor":       "black",

        # --- Grid ---
        "axes.grid":            True,
        "grid.color":           _C["grid"],
        "grid.linewidth":       0.45,
        "grid.linestyle":       ":",
        "grid.alpha":           0.85,

        # --- Ticks ---
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.major.size":     4.0,
        "ytick.major.size":     4.0,
        "xtick.minor.size":     2.2,
        "ytick.minor.size":     2.2,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.minor.width":    0.45,
        "ytick.minor.width":    0.45,
        "xtick.minor.visible":  True,
        "ytick.minor.visible":  True,
        "xtick.top":            True,
        "ytick.right":          True,
        "xtick.color":          "black",
        "ytick.color":          "black",
        "xtick.labelcolor":     "black",
        "ytick.labelcolor":     "black",

        # --- Legend ---
        "legend.fontsize":      10,
        "legend.framealpha":    0.96,
        "legend.edgecolor":     "black",
        "legend.fancybox":      True,
        "legend.handlelength":  2.2,
        "legend.handleheight":  0.85,
        "legend.borderpad":     0.55,
        "legend.labelspacing":  0.4,
        "legend.shadow":        False,

        # --- Save ---
        "figure.dpi":           150,
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,

        # --- Lines ---
        "lines.linewidth":      1.8,
        "patch.linewidth":      0.5,
    })


# ──────────────────────────────────────────────────────────────────────
# Colorblind-safe palette (flat-UI, print-safe)
# ──────────────────────────────────────────────────────────────────────

_STYLE = {
    'EKF':      {'color': '#C0392B', 'ls': '-',  'marker': 's', 'label': 'EKF (nominal)',   'hatch': '//'},
    'EKF_zm':   {'color': '#C0392B', 'ls': '-',  'marker': 's', 'label': 'EKF (no margin)', 'hatch': '//'},
    'EKF_true':    {'color': '#27AE60', 'ls': '-', 'marker': '^', 'label': 'EKF (true)',    'hatch': '\\\\'},
    'EKF_true_zm': {'color': '#27AE60', 'ls': '-', 'marker': '^', 'label': 'EKF (true, no margin)', 'hatch': '\\\\'},
    'DREKF':    {'color': '#2980B9', 'ls': '-',   'marker': 'o', 'label': 'DR-EKF',          'hatch': ''},
    'DREKF_zm': {'color': '#2980B9', 'ls': '-',   'marker': 'o', 'label': 'DR-EKF (no margin)', 'hatch': ''},
}

CONDITION_ORDER_DEFAULT = ['EKF', 'EKF_true', 'DREKF']
CONDITION_ORDER_ZM = ['EKF_zm', 'EKF_true_zm', 'DREKF_zm']


def _get_condition_order(conds):
    if any(k in conds for k in CONDITION_ORDER_DEFAULT):
        return CONDITION_ORDER_DEFAULT
    if any(k in conds for k in CONDITION_ORDER_ZM):
        return CONDITION_ORDER_ZM
    return list(conds.keys())


def _sty(key):
    """Return style dict for a condition key; fall back gracefully."""
    return _STYLE.get(key, {'color': '#888888', 'ls': '-', 'marker': 'x',
                            'label': key, 'hatch': ''})


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────

def _tint(hex_color, amount=0.58):
    """Blend hex_color toward white by `amount` (0 = original, 1 = white)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - (1 - r) * (1 - amount),
            1 - (1 - g) * (1 - amount),
            1 - (1 - b) * (1 - amount))


def _fill_band(ax, x, lo, hi, color):
    """Draw a tinted fill band with subtle border lines (reference style)."""
    ax.fill_between(x, lo, hi, color=_tint(color, 0.62),
                    zorder=2, linewidth=0)
    ax.plot(x, hi, color=_tint(color, 0.30),
            linewidth=0.7, alpha=0.35, zorder=3)
    ax.plot(x, lo, color=_tint(color, 0.30),
            linewidth=0.7, alpha=0.35, zorder=3)


def _time_axis(params):
    return np.arange(params['T'] + 1) * params['DT']


def _min_clearance(trial, obs_center, obs_radius):
    pts = trial['x_true'][:, :2, 0]
    dists_to_centre = np.linalg.norm(pts - obs_center[np.newaxis, :], axis=1)
    return float(np.min(dists_to_centre - obs_radius))


def _compute_pdop_grid(beacon_pos, x_range, y_range, resolution=150):
    x_grid = np.linspace(*x_range, resolution)
    y_grid = np.linspace(*y_range, resolution)
    pdop = np.full((resolution, resolution), np.nan)
    n_b = len(beacon_pos)
    for i, py in enumerate(y_grid):
        for j, px in enumerate(x_grid):
            H_pos = np.zeros((n_b, 2))
            for k in range(n_b):
                dx = px - beacon_pos[k, 0]
                dy = py - beacon_pos[k, 1]
                r = max(np.sqrt(dx**2 + dy**2), 0.5)
                H_pos[k, 0] = dx / r
                H_pos[k, 1] = dy / r
            F = H_pos.T @ H_pos
            try:
                pdop[i, j] = np.sqrt(np.trace(np.linalg.inv(F)))
            except np.linalg.LinAlgError:
                pass
    return x_grid, y_grid, pdop


def _draw_beacons(ax, beacon_pos, fontsize=9, label_offsets=None, color='0.35',
                  show_labels=True):
    """label_offsets: list of (dx, dy) in offset points per beacon, or None for default."""
    for i, b in enumerate(beacon_pos):
        ax.plot(b[0], b[1], 'D', color=color, markersize=10,
                markeredgecolor='black', markeredgewidth=0.8, zorder=10)
        if show_labels:
            ofs = label_offsets[i] if label_offsets is not None else (10, 0)
            ax.annotate(f'$b_{i+1}$', (b[0], b[1]),
                        textcoords='offset points', xytext=ofs,
                        fontsize=fontsize, fontweight='bold', color=color,
                        va='center', zorder=11,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.85),
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def _draw_obstacle(ax, obs_c, obs_r, label=False):
    circle = plt.Circle(obs_c, obs_r, facecolor='0.75', alpha=0.7,
                        edgecolor='0.3', linewidth=1.2, zorder=3)
    ax.add_patch(circle)
    if label:
        ax.text(obs_c[0], obs_c[1] - obs_r - 0.25, 'Obstacle',
                ha='center', va='top', fontsize=8, color='#8B4513', style='italic')


def _draw_start_goal(ax, start, goal, goal_radius):
    ax.plot(*start, marker='^', markersize=20, markerfacecolor='black',
            markeredgecolor='white', markeredgewidth=1.6,
            linestyle='none', zorder=8)
    ax.plot(*goal, marker='*', markersize=22, markerfacecolor='black',
            markeredgecolor='white', markeredgewidth=1.6,
            linestyle='none', zorder=8)


def _panel_label(ax, label, loc='upper left'):
    """Add (a), (b), ... panel label in corner."""
    positions = {
        'upper left':  (0.025, 0.965, 'left', 'top'),
        'upper right': (0.975, 0.965, 'right', 'top'),
        'lower left':  (0.025, 0.035, 'left', 'bottom'),
        'lower right': (0.975, 0.035, 'right', 'bottom'),
    }
    x, y, ha, va = positions[loc]
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va=va, ha=ha,
            zorder=20, color="black",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def _obs_zone_span(ax, params, alpha=0.08):
    obs_c = np.array(params['OBS_CENTER'])
    obs_r = params['OBS_RADIUS']
    v_approx = 1.0
    # Primary travel axis: x if goal_x > goal_y, else y
    goal = np.array(params['GOAL_POS'])
    travel_dist = max(goal[0], goal[1])
    travel_coord = obs_c[0] if goal[0] >= goal[1] else obs_c[1]
    t_enter = (travel_coord - obs_r) / v_approx
    t_exit = (travel_coord + obs_r) / v_approx
    ax.axvspan(t_enter, t_exit, color='#8B4513', alpha=alpha, zorder=0)


def _set_map_axes(ax, params):
    hw = params['ARENA_HALF_WIDTH']
    goal = np.array(params['GOAL_POS'])
    ax.set_xlim(-0.5, goal[0] + 1.5)
    ax.set_ylim(-hw, hw)
    ax.set_xlabel(r'x (m)')
    ax.set_ylabel(r'y (m)')
    ax.set_aspect('equal')


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — nav_main_4panel.pdf  (hero figure)
# ──────────────────────────────────────────────────────────────────────

def plot_main_4panel(data, save_dir):
    params = data['params']
    conds = data['conditions']
    time_ax = _time_axis(params)
    obs_c = np.array(params['OBS_CENTER'])
    obs_r = params['OBS_RADIUS']
    goal = np.array(params['GOAL_POS'])
    start = np.array(params['START_POS'])
    beacon_pos = np.array(params['BEACON_POS'])
    ORDER = _get_condition_order(conds)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.5), constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.08,
                        hspace=0.32, wspace=0.30)

    # ---- (a) Trajectories ----
    ax = axes[0, 0]
    _draw_obstacle(ax, obs_c, obs_r)
    _draw_beacons(ax, beacon_pos)
    _draw_start_goal(ax, start, goal, params['GOAL_RADIUS'])

    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        for r in c['results']:
            col_alpha = 0.40 if r['any_collision'] else 0.15
            col_ls = ':' if r['any_collision'] else '-'
            ax.plot(r['x_true'][:, 0, 0], r['x_true'][:, 1, 0],
                    color=s['color'], alpha=col_alpha, linewidth=0.6,
                    linestyle=col_ls, zorder=2)
        ax.plot([], [], color=s['color'], linewidth=1.8, linestyle=s['ls'],
                label=s['label'])

    _set_map_axes(ax, params)
    _panel_label(ax, '(a)', loc='upper left')

    # ---- (b) Safety margin ----
    ax = axes[0, 1]
    _obs_zone_span(ax, params)
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        margins_all = np.array([r['margins'] for r in c['results']])
        m_mean = np.nanmean(margins_all, axis=0)
        m_lo = np.nanpercentile(margins_all, 10, axis=0)
        m_hi = np.nanpercentile(margins_all, 90, axis=0)
        ax.plot(time_ax, m_mean, color=s['color'], linewidth=1.8,
                linestyle=s['ls'], label=s['label'], zorder=4)
        _fill_band(ax, time_ax, m_lo, m_hi, s['color'])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Safety margin $\delta_t$ (m)')
    ax.set_ylim(bottom=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    _panel_label(ax, '(b)')

    # ---- (c) Distance to goal ----
    ax = axes[1, 0]
    ax.axhline(params['GOAL_RADIUS'], color='#B8860B', linewidth=1.0,
               linestyle=':', alpha=0.8)
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        dists_all = []
        for r in c['results']:
            pts = r['x_true'][:, :2, 0]
            dists_all.append(np.linalg.norm(pts - goal[np.newaxis, :], axis=1))
        dists_all = np.array(dists_all)
        d_mean = np.nanmean(dists_all, axis=0)
        d_lo = np.nanpercentile(dists_all, 10, axis=0)
        d_hi = np.nanpercentile(dists_all, 90, axis=0)
        ax.plot(time_ax, d_mean, color=s['color'], linewidth=1.8,
                linestyle=s['ls'], label=s['label'], zorder=4)
        _fill_band(ax, time_ax, d_lo, d_hi, s['color'])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to goal (m)')
    ax.set_ylim(bottom=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    _panel_label(ax, '(c)')

    # ---- (d) Collision rate ----
    ax = axes[1, 1]
    keys_present = [k for k in ORDER if k in conds]
    labels_bar = [_sty(k)['label'] for k in keys_present]
    rates = [conds[k]['collision_rate'] * 100.0 for k in keys_present]
    colors_bar = [_sty(k)['color'] for k in keys_present]
    hatches = [_sty(k)['hatch'] for k in keys_present]

    x_pos = np.arange(len(labels_bar))
    bars = ax.bar(x_pos, rates, color=colors_bar, alpha=0.8, width=0.5,
                  edgecolor='black', linewidth=0.8)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_bar, fontsize=8, rotation=12, ha='right')
    ax.set_ylabel('Collision rate (%)')
    ax.set_ylim(0, max(max(rates) * 1.4, 10))
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f'{rate:.0f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(True, axis='y', color=_C["grid"], linewidth=0.45,
            linestyle=":", alpha=0.85)
    ax.grid(False, axis='x')
    ax.tick_params(which="both", direction="in", top=True, right=True)
    _panel_label(ax, '(d)')

    # Shared legend across the top of the figure
    legend_handles = []
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        legend_handles.append(Line2D([0], [0], color=s['color'], linewidth=1.8,
                                      linestyle=s['ls'], label=s['label']))
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.0), ncol=len(legend_handles),
               fontsize=10, frameon=False, columnspacing=2.0)

    save_path = os.path.join(save_dir, 'nav_main_4panel.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 1b — nav_trajectories_overlay.pdf  (standalone trajectory map)
# ──────────────────────────────────────────────────────────────────────

def _add_arrow(ax, px, py, color, interval=15, size=12, zorder=6):
    """Add direction arrows along a trajectory."""
    for i in range(interval, len(px), interval):
        dx = px[i] - px[i - 1]
        dy = py[i] - py[i - 1]
        if abs(dx) + abs(dy) < 1e-6:
            continue
        ax.annotate('', xy=(px[i], py[i]),
                    xytext=(px[i] - dx * 0.5, py[i] - dy * 0.5),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5, shrinkA=0, shrinkB=0),
                    zorder=zorder)


def plot_trajectories_overlay(data, save_dir, n_show=5):
    """Trajectory overlay with PDOP heatmap background."""
    params = data['params']
    conds = data['conditions']
    obs_c = np.array(params['OBS_CENTER'])
    obs_r = params['OBS_RADIUS']
    goal = np.array(params['GOAL_POS'])
    start = np.array(params['START_POS'])
    beacon_pos = np.array(params['BEACON_POS'])
    hw = params['ARENA_HALF_WIDTH']
    ORDER = _get_condition_order(conds)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.12)

    # Obstacle with shadow effect
    shadow = plt.Circle((obs_c[0] + 0.08, obs_c[1] - 0.08), obs_r,
                         facecolor='black', alpha=0.15, zorder=2)
    ax.add_patch(shadow)
    _draw_obstacle(ax, obs_c, obs_r)

    # Trajectories (drawn before beacons/start/goal so those sit on top)
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        results = c['results']
        indices = np.linspace(0, len(results) - 1, n_show, dtype=int)
        for i in indices:
            r = results[i]
            px = r['x_true'][:, 0, 0]
            py = r['x_true'][:, 1, 0]
            if r['any_collision']:
                coll_idx = int(np.argmax(r['collisions']))
                ax.plot(px[:coll_idx + 1], py[:coll_idx + 1],
                        color=s['color'], alpha=0.85, linewidth=1.8,
                        linestyle='-', zorder=4)
                ax.plot(px[coll_idx], py[coll_idx],
                        marker='x', color=s['color'], markersize=12,
                        markeredgewidth=2.5,
                        alpha=0.95, zorder=7)
            else:
                ax.plot(px, py, color=s['color'], alpha=0.7,
                        linewidth=1.8, linestyle='-', zorder=4)

    # Beacons and start/goal on top of trajectories
    _draw_beacons(ax, beacon_pos, fontsize=10)
    _draw_start_goal(ax, start, goal, params['GOAL_RADIUS'])

    # Start / Goal labels
    ax.text(start[0] + 0.3, start[1] + 0.5, 'Start', fontsize=10,
            fontweight='bold', ha='center', va='bottom', zorder=12,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    ax.text(goal[0], goal[1] + 0.5, 'Goal', fontsize=10,
            fontweight='bold', ha='center', va='bottom', zorder=12,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # Legend inside plot
    legend_handles = []
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        legend_handles.append(Line2D([0], [0], color=s['color'], linewidth=1.8,
                                      linestyle=s['ls'], label=s['label']))
    legend_handles.append(Line2D([0], [0], marker='x', color='black',
                                  linestyle='', markersize=8,
                                  markeredgewidth=1.8,
                                  label='Collision'))
    ax.legend(handles=legend_handles, loc='lower right',
              frameon=False, fontsize=10)

    ax.set_xlim(-0.5, goal[0] + 2.0)
    ax.set_ylim(-hw, hw)
    ax.set_xlabel(r'x (m)')
    ax.set_ylabel(r'y (m)')
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_aspect('equal')

    save_path = os.path.join(save_dir, 'nav_trajectories_overlay.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 1c — nav_traj_margin_combined.pdf  (trajectory + margin)
# ──────────────────────────────────────────────────────────────────────

def plot_traj_margin_combined(data, save_dir, n_show=5):
    """Combined landscape figure: trajectories (left) + safety margin (right)."""
    params = data['params']
    conds = data['conditions']
    obs_c = np.array(params['OBS_CENTER'])
    obs_r = params['OBS_RADIUS']
    goal = np.array(params['GOAL_POS'])
    start = np.array(params['START_POS'])
    beacon_pos = np.array(params['BEACON_POS'])
    hw = params['ARENA_HALF_WIDTH']
    time_ax = _time_axis(params)
    ORDER = _get_condition_order(conds)

    # --- Compute panel sizes so (a) and (b) have identical height ---
    x_span = (goal[0] + 2.0) - (-0.5)           # data x-range
    y_span = 2 * hw                              # data y-range
    traj_w = 6.0                                  # trajectory panel width (inches)
    traj_h = traj_w * (y_span / x_span)          # height from equal aspect ratio
    margin_w = traj_w                              # same width as trajectory panel
    gap = 1.2                                     # gap between panels (inches)
    left_pad = 0.72
    right_pad = 0.25
    bot_pad = 1.1                                 # space for legend below
    top_pad = 0.15
    fig_w = left_pad + traj_w + gap + margin_w + right_pad
    fig_h = bot_pad + traj_h + top_pad

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Both panels share the same bottom and height (in figure fractions)
    bot = bot_pad / fig_h
    ax_h = traj_h / fig_h
    ax_traj = fig.add_axes([left_pad / fig_w, bot,
                            traj_w / fig_w, ax_h])
    ax_marg = fig.add_axes([(left_pad + traj_w + gap) / fig_w, bot,
                            margin_w / fig_w, ax_h])

    # ---- Left panel: trajectory overlay ----
    shadow = plt.Circle((obs_c[0] + 0.08, obs_c[1] - 0.08), obs_r,
                         facecolor='black', alpha=0.15, zorder=2)
    ax_traj.add_patch(shadow)
    _draw_obstacle(ax_traj, obs_c, obs_r)

    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        results = c['results']
        indices = np.linspace(0, len(results) - 1, n_show, dtype=int)
        for i in indices:
            r = results[i]
            px = r['x_true'][:, 0, 0]
            py = r['x_true'][:, 1, 0]
            if r['any_collision']:
                coll_idx = int(np.argmax(r['collisions']))
                ax_traj.plot(px[:coll_idx + 1], py[:coll_idx + 1],
                             color=s['color'], alpha=0.85, linewidth=1.8,
                             linestyle='-', zorder=4)
                ax_traj.plot(px[coll_idx], py[coll_idx],
                             marker='x', color='black', markersize=8,
                             markeredgewidth=1.8,
                             alpha=0.95, zorder=7)
            else:
                ax_traj.plot(px, py, color=s['color'], alpha=0.7,
                             linewidth=1.8, linestyle='-', zorder=4)

    _draw_beacons(ax_traj, beacon_pos, show_labels=False)
    _draw_start_goal(ax_traj, start, goal, params['GOAL_RADIUS'])
    ax_traj.text(start[0] + 0.8, start[1] + 0.5, 'Start', fontsize=15,
                 fontweight='bold', ha='center', va='bottom', zorder=12,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    ax_traj.text(goal[0], goal[1] + 0.5, 'Goal', fontsize=15,
                 fontweight='bold', ha='center', va='bottom', zorder=12,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    ax_traj.set_xlim(-0.5, goal[0] + 2.0)
    ax_traj.set_ylim(-hw, hw)
    ax_traj.set_xlabel(r'x (m)', fontsize=15)
    ax_traj.set_ylabel(r'y (m)', fontsize=15)
    ax_traj.tick_params(which="major", direction="in", top=True, right=True,
                        labelsize=13, length=6, width=1.0)
    ax_traj.tick_params(which="minor", direction="in", top=True, right=True,
                        length=3.5, width=0.6)
    ax_traj.xaxis.set_major_locator(plt.MultipleLocator(3))
    ax_traj.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax_traj.set_aspect('equal')
    ax_traj.text(0.025, 0.965, '(a)', transform=ax_traj.transAxes,
                 fontsize=16, fontweight='bold', va='top', ha='left',
                 zorder=20, color="black",
                 path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # ---- Right panel: safety margin time series ----
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        margins_all = np.array([r['margins'] for r in c['results']])
        m_mean = np.nanmean(margins_all, axis=0)
        m_lo = np.nanpercentile(margins_all, 10, axis=0)
        m_hi = np.nanpercentile(margins_all, 90, axis=0)
        ax_marg.plot(time_ax, m_mean, color=s['color'], linewidth=1.8,
                     linestyle=s['ls'], zorder=4)
        _fill_band(ax_marg, time_ax, m_lo, m_hi, s['color'])

    ax_marg.set_xlabel('Time (s)', fontsize=15)
    ax_marg.set_ylabel(r'Safety margin $\delta_t$ (m)', fontsize=15)
    ax_marg.tick_params(which="major", direction="in", top=True, right=True,
                        labelsize=13, length=6, width=1.0)
    ax_marg.tick_params(which="minor", direction="in", top=True, right=True,
                        length=3.5, width=0.6)
    ax_marg.xaxis.set_major_locator(plt.MultipleLocator(3))
    ax_marg.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    # x-axis max = longest EKF_true trial time
    ekf_true_key = 'EKF_true' if 'EKF_true' in conds else 'EKF_true_zm'
    if ekf_true_key in conds:
        t_maxes = [r['t_reach'] for r in conds[ekf_true_key]['results']
                   if r['t_reach'] is not None]
        t_max_steps = max(t_maxes) if t_maxes else params['T']
    else:
        t_max_steps = params['T']
    ax_marg.set_xlim(0, t_max_steps * params['DT'])
    ax_marg.set_ylim(bottom=0)
    ax_marg.text(0.025, 0.965, '(b)', transform=ax_marg.transAxes,
                 fontsize=16, fontweight='bold', va='top', ha='left',
                 zorder=20, color="black",
                 path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # ---- Shared legend at bottom ----
    legend_handles = []
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        legend_handles.append(Line2D([0], [0], color=s['color'], linewidth=1.8,
                                      linestyle=s['ls'], label=s['label']))
    legend_handles.append(Line2D([0], [0], marker='x', color='black',
                                  linestyle='', markersize=8,
                                  markeredgewidth=1.8,
                                  label='Collision'))
    legend_handles.append(Line2D([0], [0], marker='D', color='0.35',
                                  linestyle='', markersize=7,
                                  markeredgecolor='black', markeredgewidth=0.8,
                                  label='UWB beacon'))
    fig.legend(handles=legend_handles, fontsize=16,
               loc='lower center', bbox_to_anchor=(0.50, 0.01),
               ncol=5, frameon=False, columnspacing=1.5)
    save_path = os.path.join(save_dir, 'nav_traj_margin_combined.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 2 — nav_trajectories.pdf  (side-by-side per filter)
# ──────────────────────────────────────────────────────────────────────

def plot_trajectories(data, save_dir):
    params = data['params']
    conds = data['conditions']
    obs_c = np.array(params['OBS_CENTER'])
    obs_r = params['OBS_RADIUS']
    goal = np.array(params['GOAL_POS'])
    start = np.array(params['START_POS'])
    beacon_pos = np.array(params['BEACON_POS'])
    ORDER = _get_condition_order(conds)
    keys_present = [k for k in ORDER if k in conds]
    n_panels = len(keys_present)

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(7.5, 3.5 * n_panels + 1.0),
                             squeeze=False, constrained_layout=False)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for idx, key in enumerate(keys_present):
        ax = axes[idx]
        s = _sty(key)
        c = conds[key]

        _draw_obstacle(ax, obs_c, obs_r, label=(idx == 0))
        _draw_beacons(ax, beacon_pos, fontsize=8)
        _draw_start_goal(ax, start, goal, params['GOAL_RADIUS'])

        # Inflated obstacle (mean margin)
        marg = c.get('mean_margin', 0.0)
        if marg > 0:
            inf_c = plt.Circle(obs_c, obs_r + marg, color=s['color'],
                               fill=False, linewidth=1.2, linestyle=':',
                               alpha=0.6, zorder=2)
            ax.add_patch(inf_c)

        n_coll = 0
        n_safe = 0
        for r in c['results']:
            if r['any_collision']:
                ax.plot(r['x_true'][:, 0, 0], r['x_true'][:, 1, 0],
                        color=s['color'], alpha=0.45, linewidth=0.7,
                        linestyle=':', zorder=2)
                n_coll += 1
            else:
                ax.plot(r['x_true'][:, 0, 0], r['x_true'][:, 1, 0],
                        color=s['color'], alpha=0.18, linewidth=0.6, zorder=2)
                n_safe += 1

        _set_map_axes(ax, params)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        # Only show x-label on bottom panel
        if idx < n_panels - 1:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
        n_total = len(c['results'])
        coll_pct = 100.0 * n_coll / n_total if n_total > 0 else 0
        ax.set_title(f'{s["label"]}\n({n_total} trials, {coll_pct:.0f}% collision)',
                     fontsize=9, pad=8)
        _panel_label(ax, f'({chr(ord("a") + idx)})')

    # Shared legend at bottom
    handles = [
        Line2D([0], [0], color='#8B4513', linewidth=6, alpha=0.75,
               label='Obstacle'),
        Line2D([0], [0], color='gray', linewidth=1.0, linestyle=':',
               label='Inflated radius'),
        Line2D([0], [0], marker='^', color='#E69F00', markersize=8,
               markeredgecolor='black', linestyle='', label='UWB beacon'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9,
               frameon=False)

    plt.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.06,
                        hspace=0.25)
    save_path = os.path.join(save_dir, 'nav_trajectories.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 3 — margin_time_series.pdf
# ──────────────────────────────────────────────────────────────────────

def plot_margin_time_series(data, save_dir):
    params = data['params']
    conds = data['conditions']
    time_ax = _time_axis(params)
    ORDER = _get_condition_order(conds)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.14)
    _obs_zone_span(ax, params, alpha=0.10)

    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        margins_all = np.array([r['margins'] for r in c['results']])
        m_mean = np.nanmean(margins_all, axis=0)
        m_lo = np.nanpercentile(margins_all, 10, axis=0)
        m_hi = np.nanpercentile(margins_all, 90, axis=0)
        ax.plot(time_ax, m_mean, color=s['color'], linewidth=1.8,
                linestyle=s['ls'], label=s['label'], zorder=4)
        _fill_band(ax, time_ax, m_lo, m_hi, s['color'])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Safety margin $\delta_t$ (m)')
    ax.set_ylim(bottom=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc='upper right', frameon=False, fontsize=10)

    save_path = os.path.join(save_dir, 'margin_time_series.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 4 — outcome_bar.pdf
# ──────────────────────────────────────────────────────────────────────

def plot_outcome_bar(data, save_dir):
    conds = data['conditions']
    ORDER = _get_condition_order(conds)
    keys_present = [k for k in ORDER if k in conds]

    labels_bar, coll_pcts, safe_pcts, timeout_pcts = [], [], [], []
    for key in keys_present:
        c = conds[key]
        results = c['results']
        n = len(results)
        n_coll = sum(1 for r in results if r['any_collision'])
        n_safe = sum(1 for r in results if r['goal_reached'] and not r['any_collision'])
        n_to = n - n_coll - n_safe
        labels_bar.append(_sty(key)['label'])
        coll_pcts.append(100.0 * n_coll / n)
        safe_pcts.append(100.0 * n_safe / n)
        timeout_pcts.append(100.0 * n_to / n)

    x_pos = np.arange(len(labels_bar))
    width = 0.50
    c_safe, c_to, c_coll = '#009E73', '#E69F00', '#D55E00'

    fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.14)

    b1 = ax.bar(x_pos, safe_pcts, width, color=c_safe, edgecolor='black',
                linewidth=0.6, label='Safe goal-reach')
    b2 = ax.bar(x_pos, timeout_pcts, width, bottom=safe_pcts,
                color=c_to, edgecolor='black', linewidth=0.6, label='Timed out')
    bottoms3 = [s + t for s, t in zip(safe_pcts, timeout_pcts)]
    b3 = ax.bar(x_pos, coll_pcts, width, bottom=bottoms3,
                color=c_coll, edgecolor='black', linewidth=0.6, label='Collision')

    def _ann(bars, vals, bots):
        for bar, v, b in zip(bars, vals, bots):
            if v > 4:
                ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                        f'{v:.0f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')

    _ann(b1, safe_pcts, [0] * len(safe_pcts))
    _ann(b2, timeout_pcts, safe_pcts)
    _ann(b3, coll_pcts, bottoms3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_bar, fontsize=9, rotation=10, ha='right')
    ax.set_ylabel('Trials (%)')
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9, loc='upper right', frameon=False)
    ax.grid(True, axis='y', color=_C["grid"], linewidth=0.45,
            linestyle=":", alpha=0.85)
    ax.grid(False, axis='x')
    ax.tick_params(which="both", direction="in", top=True, right=True)

    save_path = os.path.join(save_dir, 'outcome_bar.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 5 — clearance_hist.pdf
# ──────────────────────────────────────────────────────────────────────

def plot_clearance_hist(data, save_dir):
    params = data['params']
    conds = data['conditions']
    obs_c = np.array(params['OBS_CENTER'])
    obs_r = params['OBS_RADIUS']
    ORDER = _get_condition_order(conds)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.14)

    all_clearances = {}
    for key in ORDER:
        if key not in conds:
            continue
        c = conds[key]
        all_clearances[key] = [_min_clearance(r, obs_c, obs_r) for r in c['results']]

    all_vals = np.concatenate(list(all_clearances.values()))
    lo = min(all_vals.min() - 0.05, -0.1)
    hi = max(all_vals.max() + 0.05, 1.5)
    bins = np.linspace(lo, hi, 30)

    for key in ORDER:
        if key not in all_clearances:
            continue
        s = _sty(key)
        ax.hist(all_clearances[key], bins=bins, color=s['color'],
                alpha=0.50, edgecolor='white', linewidth=0.5,
                label=s['label'])

    ax.axvline(0.0, color='black', linewidth=1.5, linestyle='--', zorder=5)
    ax.axvspan(lo, 0.0, color='red', alpha=0.05, zorder=0)
    ax.text(lo * 0.6, ax.get_ylim()[1] * 0.85, 'Collision\nzone',
            fontsize=8, color='#D55E00', ha='center', style='italic')

    ax.set_xlabel('Min. clearance to obstacle (m)')
    ax.set_ylabel('Number of trials')
    ax.legend(fontsize=9, loc='upper right', frameon=False)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    save_path = os.path.join(save_dir, 'clearance_hist.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 6 — covariance_and_theta.pdf
# ──────────────────────────────────────────────────────────────────────

def plot_covariance_and_theta(data, save_dir):
    params = data['params']
    conds = data['conditions']
    time_ax = _time_axis(params)
    ORDER = _get_condition_order(conds)

    sample_results = next(iter(conds.values()))['results']
    if 'trace_P' not in sample_results[0]:
        print('Skipping covariance_and_theta plot (no trace_P data)')
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.10,
                        hspace=0.30, wspace=0.30)

    def _plot_ts(ax, data_key, ylabel, panel_lbl):
        _obs_zone_span(ax, params)
        for key in ORDER:
            if key not in conds:
                continue
            s = _sty(key)
            c = conds[key]
            vals = np.array([r[data_key] for r in c['results']])
            m = np.nanmean(vals, axis=0)
            lo = np.nanpercentile(vals, 10, axis=0)
            hi = np.nanpercentile(vals, 90, axis=0)
            ax.plot(time_ax, m, color=s['color'], linewidth=1.5,
                    linestyle=s['ls'], label=s['label'], zorder=4)
            _fill_band(ax, time_ax, lo, hi, s['color'])
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(fontsize=8, loc='upper right', frameon=False)
        _panel_label(ax, panel_lbl)

    _plot_ts(axes[0, 0], 'trace_P', r'$\mathrm{Tr}(\Sigma_{x,t})$', '(a)')

    if 'trace_P_pos' in sample_results[0]:
        _plot_ts(axes[0, 1], 'trace_P_pos',
                 r'$[\Sigma_{x,t}]_{11} + [\Sigma_{x,t}]_{22}$', '(b)')
    else:
        axes[0, 1].text(0.5, 0.5, 'No data', transform=axes[0, 1].transAxes,
                        ha='center', fontsize=10, color='gray')

    if 'P_theta' in sample_results[0]:
        _plot_ts(axes[1, 0], 'P_theta', r'$[\Sigma_{x,t}]_{33}$', '(c)')
    else:
        axes[1, 0].text(0.5, 0.5, 'No data', transform=axes[1, 0].transAxes,
                        ha='center', fontsize=10, color='gray')

    # (d) theta_eff
    ax = axes[1, 1]
    _obs_zone_span(ax, params)
    has_theta = False
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        theta_all = np.array([r['theta_eff'] for r in c['results']])
        if np.nanmax(theta_all) < 1e-12:
            continue
        has_theta = True
        th_m = np.nanmean(theta_all, axis=0)
        th_lo = np.nanpercentile(theta_all, 10, axis=0)
        th_hi = np.nanpercentile(theta_all, 90, axis=0)
        ax.plot(time_ax, th_m, color=s['color'], linewidth=1.5,
                linestyle=s['ls'], label=s['label'], zorder=4)
        _fill_band(ax, time_ax, th_lo, th_hi, s['color'])
    ax.set_ylabel(r'$\theta_{\epsilon,t}^{\mathrm{eff}}$')
    ax.set_ylim(bottom=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if has_theta:
        ax.legend(fontsize=8, loc='upper right', frameon=False)
    else:
        ax.text(0.5, 0.5, 'No DR-EKF data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
    _panel_label(ax, '(d)')

    for a in axes[1, :]:
        a.set_xlabel('Time (s)')

    save_path = os.path.join(save_dir, 'covariance_and_theta.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 7 — pdop_heatmap.pdf
# ──────────────────────────────────────────────────────────────────────

def plot_pdop_heatmap(data, save_dir):
    params = data['params']
    conds = data['conditions']
    obs_c = np.array(params['OBS_CENTER'])
    obs_r = params['OBS_RADIUS']
    goal = np.array(params['GOAL_POS'])
    start = np.array(params['START_POS'])
    beacon_pos = np.array(params['BEACON_POS'])
    hw = params['ARENA_HALF_WIDTH']
    ORDER = _get_condition_order(conds)

    print('  Computing PDOP heatmap...')
    x_grid, y_grid, pdop = _compute_pdop_grid(
        beacon_pos, x_range=(-0.5, goal[0] + 1.5),
        y_range=(-hw, hw), resolution=200)

    pdop_clipped = np.clip(pdop, 0.5, 8.0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.12)

    im = ax.pcolormesh(x_grid, y_grid, pdop_clipped,
                       cmap='inferno_r', shading='auto',
                       vmin=0.5, vmax=5.0, zorder=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=30)
    cbar.set_label('Position DOP', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    obs_patch = plt.Circle(obs_c, obs_r, facecolor='white', alpha=0.85,
                           zorder=5, edgecolor='black', linewidth=1.2)
    ax.add_patch(obs_patch)
    ax.text(obs_c[0], obs_c[1], 'Obs.', ha='center', va='center',
            fontsize=8, color='black', fontweight='bold', zorder=6)

    _draw_start_goal(ax, start, goal, params['GOAL_RADIUS'])
    _draw_beacons(ax, beacon_pos, fontsize=9)

    # Mean trajectories with white outline for visibility
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        all_px = np.array([r['x_true'][:, 0, 0] for r in c['results']])
        all_py = np.array([r['x_true'][:, 1, 0] for r in c['results']])
        mean_px = np.mean(all_px, axis=0)
        mean_py = np.mean(all_py, axis=0)
        ax.plot(mean_px, mean_py, color=s['color'], linewidth=2.0,
                linestyle=s['ls'], label=s['label'], zorder=7,
                path_effects=[pe.withStroke(linewidth=3.5, foreground='white',
                                            alpha=0.7)])

    _set_map_axes(ax, params)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(fontsize=9, loc='upper right', frameon=False)

    save_path = os.path.join(save_dir, 'pdop_heatmap.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 8 — dop_vs_margin.pdf
# ──────────────────────────────────────────────────────────────────────

def plot_dop_vs_margin(data, save_dir):
    params = data['params']
    conds = data['conditions']
    time_ax = _time_axis(params)
    beacon_pos = np.array(params['BEACON_POS'])
    ORDER = _get_condition_order(conds)

    first_key = next((k for k in ORDER if k in conds), None)
    if first_key is None:
        return
    c0 = conds[first_key]
    all_px = np.array([r['x_true'][:, 0, 0] for r in c0['results']])
    all_py = np.array([r['x_true'][:, 1, 0] for r in c0['results']])
    mean_px = np.mean(all_px, axis=0)
    mean_py = np.mean(all_py, axis=0)

    n_b = len(beacon_pos)
    pdop_traj = np.zeros(len(mean_px))
    for t_idx in range(len(mean_px)):
        H_pos = np.zeros((n_b, 2))
        for k in range(n_b):
            dx = mean_px[t_idx] - beacon_pos[k, 0]
            dy = mean_py[t_idx] - beacon_pos[k, 1]
            r = max(np.sqrt(dx**2 + dy**2), 0.5)
            H_pos[k, 0] = dx / r
            H_pos[k, 1] = dy / r
        F = H_pos.T @ H_pos
        try:
            pdop_traj[t_idx] = np.sqrt(np.trace(np.linalg.inv(F)))
        except np.linalg.LinAlgError:
            pdop_traj[t_idx] = np.nan

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.5), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1.2]},
                                    constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.12,
                        hspace=0.12)

    _obs_zone_span(ax1, params)
    ax1.plot(time_ax, pdop_traj, color='black', linewidth=1.5,
             label='PDOP along mean path', zorder=4)
    ax1.set_ylabel('PDOP')
    ax1.legend(fontsize=9, loc='upper right', frameon=False)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(which="both", direction="in", top=True, right=True)
    _panel_label(ax1, '(a)')

    _obs_zone_span(ax2, params)
    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        margins_all = np.array([r['margins'] for r in c['results']])
        m_mean = np.nanmean(margins_all, axis=0)
        ax2.plot(time_ax, m_mean, color=s['color'], linewidth=1.5,
                 linestyle=s['ls'], label=s['label'], zorder=4)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'Safety margin $\delta_t$ (m)')
    ax2.legend(fontsize=9, loc='upper right', frameon=False)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(which="both", direction="in", top=True, right=True)
    _panel_label(ax2, '(b)')

    save_path = os.path.join(save_dir, 'dop_vs_margin.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Figure 9 — mse_comparison.pdf
# ──────────────────────────────────────────────────────────────────────

def plot_mse_comparison(data, save_dir):
    """MSE of position estimation (EKF vs DR-EKF) over time."""
    params = data['params']
    conds = data['conditions']
    time_ax = _time_axis(params)
    ORDER = _get_condition_order(conds)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.14)

    for key in ORDER:
        if key not in conds:
            continue
        s = _sty(key)
        c = conds[key]
        # Compute per-trial MSE of position (x, y) at each timestep
        mse_all = []
        for r in c['results']:
            err = r['x_true'][:, :2, 0] - r['x_est'][:, :2, 0]  # (T+1, 2)
            mse_all.append(np.sum(err**2, axis=1))                # (T+1,)
        mse_all = np.array(mse_all)                               # (n_trials, T+1)
        mse_mean = np.nanmean(mse_all, axis=0)
        mse_lo = np.nanpercentile(mse_all, 10, axis=0)
        mse_hi = np.nanpercentile(mse_all, 90, axis=0)
        ax.plot(time_ax, mse_mean, color=s['color'], linewidth=1.5,
                linestyle=s['ls'], label=s['label'], zorder=4)
        _fill_band(ax, time_ax, mse_lo, mse_hi, s['color'])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Position MSE (m$^2$)')
    ax.set_ylim(bottom=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc='upper right', frameon=False, fontsize=10)

    save_path = os.path.join(save_dir, 'mse_comparison.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plots for UWB beacon safe navigation experiment.')
    parser.add_argument('--results_dir', default='results/safe_navigation_uwb',
                        help='Directory containing navigation_uwb_results.pkl')
    args = parser.parse_args()

    pkl_path = os.path.join(args.results_dir, 'navigation_uwb_results.pkl')
    if not os.path.exists(pkl_path):
        print(f'Error: {pkl_path} not found.  Run safe_navigation_uwb.py first.')
        return

    _setup_style()

    print(f'Loading results from: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    os.makedirs(args.results_dir, exist_ok=True)

    params = data['params']
    conds = data['conditions']
    print(f"  T={params['T']}, DT={params['DT']}")
    print(f"  Obstacle: centre={params['OBS_CENTER']}, radius={params['OBS_RADIUS']} m")
    print(f"  Beacons: {params['N_BEACONS']}")
    for i, b in enumerate(params['BEACON_POS']):
        print(f"    b{i+1} = ({b[0]:.1f}, {b[1]:.1f})")
    print()
    for key in _get_condition_order(conds):
        if key not in conds:
            continue
        c = conds[key]
        n = len(c['results'])
        cr = c['collision_rate'] * 100.0
        gr = c['goal_reach_rate'] * 100.0
        mm = c.get('mean_margin', float('nan'))
        tr = c.get('mean_t_reach', float('nan'))
        pl = c.get('mean_path_length', float('nan'))
        print(f"  {c['label']:18s}  n={n}  margin={mm:.3f}m  "
              f"coll={cr:.1f}%  goal={gr:.1f}%  t_reach={tr:.1f}  path={pl:.2f}m")
    print()

    print('Generating figures...')
    plot_main_4panel(data, args.results_dir)
    plot_trajectories_overlay(data, args.results_dir)
    plot_traj_margin_combined(data, args.results_dir)
    plot_trajectories(data, args.results_dir)
    plot_margin_time_series(data, args.results_dir)
    plot_outcome_bar(data, args.results_dir)
    plot_clearance_hist(data, args.results_dir)
    plot_covariance_and_theta(data, args.results_dir)
    plot_pdop_heatmap(data, args.results_dir)
    plot_dop_vs_margin(data, args.results_dir)
    plot_mse_comparison(data, args.results_dir)

    print(f'\nAll figures saved to: {args.results_dir}')


if __name__ == '__main__':
    main()
