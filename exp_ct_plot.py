#!/usr/bin/env python3
"""
exp_ct_plot.py: Trajectory and result plots for EKF vs DR_EKF_trace comparison from exp_ct_tracking.py.
CT model with radar measurements. Results loaded from ./results/exp_ct_tracking/.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Use TrueType (Type 42) fonts in PDF/PS to avoid Type 3 font errors on submission
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

RESULTS_DIR = "./results/exp_ct_tracking"
FILTERS_ORDER = ['EKF', 'DR_EKF_trace', 'DR_EKF_trace_multipass']


def load_data(results_path, dist):
    """Load optimal and all results from exp_ct_tracking.py."""
    optimal_file = os.path.join(results_path, f'optimal_results_{dist}.pkl')
    all_results_file = os.path.join(results_path, f'all_results_{dist}.pkl')

    if not os.path.exists(optimal_file):
        raise FileNotFoundError(f"Optimal results file not found: {optimal_file}")

    with open(optimal_file, 'rb') as f:
        optimal_results = pickle.load(f)

    all_results = None
    if os.path.exists(all_results_file):
        with open(all_results_file, 'rb') as f:
            all_results = pickle.load(f)

    return optimal_results, all_results


def load_detailed_results_for_filter(results_path, filter_name, theta_vals, dist):
    """Load detailed trajectory data for EKF or DR_EKF_trace variants (theta_vals has theta_eff for trace)."""
    if filter_name == 'EKF':
        filename = f'detailed_results_EKF_{dist}.pkl'
    elif filter_name == 'DR_EKF_trace':
        theta_eff = theta_vals['theta_eff']
        filename = f'detailed_results_DR_EKF_trace_teff{theta_eff:.4f}_{dist}.pkl'
    elif filter_name == 'DR_EKF_trace_multipass':
        theta_eff = theta_vals['theta_eff']
        filename = f'detailed_results_DR_EKF_trace_multipass_teff{theta_eff:.4f}_{dist}.pkl'
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    detailed_file = os.path.join(results_path, filename)
    if not os.path.exists(detailed_file):
        raise FileNotFoundError(f"Detailed results file not found: {detailed_file}")

    with open(detailed_file, 'rb') as f:
        return pickle.load(f)


def generate_desired_trajectory(T_total=10.0, dt=0.2):
    """Placeholder desired trajectory (CT plot uses true trajectory as reference)."""
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)
    Amp, slope, omega = 5.0, 1.0, 0.5
    px_d = Amp * np.sin(omega * time)
    py_d = slope * time
    vx_d = Amp * omega * np.cos(omega * time)
    vy_d = slope * np.ones(time_steps)
    omega_d = 0.1 * np.sin(0.3 * time)
    return np.array([px_d, py_d, vx_d, vy_d, omega_d]), time


def extract_trajectory_data_from_saved(optimal_results, results_path, dist):
    """Extract trajectory data for EKF and DR_EKF_trace from saved detailed results."""
    trajectory_data = {}

    for filt in FILTERS_ORDER:
        if filt not in optimal_results:
            print(f"Warning: Filter '{filt}' not in optimal results, skipping...")
            continue

        optimal_stats = optimal_results[filt]
        if filt == 'EKF':
            theta_vals = {}
            theta_str = "N/A"
        else:
            theta_vals = {'theta_eff': optimal_stats['theta_eff']}
            theta_str = f"θ_eff={optimal_stats['theta_eff']:.4f}"

        print(f"Loading trajectory data for {filt} with {theta_str}")

        try:
            detailed_results = load_detailed_results_for_filter(results_path, filt, theta_vals, dist)
            if filt not in detailed_results:
                continue

            filter_results = detailed_results[filt]
            sim_results = filter_results['results']

            est_trajectories = []
            true_trajectories = []
            for result in sim_results:
                est_trajectories.append(np.squeeze(result['est_state_traj'], axis=-1))
                true_trajectories.append(np.squeeze(result['state_traj'], axis=-1))

            if est_trajectories:
                est_trajectories = np.array(est_trajectories)
                true_trajectories = np.array(true_trajectories)
                trajectory_data[filt] = {
                    'mean': np.mean(est_trajectories, axis=0),
                    'std': np.std(est_trajectories, axis=0),
                    'theta_vals': theta_vals,
                    'theta_str': theta_str,
                    'true_mean': np.mean(true_trajectories, axis=0),
                    'num_sims': len(est_trajectories),
                    'all_trajs': est_trajectories,
                    'all_true_trajs': true_trajectories,
                }
                print(f"Successfully loaded {len(est_trajectories)} trajectories for {filt}")
        except FileNotFoundError as e:
            print(f"Could not load detailed results for {filt}: {e}")
        except Exception as e:
            print(f"Error loading data for {filt}: {e}")
            import traceback
            traceback.print_exc()

    return trajectory_data, FILTERS_ORDER


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
def _colors():
    return {
        'EKF': '#1f77b4',
        'DR_EKF_trace': '#d62728',
        'DR_EKF_trace_multipass': '#2ca02c',
    }


def _filter_names():
    return {
        'EKF': "Extended Kalman Filter (EKF)",
        'DR_EKF_trace': "DR-EKF (trace-tube, one-pass)",
        'DR_EKF_trace_multipass': "DR-EKF (trace-tube, multipass)",
    }


def plot_trajectory_subplots(trajectory_data, filters_order, desired_traj, time, dist):
    """Individual 2D X-Y trajectory plots with 1/20-std tube and true trajectory, using the same reference trajectory."""
    colors = _colors()
    filter_names = _filter_names()
    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    saved_files = []

    reference_xy = None
    for f in filters_order:
        if f in trajectory_data and 'true_mean' in trajectory_data[f]:
            reference_xy = trajectory_data[f]['true_mean'][:, :2]
            break

    for filt in filters_order:
        if filt not in trajectory_data:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        color = colors[filt]
        mean_traj = trajectory_data[filt]['mean']
        std_traj = trajectory_data[filt]['std']
        theta_str = trajectory_data[filt]['theta_str']

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

        ax.fill(tube_x, tube_y, color=color, alpha=0.3, label='1/20-std tube')
        ax.plot(x_mean, y_mean, '-', color=color, linewidth=2.5, label='Estimated Trajectory')

        if reference_xy is not None:
            ax.plot(reference_xy[:, 0], reference_xy[:, 1], ':', color='red', linewidth=2.0, alpha=0.8, label='True Trajectory')
            ax.scatter(reference_xy[0, 0], reference_xy[0, 1], marker='X', s=150, color='red', linewidth=3)
            ax.scatter(reference_xy[-1, 0], reference_xy[-1, 1], marker='X', s=150, color='red', linewidth=3)

        ax.set_xlabel('X position', fontsize=28)
        ax.set_ylabel('Y position', fontsize=28)
        title_text = filter_names[filt] if filt == 'EKF' else f"{filter_names[filt]}\n({theta_str})"
        ax.set_title(title_text, fontsize=28, pad=15)
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([0, 5, 10])
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(['1/20-std tube', 'Estimated Trajectory', 'True Trajectory'], fontsize=22, loc='best')
        plt.tight_layout()

        save_path = os.path.join(results_dir, f"traj_2d_EKF_{filt}_{dist}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        saved_files.append(os.path.basename(save_path))
        plt.close(fig)

    print(f"\nTrajectory plots saved to: {results_dir}")
    for f in saved_files:
        print(f"  - {f}")


def plot_sample_trajectories(trajectory_data, filters_order, desired_traj, time, dist, num_samples=20):
    """Plot multiple sample trajectories per filter."""
    colors = _colors()
    filter_names = _filter_names()
    target_filters = [f for f in filters_order if f in trajectory_data and 'all_trajs' in trajectory_data[f]]
    if not target_filters:
        print("No individual trajectory data available.")
        return

    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    reference_xy = None
    for f in filters_order:
        if f in trajectory_data and trajectory_data[f].get('true_mean') is not None:
            reference_xy = trajectory_data[f]['true_mean'][:, :2]
            break
    n_filters = len(target_filters)
    fig, axes = plt.subplots(1, n_filters, figsize=(6 * n_filters, 6), squeeze=False)
    axes = axes.flatten()

    for idx, filt in enumerate(target_filters):
        ax = axes[idx]
        color = colors[filt]
        all_trajs = trajectory_data[filt]['all_trajs']
        num_runs = all_trajs.shape[0]
        k = min(num_samples, num_runs)

        if reference_xy is not None:
            ax.plot(reference_xy[:, 0], reference_xy[:, 1], ':', color='red', linewidth=2.0, alpha=0.8, label='True')
        for ridx in range(k):
            traj = all_trajs[ridx]
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.25, linewidth=1.0)
        if reference_xy is not None:
            ax.scatter(reference_xy[0, 0], reference_xy[0, 1], marker='X', s=80, color='red', linewidth=2)
            ax.scatter(reference_xy[-1, 0], reference_xy[-1, 1], marker='X', s=80, color='red', linewidth=2)
        ax.set_xlabel('X position', fontsize=16)
        ax.set_ylabel('Y position', fontsize=16)
        ax.set_title(f"{filter_names[filt]}\n({k} sample trajectories)", fontsize=18, pad=10)
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([0, 5, 10])
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    for j in range(n_filters, len(axes)):
        axes[j].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color='red', linestyle=':', linewidth=2.0, label='True (mean)'),
        plt.Line2D([0], [0], color='#888888', linewidth=2.0, alpha=0.5, label='Sample trajs')
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=14, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.3)
    save_path = os.path.join(results_dir, f"traj_2d_EKF_sample_trajs_{num_samples}_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Sample trajectory figure saved to: {save_path}")


def plot_subplots_all_filters(trajectory_data, filters_order, desired_traj, time, dist):
    """One figure with all filters in subplots, using the same true reference trajectory in every panel."""
    colors = _colors()
    filter_names = _filter_names()
    available_filters = [f for f in filters_order if f in trajectory_data]
    n_filters = len(available_filters)
    alphabet_mapping = {filt: chr(ord('A') + i) for i, filt in enumerate(filters_order)}

    # Use only a single true reference trajectory (from the first filter; EKF and DR_EKF share the same reference so they should match).
    reference_xy = None
    for f in filters_order:
        if f in trajectory_data and 'true_mean' in trajectory_data[f]:
            reference_xy = trajectory_data[f]['true_mean'][:, :2]
            break

    fig, axes = plt.subplots(1, n_filters, figsize=(6 * n_filters, 6), squeeze=False)
    axes = axes.flatten()

    for idx, filt in enumerate(available_filters):
        ax = axes[idx]
        color = colors[filt]
        mean_traj = trajectory_data[filt]['mean']
        std_traj = trajectory_data[filt]['std']
        theta_str = trajectory_data[filt]['theta_str']
        x_mean, y_mean = mean_traj[:, 0], mean_traj[:, 1]
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

        if reference_xy is not None:
            ax.plot(reference_xy[:, 0], reference_xy[:, 1], ':', color='red', linewidth=1.5, alpha=0.8)
            ax.scatter(reference_xy[0, 0], reference_xy[0, 1], marker='X', s=80, color='red', linewidth=2)
            ax.scatter(reference_xy[-1, 0], reference_xy[-1, 1], marker='X', s=80, color='red', linewidth=2)
        ax.fill(tube_x, tube_y, color=color, alpha=0.3)
        ax.plot(x_mean, y_mean, '-', color=color, linewidth=2)
        alphabet_label = alphabet_mapping.get(filt, '')
        ax.set_title(f"({alphabet_label}) {filter_names[filt]}", fontsize=16, pad=8)
        ax.set_xlabel('X position', fontsize=14)
        ax.set_ylabel('Y position', fontsize=14)
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([0, 5, 10])
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    legend_elements = [
        plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, label='True (mean)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, label='1/20-std tube'),
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Est. (mean)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=16, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.3)
    save_path = os.path.join(RESULTS_DIR, f"traj_2d_EKF_subplots_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Subplots figure saved to: {save_path}")


def plot_subplots_single_run(trajectory_data, filters_order, run_idx=0, dist='normal'):
    """Select a single run and show True vs EKF/DR-EKF estimated trajectories in subplots."""
    colors = _colors()
    filter_names = _filter_names()
    available_filters = [f for f in filters_order if f in trajectory_data]
    # Requires both all_true_trajs and all_trajs.
    available_filters = [f for f in available_filters
                         if 'all_true_trajs' in trajectory_data[f] and 'all_trajs' in trajectory_data[f]]
    n_filters = len(available_filters)
    if n_filters == 0:
        print("Warning: No trajectory data with per-run true/est. Skipping single-run plot.")
        return

    num_sims = trajectory_data[available_filters[0]]['num_sims']
    if run_idx >= num_sims:
        run_idx = 0
    # The true trajectory at the same run_idx is identical for all filters (same seed).
    true_xy = trajectory_data[available_filters[0]]['all_true_trajs'][run_idx][:, :2]

    fig, axes = plt.subplots(1, n_filters, figsize=(6 * n_filters, 6), squeeze=False)
    axes = axes.flatten()
    alphabet = {f: chr(ord('A') + i) for i, f in enumerate(filters_order)}

    for idx, filt in enumerate(available_filters):
        ax = axes[idx]
        color = colors[filt]
        est_xy = trajectory_data[filt]['all_trajs'][run_idx][:, :2]
        theta_str = trajectory_data[filt]['theta_str']

        ax.plot(true_xy[:, 0], true_xy[:, 1], ':', color='red', linewidth=2, alpha=0.9, label='True')
        ax.scatter(true_xy[0, 0], true_xy[0, 1], marker='X', s=120, color='red', zorder=5)
        ax.scatter(true_xy[-1, 0], true_xy[-1, 1], marker='X', s=120, color='red', zorder=5)
        ax.plot(est_xy[:, 0], est_xy[:, 1], '-', color=color, linewidth=2, label='Est.')
        ax.set_title(f"({alphabet.get(filt, '')}) {filter_names[filt]}\nRun {run_idx + 1}/{num_sims}" + (f" ({theta_str})" if theta_str != "N/A" else ""), fontsize=14, pad=8)
        ax.set_xlabel('X position', fontsize=14)
        ax.set_ylabel('Y position', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(f'Single run (run index {run_idx})', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"traj_2d_single_run_run{run_idx}_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Single-run figure saved to: {save_path}")


def plot_performance_vs_robustness(all_results, dist):
    """MSE vs θ_eff for EKF, DR_EKF_trace, and DR_EKF_trace_multipass (theta_eff is the single robustness parameter)."""
    if all_results is None:
        print("Warning: No all_results. Skipping performance vs robustness.")
        return

    colors = _colors()
    filter_names = _filter_names()
    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    ekf_mse_mean = ekf_mse_std = None
    if 'EKF' in all_results and 'no_theta' in all_results['EKF']:
        ekf_mse_mean = all_results['EKF']['no_theta']['mse_mean']
        ekf_mse_std = all_results['EKF']['no_theta'].get('mse_std', 0.0)

    # x-axis range for EKF band: union of trace and multipass theta_eff
    all_theta_eff = []

    # DR_EKF_trace: keys are theta_eff (float)
    theta_eff_vals = []
    mse_means = []
    mse_stds = []
    if 'DR_EKF_trace' in all_results:
        for theta_eff in sorted(all_results['DR_EKF_trace'].keys()):
            if isinstance(theta_eff, (int, float)):
                r = all_results['DR_EKF_trace'][theta_eff]
                theta_eff_vals.append(theta_eff)
                mse_means.append(r['mse_mean'])
                mse_stds.append(r.get('mse_std', 0.0))
                all_theta_eff.append(theta_eff)

    # DR_EKF_trace_multipass: keys are theta_eff (float)
    theta_eff_vals_mp = []
    mse_means_mp = []
    mse_stds_mp = []
    if 'DR_EKF_trace_multipass' in all_results:
        for theta_eff in sorted(all_results['DR_EKF_trace_multipass'].keys()):
            if isinstance(theta_eff, (int, float)):
                r = all_results['DR_EKF_trace_multipass'][theta_eff]
                theta_eff_vals_mp.append(theta_eff)
                mse_means_mp.append(r['mse_mean'])
                mse_stds_mp.append(r.get('mse_std', 0.0))
                all_theta_eff.append(theta_eff)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Mean Squared Error vs Robustness Parameter', fontsize=18, pad=20)
    ax.set_xlabel(r'Robustness Parameter $\theta_{\epsilon}$', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    if ekf_mse_mean is not None:
        x_min, x_max = (min(all_theta_eff), max(all_theta_eff)) if all_theta_eff else (1e-3, 1e0)
        ax.axhline(ekf_mse_mean, color=colors['EKF'], linestyle='--', linewidth=2, label=filter_names['EKF'])
        if ekf_mse_std and ekf_mse_std > 0:
            ax.fill_between([x_min, x_max],
                            ekf_mse_mean - ekf_mse_std, ekf_mse_mean + ekf_mse_std,
                            color=colors['EKF'], alpha=0.2)

    if theta_eff_vals:
        ax.errorbar(theta_eff_vals, mse_means, yerr=mse_stds,
                    marker='o', linewidth=3, markersize=10, capsize=6, color=colors['DR_EKF_trace'],
                    label=filter_names['DR_EKF_trace'])

    if theta_eff_vals_mp:
        ax.errorbar(theta_eff_vals_mp, mse_means_mp, yerr=mse_stds_mp,
                    marker='s', linewidth=3, markersize=10, capsize=6, color=colors['DR_EKF_trace_multipass'],
                    label=filter_names['DR_EKF_trace_multipass'])

    ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"mse_vs_robustness_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"MSE vs robustness plot saved to: {save_path}")


def create_violin_plots(optimal_results, results_path, dist):
    """Violin plots for MSE at optimal parameters (EKF, DR_EKF_trace)."""
    print(f"Creating violin plots for optimal parameters ({dist})...")
    colors = _colors()
    filter_names = _filter_names()
    violin_data = []
    labels = []
    positions = []
    filter_colors = []
    pos_counter = 1

    for filt in FILTERS_ORDER:
        if filt not in optimal_results:
            continue
        optimal_stats = optimal_results[filt]
        if filt == 'EKF':
            theta_vals = {}
        else:
            theta_vals = {'theta_eff': optimal_stats['theta_eff']}

        try:
            detailed_results = load_detailed_results_for_filter(results_path, filt, theta_vals, dist)
            if filt in detailed_results:
                sim_results = detailed_results[filt]['results']
                raw_data = [np.mean(r['mse']) for r in sim_results]
                if len(raw_data) > 0:
                    violin_data.append(raw_data)
                    labels.append(filter_names[filt] if filt == 'EKF' else f"{filter_names[filt]}\n(θ_eff={optimal_stats['theta_eff']:.4f})")
                    positions.append(pos_counter)
                    filter_colors.append(colors[filt])
                    pos_counter += 1
        except (FileNotFoundError, Exception) as e:
            print(f"Could not load detailed results for {filt}: {e}")

    if not violin_data:
        print("No violin data available.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True, widths=0.8)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(filter_colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14)
    ax.set_ylabel('MSE Distribution', fontsize=16, labelpad=15)
    ax.set_title(f'MSE Distribution at Optimal Parameters ({dist.title()})', fontsize=18)
    ax.set_yscale('log')
    ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f'violin_plot_mse_{dist}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved as: {output_path}")


def plot_theta_eff_eta(optimal_results, results_path, dist, time):
    """
    Plot the actual θ_ε,t^eff used by the SDP, alongside θ_ε (base) and η_eff,t.
    theta_eff saved by tracking scripts is now the true theta_eps_effective from the filter.
    """
    if 'DR_EKF_trace' not in optimal_results:
        print("Warning: DR_EKF_trace not in optimal_results. Skipping theta_eff vs eta plot.")
        return

    theta_opt = optimal_results['DR_EKF_trace'].get('theta_eff')
    if theta_opt is None:
        theta_opt = 0.01
    try:
        detailed = load_detailed_results_for_filter(results_path, 'DR_EKF_trace', {'theta_eff': theta_opt}, dist)
    except FileNotFoundError as e:
        print(f"Warning: Cannot load detailed results for theta_eff/eta plot: {e}")
        return

    if 'DR_EKF_trace' not in detailed or 'results' not in detailed['DR_EKF_trace']:
        print("Warning: No DR_EKF_trace results in detailed. Skipping theta_eff vs eta plot.")
        return

    results_list = detailed['DR_EKF_trace']['results']
    runs_with_data = [r for r in results_list if 'theta_eff_eta' in r]
    if not runs_with_data:
        print("Warning: No runs with theta_eff_eta data (re-run exp_ct_tracking.py). Skipping plot.")
        return

    # Runs may have different lengths (e.g. early SDP failure); use min length so we can stack.
    min_len = min(len(r['theta_eff_eta']['theta_eff']) for r in runs_with_data)
    if min_len == 0:
        print("Warning: All runs have empty theta_eff_eta. Skipping plot.")
        return
    # theta_eff now stores the actual theta_eps_effective used by the SDP
    theta_eff_arr = np.array([np.asarray(r['theta_eff_eta']['theta_eff']).flat[:min_len] for r in runs_with_data])
    eta_arr = np.array([np.asarray(r['theta_eff_eta']['eta']).flat[:min_len] for r in runs_with_data])

    n_steps = eta_arr.shape[1]
    t_axis = time[:n_steps] if len(time) >= n_steps else np.arange(n_steps, dtype=float) * 0.2

    # theta_eps_base = sqrt(theta_w^2 + theta_v^2) = theta_opt (from experiment loop)
    # NOTE: theta_eff_arr[0,0] is NOT the base — at t=0, eta_h0 is nonzero so it's already inflated.
    theta_eps_const = theta_opt
    eta_mean = np.mean(eta_arr, axis=0)
    eta_std = np.std(eta_arr, axis=0)
    # theta_eff_arr IS the actual theta_eps_effective (no need to recompute)
    theta_eff_mean = np.mean(theta_eff_arr, axis=0)
    theta_eff_std = np.std(theta_eff_arr, axis=0)

    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    theta_eff_cap = 5.0 * theta_eps_const
    ax.plot(t_axis, np.full_like(t_axis, theta_eps_const), 'b-', linewidth=2.5, label=r'$\theta_{\epsilon}$ (base)')
    ax.axhline(y=theta_eff_cap, color='gray', linestyle='--', linewidth=1.5, label=r'cap ($5\times\theta_{\epsilon}$)')
    ax.plot(t_axis, eta_mean, 'r-', linewidth=2, label=r'$\eta_{\mathrm{eff},t}$ (mean)')
    ax.fill_between(t_axis, eta_mean - eta_std, eta_mean + eta_std, color='red', alpha=0.25)
    ax.plot(t_axis, theta_eff_mean, 'g-', linewidth=2.5, label=r'$\theta_{\epsilon,t}^{\mathrm{eff}}$ (mean)')
    ax.fill_between(t_axis, theta_eff_mean - theta_eff_std, theta_eff_mean + theta_eff_std, color='green', alpha=0.25)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Magnitude', fontsize=14)
    bw = optimal_results.get('_bw_distances', {})
    d_bw_w = bw.get('d_bw_w', None)
    d_bw_v = bw.get('d_bw_v', None)
    bw_str = ""
    if d_bw_w is not None and d_bw_v is not None:
        bw_str = f", $d_{{BW,w}}$={d_bw_w:.4f}, $d_{{BW,v}}$={d_bw_v:.4f}"
    theta_w_val = optimal_results['DR_EKF_trace'].get('theta_w', theta_opt / np.sqrt(2))
    ax.set_title(r'DR-EKF trace (CT): $\theta_{{\epsilon,t}}^{{\mathrm{{eff}}}}$ ($\theta_w = \theta_v = {:.4f}$, $\theta_{{\epsilon}} = {:.4f}${})'.format(theta_w_val, theta_opt, bw_str), fontsize=14, pad=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"theta_eff_eta_comparison_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"theta_eff vs eta comparison (CT) saved to: {save_path}")

    # --- Separate plot: theta_eps (base) and theta_eff only (no eta_eff) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(t_axis, np.full_like(t_axis, theta_eps_const), 'b-', linewidth=2.5, label=r'$\theta_{\epsilon}$ (base)')
    ax2.axhline(y=theta_eff_cap, color='gray', linestyle='--', linewidth=1.5, label=r'cap ($5\times\theta_{\epsilon}$)')
    ax2.plot(t_axis, theta_eff_mean, 'g-', linewidth=2.5, label=r'$\theta_{\epsilon,t}^{\mathrm{eff}}$ (mean)')
    ax2.fill_between(t_axis, theta_eff_mean - theta_eff_std, theta_eff_mean + theta_eff_std, color='green', alpha=0.25)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Magnitude', fontsize=14)
    ax2.set_title(r'DR-EKF trace (CT): $\theta_{{\epsilon,t}}^{{\mathrm{{eff}}}}$ vs $\theta_{{\epsilon}}$ ($\theta_w = \theta_v = {:.4f}$, $\theta_{{\epsilon}} = {:.4f}${})'.format(theta_w_val, theta_opt, bw_str), fontsize=14, pad=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    save_path2 = os.path.join(results_dir, f"theta_eff_comparison_{dist}.pdf")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig2)
    print(f"theta_eff comparison (no eta, CT) saved to: {save_path2}")

def plot_theta_eff_multipass(optimal_results, results_path, dist, time):
    """Plot theta_eff vs time for both one-pass and multipass on the same figure."""
    if 'DR_EKF_trace_multipass' not in optimal_results:
        print("DR_EKF_trace_multipass not in optimal_results. Skipping.")
        return

    theta_opt_mp = optimal_results['DR_EKF_trace_multipass'].get('theta_eff')
    if theta_opt_mp is None:
        theta_opt_mp = 0.01

    # Load multipass detailed results
    try:
        detailed_mp = load_detailed_results_for_filter(
            results_path,
            'DR_EKF_trace_multipass',
            {'theta_eff': theta_opt_mp},
            dist
        )
    except FileNotFoundError as e:
        print(f"Cannot load multipass detailed results: {e}")
        return

    if 'DR_EKF_trace_multipass' not in detailed_mp or 'results' not in detailed_mp['DR_EKF_trace_multipass']:
        print("No DR_EKF_trace_multipass results in detailed.")
        return

    results_list_mp = detailed_mp['DR_EKF_trace_multipass']['results']
    runs_mp = [r for r in results_list_mp if 'theta_eff_eta' in r]
    if not runs_mp:
        print("No runs with theta_eff_eta for multipass. Re-run exp_ct_tracking.py.")
        return

    min_len_mp = min(len(r['theta_eff_eta']['theta_eff']) for r in runs_mp)
    if min_len_mp == 0:
        print("All multipass runs have empty theta_eff_eta.")
        return

    theta_eff_arr_mp = np.array(
        [np.asarray(r['theta_eff_eta']['theta_eff']).flat[:min_len_mp] for r in runs_mp]
    )
    theta_eff_mean_mp = np.mean(theta_eff_arr_mp, axis=0)
    theta_eff_std_mp = np.std(theta_eff_arr_mp, axis=0)

    # Load one-pass (DR_EKF_trace) detailed results at its optimal theta for comparison
    theta_eff_mean_op = None
    theta_eff_std_op = None
    if 'DR_EKF_trace' in optimal_results:
        theta_opt_op = optimal_results['DR_EKF_trace'].get('theta_eff')
        if theta_opt_op is not None:
            try:
                detailed_op = load_detailed_results_for_filter(
                    results_path, 'DR_EKF_trace', {'theta_eff': theta_opt_op}, dist
                )
                if 'DR_EKF_trace' in detailed_op and 'results' in detailed_op['DR_EKF_trace']:
                    results_list_op = detailed_op['DR_EKF_trace']['results']
                    runs_op = [r for r in results_list_op if 'theta_eff_eta' in r]
                    if runs_op:
                        min_len_op = min(len(r['theta_eff_eta']['theta_eff']) for r in runs_op)
                        if min_len_op > 0:
                            theta_eff_arr_op = np.array(
                                [np.asarray(r['theta_eff_eta']['theta_eff']).flat[:min_len_op] for r in runs_op]
                            )
                            theta_eff_mean_op = np.mean(theta_eff_arr_op, axis=0)
                            theta_eff_std_op = np.std(theta_eff_arr_op, axis=0)
            except FileNotFoundError:
                pass

    # Use common length for time axis (min of one-pass and multipass)
    n_steps = theta_eff_arr_mp.shape[1]
    if theta_eff_mean_op is not None:
        n_steps = min(n_steps, len(theta_eff_mean_op))
        theta_eff_mean_op = theta_eff_mean_op[:n_steps]
        theta_eff_std_op = theta_eff_std_op[:n_steps]
    theta_eff_mean_mp = theta_eff_mean_mp[:n_steps]
    theta_eff_std_mp = theta_eff_std_mp[:n_steps]

    t_axis = time[:n_steps] if len(time) >= n_steps else np.arange(n_steps, dtype=float) * 0.2

    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    colors = _colors()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_axis, theta_eff_mean_mp, '-', color=colors['DR_EKF_trace_multipass'], linewidth=2.5,
            label=r'$\theta_{\epsilon,t}^{\mathrm{eff}}$ (multipass)')
    ax.fill_between(t_axis, theta_eff_mean_mp - theta_eff_std_mp, theta_eff_mean_mp + theta_eff_std_mp,
                    color=colors['DR_EKF_trace_multipass'], alpha=0.25)
    if theta_eff_mean_op is not None:
        ax.plot(t_axis, theta_eff_mean_op, '-', color=colors['DR_EKF_trace'], linewidth=2.5,
                label=r'$\theta_{\epsilon,t}^{\mathrm{eff}}$ (one-pass)')
        ax.fill_between(t_axis, theta_eff_mean_op - theta_eff_std_op, theta_eff_mean_op + theta_eff_std_op,
                        color=colors['DR_EKF_trace'], alpha=0.25)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Magnitude', fontsize=14)
    ax.set_title(r'DR-EKF trace: $\theta_{\epsilon,t}^{\mathrm{eff}}$ — one-pass vs multipass', fontsize=14, pad=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"theta_eff_multipass_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"theta_eff (one-pass vs multipass) plot saved to: {save_path}")


def plot_prior_trace_proxy_vs_sdp(optimal_results, results_path, dist, time):
    """
    Plot proxy prior trace \\bar{t}_{x,t}^- (one-pass formula) vs
    SDP-based worst-case prior trace Tr(Sigma_{x,t}^-) for the multipass filter.
    Uses the 'prior_trace_history' saved in exp_ct_tracking.py.
    """
    if 'DR_EKF_trace_multipass' not in optimal_results:
        print("DR_EKF_trace_multipass not in optimal_results. Skipping prior-trace plot.")
        return

    theta_opt_mp = optimal_results['DR_EKF_trace_multipass'].get('theta_eff')
    if theta_opt_mp is None:
        theta_opt_mp = 0.01

    try:
        detailed_mp = load_detailed_results_for_filter(
            results_path,
            'DR_EKF_trace_multipass',
            {'theta_eff': theta_opt_mp},
            dist,
        )
    except FileNotFoundError as e:
        print(f"Cannot load multipass detailed results for prior-trace plot: {e}")
        return

    if 'DR_EKF_trace_multipass' not in detailed_mp or 'results' not in detailed_mp['DR_EKF_trace_multipass']:
        print("No DR_EKF_trace_multipass results in detailed (prior-trace plot).")
        return

    results_list_mp = detailed_mp['DR_EKF_trace_multipass']['results']
    runs_with_prior = [
        r for r in results_list_mp
        if 'prior_trace_history' in r
        and r['prior_trace_history'].get('tbar_prior_proxy') is not None
        and r['prior_trace_history'].get('Tr_prior_sdp') is not None
    ]
    if not runs_with_prior:
        print("No runs with prior_trace_history for multipass. Re-run exp_ct_tracking.py.")
        return

    # Align lengths across runs
    tbar_list = []
    tr_sdp_list = []
    for r in runs_with_prior:
        hist = r['prior_trace_history']
        tbar = np.asarray(hist['tbar_prior_proxy'], dtype=float).ravel()
        tr_sdp = np.asarray(hist['Tr_prior_sdp'], dtype=float).ravel()
        L = min(len(tbar), len(tr_sdp))
        if L == 0:
            continue
        tbar_list.append(tbar[:L])
        tr_sdp_list.append(tr_sdp[:L])

    if not tbar_list:
        print("prior_trace_history arrays are empty. Skipping prior-trace plot.")
        return

    # Use minimum length over runs to stack
    min_len = min(len(a) for a in tbar_list)
    tbar_arr = np.array([a[:min_len] for a in tbar_list])
    tr_sdp_arr = np.array([a[:min_len] for a in tr_sdp_list])

    tbar_mean = np.mean(tbar_arr, axis=0)
    tbar_std = np.std(tbar_arr, axis=0)
    tr_sdp_mean = np.mean(tr_sdp_arr, axis=0)
    tr_sdp_std = np.std(tr_sdp_arr, axis=0)

    n_steps = min_len
    t_axis = time[:n_steps] if len(time) >= n_steps else np.arange(n_steps, dtype=float) * 0.2

    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    # Figure 1: absolute traces (proxy vs SDP), as before.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_axis, tbar_mean, 'b-', linewidth=2.5, label=r'Proxy $\bar{t}_{x,t}^-$ (one-pass)')
    ax.fill_between(t_axis, tbar_mean - tbar_std, tbar_mean + tbar_std, color='blue', alpha=0.25)
    # Use single backslashes inside mathtext; double backslashes break the parser.
    ax.plot(t_axis, tr_sdp_mean, 'r-', linewidth=2.5, label=r'$\mathrm{Tr}(\Sigma_{x,t}^{-})$ (SDP, multipass)')
    ax.fill_between(t_axis, tr_sdp_mean - tr_sdp_std, tr_sdp_mean + tr_sdp_std, color='red', alpha=0.25)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Prior trace magnitude', fontsize=14)
    ax.set_title(r'Prior trace: proxy $\bar{t}_{x,t}^{-}$ vs SDP $\mathrm{Tr}(\Sigma_{x,t}^{-})$', fontsize=14, pad=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"prior_trace_proxy_vs_sdp_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Prior-trace proxy vs SDP plot saved to: {save_path}")

    # Figure 2: zoomed difference / relative gap so even tiny deviations are visible.
    # Avoid division by zero by clipping very small denominators.
    eps = 1e-12
    denom = np.clip(tbar_mean, eps, None)
    abs_gap = tbar_mean - tr_sdp_mean
    rel_gap = abs_gap / denom

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: absolute gap
    ax1.plot(t_axis, abs_gap, 'k-', linewidth=2)
    ax1.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax1.set_ylabel(r'$\bar{t}_{x,t}^{-} - \mathrm{Tr}(\Sigma_{x,t}^{-})$', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Bottom: relative gap (%)
    ax2.plot(t_axis, 100.0 * rel_gap, 'k-', linewidth=2)
    ax2.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Relative gap [%]', fontsize=13)
    ax2.grid(True, alpha=0.3)

    for ax_ in (ax1, ax2):
        ax_.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    save_path2 = os.path.join(results_dir, f"prior_trace_gap_proxy_vs_sdp_{dist}.pdf")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig2)
    print(f"Prior-trace proxy vs SDP gap plot saved to: {save_path2}")

    # Figure 3: combined view (absolute traces + abs gap + relative gap in one figure).
    fig3, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True, gridspec_kw={"hspace": 0.15}
    )

    # Top: absolute proxy vs SDP
    ax_top.plot(t_axis, tbar_mean, 'b-', linewidth=2.0, label=r'Proxy $\bar{t}_{x,t}^-$')
    ax_top.plot(t_axis, tr_sdp_mean, 'r--', linewidth=2.0, label=r'$\mathrm{Tr}(\Sigma_{x,t}^{-})$')
    ax_top.set_ylabel('Trace', fontsize=12)
    ax_top.set_title('Proxy vs SDP prior trace (combined view)', fontsize=13, pad=8)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=10, loc='best')

    # Middle: absolute gap
    ax_mid.plot(t_axis, abs_gap, 'k-', linewidth=2.0)
    ax_mid.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax_mid.set_ylabel(r'$\bar{t}_{x,t}^{-} - \mathrm{Tr}(\Sigma_{x,t}^{-})$', fontsize=12)
    ax_mid.grid(True, alpha=0.3)

    # Bottom: relative gap (%)
    ax_bot.plot(t_axis, 100.0 * rel_gap, 'k-', linewidth=2.0)
    ax_bot.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax_bot.set_xlabel('Time', fontsize=13)
    ax_bot.set_ylabel(r'Relative gap [%]', fontsize=12)
    ax_bot.grid(True, alpha=0.3)

    for ax_ in (ax_top, ax_mid, ax_bot):
        ax_.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    save_path3 = os.path.join(results_dir, f"prior_trace_combo_proxy_vs_sdp_{dist}.pdf")
    plt.savefig(save_path3, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig3)
    print(f"Prior-trace combined (proxy/SDP/ gaps) plot saved to: {save_path3}")

def print_optimal_results_summary(optimal_results):
    """Print summary for EKF and DR_EKF_trace."""
    print("\nOptimal Results Summary:")
    print("=" * 60)
    for filter_name in FILTERS_ORDER:
        if filter_name not in optimal_results:
            continue
        results = optimal_results[filter_name]
        mse_mean = results.get('mse_mean', 'N/A')
        mse_std = results.get('mse_std', 'N/A')
        if filter_name == 'EKF':
            theta_str = "N/A"
        else:
            te = results.get('theta_eff')
            theta_str = f"θ_eff={te:.4f}" if isinstance(te, (int, float)) else "N/A"
        print(f"{filter_name:18s}: {theta_str:30s} MSE={mse_mean:.6f}±{mse_std:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Plot EKF vs DR_EKF_trace results from exp_ct_tracking.py')
    parser.add_argument('--dist', default='normal', choices=['normal', 'quadratic'])
    parser.add_argument('--individual_only', action='store_true', help='Only individual trajectory plots')
    parser.add_argument('--subplots_only', action='store_true', help='Only subplot figure')
    parser.add_argument('--heatmaps_only', action='store_true', help='Only MSE heatmap and robustness plot')
    parser.add_argument('--sample_trajectories', action='store_true', help='Plot sample trajectories')
    parser.add_argument('--num_samples', default=20, type=int, help='Number of sample trajectories')
    args = parser.parse_args()

    try:
        results_path = os.path.join(".", "results", "exp_ct_tracking")
        optimal_results, all_results = load_data(results_path, args.dist)
        print_optimal_results_summary(optimal_results)

        print("\nCreating performance vs robustness (MSE vs θ_eff) plot...")
        plot_performance_vs_robustness(all_results, args.dist)

        print("\nCreating violin plots...")
        create_violin_plots(optimal_results, results_path, args.dist)

        if not args.heatmaps_only:
            desired_traj, time = generate_desired_trajectory(10.0)
            print("\nCreating theta_eff vs eta comparison (DR_EKF_trace)...")
            plot_theta_eff_eta(optimal_results, results_path, args.dist, time)
            print("\nCreating theta_eff (multipass) vs time...")
            plot_theta_eff_multipass(optimal_results, results_path, args.dist, time)
            print("\nCreating prior-trace proxy vs SDP trace (multipass)...")
            plot_prior_trace_proxy_vs_sdp(optimal_results, results_path, args.dist, time)
            trajectory_data, filters_order = extract_trajectory_data_from_saved(optimal_results, results_path, args.dist)
            if trajectory_data:
                dt = 0.2
                first_key = next(iter(trajectory_data.keys()))
                actual_num_steps = len(trajectory_data[first_key]['mean'])
                actual_time = (actual_num_steps - 1) * dt
                desired_traj, time = generate_desired_trajectory(actual_time, dt)
            else:
                desired_traj, time = generate_desired_trajectory(10.0)

            if len(trajectory_data) == 0:
                print("No trajectory data. Run exp_ct_tracking.py first.")
                return

            if args.sample_trajectories:
                plot_sample_trajectories(trajectory_data, filters_order, desired_traj, time, args.dist, num_samples=args.num_samples)
            if not args.subplots_only:
                plot_trajectory_subplots(trajectory_data, filters_order, desired_traj, time, args.dist)
            if not args.individual_only:
                plot_subplots_all_filters(trajectory_data, filters_order, desired_traj, time, args.dist)
                # Create single-run subplots for all runs.
                first_key = next(iter(trajectory_data.keys()))
                num_sims = trajectory_data[first_key]['num_sims']

                for ridx in range(num_sims):
                    plot_subplots_single_run(
                        trajectory_data,
                        filters_order,
                        run_idx=ridx,
                        dist=args.dist
                    )
                # plot_subplots_single_run(trajectory_data, filters_order, run_idx=3, dist=args.dist)

    except FileNotFoundError as e:
        print(f"Error: Could not find results for distribution '{args.dist}'.")
        print(f"Run exp_ct_tracking.py with --dist {args.dist} first.")
        print(f"Missing: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
