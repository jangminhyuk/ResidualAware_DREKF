#!/usr/bin/env python3
"""
exp_ct_tracking.py: EKF vs DR_EKF_trace comparison using 2D coordinated-turn (CT) dynamics
with radar measurements.
Results saved to ./results/exp_ct_tracking/.
"""


import numpy as np
import argparse
import os
from joblib import Parallel, delayed

from estimator.EKF import EKF
from estimator.DR_EKF_trace import DR_EKF_trace
from common_utils import (save_data, wrap_angle, bures_wasserstein_distance, bures_covariance_distance)
from ct_radar_models import (ct_dynamics, ct_jacobian,
                              radar_observation_function, radar_observation_jacobian,
                              wrap_bearing_measurement, sample_from_distribution,
                              _as_col)


# ---------------------------------------------------------------------------
# Single simulation (no controller; autonomous CT)
# main0_CT_joint.py style: each filter samples its own internal noise,
# while run_experiment aligns np.random.seed(seed_val) for each filter
# so that EKF and DR_EKF_trace experience identical noise sequences.
# ---------------------------------------------------------------------------
def run_single_simulation(estimator, T, dt):
    """Run CT simulation: EKF or DR_EKF_trace with shared noise indexing pattern."""
    nx, ny = estimator.nx, estimator.ny
    nu = 2

    x = np.zeros((T+1, nx, 1))
    y = np.zeros((T+1, ny, 1))
    x_est = np.zeros((T+1, nx, 1))
    u_traj = np.zeros((T, nu, 1))
    mse = np.zeros(T+1)

    estimator._noise_index = 0
    x[0] = estimator.sample_initial_state()
    x_est[0] = estimator.nominal_x0_mean.copy()

    v0 = estimator.sample_measurement_noise()
    y_raw0 = radar_observation_function(x[0]) + v0
    if hasattr(estimator, 'h'):
        y_pred0 = estimator.h(x_est[0]) + estimator.nominal_mu_v
        y[0] = wrap_bearing_measurement(y_raw0, y_pred0)
    else:
        y[0] = y_raw0

    # Radar singularity: at (0,0) the observation Jacobian is zero -> no correction.
    # Initialize position from first measurement so we linearize at a non-singular point.
    r0, b0 = float(y[0][0, 0]), float(y[0][1, 0])
    r0_safe = max(r0, 1e-3)
    x_est[0][0, 0] = r0_safe * np.cos(b0)
    x_est[0][1, 0] = r0_safe * np.sin(b0)

    x_est[0] = estimator._initial_update(x_est[0], y[0])

    mse[0] = np.linalg.norm(x_est[0] - x[0])**2

    theta_eff_list = []
    eta_list = []
    if hasattr(estimator, '_last_theta_eps_effective'):
        theta_eff_list.append(estimator._last_theta_eps_effective)
        eta_list.append(getattr(estimator, '_last_eta_eps', 0.0))

    for t in range(T):
        u = np.zeros((nu, 1))
        u_traj[t] = u.copy()
        w = estimator.sample_process_noise()
        x[t+1] = ct_dynamics(x[t], u, dt=dt) + w
        estimator._noise_index += 1
        v = estimator.sample_measurement_noise()
        y_raw = radar_observation_function(x[t+1]) + v
        if hasattr(estimator, 'h'):
            x_pred = estimator.f(x_est[t], u) + estimator.nominal_mu_w
            y_pred = estimator.h(x_pred) + estimator.nominal_mu_v
            y[t+1] = wrap_bearing_measurement(y_raw, y_pred)
        else:
            y[t+1] = y_raw

        try:
            x_est[t+1] = estimator.update_step(x_est[t], y[t+1], t+1, u)
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
            if hasattr(estimator, '_last_theta_eps_effective'):
                theta_eff_list.append(estimator._last_theta_eps_effective)
                eta_list.append(getattr(estimator, '_last_eta_eps', 0.0))
        except Exception as e:
            # Even if DR_EKF_trace SDP fails (e.g., unbounded),
            # keep partial trajectories and theta/eta history up to this time.
            print(f"{type(estimator).__name__} failed at time step {t+1}: {e}")
            break

    out = {
        'mse': mse,
        'state_traj': x,
        'est_state_traj': x_est,
        'input_traj': u_traj,
    }
    if theta_eff_list:
        out['theta_eff_eta'] = {
            'theta_eff': np.array(theta_eff_list),
            'eta': np.array(eta_list),
        }
    return out


# ---------------------------------------------------------------------------
# I/O dataset for nominal EM (CT, autonomous)
# ---------------------------------------------------------------------------
def generate_io_dataset_ct(
    T_em, dt, num_rollouts, dist,
    true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
    x0_scale=None, w_scale=None, v_scale=None, seed=None
):
    T = int(T_em / dt)
    nx, ny, nu = 5, 2, 2
    u_data = np.zeros((num_rollouts, T, nu, 1))
    y_data = np.zeros((num_rollouts, T+1, ny, 1))
    if seed is not None:
        np.random.seed(seed)

    for k in range(num_rollouts):
        if dist == "normal":
            x0 = sample_from_distribution(true_x0_mean, true_x0_cov, "normal", N=1)
        elif dist == "laplace":
            x0 = sample_from_distribution(true_x0_mean, None, "laplace", scale=x0_scale, N=1)
        else:
            raise ValueError(f"Unsupported dist={dist}")
        x = _as_col(x0[:, 0])

        if dist == "normal":
            v0 = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=1)
        elif dist == "laplace":
            v0 = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=1)
        y_raw_0 = radar_observation_function(x) + _as_col(v0[:, 0])
        y_data[k, 0] = y_raw_0
        prev_bearing = y_raw_0[1, 0]

        for t in range(T):
            u = np.zeros((nu, 1))
            u_data[k, t] = u
            if dist == "normal":
                w = sample_from_distribution(true_mu_w, true_Sigma_w, "normal", N=1)
            elif dist == "laplace":
                w = sample_from_distribution(true_mu_w, None, "laplace", scale=w_scale, N=1)
            w = _as_col(w[:, 0])
            x = ct_dynamics(x, u, dt=dt) + w

            if dist == "normal":
                v = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=1)
            elif dist == "laplace":
                v = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=1)
            v = _as_col(v[:, 0])
            y_raw = radar_observation_function(x) + v
            current_bearing = y_raw[1, 0]
            wrapped_diff = wrap_angle(current_bearing - prev_bearing)
            continuous_bearing = prev_bearing + wrapped_diff
            y_data[k, t+1] = y_raw.copy()
            y_data[k, t+1, 1, 0] = continuous_bearing
            prev_bearing = continuous_bearing

    return u_data, y_data


# ---------------------------------------------------------------------------
# Experiment runner: EKF and DR_EKF_trace only
# ---------------------------------------------------------------------------
def run_experiment(exp_idx, dist, num_sim, seed_base, theta_vals, filters_to_execute, T_steps,
                  nominal_params, true_params, num_samples, tube_params=None, eta_scale=1.0):
    experiment_seed = seed_base + exp_idx * 12345
    np.random.seed(experiment_seed)
    T = T_steps
    dt = 0.2
    nx, ny, nu = 5, 2, 2
    A = np.eye(nx)
    B = np.zeros((nx, nu))
    C = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    system_data = (A, C)

    (x0_mean, x0_cov, mu_w, Sigma_w, mu_v, Sigma_v) = true_params
    (nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w,
     nominal_mu_v, nominal_Sigma_v) = nominal_params

    theta_v = theta_vals['theta_v']
    theta_w = theta_vals['theta_w']
    L_f, L_h, R_f, R_h = tube_params if tube_params is not None else (0.01, 0.01, 1.0, 1.5)

    results = {fn: [] for fn in filters_to_execute}

    for sim_idx in range(num_sim):
        seed_val = (experiment_seed + sim_idx * 10) % (2**32 - 1)
        sim_results = {}
        for filter_name in filters_to_execute:
            # main0_CT_joint style: reset the same seed for each filter
            # so they experience identical noise sequences.
            np.random.seed(seed_val)
            if filter_name == 'EKF':
                estimator = EKF(
                    T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
                    true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                    true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                    true_mu_v=mu_v, true_Sigma_v=Sigma_v,
                    nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                    nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                    nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
                    nonlinear_dynamics=ct_dynamics,
                    dynamics_jacobian=ct_jacobian,
                    observation_function=radar_observation_function,
                    observation_jacobian=radar_observation_jacobian,
                )
            elif filter_name == 'DR_EKF_trace':
                estimator = DR_EKF_trace(
                    T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
                    true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                    true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                    true_mu_v=mu_v, true_Sigma_v=Sigma_v,
                    nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                    nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                    nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
                    nonlinear_dynamics=ct_dynamics,
                    dynamics_jacobian=ct_jacobian,
                    observation_function=radar_observation_function,
                    observation_jacobian=radar_observation_jacobian,
                    theta_w=theta_w, theta_v=theta_v,
                    L_f=L_f, L_h=L_h, R_f=R_f, R_h=R_h,
                    eta_scale=eta_scale,
                )
            else:
                continue
            try:
                result = run_single_simulation(estimator, T, dt)
                sim_results[filter_name] = result
            except Exception as e:
                print(f"Simulation failed for {filter_name} (sim {sim_idx}): {e}")
        for fn in sim_results:
            results[fn].append(sim_results[fn])

    final_results = {}
    for filter_name in filters_to_execute:
        if results[filter_name]:
            filter_results = results[filter_name]
            final_results[filter_name] = {
                'mse_mean': np.mean([np.mean(r['mse']) for r in filter_results]),
                'results': filter_results,
            }
    return final_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
RESULTS_PATH = "./results/exp_ct_tracking"
FILTERS_TO_EXECUTE = ['EKF', 'DR_EKF_trace']


def main(dist, num_sim, num_exp, T_total=10.0, num_samples=100, seed_base=2026):
    dt = 0.2
    T_steps = int(T_total / dt)
    nx, ny = 5, 2

    # Robustness parameter: theta_w = theta_v = theta_eff/sqrt(2).
    theta_eff_vals = [0.001]
    # tube_params (L_f, L_h, R_f, R_h)
    tube_params = (0.3, 0.2, 1.7, 1.7)
    # Heuristic scaling for linearization residuals eta_f, eta_h (1.0 = certified bound)
    eta_scale = 1

    # True parameters (same as main0_CT_joint). x0_mean[4] = omega0 (rad/s)
    x0_mean = np.array([[0.0], [0.0], [2.0], [0.0], [0.30]])
    x0_cov = np.diag([0.2**2, 0.2**2, 0.5**2, 0.5**2, 0.05**2])

    mu_w = np.zeros((nx, 1))
    mu_v = np.zeros((ny, 1))
    Sigma_w = np.diag([0.01**2, 0.01**2, 0.05**2, 0.05**2, 0.02**2])
    Sigma_v = np.diag([0.01**2, np.deg2rad(0.5)**2])

    true_params = (x0_mean, x0_cov, mu_w, Sigma_w, mu_v, Sigma_v)

    # Nominal parameters: non-uniform per-dimension scaling (smaller than true).
    # Simulates a practitioner who underestimates noise with wrong relative magnitudes.
    # State: [px, py, vx, vy, omega]
    #   x0: severely underestimate position, mildly underestimate velocity/omega
    #   w:  severely underestimate position process noise, mildly underestimate velocity/omega
    #   v:  underestimate range more than bearing
    nom_x0_mean = x0_mean.copy()
    nom_x0_cov = x0_cov * np.diag([0.1, 0.1, 0.1, 0.1, 0.1])   # px,py ×0.05; vx,vy ×0.3; omega ×0.2
    nom_mu_w = mu_w.copy()
    nom_Sigma_w = Sigma_w * np.diag([0.1, 0.1, 0.1, 0.1, 0.1])  
    nom_mu_v = mu_v.copy()
    nom_Sigma_v = Sigma_v * np.diag([0.1, 0.1])                   # range ×0.05; bearing ×0.3
    nominal_params = (nom_x0_mean, nom_x0_cov, nom_mu_w, nom_Sigma_w, nom_mu_v, nom_Sigma_v)

    # Compute actual BW distances between true and nominal
    d_bw_x0_actual = bures_wasserstein_distance(x0_mean, x0_cov, nom_x0_mean, nom_x0_cov)
    d_bw_w_actual = bures_wasserstein_distance(mu_w, Sigma_w, nom_mu_w, nom_Sigma_w)
    d_bw_v_actual = bures_wasserstein_distance(mu_v, Sigma_v, nom_mu_v, nom_Sigma_v)
    print(f"Nominal parameters: non-uniform per-dimension scaling")
    print(f"  True  Sigma_x0 diag: {np.diag(x0_cov)}")
    print(f"  Nom   Sigma_x0 diag: {np.diag(nom_x0_cov)}")
    print(f"  True  Sigma_w  diag: {np.diag(Sigma_w)}")
    print(f"  Nom   Sigma_w  diag: {np.diag(nom_Sigma_w)}")
    print(f"  True  Sigma_v  diag: {np.diag(Sigma_v)}")
    print(f"  Nom   Sigma_v  diag: {np.diag(nom_Sigma_v)}")
    print(f"  d_BW(x0) = {d_bw_x0_actual:.6f},  d_BW(w) = {d_bw_w_actual:.6f},  d_BW(v) = {d_bw_v_actual:.6f}")

    all_results = {fn: {} for fn in FILTERS_TO_EXECUTE}

    # Paired comparison: same experiment = same seed = same true trajectory for EKF and DR_EKF_trace.
    # So MSE and plots are comparable (same N runs, run index i = same true for both filters).
    print("\n" + "="*60)
    print("Running EKF + DR_EKF_trace (paired, same true per run)")
    print("="*60)
    for theta_eff in theta_eff_vals:
        theta_v = theta_w = float(theta_eff) / np.sqrt(2)
        theta_vals = {'theta_v': theta_v, 'theta_w': theta_w}
        print(f"\nθ_eff={theta_eff:.4f} (θ_v=θ_w={theta_v:.4f})")
        experiments = Parallel(n_jobs=-1, backend='loky')(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, theta_vals,
                                   ['EKF', 'DR_EKF_trace'], T_steps, nominal_params, true_params, num_samples,
                                   tube_params, eta_scale)
            for exp_idx in range(num_exp)
        )
        ekf_results = []
        drekf_results = []
        for exp in experiments:
            # Keep only paired runs: same experiment must have both filters succeed.
            if 'EKF' in exp and 'DR_EKF_trace' in exp and exp['EKF']['results'] and exp['DR_EKF_trace']['results']:
                ekf_results.extend(exp['EKF']['results'])
                drekf_results.extend(exp['DR_EKF_trace']['results'])
        if ekf_results:
            ekf_mse_per_run = [np.mean(r['mse']) for r in ekf_results]
            all_results['EKF']['no_theta'] = {
                'mse_mean': np.mean(ekf_mse_per_run),
                'mse_std': np.std(ekf_mse_per_run),
                'results': ekf_results,
            }
            if len(theta_eff_vals) == 1 or theta_eff == theta_eff_vals[0]:
                print(f"  EKF       MSE: {np.mean(ekf_mse_per_run):.4f} ± {np.std(ekf_mse_per_run):.4f}")
        if drekf_results:
            drekf_mse_per_run = [np.mean(r['mse']) for r in drekf_results]
            all_results['DR_EKF_trace'][float(theta_eff)] = {
                'mse_mean': np.mean(drekf_mse_per_run),
                'mse_std': np.std(drekf_mse_per_run),
                'theta_eff': float(theta_eff),
                'theta_v': theta_v,
                'theta_w': theta_w,
                'results': drekf_results,
            }
            print(f"  DR_EKF    MSE: {np.mean(drekf_mse_per_run):.4f} ± {np.std(drekf_mse_per_run):.4f}")

    # Optimal theta_eff for DR_EKF_trace
    optimal_results = {}
    if all_results['EKF']:
        optimal_results['EKF'] = all_results['EKF']['no_theta']

    best_mse = np.inf
    best_theta_eff = None
    best_stats = None
    for theta_eff, data in all_results['DR_EKF_trace'].items():
        if data['mse_mean'] < best_mse:
            best_mse = data['mse_mean']
            best_theta_eff = theta_eff
            best_stats = data
    if best_stats is not None:
        optimal_results['DR_EKF_trace'] = best_stats
        print(f"\nDR_EKF_trace optimal θ_eff={best_theta_eff:.4f}, MSE(mean)={best_mse:.4f}")

    # Per-run MSE diagnostic
    if optimal_results.get('EKF') and optimal_results.get('DR_EKF_trace'):
        ekf_per = [np.mean(r['mse']) for r in optimal_results['EKF']['results']]
        dr_per = [np.mean(r['mse']) for r in optimal_results['DR_EKF_trace']['results']]
        n_runs = len(ekf_per)
        print("\n--- Per-run MSE (full 5D state) ---")
        print(f"  EKF    : min={np.min(ekf_per):.4f}, median={np.median(ekf_per):.4f}, max={np.max(ekf_per):.4f}")
        print(f"  DR_EKF : min={np.min(dr_per):.4f}, median={np.median(dr_per):.4f}, max={np.max(dr_per):.4f}  (n={n_runs})")
        if np.max(dr_per) > 10 * np.median(dr_per):
            bad = [i for i, m in enumerate(dr_per) if m > 10 * np.median(dr_per)]
            print(f"  → DR_EKF outlier run indices: {bad}")

    # Store BW distances in optimal_results for plotting
    (nom_x0_mean_, nom_x0_cov_, nom_mu_w_, nom_Sigma_w_,
     nom_mu_v_, nom_Sigma_v_) = nominal_params
    d_bw_w = bures_wasserstein_distance(mu_w, Sigma_w, nom_mu_w_, nom_Sigma_w_)
    d_bw_v = bures_wasserstein_distance(mu_v, Sigma_v, nom_mu_v_, nom_Sigma_v_)
    optimal_results['_bw_distances'] = {'d_bw_w': d_bw_w, 'd_bw_v': d_bw_v}

    os.makedirs(RESULTS_PATH, exist_ok=True)
    save_data(os.path.join(RESULTS_PATH, f'all_results_{dist}.pkl'), all_results)
    save_data(os.path.join(RESULTS_PATH, f'optimal_results_{dist}.pkl'), optimal_results)

    # Detailed results for plotting
    if optimal_results.get('EKF'):
        save_data(os.path.join(RESULTS_PATH, f'detailed_results_EKF_{dist}.pkl'),
                 {'EKF': {'mse_mean': optimal_results['EKF']['mse_mean'], 'results': optimal_results['EKF']['results']}})
    if optimal_results.get('DR_EKF_trace'):
        st = optimal_results['DR_EKF_trace']
        teff = st['theta_eff']
        save_data(os.path.join(RESULTS_PATH, f'detailed_results_DR_EKF_trace_teff{teff:.4f}_{dist}.pkl'),
                 {'DR_EKF_trace': {'mse_mean': st['mse_mean'], 'results': st['results']}})

    print("\n" + "="*60)
    print("FINAL SUMMARY (paired: same true trajectory per run index)")
    print("="*60)
    print(f"{'Filter':<18} {'θ_eff':<20} {'MSE':<12}")
    print("-"*60)
    for fn in FILTERS_TO_EXECUTE:
        if fn not in optimal_results:
            continue
        st = optimal_results[fn]
        theta_str = "N/A" if fn == 'EKF' else f"{st['theta_eff']:.4f}"
        print(f"{fn:<18} {theta_str:<20} {st['mse_mean']:.4f}")
    print(f"\nResults saved to {RESULTS_PATH}")

    # Print BW distances (already computed above)
    d_b_w = bures_covariance_distance(Sigma_w, nom_Sigma_w_)
    d_b_v = bures_covariance_distance(Sigma_v, nom_Sigma_v_)
    print("\n" + "=" * 60)
    print("Bures-Wasserstein Distance: True vs Nominal")
    print("=" * 60)
    print(f"  Process noise w:     d_BW = {d_bw_w:.6f},  d_Bures(cov) = {d_b_w:.6f}")
    print(f"  Observation noise v: d_BW = {d_bw_v:.6f},  d_Bures(cov) = {d_b_v:.6f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT experiment: EKF vs DR_EKF_trace")
    parser.add_argument('--dist', default="normal", type=str)
    parser.add_argument('--num_sim', default=1, type=int)
    parser.add_argument('--num_exp', default=100, type=int)
    parser.add_argument('--T_total', default=50.0, type=float)
    parser.add_argument('--num_samples', default=20, type=int)
    parser.add_argument('--seed_base', default=2024, type=int,
                        help="Base random seed for experiments (default: 2026)")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.T_total, args.num_samples, args.seed_base)
