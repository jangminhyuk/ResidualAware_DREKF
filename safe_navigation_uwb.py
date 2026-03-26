#!/usr/bin/env python3
"""
safe_navigation_uwb.py — Goal-reaching with obstacle avoidance using
UWB beacon + compass self-localization.

Experiment summary
------------------
A unicycle robot must navigate from a start position to a goal position
while avoiding a circular obstacle.  Unlike safe_navigation.py (which uses
direct position measurements y = [px, py]), the robot localises itself via
range measurements to three fixed UWB beacons plus a noisy compass:

    y = [||p - b_1||, ||p - b_2||, ||p - b_3||, theta + v_compass]^T

The range observations are highly nonlinear, activating the L_h / eta_h
correction in the DR-EKF effective radius.  The compass weakly observes
heading, preventing P_theta from diverging (range-only cannot observe
heading at all).  Localisation quality varies spatially (Geometric
Dilution of Precision), making the DR-EKF's covariance inflation
geometry-dependent — conservative precisely where localisation is hardest.

Beacon placement
----------------
Two beacons near the start line and one near the goal create deliberately
poor DOP in the obstacle region (x ~ 8–12 m), where b1/b2 are distant and
nearly collinear as seen from the robot.

  b1 = (0.5, -3.5)    below-near-start
  b2 = (0.5,  3.5)    above-near-start
  b3 = (14.0, -1.5)   below-near-goal

Filter conditions
-----------------
  EKF (misspecified noise):
    Nominal noise underestimated (8x variance for range, 4x for compass).
    Safety margin is small; frequent collisions.

  DR-EKF (Wasserstein ambiguity):
    L_h = 0.5 activates the observation-nonlinearity correction (eta_h).
    Covariance P inflates in the high-DOP zone near the obstacle.

  EKF-Oracle (true noise covariances):
    Upper bound for a correctly tuned EKF.

Controller
----------
Same MPC as safe_navigation.py: horizon N_MPC=10, SLSQP, obstacle inflated
by K_sigma * sqrt(tr(P_pos)), where P_pos is the 2x2 position block of P.
"""

import argparse
import numpy as np
import os
import pickle
from collections import OrderedDict
from scipy.optimize import minimize

from estimator.EKF import EKF
from estimator.DR_EKF_trace import DR_EKF_trace

# ======================================================================
# Unicycle dynamics (identical to safe_navigation.py)
# ======================================================================

DT = 0.2  # timestep (seconds)


def unicycle_dynamics(x, u, dt=DT):
    """Discrete-time unicycle: x = [px, py, theta], u = [v, omega]."""
    px, py, theta = x[0, 0], x[1, 0], x[2, 0]
    v, omega = u[0, 0], u[1, 0]
    return np.array([[px + v * np.cos(theta) * dt],
                     [py + v * np.sin(theta) * dt],
                     [theta + omega * dt]])


def unicycle_jacobian(x, u, dt=DT):
    """Jacobian d f / d x."""
    theta = x[2, 0]
    v = u[0, 0]
    return np.array([[1, 0, -v * np.sin(theta) * dt],
                     [0, 1,  v * np.cos(theta) * dt],
                     [0, 0,  1]])


# ======================================================================
# UWB Beacon + compass observation model
# ======================================================================

BEACON_POS = np.array([[0.5, -3.5],     # below-near-start
                        [0.5,  3.5],     # above-near-start
                        [14.0, -1.5]])   # below-near-goal
N_BEACONS = BEACON_POS.shape[0]
NY = N_BEACONS + 1   # 3 ranges + 1 compass = 4 measurements
R_MIN_JAC = 0.5      # clamp to prevent Jacobian singularity at zero range


def observation_function(x):
    """UWB ranges + compass: y = [r1, r2, r3, theta]^T.

    First N_BEACONS entries are ranges to beacons (nonlinear).
    Last entry is a direct heading measurement (linear).
    """
    px, py, theta = x[0, 0], x[1, 0], x[2, 0]
    y = np.zeros((NY, 1))
    for i in range(N_BEACONS):
        dx = px - BEACON_POS[i, 0]
        dy = py - BEACON_POS[i, 1]
        y[i, 0] = np.sqrt(dx**2 + dy**2)
    y[N_BEACONS, 0] = theta   # compass
    return y


def observation_jacobian(x):
    """Jacobian H is (NY, 3).

    Range rows:   H_{i,:} = [(px-bx_i)/r_i, (py-by_i)/r_i, 0]
    Compass row:  H_{3,:} = [0, 0, 1]
    """
    px, py = x[0, 0], x[1, 0]
    H = np.zeros((NY, 3))
    for i in range(N_BEACONS):
        dx = px - BEACON_POS[i, 0]
        dy = py - BEACON_POS[i, 1]
        r = max(np.sqrt(dx**2 + dy**2), R_MIN_JAC)
        H[i, 0] = dx / r
        H[i, 1] = dy / r
    H[N_BEACONS, 2] = 1.0     # compass: d(theta)/d(theta) = 1
    return H


# ======================================================================
# Experiment constants
# ======================================================================

T = 110              # Navigation steps (22 s at DT=0.2)
N_TRIALS = 100       # Monte Carlo trials

# Confidence multiplier: K_sigma = Phi^{-1}(0.95)
K_SIGMA = 1.645

# Goal-reaching threshold
GOAL_RADIUS = 0.30   # (m)

# Geometry — robot travels in +x direction
START_POS = np.array([0.0, 0.0])
GOAL_POS  = np.array([15.0, 0.0])
START_HEADING = 0.0                 # heading = east (+x)

OBS_CENTER = np.array([10.0, 0.0])  # obstacle centre
OBS_RADIUS = 2.0                    # obstacle physical radius (m)

ARENA_HALF_WIDTH = 5.0              # (m) — half-height of arena (y-axis)

# ----- Noise covariances -----
# Process noise: same as safe_navigation.py
TRUE_SIGMA_W = np.diag([0.008, 0.008, 0.002])
NOM_SIGMA_W  = np.diag([0.002, 0.002, 0.0005])   # 4x underestimate

# Measurement noise: [range1, range2, range3, compass]
# True σ_range = 0.20 m (realistic UWB in multipath / NLOS)
# Nominal σ_range = 0.071 m (lab calibration, LOS) → 8x variance underestimate
# True σ_compass = 0.17 rad (~10 deg, cheap IMU)
# Nominal σ_compass = 0.087 rad (~5 deg) → ~4x variance underestimate
TRUE_SIGMA_V = np.diag([0.04, 0.04, 0.04, 0.03])       # [σ²_r, σ²_r, σ²_r, σ²_compass]
NOM_SIGMA_V  = np.diag([0.005, 0.005, 0.005, 0.0075])  # underestimated

MU_W = np.zeros((3, 1))
MU_V = np.zeros((NY, 1))

X0_MEAN = np.array([[START_POS[0]], [START_POS[1]], [START_HEADING]])
X0_COV_TRUE = np.diag([0.01, 0.01, 0.001])
X0_COV_NOM  = np.diag([0.01, 0.01, 0.001])

# DR-EKF Wasserstein parameters
THETA_EPS = 0.25
THETA_W = THETA_EPS / np.sqrt(2)
THETA_V = THETA_EPS / np.sqrt(2)
L_F = 0.3
L_H = 0.5             # KEY: nonzero — activates eta_h for observation nonlinearity

# Shared matrix placeholders (BaseFilter uses these for nx/ny dimensions)
SYS_DATA = (np.eye(3), np.eye(NY, 3))   # (A_3x3, C_4x3)
B_MAT    = np.eye(3, 2)

# MPC parameters (same as safe_navigation.py)
N_MPC       = 10
Q_GOAL      = 5.0
Q_TERM      = 20.0
R_V         = 0.5
R_W         = 0.5
V_MAX       = 1.5
OMEGA_MAX   = 2.0


# ======================================================================
# Safety margin computation
# ======================================================================


def compute_safety_margin(P):
    """Safety margin from the position block of the state covariance.

    margin = K_sigma * sqrt(tr(P_pos))   where P_pos = P[0:2, 0:2]
    """
    P_pos = P[0:2, 0:2]
    return K_SIGMA * np.sqrt(max(float(np.trace(P_pos)), 0.0))


# ======================================================================
# MPC controller (identical to safe_navigation.py)
# ======================================================================


def _dist_to_obstacle(px, py):
    return np.sqrt((px - OBS_CENTER[0])**2 + (py - OBS_CENTER[1])**2)


def compute_safe_control(x_est, margin):
    """MPC: reach goal while avoiding inflated obstacle."""
    inflated_radius = OBS_RADIUS + margin

    px0    = x_est[0, 0]
    py0    = x_est[1, 0]
    theta0 = x_est[2, 0]
    goal_px, goal_py = GOAL_POS

    def _rollout(u_vec):
        states = []
        px, py, theta = px0, py0, theta0
        for k in range(N_MPC):
            v_k     = u_vec[2 * k]
            omega_k = u_vec[2 * k + 1]
            px    += v_k * np.cos(theta) * DT
            py    += v_k * np.sin(theta) * DT
            theta += omega_k * DT
            states.append((px, py, theta))
        return states

    def mpc_cost(u_vec):
        px, py, theta = px0, py0, theta0
        cost = 0.0
        for k in range(N_MPC):
            v_k     = u_vec[2 * k]
            omega_k = u_vec[2 * k + 1]
            cost += Q_GOAL * ((px - goal_px)**2 + (py - goal_py)**2)
            cost += R_V * v_k**2 + R_W * omega_k**2
            px    += v_k * np.cos(theta) * DT
            py    += v_k * np.sin(theta) * DT
            theta += omega_k * DT
        cost += Q_TERM * ((px - goal_px)**2 + (py - goal_py)**2)
        return cost

    def obs_constraint(u_vec):
        states = _rollout(u_vec)
        vals = []
        for px, py, _ in states:
            d_sq = (px - OBS_CENTER[0])**2 + (py - OBS_CENTER[1])**2
            vals.append(d_sq - inflated_radius**2)
        return np.array(vals)

    bounds = []
    for _ in range(N_MPC):
        bounds.append((0.0, V_MAX))
        bounds.append((-OMEGA_MAX, OMEGA_MAX))

    constraints = [{'type': 'ineq', 'fun': obs_constraint}]

    best_result = None
    for omega_bias in [-0.4, 0.4]:
        u0 = np.zeros(2 * N_MPC)
        u0[0::2] = 1.0
        d_to_obs = _dist_to_obstacle(px0, py0)
        if d_to_obs < inflated_radius + 3.0:
            u0[1::2] = omega_bias

        res = minimize(mpc_cost, u0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'ftol': 1e-6, 'maxiter': 300})
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    if best_result.success or best_result.fun < 1e8:
        v_star     = float(np.clip(best_result.x[0], 0.0, V_MAX))
        omega_star = float(np.clip(best_result.x[1], -OMEGA_MAX, OMEGA_MAX))
    else:
        angle_to_goal = np.arctan2(goal_py - py0, goal_px - px0)
        heading_err   = angle_to_goal - theta0
        heading_err   = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        v_star     = 0.8
        omega_star = float(np.clip(3.0 * heading_err, -OMEGA_MAX, OMEGA_MAX))

    return np.array([[v_star], [omega_star]])


# ======================================================================
# Collision check
# ======================================================================


def is_collision(x_true):
    px = x_true[0, 0]
    py = x_true[1, 0]
    return _dist_to_obstacle(px, py) < OBS_RADIUS


# ======================================================================
# Filter factories
# ======================================================================


def _make_ekf_base(nom_Sigma_w, nom_Sigma_v):
    """Construct an EKF with specified nominal noise covariances."""
    return EKF(
        T=T + 10,
        dist='normal', noise_dist='normal',
        system_data=SYS_DATA, B=B_MAT,
        true_x0_mean=X0_MEAN,       true_x0_cov=X0_COV_TRUE,
        true_mu_w=MU_W,             true_Sigma_w=TRUE_SIGMA_W,
        true_mu_v=MU_V,             true_Sigma_v=TRUE_SIGMA_V,
        nominal_x0_mean=X0_MEAN.copy(),   nominal_x0_cov=X0_COV_NOM.copy(),
        nominal_mu_w=MU_W.copy(),         nominal_Sigma_w=nom_Sigma_w.copy(),
        nominal_mu_v=MU_V.copy(),         nominal_Sigma_v=nom_Sigma_v.copy(),
        nonlinear_dynamics=unicycle_dynamics,
        dynamics_jacobian=unicycle_jacobian,
        observation_function=observation_function,
        observation_jacobian=observation_jacobian,
    )


def make_ekf():
    """EKF with misspecified (underestimated) nominal noise."""
    return _make_ekf_base(NOM_SIGMA_W, NOM_SIGMA_V)


def make_ekf_oracle():
    """EKF with true noise covariances (oracle upper bound)."""
    return _make_ekf_base(TRUE_SIGMA_W, TRUE_SIGMA_V)


def make_drekf():
    """DR-EKF with Wasserstein ambiguity ball."""
    return DR_EKF_trace(
        T=T + 10,
        dist='normal', noise_dist='normal',
        system_data=SYS_DATA, B=B_MAT,
        true_x0_mean=X0_MEAN,       true_x0_cov=X0_COV_TRUE,
        true_mu_w=MU_W,             true_Sigma_w=TRUE_SIGMA_W,
        true_mu_v=MU_V,             true_Sigma_v=TRUE_SIGMA_V,
        nominal_x0_mean=X0_MEAN.copy(),   nominal_x0_cov=X0_COV_NOM.copy(),
        nominal_mu_w=MU_W.copy(),         nominal_Sigma_w=NOM_SIGMA_W.copy(),
        nominal_mu_v=MU_V.copy(),         nominal_Sigma_v=NOM_SIGMA_V.copy(),
        nonlinear_dynamics=unicycle_dynamics,
        dynamics_jacobian=unicycle_jacobian,
        observation_function=observation_function,
        observation_jacobian=observation_jacobian,
        theta_w=THETA_W, theta_v=THETA_V,
        L_f=L_F, L_h=L_H,
    )


# ======================================================================
# Single trial
# ======================================================================


def run_trial(make_filter_fn, seed, zero_margin=False):
    """Run one trial: goal-reaching with obstacle avoidance.

    Returns dict with trajectory, margins, collisions, and diagnostics.
    """
    rng  = np.random.RandomState(seed)
    filt = make_filter_fn()

    # ---- Initial state (perturbed from X0_MEAN) ----
    x_true = X0_MEAN + rng.multivariate_normal(
        np.zeros(3), X0_COV_TRUE).reshape(3, 1)
    x_est = X0_MEAN.copy()

    v0 = rng.multivariate_normal(np.zeros(NY), TRUE_SIGMA_V).reshape(NY, 1)
    y0 = observation_function(x_true) + v0
    x_est = filt._initial_update(x_est, y0)

    # ---- Navigation ----
    nav_x_true   = np.zeros((T + 1, 3, 1))
    nav_x_est    = np.zeros((T + 1, 3, 1))
    nav_margins    = np.zeros(T + 1)
    nav_trace_P    = np.zeros(T + 1)
    nav_trace_Ppos = np.zeros(T + 1)
    nav_P_theta    = np.zeros(T + 1)
    nav_theta_eff  = np.zeros(T + 1)
    nav_collisions = np.zeros(T + 1, dtype=bool)
    nav_controls   = np.zeros((T, 2, 1))

    def _log_cov(P, idx):
        if P is None:
            return
        nav_trace_P[idx]    = float(np.trace(P))
        nav_trace_Ppos[idx] = float(P[0, 0] + P[1, 1])
        nav_P_theta[idx]    = float(P[2, 2])
        nav_theta_eff[idx]  = getattr(filt, '_last_theta_eps_effective', 0.0)

    nav_x_true[0]   = x_true.copy()
    nav_x_est[0]    = x_est.copy()
    nav_margins[0]  = 0.0 if zero_margin else compute_safety_margin(filt._P)
    _log_cov(filt._P, 0)
    nav_collisions[0] = is_collision(x_true)

    goal_reached = False
    t_reach      = None

    for t_nav in range(T):
        px_now = x_true[0, 0]
        py_now = x_true[1, 0]
        if np.sqrt((px_now - GOAL_POS[0])**2 + (py_now - GOAL_POS[1])**2) < GOAL_RADIUS:
            for t_fill in range(t_nav, T):
                nav_x_true[t_fill + 1]      = x_true.copy()
                nav_x_est[t_fill + 1]       = x_est.copy()
                nav_margins[t_fill + 1]     = np.nan
                nav_trace_P[t_fill + 1]     = np.nan
                nav_trace_Ppos[t_fill + 1]  = np.nan
                nav_P_theta[t_fill + 1]     = np.nan
                nav_theta_eff[t_fill + 1]   = np.nan
                nav_collisions[t_fill + 1]  = False
            goal_reached = True
            t_reach      = t_nav
            break

        # Safety margin from current filter covariance
        margin = 0.0 if zero_margin else compute_safety_margin(filt._P)

        # MPC control
        u = compute_safe_control(x_est, margin)
        nav_controls[t_nav] = u.copy()

        # True state propagation
        w = rng.multivariate_normal(np.zeros(3), TRUE_SIGMA_W).reshape(3, 1)
        x_true = unicycle_dynamics(x_true, u) + w

        # Measurement: ranges + compass
        v = rng.multivariate_normal(np.zeros(NY), TRUE_SIGMA_V).reshape(NY, 1)
        y = observation_function(x_true) + v

        # Filter update
        x_est = filt.update_step(x_est, y, 1 + t_nav, u)

        nav_x_true[t_nav + 1]      = x_true.copy()
        nav_x_est[t_nav + 1]       = x_est.copy()
        nav_margins[t_nav + 1]     = 0.0 if zero_margin else compute_safety_margin(filt._P)
        _log_cov(filt._P, t_nav + 1)
        nav_collisions[t_nav + 1]  = is_collision(x_true)

    # Check goal at final step
    if not goal_reached:
        px_fin = nav_x_true[T, 0, 0]
        py_fin = nav_x_true[T, 1, 0]
        if np.sqrt((px_fin - GOAL_POS[0])**2 + (py_fin - GOAL_POS[1])**2) < GOAL_RADIUS:
            goal_reached = True
            t_reach      = T

    return {
        'x_true':       nav_x_true,
        'x_est':        nav_x_est,
        'margins':      nav_margins,
        'trace_P':      nav_trace_P,
        'trace_P_pos':  nav_trace_Ppos,
        'P_theta':      nav_P_theta,
        'theta_eff':    nav_theta_eff,
        'collisions':   nav_collisions,
        'any_collision': bool(np.any(nav_collisions)),
        'goal_reached': goal_reached,
        't_reach':      t_reach,
        'controls':     nav_controls,
    }


# ======================================================================
# Monte Carlo
# ======================================================================


def run_monte_carlo(make_filter_fn, n_trials, label='', zero_margin=False):
    results = []
    for i in range(n_trials):
        seed = 10000 + i
        if label:
            print(f'  {label} trial {i+1}/{n_trials}', end='\r')
        results.append(run_trial(make_filter_fn, seed, zero_margin=zero_margin))
    if label:
        print()
    return results


# ======================================================================
# Metrics
# ======================================================================


def compute_metrics(results):
    """Compute collision rate, goal-reach rate, mean arrival time, path length."""
    collision_rate  = float(np.mean([r['any_collision'] for r in results]))
    goal_reach_rate = float(np.mean([r['goal_reached'] for r in results]))

    t_reaches = [r['t_reach'] for r in results
                 if r['goal_reached'] and not r['any_collision']]
    mean_t_reach = float(np.mean(t_reaches)) if t_reaches else float('nan')

    path_lengths = []
    for r in results:
        pts = r['x_true'][:, :2, 0]
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        path_lengths.append(float(np.sum(dists)))
    mean_path_length = float(np.mean(path_lengths))

    return collision_rate, goal_reach_rate, mean_t_reach, mean_path_length


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Safe navigation with UWB beacon self-localization')
    parser.add_argument('--zero-margin', action='store_true',
                        help='Force safety margin to 0')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS,
                        help=f'Number of Monte Carlo trials (default {N_TRIALS})')
    args = parser.parse_args()
    n_trials = args.n_trials

    print('=' * 65)
    print('Safe Navigation with UWB Beacon + Compass Self-Localization')
    if args.zero_margin:
        print('  ** ZERO-MARGIN MODE **')
    print('=' * 65)
    print(f'Start: {START_POS}  Goal: {GOAL_POS}  '
          f'Heading: {np.degrees(START_HEADING):.0f} deg')
    print(f'Obstacle: centre={OBS_CENTER}, radius={OBS_RADIUS} m')
    print(f'Observation: {N_BEACONS} UWB ranges + 1 compass (ny={NY})')
    print(f'Beacons:')
    for i, b in enumerate(BEACON_POS):
        print(f'  b{i+1} = ({b[0]:.1f}, {b[1]:.1f})')
    print(f'Time: T={T} steps ({T*DT:.0f}s)')
    print(f'Trials: {n_trials}  MPC horizon: {N_MPC}')
    print(f'L_H = {L_H}  (observation nonlinearity bound)')
    print()

    results_dir = os.path.join('results', 'safe_navigation_uwb')
    os.makedirs(results_dir, exist_ok=True)

    # ---- Conditions ----
    conditions = OrderedDict()
    if args.zero_margin:
        conditions['EKF_zm'] = {
            'label':   'EKF (no margin)',
            'color':   '#ff7f0e',
            'make_fn': make_ekf,
            'zero_margin': True,
        }
        conditions['DREKF_zm'] = {
            'label':   'DR-EKF (no margin)',
            'color':   '#9467bd',
            'make_fn': make_drekf,
            'zero_margin': True,
        }
        conditions['EKF_true_zm'] = {
            'label':   'EKF-true (no margin)',
            'color':   '#8c564b',
            'make_fn': make_ekf_oracle,
            'zero_margin': True,
        }
    else:
        conditions['EKF'] = {
            'label':   'EKF (nominal)',
            'color':   '#d62728',
            'make_fn': make_ekf,
        }
        conditions['DREKF'] = {
            'label':   'DR-EKF',
            'color':   '#1f77b4',
            'make_fn': make_drekf,
        }
        conditions['EKF_true'] = {
            'label':   'EKF (true)',
            'color':   '#2ca02c',
            'make_fn': make_ekf_oracle,
        }

    # ---- Run Monte Carlo ----
    cond_results = OrderedDict()
    for key, cond in conditions.items():
        label = cond['label']
        zm = cond.get('zero_margin', False)
        print(f'Running {label} ({n_trials} trials)...')
        results = run_monte_carlo(cond['make_fn'], n_trials, label=label,
                                  zero_margin=zm)
        cr, gr, mt, pl = compute_metrics(results)
        mean_margin = float(np.nanmean([np.nanmean(r['margins']) for r in results]))

        cond_results[key] = {
            'label':          cond['label'],
            'color':          cond['color'],
            'results':        results,
            'collision_rate': cr,
            'goal_reach_rate': gr,
            'mean_t_reach':   mt,
            'mean_path_length': pl,
            'mean_margin':    mean_margin,
        }
        print(f'  -> collision: {100*cr:.1f}%  '
              f'goal reached: {100*gr:.1f}%  '
              f'mean t_reach: {mt:.1f}  '
              f'path: {pl:.2f} m  '
              f'mean margin: {mean_margin:.3f} m')
        print()

    # ---- Summary table ----
    print('=' * 78)
    print(f'{"Condition":<18} {"Margin":>9} {"Collision":>10} '
          f'{"Goal%":>8} {"t_reach":>9} {"Path (m)":>10}')
    print('-' * 78)
    for key, cr in cond_results.items():
        mm   = cr['mean_margin']
        coll = f'{100*cr["collision_rate"]:.1f}%'
        goal = f'{100*cr["goal_reach_rate"]:.1f}%'
        t_r  = f'{cr["mean_t_reach"]:.1f}' if not np.isnan(cr["mean_t_reach"]) else 'n/a'
        pl   = f'{cr["mean_path_length"]:.2f}'
        print(f'{cr["label"]:<18} {mm:>9.3f}m {coll:>10} {goal:>8} {t_r:>9} {pl:>10}')
    print('=' * 78)
    print()

    # ---- Save ----
    save_dict = {
        'conditions': {k: {kk: vv for kk, vv in v.items()}
                       for k, v in cond_results.items()},
        'params': {
            'T': T, 'DT': DT, 'N_TRIALS': n_trials,
            'K_SIGMA': K_SIGMA, 'N_MPC': N_MPC,
            'GOAL_RADIUS': GOAL_RADIUS,
            'START_POS': START_POS, 'GOAL_POS': GOAL_POS,
            'OBS_CENTER': OBS_CENTER, 'OBS_RADIUS': OBS_RADIUS,
            'ARENA_HALF_WIDTH': ARENA_HALF_WIDTH,
            'BEACON_POS': BEACON_POS, 'N_BEACONS': N_BEACONS, 'NY': NY,
            'TRUE_SIGMA_W': TRUE_SIGMA_W, 'TRUE_SIGMA_V': TRUE_SIGMA_V,
            'NOM_SIGMA_W':  NOM_SIGMA_W,  'NOM_SIGMA_V':  NOM_SIGMA_V,
            'THETA_W': THETA_W, 'THETA_V': THETA_V,
            'L_F': L_F, 'L_H': L_H,
            'Q_GOAL': Q_GOAL, 'Q_TERM': Q_TERM,
            'R_V': R_V, 'R_W': R_W,
            'V_MAX': V_MAX, 'OMEGA_MAX': OMEGA_MAX,
        },
    }

    pkl_path = os.path.join(results_dir, 'navigation_uwb_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f'Results saved to {pkl_path}')
    print('Run plot_safe_navigation_uwb.py to generate figures.')


if __name__ == '__main__':
    main()
