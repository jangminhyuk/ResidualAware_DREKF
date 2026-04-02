# Residual-Aware Distributionally Robust Extended Kalman Filter (DR-EKF)

This repository implements the **Residual-Aware DR-EKF**, a distributionally robust extension of the Extended Kalman Filter that accounts for both model uncertainty and EKF linearization error. The filter maintains correctness guarantees under Wasserstein ambiguity sets while automatically inflating the effective robustness radius to absorb linearization residuals from nonlinear dynamics and observations.

## Method Overview

The DR-EKF solves a semidefinite program (SDP) at each time step to compute worst-case noise covariances within a Bures-Wasserstein ball around the nominal distribution. The key contribution is the **residual-aware effective radius**:

```
theta_eff = theta_eps + eta_eps
```

where `theta_eps` is the base distributional radius and `eta_eps = sqrt(eta_f^2 + eta_h^2)` absorbs EKF linearization residuals from dynamics (`eta_f`) and observations (`eta_h`). This ensures the ambiguity set covers both distributional mismatch and linearization error.

## Repository Structure

```
estimator/                        # Filter implementations
    base_filter.py                # Base class: noise sampling, simulation loops, MPC
    EKF.py                        # Extended Kalman Filter
    DR_EKF_trace.py               # DR-EKF with joint-ball Wasserstein ambiguity (one-pass)

ct_radar_models.py                # Shared CT dynamics and radar observation models

common_utils.py                   # Shared utilities: EM estimation, Bures-Wasserstein
                                  #   distance, covariance perturbation, LQR cost

exp_ct_tracking.py                # Experiment: 2D coordinated-turn tracking with radar
exp_ct_tracking_nonlinear.py      # Experiment: CT tracking, sweeping initial turn rate
plot_trajectories_combined.py     # Combined trajectory + nonlinear MSE sweep figure

safe_navigation_uwb.py            # Experiment: unicycle obstacle avoidance with UWB beacons
plot_safe_navigation_uwb.py       # Plotting for UWB navigation experiment

results/                          # Saved experiment outputs (.pkl, .pdf)
```

## Experiments

### 1. Coordinated-Turn Radar Tracking (`exp_ct_tracking.py`, `exp_ct_tracking_nonlinear.py`)

A 5-state coordinated-turn (CT) model `[px, py, vx, vy, omega]` with nonlinear radar observations `[range, bearing]`. The EKF uses deliberately misspecified (underestimated) nominal noise covariances; the DR-EKF hedges against this mismatch via its Wasserstein ambiguity set.

`exp_ct_tracking.py` runs the baseline comparison. `exp_ct_tracking_nonlinear.py` sweeps the true initial turn rate `omega_0` to amplify linearization error, demonstrating that the DR-EKF's residual-aware radius inflation provides increasing benefit as nonlinearity grows. Both produce results consumed by `plot_trajectories_combined.py`.

```bash
py exp_ct_tracking.py --num_exp 10
py exp_ct_tracking_nonlinear.py --num_exp 10
py plot_trajectories_combined.py
```

### 2. Safe Navigation with UWB Beacons (`safe_navigation_uwb.py`)

A unicycle robot navigates from start to goal while avoiding a circular obstacle. Self-localization uses 3 UWB range beacons + compass (4 nonlinear measurements). The DR-EKF's covariance inflation is geometry-dependent via the observation Hessian bound `L_h`, producing larger safety margins in poor-DOP regions near the obstacle.

```bash
py safe_navigation_uwb.py --n-trials 10
py plot_safe_navigation_uwb.py
```

## 3. Safe Navigation in Crowds scenes

This repository contains experiments for distributionally robust EKF (DR-EKF) localization with UWB beacons and a compass, combined with unicycle navigation on ETH/UCY-style pedestrian datasets. Pedestrian motion is forecast with CANVAS predictors (e.g. Trajectron) from non-ego agent histories; a sample-based MPC plans controls under predicted agents and static geometry. Static and interactive scripts reproduce trajectory plots and canvas rollouts used in the paper-style analysis.

```bash
export CANVAS_ROOT="/path/to/CANVAS"
python navigation_trajectories.py
```







## Requirements

- Python 3.8+
- NumPy
- SciPy
- CVXPY
- MOSEK (SDP solver)
- joblib (parallel experiment execution)
- matplotlib (plotting)

### MOSEK License

The DR-EKF solves SDPs using MOSEK via CVXPY. A valid MOSEK license file (`mosek.lic`) must be available on your system. Typical locations:

- **Linux/macOS:** `~/mosek/mosek.lic`
- **Windows:** `%USERPROFILE%\mosek\mosek.lic`

Free academic licenses are available at [https://www.mosek.com/products/academic-licenses/](https://www.mosek.com/products/academic-licenses/). Without a MOSEK license, the DR-EKF experiments will fail at the SDP solve step. The standard EKF and UKF do not require MOSEK.

## Installation

```bash
pip install numpy scipy cvxpy mosek joblib matplotlib
```

Place your MOSEK license file at the appropriate path (see above).

## Key Design Choices

- **Joint ambiguity set:** The Wasserstein ball is defined over the joint noise `(w, v)`, allowing the solver to find worst-case cross-correlations between process and measurement noise.
- **Cached SDP structure:** The CVXPY problem is constructed once and reused with warm-starting across time steps for efficiency.
- **Paired Monte Carlo:** Experiment scripts reset the random seed per filter so that EKF and DR-EKF experience identical noise realizations, enabling fair paired comparisons.
