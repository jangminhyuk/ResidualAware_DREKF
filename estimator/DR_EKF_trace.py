#!/usr/bin/env python3
"""
DR_EKF_trace.py — Joint-ball DR-EKF with one-pass effective radius
                   and recursive Pbar certificate.

Implements a distributionally robust Extended Kalman filter (DR-EKF) with
joint ambiguity sets.  The effective Wasserstein radius is inflated at each
time step to absorb EKF linearization residuals via a recursive scalar
bound Pbar_t (not trace(P)).

Assumptions:
  (A1) Noise-centric ambiguity: the true joint noise (w_t, v_t) lies in
       a Bures-Wasserstein ball of radius theta_eps around the nominal.
  (A2) Raw-noise independence: w_t and v_t are independent.
  (A3) Local Hessian bounds: ||nabla^2 f|| <= L_f, ||nabla^2 h|| <= L_h.

Initial stage (t = 0)
----------------------
  theta_eps0 = sqrt(theta_x0^2 + theta_v^2)
  gamma0     = (sqrt(Tr(Sigma_x0_hat)) + theta_eps0)^2
  eta_f0     = 0
  eta_h0     = 0.5 * L_h * alpha_h * gamma0
  theta_eff_eps0 = theta_eps0 + eta_h0

  After SDP solve and K0:
    rbar0 = 2 * ||K0||^2 * eta_h0^2
    rho0  = sqrt(rbar0)
    Pbar0 = (sqrt(Tr(Sigma_x_post_star_0)) + rho0)^2

Radius computation (t >= 1)
----------------------------
  theta_eps  = sqrt(theta_w^2 + theta_v^2)
  eta_f      = 0.5 * L_f * alpha_f * Pbar_{t-1}
  gamma_t    = (||A_t||_2 * sqrt(Pbar_{t-1})
               + sqrt(Tr(Sigma_w_hat)) + theta_eps + eta_f)^2
  eta_h      = 0.5 * L_h * alpha_h * gamma_t
  theta_eff_eps = theta_eps + sqrt(eta_f^2 + eta_h^2)

  After SDP solve and K_t:
    kappa_t = ||(I - K_t C_t) A_t||_2
    rbar_t  = 2 ||I - K_t C_t||^2 eta_f^2 + 2 ||K_t||^2 eta_h^2
    rho_t   = kappa_t * rho_{t-1} + sqrt(rbar_t)
    Pbar_t  = (sqrt(Tr(Sigma_x_post_star_t)) + rho_t)^2
"""

import numpy as np
import cvxpy as cp
import warnings
from .base_filter import BaseFilter

# Suppress non-DPP parameterized problem warnings from CVXPY/MOSEK
warnings.filterwarnings(
    "ignore",
    message="You are solving a parameterized problem that is not DPP.*",
    category=UserWarning,
)


class DR_EKF_trace(BaseFilter):
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 nonlinear_dynamics=None,
                 dynamics_jacobian=None,
                 observation_function=None,
                 observation_jacobian=None,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None,
                 theta_w=None, theta_v=None,
                 theta_x0=None,
                 L_f=None, L_h=None, R_f=None, R_h=None,
                 alpha_f=None, alpha_h=None,
                 rho_bar_process=1.0,
                 rho_bar_joint=1.0,
                 theta_eff_cap=None,
                 eta_scale=1.0,
                 input_lower_bound=None, input_upper_bound=None):
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, None,
                        input_lower_bound, input_upper_bound)

        # Store nonlinear dynamics and jacobians (required for DR-EKF)
        if not all([nonlinear_dynamics, dynamics_jacobian, observation_function, observation_jacobian]):
            raise ValueError("DR-EKF requires all nonlinear functions: dynamics, dynamics_jacobian, observation, observation_jacobian")

        self.f = nonlinear_dynamics
        self.F_jacobian = dynamics_jacobian
        self.h = observation_function
        self.C_jacobian = observation_jacobian

        # Wasserstein radii for process noise and measurement noise
        self.theta_w = theta_w
        self.theta_v = theta_v

        # Initial-state Wasserstein radius
        self.theta_x0 = theta_x0 if theta_x0 is not None else (theta_w if theta_w is not None else 0.0)

        # Base joint radius (static, used as the distributional component)
        self.theta_eps_base = (np.sqrt(theta_w**2 + theta_v**2)
                               if (theta_w is not None and theta_v is not None)
                               else None)

        # Correlation coefficients for cross-term in radius inflation.
        # rho_bar = 1.0 (default) => certified triangle-inequality bound.
        # rho_bar < 1.0 => tighter but requires additional assumptions (see docstring).
        if not (0.0 <= rho_bar_process <= 1.0):
            raise ValueError(f"rho_bar_process must be in [0, 1], got {rho_bar_process}")
        if not (0.0 <= rho_bar_joint <= 1.0):
            raise ValueError(f"rho_bar_joint must be in [0, 1], got {rho_bar_joint}")
        self.rho_bar_process = rho_bar_process
        self.rho_bar_joint = rho_bar_joint

        # Cap on effective radius to prevent positive-feedback blowup.
        # Default: 5x the static base radius.  Set higher for looser control.
        if theta_eff_cap is not None:
            self.theta_eff_cap = theta_eff_cap
        elif self.theta_eps_base is not None:
            self.theta_eff_cap = 3.0 * self.theta_eps_base
        else:
            self.theta_eff_cap = None

        # Heuristic scaling factor for linearization residuals eta_f, eta_h.
        # c = 1.0 uses the certified (conservative) bound; c < 1 tightens it.
        self.eta_scale = eta_scale

        # Trust region parameters
        self.L_f = L_f if L_f is not None else 0.3
        self.L_h = L_h if L_h is not None else 0.01
        self.R_f = R_f if R_f is not None else (alpha_f if alpha_f is not None else np.sqrt(3))
        self.R_h = R_h if R_h is not None else (alpha_h if alpha_h is not None else np.sqrt(3))

        # Explicit alpha parameters for one-pass radius (backward compat: fall back to R_f/R_h)
        self.alpha_f = alpha_f if alpha_f is not None else (R_f if R_f is not None else np.sqrt(3))
        self.alpha_h = alpha_h if alpha_h is not None else (R_h if R_h is not None else np.sqrt(3))

        # Initialize posterior covariance for online computation
        self._P = None

        # Recursive certificate state (scalar)
        self._Pbar = None       # scalar Pbar_t
        self._rho_cert = None   # scalar rho_t

        # Pre-created SDP problems and parameter references for efficiency
        self._sdp_problem_initial = None
        self._sdp_params_initial = None
        self._sdp_problem_regular = None
        self._sdp_params_regular = None
        self._warm_start_vars_initial = None
        self._warm_start_vars_regular = None

    # ------------------------------------------------------------------
    # One-pass effective-radius computation (t >= 1)
    # ------------------------------------------------------------------
    def _compute_theta_eps_effective(self, Pbar_prev, A_t):
        """One-pass effective radius for t >= 1 using recursive Pbar bound.

        theta_eps  = sqrt(theta_w^2 + theta_v^2)
        eta_f      = 0.5 * L_f * alpha_f * Pbar_prev
        gamma_t    = (||A_t||_2 * sqrt(Pbar_prev)
                     + sqrt(Tr(Sigma_w_hat)) + theta_eps + eta_f)^2
        eta_h      = 0.5 * L_h * alpha_h * gamma_t
        theta_eff  = theta_eps + sqrt(eta_f^2 + eta_h^2)
        """
        # Joint noise radius
        theta_eps = self.theta_eps_base  # sqrt(theta_w^2 + theta_v^2)

        # Dynamics residual from Pbar bound
        eta_f = 0.5 * self.L_f * self.alpha_f * Pbar_prev

        # Prior second-moment bound (gamma uses theta_eps, not theta_w)
        A_norm = np.linalg.norm(A_t, ord=2)
        tr_Sigma_w = max(float(np.trace(self.nominal_Sigma_w)), 0.0)
        gamma_t = (A_norm * np.sqrt(Pbar_prev)
                   + np.sqrt(tr_Sigma_w)
                   + theta_eps
                   + eta_f) ** 2

        # Observation residual
        eta_h = 0.5 * self.L_h * self.alpha_h * gamma_t

        # Cap individual residuals to prevent overflow in eta_f**2 + eta_h**2
        if self.theta_eff_cap is not None:
            eta_f = min(eta_f, self.theta_eff_cap)
            eta_h = min(eta_h, self.theta_eff_cap)

        # Effective radius (with empirical cap)
        theta_eps_effective = theta_eps + np.sqrt(eta_f**2 + eta_h**2)
        if self.theta_eff_cap is not None and theta_eps_effective > self.theta_eff_cap:
            theta_eps_effective = self.theta_eff_cap

        # Diagnostics
        self._last_theta_eps = theta_eps
        self._last_gamma = gamma_t
        self._last_eta_f = eta_f
        self._last_eta_h = eta_h
        self._last_theta_eps_effective = theta_eps_effective
        return theta_eps_effective

    # ------------------------------------------------------------------
    # One-pass effective-radius computation (t = 0)
    # ------------------------------------------------------------------
    def _compute_theta_eps0_effective(self):
        """One-pass effective radius for t=0.

        theta_eps0 = sqrt(theta_x0^2 + theta_v^2)
        gamma0     = (sqrt(Tr(Sigma_x0_hat)) + theta_eps0)^2
        eta_f0     = 0
        eta_h0     = 0.5 * L_h * alpha_h * gamma0
        theta_eff_eps0 = theta_eps0 + eta_h0
        """
        # Joint noise radius at t=0
        theta_eps0 = np.sqrt(self.theta_x0**2 + self.theta_v**2)

        # Prior second-moment bound
        tr_x0_cov = max(float(np.trace(self.nominal_x0_cov)), 0.0)
        gamma0 = (np.sqrt(tr_x0_cov) + theta_eps0)**2

        # Linearization residuals
        eta_f0 = 0.0
        eta_h0 = 0.5 * self.L_h * self.alpha_h * gamma0

        # Cap observation residual to prevent overflow
        if self.theta_eff_cap is not None:
            eta_h0 = min(eta_h0, self.theta_eff_cap)

        # Effective radius (with empirical cap)
        theta_eff_eps0 = theta_eps0 + eta_h0
        if self.theta_eff_cap is not None and theta_eff_eps0 > self.theta_eff_cap:
            theta_eff_eps0 = self.theta_eff_cap

        # Diagnostics
        self._last_theta_eps = theta_eps0
        self._last_gamma = gamma0
        self._last_eta_f = eta_f0
        self._last_eta_h = eta_h0
        self._last_theta_eps_effective = theta_eff_eps0
        return theta_eff_eps0

    # ------------------------------------------------------------------
    # SDP construction — initial step (t = 0)
    # ------------------------------------------------------------------
    def _create_and_cache_sdp_initial(self):
        """Create initial joint-ball SDP structure once and cache for reuse."""
        if self._sdp_problem_initial is not None:
            return self._sdp_problem_initial, self._sdp_params_initial

        n_eps = self.nx + self.ny

        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        Sigma_xv = cp.Variable((self.nx, self.ny), name='Sigma_xv')
        Y_joint = cp.Variable((n_eps, n_eps), name='Y_joint')

        X_pred_hat = cp.Parameter((self.nx, self.nx), name='X_pred_hat')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')
        theta_eps0 = cp.Parameter(nonneg=True, name='theta_eps0')
        lam_min_eps_nom = cp.Parameter(nonneg=True, name='lam_min_eps')
        lam_min_v_nom = cp.Parameter(nonneg=True, name='lam_min_v')
        C_t = cp.Parameter((self.ny, self.nx), name='C_t')

        Sigma_eps0 = cp.bmat([[X_pred, Sigma_xv],
                              [Sigma_xv.T, Sigma_v]])
        Sigma_hat_eps0 = cp.bmat([[X_pred_hat, np.zeros((self.nx, self.ny))],
                                  [np.zeros((self.ny, self.nx)), Sigma_v_hat]])

        T0 = X_pred @ C_t.T + Sigma_xv
        S0 = C_t @ X_pred @ C_t.T + Sigma_v + C_t @ Sigma_xv + Sigma_xv.T @ C_t.T

        obj = cp.Maximize(cp.trace(X))
        constraints = [
            # Schur complement PSD for posterior covariance
            cp.bmat([[X_pred - X, T0],
                     [T0.T, S0]
                    ]) >> 0,
            # Joint Bures constraint
            cp.bmat([[Sigma_hat_eps0, Y_joint],
                     [Y_joint.T, Sigma_eps0]
                    ]) >> 0,
            cp.trace(Sigma_eps0 + Sigma_hat_eps0 - 2*Y_joint) <= theta_eps0**2,
            # PSD and lower bound constraints
            X >> 0,
            Sigma_eps0 >> lam_min_eps_nom * np.eye(n_eps),
            Sigma_v >> lam_min_v_nom * np.eye(self.ny),
        ]

        prob = cp.Problem(obj, constraints)

        self._sdp_problem_initial = prob
        self._sdp_params_initial = {
            'X_pred_hat': X_pred_hat,
            'Sigma_v_hat': Sigma_v_hat,
            'theta_eps0': theta_eps0,
            'lam_min_eps_nom': lam_min_eps_nom,
            'lam_min_v_nom': lam_min_v_nom,
            'C_t': C_t
        }

        self._warm_start_vars_initial = {
            'X': X,
            'X_pred': X_pred,
            'Sigma_v': Sigma_v,
            'Sigma_xv': Sigma_xv,
            'Y_joint': Y_joint
        }

        return prob, self._sdp_params_initial

    def solve_sdp_online_initial(self, X_pred_hat, C_t):
        """Solve joint-ball SDP for t=0 with certified residual-aware radius."""
        prob, params = self._create_and_cache_sdp_initial()

        # Compute certified effective radius for initial step
        theta_eps0_effective = self._compute_theta_eps0_effective()

        params['X_pred_hat'].value = X_pred_hat
        params['Sigma_v_hat'].value = self.nominal_Sigma_v
        params['theta_eps0'].value = theta_eps0_effective
        Sigma_hat_eps0_val = np.block([[X_pred_hat, np.zeros((self.nx, self.ny))],
                                       [np.zeros((self.ny, self.nx)), self.nominal_Sigma_v]])
        # Clamp min eigenvalue to small positive for nonneg parameter (avoid solver failure)
        lam_eps = np.min(np.real(np.linalg.eigvals(Sigma_hat_eps0_val)))
        lam_v = np.min(np.real(np.linalg.eigvals(self.nominal_Sigma_v)))
        params['lam_min_eps_nom'].value = max(1e-8, lam_eps)
        params['lam_min_v_nom'].value = max(1e-8, lam_v)
        params['C_t'].value = C_t

        if self._warm_start_vars_initial is not None:
            for var_name, var in self._warm_start_vars_initial.items():
                if var.value is not None:
                    var.value = var.value

        prob.solve(solver=cp.MOSEK, warm_start=True)

        if prob.status in ["infeasible", "unbounded"]:
            print(f'DR-EKF trace SDP initial problem: {prob.status}')
            return None, None, None, None

        vars_ = self._warm_start_vars_initial
        worst_case_Xpost = vars_['X'].value
        worst_case_Xprior = vars_['X_pred'].value
        worst_case_Sigma_v = vars_['Sigma_v'].value
        worst_case_Sigma_xv = vars_['Sigma_xv'].value

        return worst_case_Sigma_v, worst_case_Sigma_xv, worst_case_Xprior, worst_case_Xpost

    # ------------------------------------------------------------------
    # SDP construction — regular steps (t > 0)
    # ------------------------------------------------------------------
    def _create_and_cache_sdp_regular(self):
        """Create regular joint-ball SDP structure for t>0 once and cache for reuse."""
        if self._sdp_problem_regular is not None:
            return self._sdp_problem_regular, self._sdp_params_regular

        n_eps = self.nx + self.ny

        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        Sigma_w = cp.Variable((self.nx, self.nx), symmetric=True, name='Sigma_w')
        Sigma_wv = cp.Variable((self.nx, self.ny), name='Sigma_wv')
        Y_joint = cp.Variable((n_eps, n_eps), name='Y_joint')

        Sigma_w_hat = cp.Parameter((self.nx, self.nx), name='Sigma_w_hat')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')
        theta_eps = cp.Parameter(nonneg=True, name='theta_eps')
        X_post_prev = cp.Parameter((self.nx, self.nx), name='X_post_prev')
        lam_min_eps_nom = cp.Parameter(nonneg=True, name='lam_min_eps')
        lam_min_v_nom = cp.Parameter(nonneg=True, name='lam_min_v')
        lam_min_w_nom = cp.Parameter(nonneg=True, name='lam_min_w')
        A_t = cp.Parameter((self.nx, self.nx), name='A_t')
        C_t = cp.Parameter((self.ny, self.nx), name='C_t')

        Sigma_eps = cp.bmat([[Sigma_w, Sigma_wv],
                             [Sigma_wv.T, Sigma_v]])
        Sigma_hat_eps = cp.bmat([[Sigma_w_hat, np.zeros((self.nx, self.ny))],
                                 [np.zeros((self.ny, self.nx)), Sigma_v_hat]])

        T = X_pred @ C_t.T + Sigma_wv
        S = C_t @ X_pred @ C_t.T + Sigma_v + C_t @ Sigma_wv + Sigma_wv.T @ C_t.T

        obj = cp.Maximize(cp.trace(X))
        constraints = [
            # Schur complement PSD for posterior covariance
            cp.bmat([[X_pred - X, T],
                     [T.T, S]
                    ]) >> 0,
            # Prior covariance constraint
            X_pred == A_t @ X_post_prev @ A_t.T + Sigma_w,
            # Joint Bures constraint
            cp.bmat([[Sigma_hat_eps, Y_joint],
                     [Y_joint.T, Sigma_eps]
                    ]) >> 0,
            cp.trace(Sigma_eps + Sigma_hat_eps - 2*Y_joint) <= theta_eps**2,
            # PSD and lower bound constraints
            X >> 0,
            Sigma_eps >> lam_min_eps_nom * np.eye(n_eps),
            Sigma_v >> lam_min_v_nom * np.eye(self.ny),
            Sigma_w >> lam_min_w_nom * np.eye(self.nx),
        ]

        prob = cp.Problem(obj, constraints)

        self._sdp_problem_regular = prob
        self._sdp_params_regular = {
            'Sigma_w_hat': Sigma_w_hat,
            'Sigma_v_hat': Sigma_v_hat,
            'theta_eps': theta_eps,
            'X_post_prev': X_post_prev,
            'lam_min_eps_nom': lam_min_eps_nom,
            'lam_min_v_nom': lam_min_v_nom,
            'lam_min_w_nom': lam_min_w_nom,
            'A_t': A_t,
            'C_t': C_t
        }

        self._warm_start_vars_regular = {
            'X': X,
            'X_pred': X_pred,
            'Sigma_v': Sigma_v,
            'Sigma_w': Sigma_w,
            'Sigma_wv': Sigma_wv,
            'Y_joint': Y_joint
        }

        return prob, self._sdp_params_regular

    def solve_sdp_online(self, X_post_prev, A_t, C_t):
        """Solve joint-ball SDP for t>0 with certified theta_eps_effective."""
        prob, params = self._create_and_cache_sdp_regular()

        theta_eps_effective = self._compute_theta_eps_effective(self._Pbar, A_t)

        params['Sigma_w_hat'].value = self.nominal_Sigma_w
        params['Sigma_v_hat'].value = self.nominal_Sigma_v
        params['theta_eps'].value = theta_eps_effective
        params['X_post_prev'].value = X_post_prev
        Sigma_hat_eps_val = np.block([[self.nominal_Sigma_w, np.zeros((self.nx, self.ny))],
                                      [np.zeros((self.ny, self.nx)), self.nominal_Sigma_v]])
        # Clamp min eigenvalues to small positive for nonneg parameters
        lam_eps = np.min(np.real(np.linalg.eigvals(Sigma_hat_eps_val)))
        lam_v = np.min(np.real(np.linalg.eigvals(self.nominal_Sigma_v)))
        lam_w = np.min(np.real(np.linalg.eigvals(self.nominal_Sigma_w)))
        params['lam_min_eps_nom'].value = max(1e-8, lam_eps)
        params['lam_min_v_nom'].value = max(1e-8, lam_v)
        params['lam_min_w_nom'].value = max(1e-8, lam_w)
        params['A_t'].value = A_t
        params['C_t'].value = C_t

        if self._warm_start_vars_regular is not None:
            for var_name, var in self._warm_start_vars_regular.items():
                if var.value is not None:
                    var.value = var.value

        prob.solve(solver=cp.MOSEK, warm_start=True)

        if prob.status in ["infeasible", "unbounded"]:
            print(f'DR-EKF trace SDP problem: {prob.status}')
            return None, None, None, None, None

        vars_ = self._warm_start_vars_regular
        worst_case_Xpost = vars_['X'].value
        worst_case_Xprior = vars_['X_pred'].value
        worst_case_Sigma_v = vars_['Sigma_v'].value
        worst_case_Sigma_w = vars_['Sigma_w'].value
        worst_case_Sigma_wv = vars_['Sigma_wv'].value

        return worst_case_Sigma_v, worst_case_Sigma_w, worst_case_Sigma_wv, worst_case_Xprior, worst_case_Xpost

    # ------------------------------------------------------------------
    # DR-EKF measurement update
    # ------------------------------------------------------------------
    def DR_kalman_filter(self, v_mean_hat, x_prior, y, t, u_prev=None, x_post_prev=None):
        """DR-EKF with joint ambiguity and certified residual-aware trace-tube."""
        C_t = self.C_jacobian(x_prior)

        if t == 0:
            X_prior_nom = self.nominal_x0_cov.copy()
            result = self.solve_sdp_online_initial(X_prior_nom, C_t)
            wc_Sigma_v, wc_Sigma_xv, wc_Xprior, wc_Xpost = result
            wc_Sigma_cross = wc_Sigma_xv
        else:
            if x_post_prev is not None and u_prev is not None:
                A_t = self.F_jacobian(x_post_prev, u_prev)
                result = self.solve_sdp_online(self._P, A_t, C_t)
                wc_Sigma_v, wc_Sigma_w, wc_Sigma_wv, wc_Xprior, wc_Xpost = result
                wc_Sigma_cross = wc_Sigma_wv
            else:
                raise RuntimeError(f"DR-EKF trace requires previous state and control input for t > 0")

        if wc_Sigma_v is None:
            raise RuntimeError(f"DR-EKF trace SDP optimization failed at time step {t}.")

        T = wc_Xprior @ C_t.T + wc_Sigma_cross
        S = C_t @ wc_Xprior @ C_t.T + wc_Sigma_v + C_t @ wc_Sigma_cross + wc_Sigma_cross.T @ C_t.T
        K_star = np.linalg.solve(S, T.T).T

        # ----------------------------------------------------------
        # Recursive certificate state update (rho_t, Pbar_t)
        # ----------------------------------------------------------
        tr_Xpost = max(float(np.trace(wc_Xpost)), 0.0)

        if t == 0:
            eta_h0 = self._last_eta_h
            K_norm = np.linalg.norm(K_star, ord=2)
            rbar0 = 2.0 * K_norm**2 * eta_h0**2
            rho0 = np.sqrt(rbar0)
            if self.theta_eff_cap is not None:
                rho0 = min(rho0, self.theta_eff_cap)
            Pbar0 = (np.sqrt(tr_Xpost) + rho0)**2

            self._rho_cert = rho0
            self._Pbar = Pbar0

            # Diagnostics
            self._last_rbar = rbar0
            self._last_kappa = 0.0
            self._last_rho_cert = rho0
            self._last_Pbar = Pbar0
        else:
            eta_f = self._last_eta_f
            eta_h = self._last_eta_h
            rho_prev = self._rho_cert

            I_nx = np.eye(self.nx)
            IKC = I_nx - K_star @ C_t
            kappa_t = np.linalg.norm(IKC @ A_t, ord=2)
            rbar_t = (2.0 * np.linalg.norm(IKC, ord=2)**2 * eta_f**2
                      + 2.0 * np.linalg.norm(K_star, ord=2)**2 * eta_h**2)
            rho_t = kappa_t * rho_prev + np.sqrt(rbar_t)
            if self.theta_eff_cap is not None:
                rho_t = min(rho_t, self.theta_eff_cap)
            Pbar_t = (np.sqrt(tr_Xpost) + rho_t)**2

            self._rho_cert = rho_t
            self._Pbar = Pbar_t

            # Diagnostics
            self._last_rbar = rbar_t
            self._last_kappa = kappa_t
            self._last_rho_cert = rho_t
            self._last_Pbar = Pbar_t

        innovation = y - (self.h(x_prior) + v_mean_hat)
        x_post = x_prior + K_star @ innovation

        self._P = wc_Xpost
        return x_post

    def _initial_update(self, x_est_init, y0):
        return self.DR_kalman_filter(self.nominal_mu_v, x_est_init, y0, 0, None, None)

    def _drkf_finite_update(self, x_prior, y, t, u_prev=None, x_post_prev=None):
        return self.DR_kalman_filter(self.nominal_mu_v, x_prior, y, t, u_prev, x_post_prev)

    def forward(self):
        return self._run_simulation_loop(self._drkf_finite_update)

    def forward_track(self, desired_trajectory):
        return self._run_simulation_loop(self._drkf_finite_update, desired_trajectory)

    def forward_track_MPC(self, desired_trajectory):
        return self._run_simulation_loop_MPC(self._drkf_finite_update, desired_trajectory)

    def update_step(self, x_est_prev, y_curr, t, u_prev):
        """Common interface for filter update step."""
        x_pred = self.f(x_est_prev, u_prev) + self.nominal_mu_w
        return self._drkf_finite_update(x_pred, y_curr, t, u_prev, x_est_prev)
