#!/usr/bin/env python3
"""
Extended Kalman Filter (EKF) for nonlinear state estimation.
"""

import numpy as np
from .base_filter import BaseFilter

class EKF(BaseFilter):
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
                 input_lower_bound=None, input_upper_bound=None):
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, None,
                        input_lower_bound, input_upper_bound)
        
        # Store nonlinear dynamics and jacobians
        self.f = nonlinear_dynamics if nonlinear_dynamics else self._default_dynamics
        self.F_jacobian = dynamics_jacobian if dynamics_jacobian else self._default_dynamics_jacobian
        self.h = observation_function if observation_function else self._default_observation
        self.C_jacobian = observation_jacobian if observation_jacobian else self._default_observation_jacobian
        
        # For storing covariance matrices
        self._P = None

    def _default_dynamics(self, x, u):
        """Default linear dynamics for backward compatibility."""
        return self.A @ x + self.B @ u
    
    def _default_dynamics_jacobian(self, x, u):
        """Default linear dynamics jacobian."""
        return self.A
    
    def _default_observation(self, x):
        """Default linear observation."""
        return self.C @ x
    
    def _default_observation_jacobian(self, x):
        """Default linear observation jacobian."""
        return self.C

    def _initial_update(self, x_est_init, y0):
        """Initial state update using EKF."""
        P0 = self.nominal_x0_cov.copy()
        
        # Linearize observation around initial estimate
        C0 = self.C_jacobian(x_est_init)
        h0 = self.h(x_est_init)
        
        # Innovation covariance
        S0 = C0 @ P0 @ C0.T + self.nominal_Sigma_v
        
        # Kalman gain: K0 = P0 @ C0.T @ inv(S0) = solve(S0.T, (P0 @ C0.T).T).T
        K0 = np.linalg.solve(S0, (P0 @ C0.T).T).T
        
        # Innovation
        innovation0 = y0 - (h0 + self.nominal_mu_v)
        
        # Store posterior covariance
        self._P = (np.eye(self.nx) - K0 @ C0) @ P0
        
        return x_est_init + K0 @ innovation0
    
    def _ekf_update(self, x_pred, y, t, u_prev, x_prev=None):
        """Extended Kalman Filter update step."""
        # Use provided previous state or recover it
        if x_prev is None:
            # For nonlinear dynamics, we need the actual previous state estimate
            # This should be passed from the calling function
            x_prev = self.nominal_x0_mean if t == 1 else x_pred  # Fallback
        
        # Linearize dynamics around previous estimate
        F_t = self.F_jacobian(x_prev, u_prev)
        
        # Prediction covariance
        P_pred = F_t @ self._P @ F_t.T + self.nominal_Sigma_w
        
        # Linearize observation around predicted state
        C_t = self.C_jacobian(x_pred)
        h_pred = self.h(x_pred)
        
        # Innovation covariance
        S_t = C_t @ P_pred @ C_t.T + self.nominal_Sigma_v
        
        # Kalman gain: K_t = P_pred @ C_t.T @ inv(S_t) = solve(S_t.T, (P_pred @ C_t.T).T).T
        K_t = np.linalg.solve(S_t, (P_pred @ C_t.T).T).T
        
        # Innovation
        innovation = y - (h_pred + self.nominal_mu_v)
        
        # State update
        x_new = x_pred + K_t @ innovation
        
        # Covariance update
        self._P = (np.eye(self.nx) - K_t @ C_t) @ P_pred
        
        return x_new
    
    def update_step(self, x_est_prev, y_curr, t, u_prev):
        """Common interface for filter update step.
        
        Args:
            x_est_prev: Previous state estimate
            y_curr: Current measurement
            t: Time step
            u_prev: Previous control input
            
        Returns:
            x_est_new: Updated state estimate
        """
        # EKF mean prediction (add nominal process noise mean)
        x_pred = self.f(x_est_prev, u_prev) + self.nominal_mu_w
        
        # EKF measurement update
        return self._ekf_update(x_pred, y_curr, t, u_prev, x_est_prev)
    
    def _run_simulation_loop_EKF(self, desired_trajectory=None):
        """EKF-specific simulation loop that handles nonlinear dynamics."""
        # This method is kept for compatibility but not used in main0.py
        # main0.py uses run_single_simulation which handles PID control properly
        raise NotImplementedError("Use run_single_simulation in main0.py for EKF with PID control")
    
    def forward(self):
        """Run EKF simulation without trajectory tracking."""
        return self._run_simulation_loop_EKF()
    
    def forward_track(self, desired_trajectory):
        """Run EKF simulation with trajectory tracking."""
        return self._run_simulation_loop_EKF(desired_trajectory)
    
    def forward_track_MPC(self, desired_trajectory):
        """Run EKF simulation with MPC-based trajectory tracking."""
        # For now, use the regular tracking method
        # Can be extended to use MPC-specific control computation
        return self.forward_track(desired_trajectory)
