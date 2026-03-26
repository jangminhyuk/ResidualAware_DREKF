#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_filter.py contains common functionality shared across all filter implementations.
"""

import numpy as np
import time

def generate_shared_noise_sequences(T, nx, ny, dist, noise_dist, 
                                  true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                                  x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                                  x0_scale=None, w_scale=None, v_scale=None, seed=42):
    """Generate shared noise sequences for consistent experiments across filters."""
    np.random.seed(seed)
    
    # Create temporary sampler for generating sequences
    A_temp, C_temp = np.eye(nx), np.eye(ny, nx)
    B_temp = np.eye(nx, 1)
    temp_params = np.zeros((nx, 1)), np.eye(nx)
    temp_params_y = np.zeros((ny, 1)), np.eye(ny)
    
    sampler = BaseFilter(T, dist, noise_dist, (A_temp, C_temp), B_temp,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        *temp_params, *temp_params, *temp_params_y,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale)
    
    # Generate sequences
    sequences = {}
    
    # Initial state
    sequences['x0'] = sampler.sample_initial_state()
    
    # Process and measurement noise sequences
    w_seq = np.zeros((T+1, nx, 1))
    v_seq = np.zeros((T+1, ny, 1))
    
    for t in range(T+1):
        w_seq[t] = sampler.sample_process_noise()
        v_seq[t] = sampler.sample_measurement_noise()
    
    sequences['w'] = w_seq
    sequences['v'] = v_seq
    
    return sequences

class BaseFilter:
    """Base class containing common distribution sampling and simulation logic."""
    
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None, shared_noise_sequences=None,
                 input_lower_bound=None, input_upper_bound=None):
        
        self.T = T
        self.dist = dist
        self.noise_dist = noise_dist
        self.A, self.C = system_data
        self.B = B
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        
        self.true_x0_mean = true_x0_mean
        self.true_x0_cov = true_x0_cov
        self.true_mu_w = true_mu_w
        self.true_Sigma_w = true_Sigma_w
        self.true_mu_v = true_mu_v
        self.true_Sigma_v = true_Sigma_v
        
        self.nominal_x0_mean = np.asarray(nominal_x0_mean).copy()
        self.nominal_x0_cov = np.asarray(nominal_x0_cov).copy()
        self.nominal_mu_w = np.asarray(nominal_mu_w).copy()
        self.nominal_Sigma_w = np.asarray(nominal_Sigma_w).copy()
        self.nominal_mu_v = np.asarray(nominal_mu_v).copy()
        self.nominal_Sigma_v = np.asarray(nominal_Sigma_v).copy()
        
        if self.dist == "uniform":
            self.x0_max, self.x0_min = x0_max, x0_min
            self.w_max, self.w_min = w_max, w_min
        if self.noise_dist == "uniform":
            self.v_max, self.v_min = v_max, v_min
        if self.dist == "laplace":
            self.x0_scale, self.w_scale = x0_scale, w_scale
        if self.noise_dist == "laplace":
            self.v_scale = v_scale
            
        self.K_lqr = None
        self.shared_noise_sequences = shared_noise_sequences
        self._noise_index = 0
        
        self.input_lower_bound = input_lower_bound
        self.input_upper_bound = input_upper_bound
    
    def normal(self, mu, Sigma, N=1):
        return np.random.multivariate_normal(mu[:, 0], Sigma, size=N).T
    
    def uniform(self, a, b, N=1):
        n = a.shape[0]
        return a + (b - a) * np.random.rand(n, N)
    
    def laplace(self, mu, scale, N=1):
        return np.random.laplace(mu[:, 0], scale, size=(N, mu.shape[0])).T
    
    def saturate_input(self, u):
        """Apply input saturation if bounds are specified."""
        if self.input_lower_bound is not None and self.input_upper_bound is not None:
            return np.clip(u, self.input_lower_bound, self.input_upper_bound)
        return u
    
    def sample_initial_state(self):
        if self.shared_noise_sequences is not None:
            return self.shared_noise_sequences['x0']
        
        if self.dist == "normal":
            return self.normal(self.true_x0_mean, self.true_x0_cov)
        elif self.dist == "laplace":
            return self.laplace(self.true_x0_mean, self.x0_scale)
        else:
            raise ValueError("Unsupported distribution for initial state.")
    
    def sample_process_noise(self):
        if self.shared_noise_sequences is not None:
            w = self.shared_noise_sequences['w'][self._noise_index]
            return w
        
        if self.dist == "normal":
            return self.normal(self.true_mu_w, self.true_Sigma_w)
        elif self.dist == "laplace":
            return self.laplace(self.true_mu_w, self.w_scale)
        else:
            raise ValueError("Unsupported distribution for process noise.")
    
    def sample_measurement_noise(self):
        if self.shared_noise_sequences is not None:
            v = self.shared_noise_sequences['v'][self._noise_index]
            return v
        
        if self.noise_dist == "normal":
            return self.normal(self.true_mu_v, self.true_Sigma_v)
        elif self.noise_dist == "laplace":
            return self.laplace(self.true_mu_v, self.v_scale)
        else:
            raise ValueError("Unsupported distribution for measurement noise.")
    
    def _run_simulation_loop(self, forward_method, desired_trajectory=None):
        """Common simulation loop for both forward and forward_track methods."""
        start_time = time.time()
        T, nx, ny, A, C, B = self.T, self.nx, self.ny, self.A, self.C, self.B
        nu = B.shape[1]  # Number of control inputs
        
        # Allocate arrays
        x = np.zeros((T+1, nx, 1))
        y = np.zeros((T+1, ny, 1))
        x_est = np.zeros((T+1, nx, 1))
        u_traj = np.zeros((T, nu, 1))  # Input trajectory (T time steps, not T+1)
        mse = np.zeros(T+1)
        error = np.zeros((T+1, nx, 1)) if desired_trajectory is not None else None
        
        # Reset noise index for consistent sequences across experiments
        self._noise_index = 0
        
        # Initialization
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        
        # First measurement and update
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        x_est[0] = self._initial_update(x_est[0], y[0])
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # Main loop
        for t in range(T):
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            
            # Compute control
            if desired_trajectory is not None:
                desired = desired_trajectory[:, t].reshape(-1, 1)
                error[t] = x_est[t] - desired
                u = -self.K_lqr @ error[t]
            else:
                u = -self.K_lqr @ x_est[t]
            
            # Apply input saturation if bounds are specified
            u = self.saturate_input(u)
            
            # Store input trajectory
            u_traj[t] = u.copy()
            
            # State propagation
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w
            
            # Increment noise index after sampling process noise
            self._noise_index += 1
            
            # Measurement
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Filter update
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            x_est[t+1] = forward_method(x_pred, y[t+1], t+1)
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        result = {
            'comp_time': time.time() - start_time,
            'state_traj': x,
            'output_traj': y,
            'est_state_traj': x_est,
            'input_traj': u_traj,
            'mse': mse
        }
        if error is not None:
            result['tracking_error'] = error
        return result
    
    def _run_simulation_loop_MPC(self, forward_method, desired_trajectory=None, mpc_horizon=10):
        """MPC simulation loop with finite horizon predictive control."""
        import scipy.optimize as opt
        
        start_time = time.time()
        T, nx, ny, A, C, B = self.T, self.nx, self.ny, self.A, self.C, self.B
        nu = B.shape[1]  # Number of control inputs
        
        # MPC parameters
        Q_mpc = np.diag([10, 1, 10, 1])  # State cost matrix
        R_mpc = 0.1 * np.eye(nu)         # Input cost matrix
        
        # Allocate arrays
        x = np.zeros((T+1, nx, 1))
        y = np.zeros((T+1, ny, 1))
        x_est = np.zeros((T+1, nx, 1))
        u_traj = np.zeros((T, nu, 1))  # Input trajectory (T time steps, not T+1)
        mse = np.zeros(T+1)
        error = np.zeros((T+1, nx, 1)) if desired_trajectory is not None else None
        
        # Reset noise index for consistent sequences across experiments
        self._noise_index = 0
        
        # Initialization
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        
        # First measurement and update
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        x_est[0] = self._initial_update(x_est[0], y[0])
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # Main loop
        for t in range(T):
            # Compute MPC control
            if desired_trajectory is not None:
                u = self._compute_mpc_control(x_est[t], desired_trajectory, t, mpc_horizon, Q_mpc, R_mpc)
                # Track error for the current time step
                desired = desired_trajectory[:, t].reshape(-1, 1)
                error[t] = x_est[t] - desired
            else:
                # For regulation (no desired trajectory), set target to origin
                target_traj = np.zeros((nx, min(mpc_horizon, T-t)))
                u = self._compute_mpc_control(x_est[t], target_traj, 0, mpc_horizon, Q_mpc, R_mpc)
            
            # Apply input saturation if bounds are specified
            u = self.saturate_input(u)
            
            # Store input trajectory
            u_traj[t] = u.copy()
            
            # State propagation
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w
            
            # Increment noise index after sampling process noise
            self._noise_index += 1
            
            # Measurement
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Filter update
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            x_est[t+1] = forward_method(x_pred, y[t+1], t+1)
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        result = {
            'comp_time': time.time() - start_time,
            'state_traj': x,
            'output_traj': y,
            'est_state_traj': x_est,
            'input_traj': u_traj,
            'mse': mse
        }
        if error is not None:
            result['tracking_error'] = error
        return result
    
    def _compute_mpc_control(self, x_current, desired_trajectory, current_time, horizon, Q, R):
        """Compute MPC control input using finite horizon optimization."""
        import scipy.optimize as opt
        
        nx, nu = self.nx, self.B.shape[1]
        A, B = self.A, self.B
        
        # Adjust horizon if near end of trajectory
        remaining_steps = desired_trajectory.shape[1] - current_time
        effective_horizon = min(horizon, remaining_steps, self.T - current_time)
        
        if effective_horizon <= 0:
            return np.zeros((nu, 1))
        
        # For desired trajectory, handle the endpoint properly
        if desired_trajectory.shape[1] > current_time + effective_horizon:
            desired_segment = desired_trajectory[:, current_time:current_time + effective_horizon]
        else:
            # Extend with the last point if needed
            desired_segment = np.zeros((nx, effective_horizon))
            available_points = min(effective_horizon, desired_trajectory.shape[1] - current_time)
            if available_points > 0:
                desired_segment[:, :available_points] = desired_trajectory[:, current_time:current_time + available_points]
                # Fill remaining with the endpoint
                if available_points < effective_horizon:
                    endpoint = desired_trajectory[:, -1].reshape(-1, 1)
                    for i in range(available_points, effective_horizon):
                        desired_segment[:, i] = endpoint.flatten()
        
        def cost_function(u_vec):
            u_vec = u_vec.reshape(nu, effective_horizon)
            x = x_current.copy()
            cost = 0.0
            
            for k in range(effective_horizon):
                u_k = u_vec[:, k].reshape(-1, 1)
                # Tracking error cost
                ref_k = desired_segment[:, k].reshape(-1, 1)
                error_k = x - ref_k
                cost += (error_k.T @ Q @ error_k)[0, 0]
                cost += (u_k.T @ R @ u_k)[0, 0]
                
                # Predict next state (without noise for MPC)
                x = A @ x + B @ u_k + self.nominal_mu_w
            
            return cost
        
        # Initial guess: zero inputs
        u0 = np.zeros(nu * effective_horizon)
        
        # Optimization bounds (if input saturation exists)
        bounds = None
        if self.input_lower_bound is not None and self.input_upper_bound is not None:
            bounds = [(self.input_lower_bound, self.input_upper_bound)] * (nu * effective_horizon)
        
        # Solve optimization
        result = opt.minimize(cost_function, u0, method='SLSQP', bounds=bounds)
        
        if result.success:
            optimal_u = result.x.reshape(nu, effective_horizon)
            return optimal_u[:, 0].reshape(-1, 1)  # Return only first control input
        else:
            # Fallback to LQR control if optimization fails
            if hasattr(self, 'K_lqr') and self.K_lqr is not None:
                ref_current = desired_segment[:, 0].reshape(-1, 1)
                error = x_current - ref_current
                return -self.K_lqr @ error
            else:
                return np.zeros((nu, 1))
    
    def _initial_update(self, x_est_init, y0):
        """Override in subclasses for specific initial update logic."""
        raise NotImplementedError
    
    def forward(self):
        """Override in subclasses."""
        raise NotImplementedError
    
    def forward_track(self, desired_trajectory):
        """Override in subclasses."""
        raise NotImplementedError
    
    def forward_track_MPC(self, desired_trajectory):
        """Override in subclasses for MPC-based tracking."""
        raise NotImplementedError