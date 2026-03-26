#!/usr/bin/env python3
"""
ct_radar_models.py — Coordinated-turn dynamics and radar observation models.

Shared between exp_ct_tracking.py and exp_ct_tracking_nonlinear.py.
"""

import numpy as np
from common_utils import wrap_angle
from estimator.base_filter import BaseFilter

# Module-level sampler for distribution sampling helpers
_temp_A, _temp_C = np.eye(5), np.eye(2, 5)
_temp_params = np.zeros((5, 1)), np.eye(5)
_temp_params_v = np.zeros((2, 1)), np.eye(2)
_sampler = BaseFilter(1, 'normal', 'normal', (_temp_A, _temp_C), np.eye(5, 2),
                      *_temp_params, *_temp_params, *_temp_params_v,
                      *_temp_params, *_temp_params, *_temp_params_v)


def ct_dynamics(x, u, k=None, dt=0.2, omega_eps=0.2):
    """CT dynamics: x = [px, py, vx, vy, omega]^T"""
    px, py, vx, vy, omega = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0]
    phi = omega * dt
    if abs(omega) >= omega_eps:
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        A = sin_phi / omega
        B = (1 - cos_phi) / omega
        C = (cos_phi - 1) / omega
        D = sin_phi / omega
        px_next = px + A * vx + B * vy
        py_next = py + C * vx + D * vy
        vx_next = cos_phi * vx - sin_phi * vy
        vy_next = sin_phi * vx + cos_phi * vy
        omega_next = omega
    else:
        px_next = px + vx * dt
        py_next = py + vy * dt
        vx_next = vx
        vy_next = vy
        omega_next = omega
    return np.array([[px_next], [py_next], [vx_next], [vy_next], [omega_next]])


def ct_jacobian(x, u, k=None, dt=0.2, omega_eps=0.2):
    """Jacobian of CT dynamics w.r.t. state (5x5)"""
    vx, vy, omega = x[2, 0], x[3, 0], x[4, 0]
    phi = omega * dt
    if abs(omega) >= omega_eps:
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        A = sin_phi / omega
        B = (1 - cos_phi) / omega
        C = (cos_phi - 1) / omega
        D = sin_phi / omega
        dA_domega = (dt * cos_phi) / omega - sin_phi / (omega**2)
        dB_domega = (dt * sin_phi) / omega - (1 - cos_phi) / (omega**2)
        dC_domega = (-dt * sin_phi) / omega - (cos_phi - 1) / (omega**2)
        dD_domega = (dt * cos_phi) / omega - sin_phi / (omega**2)
        dcos_phi_domega = -dt * sin_phi
        dsin_phi_domega = dt * cos_phi
        F = np.array([
            [1, 0, A, B, vx * dA_domega + vy * dB_domega],
            [0, 1, C, D, vx * dC_domega + vy * dD_domega],
            [0, 0, cos_phi, -sin_phi, vx * dcos_phi_domega - vy * dsin_phi_domega],
            [0, 0, sin_phi, cos_phi, vx * dsin_phi_domega + vy * dcos_phi_domega],
            [0, 0, 0, 0, 1]
        ])
    else:
        F = np.array([
            [1, 0, dt, 0, 0],
            [0, 1, 0, dt, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
    return F


def radar_observation_function(x, sensor_pos=(0, 0)):
    """Radar: y = [range, bearing]^T"""
    px, py = x[0, 0], x[1, 0]
    sx, sy = sensor_pos
    dx, dy = px - sx, py - sy
    range_val = np.sqrt(dx**2 + dy**2)
    bearing = np.arctan2(dy, dx)
    return np.array([[range_val], [bearing]])


def radar_observation_jacobian(x, sensor_pos=(0, 0), range_eps=1e-6):
    """Radar observation Jacobian H (2x5)"""
    px, py = x[0, 0], x[1, 0]
    sx, sy = sensor_pos
    dx, dy = px - sx, py - sy
    r = max(np.sqrt(dx**2 + dy**2), range_eps)
    dr_dpx, dr_dpy = dx / r, dy / r
    db_dpx, db_dpy = -dy / (r**2), dx / (r**2)
    return np.array([
        [dr_dpx, dr_dpy, 0, 0, 0],
        [db_dpx, db_dpy, 0, 0, 0]
    ])


def wrap_bearing_measurement(y_measured, y_predicted):
    """Wrap bearing so innovation stays in (-pi, pi]."""
    y_wrapped = y_measured.copy()
    if y_wrapped.shape[0] >= 2:
        bearing_diff = y_measured[1, 0] - y_predicted[1, 0]
        wrapped_diff = wrap_angle(bearing_diff)
        y_wrapped[1, 0] = y_predicted[1, 0] + wrapped_diff
        if abs(wrapped_diff) > np.pi / 2:
            alt = wrapped_diff - np.sign(wrapped_diff) * 2 * np.pi
            if abs(alt) < abs(wrapped_diff):
                y_wrapped[1, 0] = y_predicted[1, 0] + alt
    return y_wrapped


def sample_from_distribution(mu, Sigma, dist_type,
                              max_val=None, min_val=None, scale=None, N=1):
    """Sample from the specified distribution using a module-level BaseFilter sampler."""
    if dist_type == "normal":
        return _sampler.normal(mu, Sigma, N)
    elif dist_type == "laplace":
        return _sampler.laplace(mu, scale, N)
    else:
        raise ValueError(f"Unsupported distribution: {dist_type}")


def _as_col(x):
    """Ensure x is a column vector (n, 1)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x
