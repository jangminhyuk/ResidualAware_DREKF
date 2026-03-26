#!/usr/bin/env python3
"""
Common utility functions used across the project.
"""

import numpy as np
import pickle
from scipy.linalg import solve_discrete_are, sqrtm

def save_data(path, data):
    """Save data to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def is_stabilizable(A, B, tol=1e-9):
    """Check if the pair (A, B) is stabilizable."""
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M_mat = np.hstack([eig * np.eye(n) - A, B])
            if np.linalg.matrix_rank(M_mat, tol) < n:
                return False
    return True

def is_detectable(A, C, tol=1e-9):
    """Check if the pair (A, C) is detectable."""
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M_mat = np.vstack([eig * np.eye(n) - A, C])
            if np.linalg.matrix_rank(M_mat, tol) < n:
                return False
    return True

def is_positive_definite(M, tol=1e-9):
    """Check if matrix M is positive definite."""
    if not np.allclose(M, M.T, atol=tol):
        return False
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False

def enforce_positive_definiteness(Sigma, epsilon=1e-4):
    """Enforce positive definiteness of a matrix."""
    Sigma = (Sigma + Sigma.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma)
    min_eig = np.min(eigvals)
    if min_eig < epsilon:
        Sigma += (epsilon - min_eig) * np.eye(Sigma.shape[0])
    return Sigma

def generate_desired_trajectory(T_total, Amp=5.0, slope=1.0, omega=0.5, dt=0.2):
    """Generate a sinusoidal desired trajectory."""
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)
    
    x_d = Amp * np.sin(omega * time)
    vx_d = Amp * omega * np.cos(omega * time)
    y_d = slope * time
    vy_d = slope * np.ones(time_steps)
    
    return np.vstack((x_d, vx_d, y_d, vy_d))

def compute_lqr_cost(result, Q_lqr, R_lqr, K_lqr, desired_traj):
    """Compute LQR cost for trajectory tracking."""
    x = result['state_traj']
    x_est = result['est_state_traj']
    T_steps = x.shape[0]
    cost = 0.0
    for t in range(T_steps):
        error = x[t] - desired_traj[:, t].reshape(-1, 1)
        u = -K_lqr @ (x_est[t] - desired_traj[:, t].reshape(-1, 1))
        cost += (error.T @ Q_lqr @ error)[0, 0] + (u.T @ R_lqr @ u)[0, 0]
    return cost

def compute_mpc_cost(result, Q_lqr, R_lqr, desired_traj):
    """Compute trajectory tracking cost using actual MPC control inputs."""
    x = result['state_traj']          # Actual state trajectory
    u_actual = result['input_traj']   # Actual MPC control inputs applied
    T_steps = x.shape[0] - 1          # Number of control steps (T_steps - 1)
    cost = 0.0

    # Cost over control horizon
    for t in range(T_steps):
        # State tracking error cost
        error = x[t] - desired_traj[:, t].reshape(-1, 1)
        state_cost = (error.T @ Q_lqr @ error)[0, 0]

        # Control effort cost using actual MPC input
        control_cost = (u_actual[t].T @ R_lqr @ u_actual[t])[0, 0]

        cost += state_cost + control_cost

    # Add final state cost (no control at final time)
    final_error = x[T_steps] - desired_traj[:, T_steps].reshape(-1, 1)
    cost += (final_error.T @ Q_lqr @ final_error)[0, 0]

    return cost

# ============================================================================
# EM Parameter Estimation Functions (Unified for all dynamics)
# ============================================================================

def wrap_angle(a):
    """Wrap angle to (-pi, pi]. Works with scalars or 1x1 arrays."""
    return np.arctan2(np.sin(a), np.cos(a))

def _ekf_filter_single(
    y_seq, u_seq, dt,
    x0_mean, x0_cov,
    mu_w, Q,
    mu_v, R,
    f, F_jac, h, H_jac,
    wrap_innovation_fn=None
):
    """
    EKF forward pass for one trajectory.

    Args:
        y_seq: (T+1, ny, 1) - measurement sequence
        u_seq: (T, nu, 1) - control input sequence
        dt: scalar - time step
        x0_mean, x0_cov: initial state distribution
        mu_w, Q: process noise mean and covariance
        mu_v, R: measurement noise mean and covariance
        f, F_jac: dynamics function and Jacobian
        h, H_jac: observation function and Jacobian
        wrap_innovation_fn: Optional function to wrap innovations (e.g., for bearing angles)

    Returns:
        m_pred, P_pred: predicted at each t (t=0 uses prior)
        m_filt, P_filt: filtered at each t
        F_list: list of F_t (length T) where F_t is Jacobian at (m_filt[t], u_t)
    """
    T = u_seq.shape[0]
    nx = x0_mean.shape[0]
    I = np.eye(nx)

    m_pred = np.zeros((T+1, nx, 1))
    P_pred = np.zeros((T+1, nx, nx))
    m_filt = np.zeros((T+1, nx, 1))
    P_filt = np.zeros((T+1, nx, nx))
    F_list = []

    # prior at t=0
    m_pred[0] = x0_mean.copy()
    P_pred[0] = x0_cov.copy()

    # update with y0
    H0 = H_jac(m_pred[0])
    yhat0 = h(m_pred[0]) + mu_v
    innov0 = y_seq[0] - yhat0
    if wrap_innovation_fn is not None:
        innov0 = wrap_innovation_fn(innov0)
    S0 = H0 @ P_pred[0] @ H0.T + R
    S0 = 0.5 * (S0 + S0.T)
    K0 = P_pred[0] @ H0.T @ np.linalg.solve(S0, np.eye(S0.shape[0]))
    m_filt[0] = m_pred[0] + K0 @ innov0
    P_filt[0] = (I - K0 @ H0) @ P_pred[0] @ (I - K0 @ H0).T + K0 @ R @ K0.T
    P_filt[0] = 0.5 * (P_filt[0] + P_filt[0].T)

    for t in range(T):
        # predict to t+1
        Ft = F_jac(m_filt[t], u_seq[t], dt)
        F_list.append(Ft)

        m_pred[t+1] = f(m_filt[t], u_seq[t], dt) + mu_w
        P_pred[t+1] = Ft @ P_filt[t] @ Ft.T + Q
        P_pred[t+1] = 0.5 * (P_pred[t+1] + P_pred[t+1].T)

        # update with y_{t+1}
        Ht = H_jac(m_pred[t+1])
        yhat = h(m_pred[t+1]) + mu_v
        innov = y_seq[t+1] - yhat
        if wrap_innovation_fn is not None:
            innov = wrap_innovation_fn(innov)
        S = Ht @ P_pred[t+1] @ Ht.T + R
        S = 0.5 * (S + S.T)
        K = P_pred[t+1] @ Ht.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        m_filt[t+1] = m_pred[t+1] + K @ innov
        P_filt[t+1] = (I - K @ Ht) @ P_pred[t+1] @ (I - K @ Ht).T + K @ R @ K.T
        P_filt[t+1] = 0.5 * (P_filt[t+1] + P_filt[t+1].T)

    return m_pred, P_pred, m_filt, P_filt, F_list


def _rts_smoother_single(m_pred, P_pred, m_filt, P_filt, F_list):
    """
    Extended RTS smoother (uses stored Jacobians F_list).

    Args:
        m_pred, P_pred: predicted means and covariances from forward pass
        m_filt, P_filt: filtered means and covariances from forward pass
        F_list: list of Jacobians from forward pass

    Returns:
        m_smooth, P_smooth: smoothed means and covariances
    """
    T = len(F_list)
    nx = m_filt.shape[1]
    m_smooth = m_filt.copy()
    P_smooth = P_filt.copy()

    for t in range(T-1, -1, -1):
        Ft = F_list[t]
        # smoother gain
        Ppred_next = P_pred[t+1]
        Ppred_next = 0.5 * (Ppred_next + Ppred_next.T)
        Gt = P_filt[t] @ Ft.T @ np.linalg.solve(Ppred_next, np.eye(nx))

        m_smooth[t] = m_filt[t] + Gt @ (m_smooth[t+1] - m_pred[t+1])
        P_smooth[t] = P_filt[t] + Gt @ (P_smooth[t+1] - Ppred_next) @ Gt.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    return m_smooth, P_smooth


def estimate_nominal_parameters_EM(
    u_data, y_data, dt,
    x0_mean_init, x0_cov_init,
    mu_w_init=None, Sigma_w_init=None,
    mu_v_init=None, Sigma_v_init=None,
    f=None, F_jac=None, h=None, H_jac=None,
    max_iters=50, tol=1e-4,
    estimate_means=False, estimate_x0=False,
    cov_structure="diag", reg=1e-6, verbose=True,
    wrap_innovation_fn=None,
    wrap_measurement_residual_fn=None,
    wrap_process_residual_fn=None,
    wrap_smoothed_state_fn=None,
    custom_filter_fn=None,
    custom_filter_kwargs=None
):
    """
    Unified EM-like nominal parameter estimation from input-output data.
    Works for both unicycle dynamics and CT dynamics (and other nonlinear systems).

    Args:
        u_data: Control inputs, either (N, T, nu, 1) or list of (T, nu, 1)
        y_data: Measurements, either (N, T+1, ny, 1) or list of (T+1, ny, 1)
        dt: Time step
        x0_mean_init, x0_cov_init: Initial state distribution guess
        mu_w_init, Sigma_w_init: Process noise distribution guess
        mu_v_init, Sigma_v_init: Measurement noise distribution guess
        f, F_jac: Dynamics function and Jacobian
        h, H_jac: Observation function and Jacobian
        max_iters: Maximum EM iterations
        tol: Convergence tolerance
        estimate_means: Whether to estimate noise means
        estimate_x0: Whether to estimate initial state distribution
        cov_structure: "full", "diag", or "scalar" covariance structure
        reg: Regularization for positive definiteness
        verbose: Print iteration info
        wrap_innovation_fn: Optional function to wrap innovations in filter (e.g., for bearing angles)
        wrap_measurement_residual_fn: Optional function to wrap measurement residuals in M-step
        wrap_process_residual_fn: Optional function to wrap process noise residuals in M-step
        wrap_smoothed_state_fn: Optional function to wrap smoothed states (e.g., for theta angles)
        custom_filter_fn: Optional custom EKF filter function (for special cases like angle gating)
        custom_filter_kwargs: Optional dict of extra kwargs to pass to custom_filter_fn

    Returns:
        Tuple: (nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w,
                nominal_mu_v, nominal_Sigma_v)
    """
    # Normalize shapes to (N, T, *, 1) format
    if isinstance(u_data, list):
        # Convert list format to array format (from unicycle version)
        u_data_list = u_data
        y_data_list = y_data
        N = len(u_data_list)
        T = u_data_list[0].shape[0]
        nu = u_data_list[0].shape[1]
        ny = y_data_list[0].shape[1]

        # Convert to 4D array format
        u_data = np.zeros((N, T, nu, 1))
        y_data = np.zeros((N, T+1, ny, 1))
        for i in range(N):
            u_data[i] = u_data_list[i][:, :, np.newaxis] if u_data_list[i].ndim == 2 else u_data_list[i]
            y_data[i] = y_data_list[i][:, :, np.newaxis] if y_data_list[i].ndim == 2 else y_data_list[i]
    elif u_data.ndim == 3:  # Single trajectory (T, nu, 1)
        u_data = u_data[None, ...]
        y_data = y_data[None, ...]

    N = u_data.shape[0]
    T = u_data.shape[1]
    nx = x0_mean_init.shape[0]
    ny = y_data.shape[2]

    x0_mean = x0_mean_init.copy()
    x0_cov = x0_cov_init.copy()

    mu_w = np.zeros((nx, 1)) if mu_w_init is None else mu_w_init.copy()
    Q = 0.1 * np.eye(nx) if Sigma_w_init is None else Sigma_w_init.copy()

    mu_v = np.zeros((ny, 1)) if mu_v_init is None else mu_v_init.copy()
    R = 0.1 * np.eye(ny) if Sigma_v_init is None else Sigma_v_init.copy()

    # enforce SPD at start
    x0_cov = enforce_positive_definiteness(x0_cov)
    Q = enforce_positive_definiteness(Q)
    R = enforce_positive_definiteness(R)

    def _apply_structure(S):
        if cov_structure == "full":
            return S
        if cov_structure == "diag":
            return np.diag(np.diag(S))
        if cov_structure == "scalar":
            s = float(np.trace(S) / S.shape[0])
            return s * np.eye(S.shape[0])
        raise ValueError(f"Unknown cov_structure={cov_structure}")

    for it in range(max_iters):
        # Accumulators for M-step
        w_res_list = []
        v_res_list = []
        x0_list = []
        x0_cov_list = []

        # E-step: smooth each rollout
        for k in range(N):
            y_seq = y_data[k]
            u_seq = u_data[k]

            # Use custom filter if provided, otherwise use default
            if custom_filter_fn is not None:
                # Custom filter function (e.g., for 3D radar with gating)
                extra_kwargs = custom_filter_kwargs if custom_filter_kwargs is not None else {}
                m_pred, P_pred, m_filt, P_filt, F_list = custom_filter_fn(
                    y_seq, u_seq, dt,
                    x0_mean, x0_cov,
                    mu_w, Q,
                    mu_v, R,
                    f, F_jac, h, H_jac,
                    **extra_kwargs
                )
            else:
                # Default filter function
                m_pred, P_pred, m_filt, P_filt, F_list = _ekf_filter_single(
                    y_seq, u_seq, dt,
                    x0_mean, x0_cov,
                    mu_w, Q,
                    mu_v, R,
                    f, F_jac, h, H_jac,
                    wrap_innovation_fn
                )
            m_smooth, P_smooth = _rts_smoother_single(m_pred, P_pred, m_filt, P_filt, F_list)

            # Wrap smoothed states if needed (e.g., for unicycle theta angles)
            if wrap_smoothed_state_fn is not None:
                for t in range(T+1):
                    m_smooth[t] = wrap_smoothed_state_fn(m_smooth[t])

            if estimate_x0:
                x0_list.append(m_smooth[0])
                x0_cov_list.append(P_smooth[0])

            # residuals
            for t in range(T):
                # w_t approximation
                w_hat = m_smooth[t+1] - f(m_smooth[t], u_seq[t], dt)
                if wrap_process_residual_fn is not None:
                    w_hat = wrap_process_residual_fn(w_hat)
                w_res_list.append(w_hat)

            for t in range(T+1):
                v_hat = y_seq[t] - h(m_smooth[t])
                if wrap_measurement_residual_fn is not None:
                    v_hat = wrap_measurement_residual_fn(v_hat)
                v_res_list.append(v_hat)

        # Stack residuals: shape (dim, count)
        W = np.hstack(w_res_list)  # (nx, N*T)
        V = np.hstack(v_res_list)  # (ny, N*(T+1))

        # M-step: means
        if estimate_means:
            mu_w_new = np.mean(W, axis=1, keepdims=True)
            mu_v_new = np.mean(V, axis=1, keepdims=True)
        else:
            mu_w_new = mu_w
            mu_v_new = mu_v

        # M-step: covariances (moment-matching)
        Wc = W - mu_w_new
        Vc = V - mu_v_new

        Q_new = (Wc @ Wc.T) / max(Wc.shape[1], 1)
        R_new = (Vc @ Vc.T) / max(Vc.shape[1], 1)

        # Regularize + structure + SPD
        Q_new = _apply_structure(Q_new) + reg * np.eye(nx)
        R_new = _apply_structure(R_new) + reg * np.eye(ny)

        Q_new = enforce_positive_definiteness(Q_new)
        R_new = enforce_positive_definiteness(R_new)

        # x0 updates
        if estimate_x0 and len(x0_list) > 0:
            X0 = np.hstack(x0_list)  # (nx, N)
            x0_mean_new = np.mean(X0, axis=1, keepdims=True)
            # include smoother covariance at t=0
            P0_bar = sum(x0_cov_list) / len(x0_cov_list)
            centered = X0 - x0_mean_new
            x0_cov_new = P0_bar + (centered @ centered.T) / max(centered.shape[1], 1)
            x0_cov_new = _apply_structure(x0_cov_new) + reg * np.eye(nx)
            x0_cov_new = enforce_positive_definiteness(x0_cov_new)
        else:
            x0_mean_new = x0_mean
            x0_cov_new = x0_cov

        # Convergence check (relative change)
        dQ = np.linalg.norm(Q_new - Q, ord="fro") / (np.linalg.norm(Q, ord="fro") + 1e-12)
        dR = np.linalg.norm(R_new - R, ord="fro") / (np.linalg.norm(R, ord="fro") + 1e-12)
        dx0 = np.linalg.norm(x0_mean_new - x0_mean) / (np.linalg.norm(x0_mean) + 1e-12)

        if verbose:
            print(f"[EM] iter={it:02d}  rel_change: dQ={dQ:.3e}, dR={dR:.3e}, dx0={dx0:.3e}")

        # Update params
        mu_w, Q = mu_w_new, Q_new
        mu_v, R = mu_v_new, R_new
        x0_mean, x0_cov = x0_mean_new, x0_cov_new

        if max(dQ, dR, dx0) < tol:
            if verbose:
                print(f"[EM] Converged at iter={it} (tol={tol}).")
            break

    return x0_mean, x0_cov, mu_w, Q, mu_v, R


def bures_wasserstein_distance(mu1, Sigma1, mu2, Sigma2):
    """Compute the 2-Wasserstein (Bures-Wasserstein) distance between two Gaussians.

    d_BW^2 = ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2 (Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2})

    Args:
        mu1, mu2: mean vectors (n,1) or (n,). Can be None (treated as zero).
        Sigma1, Sigma2: covariance matrices (n,n).

    Returns:
        d_BW: scalar Bures-Wasserstein distance.
    """
    n = Sigma1.shape[0]

    # Mean term
    if mu1 is not None and mu2 is not None:
        diff = np.asarray(mu1).flatten() - np.asarray(mu2).flatten()
        mean_sq = float(np.dot(diff, diff))
    else:
        mean_sq = 0.0

    # Covariance (Bures) term: Tr(S1 + S2 - 2 M) where M = (S1^{1/2} S2 S1^{1/2})^{1/2}
    S1_half = sqrtm(Sigma1)
    # Ensure real (sqrtm can return complex with tiny imaginary parts)
    S1_half = np.real(S1_half)
    inner = S1_half @ Sigma2 @ S1_half
    M = sqrtm(inner)
    M = np.real(M)

    bures_sq = float(np.trace(Sigma1) + np.trace(Sigma2) - 2.0 * np.trace(M))
    # Numerical safety
    bures_sq = max(bures_sq, 0.0)

    return np.sqrt(mean_sq + bures_sq)


def bures_covariance_distance(Sigma1, Sigma2):
    """Compute the Bures metric between two covariance matrices (no mean term).

    d_B^2 = Tr(Sigma1 + Sigma2 - 2 (Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2})

    Args:
        Sigma1, Sigma2: covariance matrices (n,n).

    Returns:
        d_B: scalar Bures distance.
    """
    return bures_wasserstein_distance(None, Sigma1, None, Sigma2)


def perturb_covariance_bures(Sigma, target_distance, rng=None):
    """Generate a randomly perturbed covariance at `target_distance` Bures distance from Sigma.

    Eigendecompose Sigma = U diag(lambda_i) U^T, then perturb in sqrt-eigenvalue
    space using a random direction on the unit sphere:
        sqrt(lambda_hat_i) = sqrt(lambda_i) + target_distance * v_i
    where v is a random unit vector in R^n drawn from the uniform distribution on S^{n-1}.

    If clamping is needed (to keep eigenvalues > 0), the perturbation is rescaled
    to hit the target distance as closely as possible.

    Args:
        Sigma: true covariance matrix (n,n), must be PSD.
        target_distance: desired Bures distance (scalar >= 0).
        rng: numpy random Generator instance (optional).

    Returns:
        Sigma_hat: perturbed covariance at approximately target_distance Bures distance.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = Sigma.shape[0]
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_eigs = np.sqrt(eigvals)

    # Random direction on unit sphere in R^n
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    # Perturb sqrt-eigenvalues
    sqrt_eigs_new = sqrt_eigs + target_distance * v

    # Clamp to keep PD, then rescale to preserve target distance
    sqrt_eigs_new = np.maximum(sqrt_eigs_new, 1e-12)
    diff = sqrt_eigs_new - sqrt_eigs
    actual_d = np.linalg.norm(diff)
    if actual_d > 1e-15:
        diff = diff * (target_distance / actual_d)
        sqrt_eigs_new = sqrt_eigs + diff
        sqrt_eigs_new = np.maximum(sqrt_eigs_new, 1e-12)

    eigs_new = sqrt_eigs_new ** 2
    Sigma_hat = eigvecs @ np.diag(eigs_new) @ eigvecs.T
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)
    return Sigma_hat


def perturb_covariance_bures_full(Sigma, target_distance, rng=None):
    """Generate a perturbed PD covariance with off-diagonal terms at target Bures distance.

    Unlike perturb_covariance_bures (eigenvalue-only, preserves eigenvectors),
    this also applies a random rotation, producing a non-diagonal result even
    when Sigma is diagonal.  Uses bisection on a convex interpolation parameter
    to hit the target Bures distance.

    Args:
        Sigma: true covariance matrix (n,n), must be PSD.
        target_distance: desired Bures distance (scalar >= 0).
        rng: numpy random Generator instance (optional).

    Returns:
        Sigma_hat: perturbed covariance at approximately target_distance Bures distance,
                   with off-diagonal structure.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = Sigma.shape[0]
    eigvals = np.linalg.eigvalsh(Sigma)
    eigvals = np.maximum(eigvals, 1e-15)

    # 1. Random orthogonal rotation (Haar-distributed via QR)
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # 2. Random eigenvalue perturbation in log-space
    log_pert = rng.standard_normal(n)
    log_pert = log_pert / np.linalg.norm(log_pert)

    # 3. Build a "far" target matrix with off-diagonal terms
    scale = 1.0
    for _ in range(15):
        new_eigs = eigvals * np.exp(scale * log_pert)
        Sigma_far = Q @ np.diag(new_eigs) @ Q.T
        Sigma_far = 0.5 * (Sigma_far + Sigma_far.T)
        d_far = bures_covariance_distance(Sigma, Sigma_far)
        if d_far >= target_distance:
            break
        scale *= 2.0

    # 4. Bisection: Sigma_hat(alpha) = (1-alpha)*Sigma + alpha*Sigma_far
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        Sigma_mid = (1 - mid) * Sigma + mid * Sigma_far
        d_mid = bures_covariance_distance(Sigma, Sigma_mid)
        if d_mid < target_distance:
            lo = mid
        else:
            hi = mid

    alpha_opt = (lo + hi) / 2
    Sigma_hat = (1 - alpha_opt) * Sigma + alpha_opt * Sigma_far
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)
    return Sigma_hat


def generate_nominal_at_bw_distance(mu_w, Sigma_w, mu_v, Sigma_v,
                                     target_bw_w, target_bw_v, seed=None,
                                     full_covariance=False):
    """Generate nominal noise parameters with separate Bures distances for w and v.

    Each covariance is randomly perturbed to achieve the requested Bures distance
    independently.  Means are kept identical (no mean mismatch).

    Args:
        mu_w, mu_v: true noise means.
        Sigma_w, Sigma_v: true noise covariances.
        target_bw_w: desired Bures distance for process noise covariance.
        target_bw_v: desired Bures distance for observation noise covariance.
        seed: optional int seed for reproducibility.
        full_covariance: if True, also rotate eigenvectors to produce non-diagonal
                         nominal covariances (more realistic mismatch).

    Returns:
        (nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v)
    """
    rng = np.random.default_rng(seed)
    perturb_fn = perturb_covariance_bures_full if full_covariance else perturb_covariance_bures
    nominal_Sigma_w = perturb_fn(Sigma_w, target_bw_w, rng)
    nominal_Sigma_v = perturb_fn(Sigma_v, target_bw_v, rng)
    return mu_w.copy(), nominal_Sigma_w, mu_v.copy(), nominal_Sigma_v