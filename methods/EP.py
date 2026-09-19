
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from kernels.kernels import kernel_rbf
from scipy.stats import norm

class EPGPC:
    """Expectation Propagation with probit likelihood. R&W (2006) Algorithm 3.5."""
    def __init__(self, kernel_fn=kernel_rbf, log_params=None,
                 max_sweeps=30, tol=1e-6, damping=0.5):
        self.kernel_fn = kernel_fn
        self.log_params = log_params if log_params is not None else [0.0, 0.0]
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.damping = damping

    def _compute_kernel(self, X1, X2):
        return self.kernel_fn(X1, X2, **self.log_params)

    def fit(self, X, y):
        self.X_train_ = X
        n = len(y)
        y_pm = 2 * y.astype(np.float64) - 1   # {0,1} -> {-1,+1}
        K = self._compute_kernel(X, X) + 1e-6 * np.eye(n)
        self.K_ = K

        tau_tilde = np.zeros(n)
        nu_tilde = np.zeros(n)
        mu_ep = np.zeros(n)
        Sigma = K.copy()

        for sweep in range(self.max_sweeps):
            tau_old = tau_tilde.copy()
            for i in range(n):
                # Cavity
                tau_cav = max(1.0 / Sigma[i, i] - tau_tilde[i], 1e-8)
                #mu_cav = mu_ep[i] / (Sigma[i, i] * tau_cav) - nu_tilde[i] / tau_cav
                denom = 1.0 - Sigma[i, i] * tau_tilde[i]
                mu_cav = (mu_ep[i] - Sigma[i, i] * nu_tilde[i]) / denom

                # Tilted moments (probit closed form)
                z = y_pm[i] * mu_cav / np.sqrt(1 + 1.0 / tau_cav)
                Phi_z = max(norm.cdf(z), 1e-10)
                phi_z = norm.pdf(z)
                ratio = phi_z / Phi_z   # Mills ratio

                mu_hat = mu_cav + y_pm[i] * ratio / (tau_cav * np.sqrt(1 + 1.0 / tau_cav))
                # FIXED: correct variance update from R&W (2006) Eq. 3.79
                sig2_hat = max(1.0 / tau_cav - ratio * (z + ratio) / (tau_cav * (1 + 1.0 / tau_cav)), 1e-8)

                tau_new_i = 1.0 / sig2_hat - tau_cav
                nu_new_i = mu_hat / sig2_hat - mu_cav * tau_cav

                tau_tilde[i] = (1 - self.damping) * tau_tilde[i] + self.damping * tau_new_i
                nu_tilde[i] = (1 - self.damping) * nu_tilde[i] + self.damping * nu_new_i

            # Recompute global posterior
            S_sq = np.sqrt(np.maximum(tau_tilde, 0))
            B = np.eye(n) + S_sq[:, None] * K * S_sq[None, :]
            L = cholesky(B + 1e-8 * np.eye(n), lower=True)
            V = solve_triangular(L, S_sq[:, None] * K, lower=True)
            Sigma = K - V.T @ V
            mu_ep = Sigma @ nu_tilde

            if np.max(np.abs(tau_tilde - tau_old)) < self.tol:
                break

        self.tau_tilde_ = tau_tilde
        self.nu_tilde_ = nu_tilde
        self.L_ep_ = L
        self.S_sq_ = S_sq
        return self

    def predict_proba(self, X_test):
        K_s = self._compute_kernel(self.X_train_, X_test)
        k_ss = np.diag(self._compute_kernel(X_test, X_test))
        mu_f = K_s.T @ self.nu_tilde_
        v = solve_triangular(self.L_ep_, self.S_sq_[:, None] * K_s, lower=True)
        var_f = np.maximum(k_ss - np.sum(v ** 2, axis=0), 1e-8)
        p_pos = norm.cdf(mu_f / np.sqrt(1 + var_f))
        return np.column_stack([1 - p_pos, p_pos])



import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, solve_triangular

class EPGPC_Sequential:
    """Sequential Expectation Propagation for probit (R&W Algorithm 3.5)."""

    def __init__(self, kernel_fn, log_params=None, max_sweeps=50, tol=1e-6, damping=0.5):
        self.kernel_fn = kernel_fn
        self.log_params = log_params if log_params is not None else [0.0, 0.0]
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.damping = damping

    def _compute_kernel(self, X1, X2):
        return self.kernel_fn(X1, X2, **self.log_params)

    def fit(self, X, y):
        self.X_train_ = X
        n = len(y)
        y_pm = 2 * y.astype(np.float64) - 1   # {0,1} -> {-1,+1}
        K = self._compute_kernel(X, X) + 1e-6 * np.eye(n)
        self.K_ = K

        # Initialise site parameters
        tau_tilde = np.zeros(n)
        nu_tilde = np.zeros(n)

        # Initial posterior = prior
        Sigma = K.copy()
        mu = np.zeros(n)

        for sweep in range(self.max_sweeps):
            tau_old = tau_tilde.copy()
            nu_old = nu_tilde.copy()

            for i in range(n):
                # 1) Cavity (GPML stable form)
                sigma_i = Sigma[i, i]
                denom = 1.0 - sigma_i * tau_tilde[i]
                if denom <= 1e-12:
                    denom = 1e-12
                mu_cav = (mu[i] - sigma_i * nu_tilde[i]) / denom
                tau_cav = 1.0 / sigma_i - tau_tilde[i]
                if tau_cav <= 1e-8:
                    tau_cav = 1e-8

                # 2) Tilted moments (probit)
                v = 1.0 + 1.0 / tau_cav
                z = y_pm[i] * mu_cav / np.sqrt(v)
                z_clip = np.clip(z, -10, 10)
                Phi_z = norm.cdf(z_clip)
                phi_z = norm.pdf(z_clip)
                Phi_z = np.clip(Phi_z, 1e-12, 1 - 1e-12)
                ratio = phi_z / Phi_z

                mu_hat = mu_cav + y_pm[i] * ratio / (tau_cav * np.sqrt(v))
                sig2_hat = 1.0 / tau_cav - ratio * (z + ratio) / (tau_cav * v)
                sig2_hat = max(sig2_hat, 1e-8)

                # 3) New site parameters
                tau_new = 1.0 / sig2_hat - tau_cav
                nu_new = mu_hat / sig2_hat - mu_cav * tau_cav

                # 4) Damped update
                delta_tau = self.damping * (tau_new - tau_tilde[i])
                delta_nu  = self.damping * (nu_new - nu_tilde[i])

                # 5) Rank‑1 update of posterior (before changing site)
                #    Compute effect of changing τ̃_i → τ̃_i + Δτ, ν̃_i → ν̃_i + Δν
                #    Standard formula (see e.g. Minka 2001, Section 4.1)
                #    Let S = Σ, m = μ.
                #    Δ = (Δν - Δτ·m_i) / (1 + Δτ·S_ii)
                #    Then S ← S - (Δτ / (1+Δτ·S_ii)) S[:,i] S[i,:]
                #         m ← m + Δ·S[:,i]
                S_ii = Sigma[i, i]
                denom_up = 1.0 + delta_tau * S_ii
                if denom_up <= 1e-12:
                    denom_up = 1e-12
                delta_m = (delta_nu - delta_tau * mu[i]) / denom_up
                # Update covariance (symmetric)
                Sigma -= (delta_tau / denom_up) * np.outer(Sigma[:, i], Sigma[i, :])
                # Update mean
                mu += delta_m * Sigma[:, i]

                # 6) Store new site parameters
                tau_tilde[i] += delta_tau
                nu_tilde[i]  += delta_nu

            # Compute global posterior once more for prediction storage
            # (optional, can store final Sigma, mu after loops)
            S_sq = np.sqrt(np.maximum(tau_tilde, 0))
            B = np.eye(n) + (S_sq[:, None] * K) * S_sq[None, :]
            L = cholesky(B + 1e-10 * np.eye(n), lower=True)
            V = solve_triangular(L, S_sq[:, None] * K, lower=True)
            Sigma = K - V.T @ V
            mu = Sigma @ nu_tilde

            # Check convergence (on site parameters)
            if np.max(np.abs(tau_tilde - tau_old)) < self.tol and np.max(np.abs(nu_tilde - nu_old)) < self.tol:
                break

        # Store final posterior for prediction
        self.nu_tilde_ = nu_tilde
        self.S_sq_ = np.sqrt(np.maximum(tau_tilde, 0))
        # Compute final Cholesky factor L for predictive variance
        B_final = np.eye(n) + (self.S_sq_[:, None] * K) * self.S_sq_[None, :]
        self.L_ep_ = cholesky(B_final + 1e-10 * np.eye(n), lower=True)
        self.K_ = K
        return self

    def predict_proba(self, X_test):
        K_s = self._compute_kernel(self.X_train_, X_test)
        k_ss = np.diag(self._compute_kernel(X_test, X_test))
        mu_f = K_s.T @ self.nu_tilde_
        v = solve_triangular(self.L_ep_, self.S_sq_[:, None] * K_s, lower=True)
        var_f = np.maximum(k_ss - np.sum(v ** 2, axis=0), 1e-8)
        p_pos = norm.cdf(mu_f / np.sqrt(1 + var_f))
        return np.column_stack([1 - p_pos, p_pos])