
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
        self.K_inv_ = np.linalg.solve(K, np.eye(n))

        tau_tilde = np.zeros(n)
        nu_tilde = np.zeros(n)
        mu_ep = np.zeros(n)
        Sigma = K.copy()

        for sweep in range(self.max_sweeps):
            tau_old = tau_tilde.copy()
            for i in range(n):
                # Cavity
                tau_cav = max(1.0 / Sigma[i, i] - tau_tilde[i], 1e-8)
                denom = 1.0 - Sigma[i, i] * tau_tilde[i]
                mu_cav = (mu_ep[i] - Sigma[i, i] * nu_tilde[i]) / denom

                # Tilted moments (probit closed form)
                z = y_pm[i] * mu_cav / np.sqrt(1 + 1.0 / tau_cav)
                Phi_z = max(norm.cdf(z), 1e-10)
                phi_z = norm.pdf(z)
                ratio = phi_z / Phi_z   # Mills ratio

                mu_hat = mu_cav + y_pm[i] * ratio / (tau_cav * np.sqrt(1 + 1.0 / tau_cav))
                sig2_hat = max(1.0 / tau_cav - ratio * (z + ratio) / (tau_cav ** 2 * (1 + 1.0 / tau_cav)), 1e-8)

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
        self.mu_ep_ = mu_ep
        return self

    def predict_proba(self, X_test):
        K_s = self._compute_kernel(self.X_train_, X_test)
        k_ss = np.diag(self._compute_kernel(X_test, X_test))
        alpha = self.K_inv_ @ self.mu_ep_
        mu_f = K_s.T @ alpha
        v = solve_triangular(self.L_ep_, self.S_sq_[:, None] * K_s, lower=True)
        var_f = np.maximum(k_ss - np.sum(v ** 2, axis=0), 1e-8)
        p_pos = norm.cdf(mu_f / np.sqrt(1 + var_f))
        return np.column_stack([1 - p_pos, p_pos])