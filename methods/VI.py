
import numpy as np
from scipy.special import expit as sigmoid
from kernels.kernels import kernel_rbf

class VIGPC:
    """Mean-field variational inference with Jaakkola-Jordan bound (logistic)."""
    def __init__(self, kernel_fn=kernel_rbf, log_params=None, n_iter=50, tol=1e-6):
        self.kernel_fn = kernel_fn
        self.log_params = log_params if log_params is not None else [0.0, 0.0]
        self.n_iter = n_iter
        self.tol = tol

    def _compute_kernel(self, X1, X2):
        return self.kernel_fn(X1, X2, **self.log_params)

    @staticmethod
    def _lambda(xi):
        xi = np.asarray(xi)
        out = np.where(np.abs(xi) < 1e-8, 0.125, (sigmoid(xi) - 0.5) / (2 * xi + 1e-10))
        return out

    def fit(self, X, y):
        self.X_train_ = X
        n = len(y)
        y_f = y.astype(np.float64)

        K = self._compute_kernel(X, X) + 1e-6 * np.eye(n)
        K_inv = np.linalg.solve(K, np.eye(n))
        self.K_ = K
        self.K_inv_ = K_inv

        xi = np.ones(n)
        m = np.zeros(n)
        S = K.copy()
        elbo_prev = -np.inf

        for it in range(self.n_iter):
            lam = self._lambda(xi)
            Lambda = np.diag(2 * lam)
            S = np.linalg.solve(K_inv + Lambda, np.eye(n))
            m = S @ (y_f - 0.5)

            xi_new = np.sqrt(m ** 2 + np.diag(S))
            xi_new = np.maximum(xi_new, 1e-8)

            lam_new = self._lambda(xi_new)
            lik_term = (np.sum(np.log(sigmoid(xi_new) + 1e-10))
                        + np.sum((m * (y_f - 0.5)) - lam_new * (m ** 2 + np.diag(S) - xi_new ** 2)))
            logdet_K = np.linalg.slogdet(K)[1]
            logdet_S = np.linalg.slogdet(S)[1]
            kl_term = 0.5 * (np.trace(K_inv @ S) + m @ K_inv @ m - n + logdet_K - logdet_S)
            elbo = lik_term - kl_term

            if np.abs(elbo - elbo_prev) < self.tol:
                xi = xi_new
                break
            elbo_prev = elbo
            xi = xi_new

        self.m_ = m
        self.S_ = S
        self.xi_ = xi
        return self

    def predict_proba(self, X_test):
        K_s = self._compute_kernel(self.X_train_, X_test)
        k_ss = np.diag(self._compute_kernel(X_test, X_test))

        # Correct predictive mean and variance (R&W 2006, Eq. 3.30)
        K_inv_m = self.K_inv_ @ self.m_
        mu_f = K_s.T @ K_inv_m

        # k_*^T K^{-1} k_* (diagonal of K_s.T @ K_inv @ K_s)
        K_inv_Ks = self.K_inv_ @ K_s
        k_diag = np.einsum('ij,ji->i', K_s.T, K_inv_Ks)

        # k_*^T K^{-1} S K^{-1} k_*
        S_Kinv_Ks = self.S_ @ K_inv_Ks
        s_diag = np.einsum('ij,ji->i', K_s.T, S_Kinv_Ks)

        var_f = np.maximum(k_ss - k_diag + s_diag, 1e-8)

        # MacKay approximation for logistic
        kappa = 1.0 / np.sqrt(1 + np.pi * var_f / 8)
        p_pos = sigmoid(kappa * mu_f)
        return np.column_stack([1 - p_pos, p_pos])