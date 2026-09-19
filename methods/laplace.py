
import numpy as np
from scipy.special import expit as sigmoid
from scipy.linalg import cholesky, solve_triangular
from kernels.kernels import kernel_rbf

class LaplaceGPC:
    """Laplace approximation with logistic likelihood. R&W (2006) Algorithm 3.1."""
    def __init__(self, kernel_fn=kernel_rbf, log_params=None, n_iter=50, tol=1e-6):
        self.kernel_fn = kernel_fn
        self.log_params = log_params if log_params is not None else [0.0, 0.0]
        self.n_iter = n_iter
        self.tol = tol

    def _compute_kernel(self, X1, X2):
        return self.kernel_fn(X1, X2, **self.log_params)

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y.astype(np.float64)
        n = len(y)
        K = self._compute_kernel(X, X) + 1e-6 * np.eye(n)
        self.K_ = K
        f = np.zeros(n)

        for _ in range(self.n_iter):
            pi = sigmoid(f)
            W = pi * (1 - pi)
            W_sq = np.sqrt(W + 1e-10)
            B = np.eye(n) + W_sq[:, None] * K * W_sq[None, :]
            L = cholesky(B + 1e-8 * np.eye(n), lower=True)
            b = W * f + (y - pi)
            tmp = solve_triangular(L, W_sq * (K @ b), lower=True)
            a = b - W_sq * solve_triangular(L.T, tmp, lower=False)
            f_new = K @ a
            if np.max(np.abs(f_new - f)) < self.tol:
                f = f_new
                break
            f = f_new

        self.f_hat_ = f
        self.pi_hat_ = sigmoid(f)
        self.W_hat_ = self.pi_hat_ * (1 - self.pi_hat_)
        self.W_sq_hat_  = np.sqrt(self.W_hat_ + 1e-10)
        B_hat = np.eye(n) + self.W_sq_hat_[:, None] * K * self.W_sq_hat_[None, :]
        self.L_hat_ = cholesky(B_hat + 1e-8 * np.eye(n), lower=True)
        return self

    def predict_proba(self, X_test):
        K_s = self._compute_kernel(self.X_train_, X_test)
        k_ss = np.diag(self._compute_kernel(X_test, X_test))
        alpha = self.y_train_ - self.pi_hat_

        #alpha = self.W_hat_ * self.f_hat_ + (self.y_train_ - self.pi_hat_)
        mu_f = K_s.T @ alpha
        v = solve_triangular(self.L_hat_, self.W_sq_hat_[:, None] * K_s, lower=True)
        var_f = np.maximum(k_ss - np.sum(v ** 2, axis=0), 1e-8)
        # MacKay approximation for logistic
        kappa = 1.0 / np.sqrt(1 + np.pi * var_f / 8)
        p_pos = sigmoid(kappa * mu_f)
        return np.column_stack([1 - p_pos, p_pos])

