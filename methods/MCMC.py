

import numpy as np
from scipy.linalg import cholesky
from kernels.kernels import kernel_rbf
from scipy.special import expit as sigmoid


class MCMCGPC:
    """Elliptical slice sampling (Murray et al., 2010) with logistic likelihood."""
    def __init__(self, kernel_fn=kernel_rbf, log_params=None,
                 n_samples=300, n_burnin=100, thin=2):
        self.kernel_fn = kernel_fn
        self.log_params = log_params if log_params is not None else [0.0, 0.0]
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.thin = thin

    def _compute_kernel(self, X1, X2):
        return self.kernel_fn(X1, X2, **self.log_params)

    def _log_lik(self, f, y):
        p = sigmoid(f)
        eps = 1e-10
        return np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y.astype(np.float64)
        n = len(y)
        K = self._compute_kernel(X, X) + 1e-6 * np.eye(n)
        self.K_ = K
        L_prior = cholesky(K, lower=True)

        f_curr = np.zeros(n)
        samples = []
        total = self.n_burnin + self.n_samples * self.thin

        for step in range(total):
            nu = L_prior @ np.random.randn(n)
            log_thresh = self._log_lik(f_curr, y) + np.log(np.random.rand() + 1e-10)
            theta = np.random.uniform(0, 2 * np.pi)
            theta_min, theta_max = theta - 2 * np.pi, theta

            for _ in range(200):
                f_prop = f_curr * np.cos(theta) + nu * np.sin(theta)
                if self._log_lik(f_prop, y) > log_thresh:
                    f_curr = f_prop
                    break
                if theta < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)

            if step >= self.n_burnin and (step - self.n_burnin) % self.thin == 0:
                samples.append(f_curr.copy())

        self.samples_ = np.array(samples)
        return self

    def predict_proba(self, X_test):
        K_s = self._compute_kernel(self.X_train_, X_test)
        K_inv = np.linalg.solve(self.K_, np.eye(len(self.X_train_)))
        alpha = K_inv @ self.samples_.T
        mu_f_star = K_s.T @ alpha
        p_pos = sigmoid(mu_f_star).mean(axis=1)
        return np.column_stack([1 - p_pos, p_pos])