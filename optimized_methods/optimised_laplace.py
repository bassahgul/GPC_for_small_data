import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from scipy.linalg  import cholesky, solve_triangular


class LaplaceGPC:
    """
    Laplace approximation with logistic likelihood.
    R&W (2006) Algorithm 3.1.
    """

    def __init__(self, kernel_fn, log_params, n_iter=50, tol=1e-6):
        self.kernel_fn  = kernel_fn
        self.log_params = log_params if log_params is not None else {'log_l': 0.0, 'log_sf': 0.0}
        self.n_iter     = n_iter
        self.tol        = tol

    def _compute_kernel(self, X1, X2):
        return self.kernel_fn(X1, X2, **self.log_params)

    def _lml(self, X, y, log_l, log_sf):
        """Compute log marginal likelihood for given hyperparameters."""


        n = len(y)
        yf = y.astype(float)
        K = self.kernel_fn(X, X, log_l=log_l, log_sf=log_sf) + 1e-6*np.eye(n)
        f = np.zeros(n)

        for _ in range(self.n_iter):
            pi  = sigmoid(f); W = pi*(1-pi); W_sq = np.sqrt(W+1e-10)
            B   = np.eye(n) + W_sq[:,None]*K*W_sq[None,:]
            L   = cholesky(B+1e-8*np.eye(n), lower=True)
            b   = W*f + (yf-pi)
            tmp = solve_triangular(L, W_sq*(K@b), lower=True)
            a   = b - W_sq*solve_triangular(L.T, tmp, lower=False)
            f_new = K@a
            if np.max(np.abs(f_new-f)) < self.tol: f=f_new; break
            f = f_new

        pi_hat = sigmoid(f); W_hat = pi_hat*(1-pi_hat)
        W_sq_hat = np.sqrt(W_hat+1e-10)
        B_hat = np.eye(n) + W_sq_hat[:,None]*K*W_sq_hat[None,:]
        L_hat = cholesky(B_hat+1e-8*np.eye(n), lower=True)

        log_lik   = np.sum(yf*np.log(pi_hat+1e-12) + (1-yf)*np.log(1-pi_hat+1e-12))
        log_prior = -0.5 * f @ a
        log_det   = -np.sum(np.log(np.diag(L_hat)+1e-15))

        return log_lik + log_prior + log_det


    def optimise(self, X, y):
        """Find best log_l and log_sf by maximising LML."""


        def neg_lml(p):
            try:    return -self._lml(X, y, p[0], p[1])
            except: return 1e9

        res = minimize(
            neg_lml,
            x0     = [0.0, 0.0],              # start at l=1, sf=1
            method = 'L-BFGS-B',
            bounds = [(-4.0, 4.0), (-3.0, 1.0)]
        )
        self.log_params = {'log_l': float(res.x[0]), 'log_sf': float(res.x[1])}
        print(f"  Laplace  log_l={res.x[0]:.3f}  log_sf={res.x[1]:.3f}")
        return self

    def fit(self, X, y):

        self.X_train_ = X
        self.y_train_ = y.astype(np.float64)
        n = len(y)
        K = self._compute_kernel(X, X) + 1e-6 * np.eye(n)
        self.K_ = K
        f = np.zeros(n)

        for _ in range(self.n_iter):
            pi  = sigmoid(f)
            W = pi*(1-pi)
            W_sq = np.sqrt(W+1e-10)
            B   = np.eye(n) + W_sq[:,None]*K*W_sq[None,:]
            L   = cholesky(B+1e-8*np.eye(n), lower=True)
            b = W * f + (y - pi)
            tmp = solve_triangular(L, W_sq*(K@b), lower=True)
            a   = b - W_sq*solve_triangular(L.T, tmp, lower=False)
            f_new = K@a
            if np.max(np.abs(f_new-f)) < self.tol:
                f=f_new
                break
            f = f_new

        self.f_hat_ = f
        self.pi_hat_ = sigmoid(f)
        self.W_hat_ = self.pi_hat_ * (1 - self.pi_hat_)
        self.W_sq_hat_ = np.sqrt(self.W_hat_ + 1e-10)
        B_hat = np.eye(n) + self.W_sq_hat_[:, None] * K * self.W_sq_hat_[None, :]
        self.L_hat_ = cholesky(B_hat + 1e-8 * np.eye(n), lower=True)
        return self

    def predict_proba(self, X_test):
        K_s = self._compute_kernel(self.X_train_, X_test)
        k_ss = np.diag(self._compute_kernel(X_test, X_test))
        # Correct alpha = y - pi_hat (gradient)
        alpha = self.y_train_ - self.pi_hat_
        mu_f = K_s.T @ alpha
        v = solve_triangular(self.L_hat_, self.W_sq_hat_[:, None] * K_s, lower=True)
        var_f = np.maximum(k_ss - np.sum(v ** 2, axis=0), 1e-8)
        kappa = 1.0 / np.sqrt(1 + np.pi * var_f / 8)
        p_pos = sigmoid(kappa * mu_f)
        return np.column_stack([1 - p_pos, p_pos])



