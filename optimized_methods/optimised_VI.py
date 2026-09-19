import numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


class VIGPC:
    """
    Variational Inference with logistic likelihood.
    Jaakkola & Jordan (2000).
    """

    def __init__(self, kernel_fn, log_params, n_iter=100, tol=1e-8):
        self.kernel_fn  = kernel_fn
        self.log_params = log_params if log_params is not None else {'log_l': 0.0, 'log_sf': 0.0}
        self.n_iter     = n_iter
        self.tol        = tol

    def _compute_kernel(self, X1, X2):
        return self.kernel_fn(X1, X2, **self.log_params)

    @staticmethod
    def _lam(xi):
        xi = np.asarray(xi, dtype=float)
        return np.where(np.abs(xi) < 1e-8, 0.125, (sigmoid(xi) - 0.5) / (2 * xi + 1e-15))


    # ELBO (for hyperparameter optimisation)
    def _elbo(self, X, y, log_l, log_sf):
        """Run VI coordinate ascent and return ELBO. Returns -inf on failure."""
        n = len(y)
        yf = y.astype(float)
        K = self.kernel_fn(X, X, log_l=log_l, log_sf=log_sf) + 1e-6 * np.eye(n)
        K_inv = np.linalg.solve(K, np.eye(n))
        xi = np.ones(n)
        elbo_prev = -np.inf

        try:
            for _ in range(self.n_iter):
                lam = self._lam(xi)
                S = np.linalg.solve(K_inv + 2 * np.diag(lam), np.eye(n))
                m = S @ (yf - 0.5)
                xi_new = np.sqrt(np.maximum(m**2 + np.diag(S), 1e-10))
                lam2 = self._lam(xi_new)
                lik = (np.sum(np.log(sigmoid(xi_new) + 1e-12))
                       + np.sum(m * (yf - 0.5))
                       - np.sum(lam2 * (m**2 + np.diag(S) - xi_new**2)))
                sign, ldK = np.linalg.slogdet(K)
                sign, ldS = np.linalg.slogdet(S)
                kl = 0.5 * (np.trace(K_inv @ S) + m @ K_inv @ m - n + ldK - ldS)
                elbo = lik - kl
                if abs(elbo - elbo_prev) < self.tol:
                    break
                elbo_prev = elbo
                xi = xi_new
        except Exception:
            return -np.inf
        return float(elbo_prev)

    # Hyperparameter optimisation
    def optimise(self, X, y):
        """Find best log_l and log_sf by maximising ELBO (single start)."""
        # Median heuristic for length scale (data‑driven start)
        dists = pdist(X, metric='euclidean')
        med = np.median(dists)
        log_l_start = np.log(max(med, 1e-6))
        log_sf_start = 0.0   # signal variance 1

        def neg_elbo(p):
            try:
                return -self._elbo(X, y, p[0], p[1])
            except:
                return 1e9

        res = minimize(
            neg_elbo,
            x0=[log_l_start, log_sf_start],
            method='L-BFGS-B',
            bounds=[(-4.0, 4.0), (-3.0, 1.0)],
            options={'maxiter': 50}
        )
        self.log_params = {'log_l': float(res.x[0]), 'log_sf': float(res.x[1])}
        print(f"  VI optimised: log_l={self.log_params['log_l']:.3f}, log_sf={self.log_params['log_sf']:.3f} (ELBO={-res.fun:.3f})")
        return self


    # Fit and predict
    def fit(self, X, y):
        n = len(y)
        yf = y.astype(float)
        K = self._compute_kernel(X, X) + 1e-6 * np.eye(n)
        K_inv = np.linalg.solve(K, np.eye(n))
        xi = np.ones(n)
        elbo_prev = -np.inf

        for _ in range(self.n_iter):
            lam = self._lam(xi)
            S = np.linalg.solve(K_inv + 2 * np.diag(lam), np.eye(n))
            m = S @ (yf - 0.5)
            xi = np.sqrt(np.maximum(m**2 + np.diag(S), 1e-10))
            lam2 = self._lam(xi)
            sign, ldK = np.linalg.slogdet(K)
            sign, ldS = np.linalg.slogdet(S)
            lik = (np.sum(np.log(sigmoid(xi) + 1e-12))
                   + np.sum(m * (yf - 0.5))
                   - np.sum(lam2 * (m**2 + np.diag(S) - xi**2)))
            kl = 0.5 * (np.trace(K_inv @ S) + m @ K_inv @ m - n + ldK - ldS)
            elbo = lik - kl
            if abs(elbo - elbo_prev) < self.tol:
                break
            elbo_prev = elbo

        self.X_ = X
        self.m_ = m
        self.S_ = S
        self.K_inv_ = K_inv
        return self

    def predict_proba(self, X_test):
        K_s = self._compute_kernel(self.X_, X_test)
        k_ss = np.diag(self._compute_kernel(X_test, X_test))
        mu_f = K_s.T @ (self.K_inv_ @ self.m_)
        Ki_k = self.K_inv_ @ K_s
        var_f = np.maximum(
            k_ss
            - np.einsum('ij,ij->j', K_s, Ki_k)
            + np.einsum('ij,ij->j', Ki_k, self.S_ @ Ki_k),
            1e-8
        )
        kappa = 1.0 / np.sqrt(1 + np.pi * var_f / 8)
        p_pos = sigmoid(kappa * mu_f)
        return np.column_stack([1 - p_pos, p_pos])