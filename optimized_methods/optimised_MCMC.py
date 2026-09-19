import numpy as np
from scipy.special import expit as sigmoid
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

class MCMCGPC:
    """
    MCMC / Elliptical Slice Sampling with logistic likelihood.
    Murray, Adams & MacKay (2010).
    Hyperparameters are optimised using a Laplace‑LML surrogate.
    """

    def __init__(self, kernel_fn, log_params,
                 n_samples=200, n_burnin=100, thin=2):
        self.kernel_fn = kernel_fn
        self.log_params = log_params if log_params is not None else {'log_l': 0.0, 'log_sf': 0.0}
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.thin = thin

    def _compute_kernel(self, X1, X2):
        return self.kernel_fn(X1, X2, **self.log_params)


    # Laplace LML surrogate (used for hyperparameter optimisation)

    def _laplace_lml(self, X, y, log_l, log_sf):
        """Compute Laplace‑approximated log marginal likelihood.
        Returns negative LML (to minimise) or inf on failure."""
        n = len(y)
        yf = y.astype(float)
        K = self.kernel_fn(X, X, log_l=log_l, log_sf=log_sf) + 1e-6 * np.eye(n)
        f = np.zeros(n)

        # Newton to find MAP f̂ and a = K⁻¹ f̂
        for _ in range(50):   # 50 iterations (similar to LaplaceGPC)
            pi = sigmoid(f)
            W = pi * (1 - pi) + 1e-10
            sqrt_w = np.sqrt(W)
            B = np.eye(n) + sqrt_w[:, None] * K * sqrt_w[None, :]
            try:
                L = cholesky(B + 1e-8 * np.eye(n), lower=True)
            except:
                return np.inf
            b = W * f + (yf - pi)
            tmp = solve_triangular(L, sqrt_w * (K @ b), lower=True)
            a = b - sqrt_w * solve_triangular(L.T, tmp, lower=False)
            f_new = K @ a
            if np.max(np.abs(f_new - f)) < 1e-6:
                f = f_new
                break
            f = f_new

        pi_hat = sigmoid(f)
        log_lik = np.sum(yf * np.log(pi_hat + 1e-12) + (1 - yf) * np.log(1 - pi_hat + 1e-12))
        quad = 0.5 * f @ a
        sign, logdet_K = np.linalg.slogdet(K)
        sqrt_w_final = np.sqrt(pi_hat * (1 - pi_hat) + 1e-10)
        B_final = np.eye(n) + sqrt_w_final[:, None] * K * sqrt_w_final[None, :]
        try:
            L_final = cholesky(B_final + 1e-8 * np.eye(n), lower=True)
            logdet_B = 2 * np.sum(np.log(np.diag(L_final) + 1e-15))
        except:
            return np.inf
        lml = log_lik - quad - 0.5  * logdet_B
        return -lml   # negative for minimisation


    # Hyperparameter optimisation (surrogate)
    def optimise(self, X, y):
        """Optimise hyperparameters using Laplace‑LML surrogate."""
        # Median heuristic for length scale
        dists = pdist(X, metric='euclidean')
        med = np.median(dists)
        log_l_start = np.log(max(med, 1e-6))
        log_sf_start = 0.0   # signal variance 1

        def objective(p):
            return self._laplace_lml(X, y, p[0], p[1])

        res = minimize(
            objective,
            x0=[log_l_start, log_sf_start],
            method='L-BFGS-B',
            bounds=[(-4.0, 4.0), (-3.0, 1.0)],
            options={'maxiter': 50}
        )
        self.log_params = {'log_l': float(res.x[0]), 'log_sf': float(res.x[1])}
        print(f"  MCMC optimised: log_l={self.log_params['log_l']:.3f}, log_sf={self.log_params['log_sf']:.3f} (Laplace-LML={-res.fun:.3f})")
        return self


    # MCMC / Elliptical slice sampling

    def fit(self, X, y):
        yf = y.astype(float)
        n = len(y)
        K = self._compute_kernel(X, X) + 1e-6 * np.eye(n)
        self.K_ = K
        L_prior = cholesky(K, lower=True)

        def log_lik(f):
            p = np.clip(sigmoid(f), 1e-12, 1-1e-12)
            return np.sum(yf * np.log(p) + (1 - yf) * np.log(1 - p))

        f_curr = np.zeros(n)
        samples = []
        total = self.n_burnin + self.n_samples * self.thin

        for step in range(total):
            nu = L_prior @ np.random.randn(n)
            log_thresh = log_lik(f_curr) + np.log(np.random.rand() + 1e-15)
            theta = np.random.uniform(0, 2 * np.pi)
            theta_min, theta_max = theta - 2 * np.pi, theta

            for _ in range(500):
                f_prop = f_curr * np.cos(theta) + nu * np.sin(theta)
                if log_lik(f_prop) > log_thresh:
                    f_curr = f_prop
                    break
                if theta < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)

            if step >= self.n_burnin and (step - self.n_burnin) % self.thin == 0:
                samples.append(f_curr.copy())

        self.X_ = X
        self.samples_ = np.array(samples)
        return self


    # Predictive probability (averaging over posterior samples)
    def predict_proba(self, X_test):
        K_s = self._compute_kernel(self.X_, X_test)
        K_inv = np.linalg.solve(self.K_, np.eye(len(self.X_)))
        f_star = K_s.T @ (K_inv @ self.samples_.T)   # shape (n_test, n_samples)
        p_pos = sigmoid(f_star).mean(axis=1)
        return np.column_stack([1 - p_pos, p_pos])