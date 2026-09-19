import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from kernels.kernels import kernel_rbf


# ── EP core ──

def run_ep(X, y, log_l, log_sf, max_sweeps=100, tol=1e-6, damping=0.7):
    """
    Run EP with probit likelihood.
    Returns dict with posterior + EP log marginal likelihood, or None if failed.
    R&W (2006) Algorithm 3.5, p.52.
    """

    n    = len(y)
    y_pm = 2.0 * y.astype(float) - 1.0          # {0,1} -> {-1,+1}
    K    = kernel_rbf(X, X, log_l, log_sf) + 1e-6 * np.eye(n)

    # site parameters — start at zero (flat approximation)
    tau  = np.zeros(n)   # site precisions
    nu   = np.zeros(n)   # site precision * mean
    mu   = np.zeros(n)   # EP posterior mean
    Sig  = K.copy()      # EP posterior covariance
    lZ   = np.zeros(n)   # log normalising constant per site (for LML)
    L    = np.eye(n)
    S_sq = np.zeros(n)

    try:
        for sweep in range(max_sweeps):
            tau_old = tau.copy()

            for i in range(n):

                # cavity: remove site i from global posterior
                # R&W eq. 3.56
                denom    = max(1.0 - Sig[i, i] * tau[i], 1e-10)
                tau_cav  = max(1.0 / Sig[i, i] - tau[i], 1e-10)
                sig2_cav = 1.0 / tau_cav
                mu_cav   = (mu[i] - Sig[i, i] * nu[i]) / denom

                # tilted moments — probit closed form
                # R&W eq. 3.58-3.59, verified against numerical integration
                z     = y_pm[i] * mu_cav / np.sqrt(1.0 + sig2_cav)
                Phi_z = max(norm.cdf(z), 1e-15)
                ratio = norm.pdf(z) / Phi_z          # Mills ratio phi/Phi

                lZ[i]    = np.log(Phi_z)             # store for LML

                mu_hat   = mu_cav + y_pm[i] * sig2_cav * ratio / np.sqrt(1.0 + sig2_cav)

                # CORRECT variance (R&W eq. 3.59)
                sig2_hat = max(
                    sig2_cav - sig2_cav**2 * ratio * (z + ratio) / (1.0 + sig2_cav),
                    1e-10
                )

                # new site parameters
                tau_new = max(1.0 / sig2_hat - tau_cav, 0.0)
                nu_new  = mu_hat / sig2_hat - mu_cav * tau_cav

                # damped update — prevents oscillation
                tau[i] = (1 - damping) * tau[i] + damping * tau_new
                nu[i]  = (1 - damping) * nu[i]  + damping * nu_new

            # recompute global posterior after full sweep
            S_sq = np.sqrt(np.maximum(tau, 0.0))
            B    = np.eye(n) + S_sq[:, None] * K * S_sq[None, :]
            L    = cholesky(B + 1e-8 * np.eye(n), lower=True)
            V    = solve_triangular(L, S_sq[:, None] * K, lower=True)
            Sig  = K - V.T @ V
            mu   = Sig @ nu

            if sweep > 0 and np.max(np.abs(tau - tau_old)) < tol:
                break

        # EP log marginal likelihood  R&W eq. 3.65
        b      = K @ nu
        v2     = solve_triangular(L, S_sq * b, lower=True)
        ep_lml = np.sum(lZ) - np.sum(np.log(np.diag(L))) - 0.5 * (nu @ b - v2 @ v2)

    except Exception:
        return None

    return {'nu': nu, 'L': L, 'S_sq': S_sq, 'K': K, 'ep_lml': ep_lml}


# ── class ──

class EPGPC:
    """
    EP-GPC with probit likelihood.
    Optionally optimises RBF hyperparameters by maximising EP-LML.

    Usage
    -----
    # fixed hyperparameters (e.g. from median heuristic)
    clf = EPGPC(log_l=np.log(l_median), optimise=False)

    # optimised hyperparameters
    clf = EPGPC(optimise=True, n_restarts=3)

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    """

    def __init__(self, log_l=0.0, log_sf=0.0,
                 max_sweeps=100, tol=1e-6, damping=0.7,
                 optimise=False, n_restarts=3):
        self.log_l      = log_l
        self.log_sf     = log_sf
        self.max_sweeps = max_sweeps
        self.tol        = tol
        self.damping    = damping
        self.optimise   = optimise
        self.n_restarts = n_restarts

    def _neg_lml(self, p, X, y):
        # objective for scipy.minimize — returns negative EP-LML
        out = run_ep(X, y, p[0], p[1],
                     self.max_sweeps, self.tol, self.damping)
        return 1e9 if out is None else -out['ep_lml']

    def _starting_points(self, X):
        # first start: median heuristic for length-scale
        med = float(np.median(pdist(X, 'euclidean')))
        starts = [(np.log(max(med, 1e-2)), 0.0)]
        # random restarts
        rng = np.random.RandomState(42)
        for _ in range(self.n_restarts):
            starts.append((rng.uniform(-2.3, 2.3),
                           rng.uniform(-1.2, 1.2)))
        return starts

    def fit(self, X, y):
        self.X_ = X
        #print("Optimizing")
        # optimise hyperparameters if requested
        if self.optimise:
            best_lml, best_p = -np.inf, (self.log_l, self.log_sf)
            for p0 in self._starting_points(X):
                try:
                    r = minimize(self._neg_lml, p0, args=(X, y),
                                 method='L-BFGS-B',
                                 bounds=[(-3, 3), (-2, 2)],
                                 options={'maxiter': 100})
                    if -r.fun > best_lml:
                        best_lml = -r.fun
                        best_p   = tuple(r.x)
                except Exception:
                    continue
            self.log_l, self.log_sf = best_p
            print(f"  EP optimised: log_l={self.log_l:.3f}, log_sf={self.log_sf:.3f}")
        # final EP run with chosen hyperparameters
        out = run_ep(X, y, self.log_l, self.log_sf,
                     self.max_sweeps, self.tol, self.damping)
        if out is None:
            raise RuntimeError("EP failed. Try damping=0.9.")

        # store what predict_proba needs
        self.nu_   = out['nu']
        self.L_    = out['L']
        self.S_sq_ = out['S_sq']
        return self

    def predict_proba(self, X_test):
        # predictive mean and variance of f*
        K_s  = kernel_rbf(self.X_, X_test, self.log_l, self.log_sf)
        k_ss = np.diag(kernel_rbf(X_test, X_test, self.log_l, self.log_sf))
        mu_f  = K_s.T @ self.nu_
        v     = solve_triangular(self.L_, self.S_sq_[:, None] * K_s, lower=True)
        var_f = np.maximum(k_ss - np.sum(v**2, axis=0), 1e-8)
        # exact probit predictive integral  R&W eq. 3.82
        p_pos = norm.cdf(mu_f / np.sqrt(1.0 + var_f))
        return np.column_stack([1.0 - p_pos, p_pos])