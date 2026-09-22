"""
Experiment: Learning from small data.
Training sizes: n = 20, 50, 100
Kernel: RBF with length-scale = median pairwise distance (per training set)
        signal variance fixed at sf = 1.0 (log_sf = 0.0)
Methods: Laplace, EP, VI, MCMC
Metrics: Predictive log-likelihood (NLL), Calibration Error (ECE)
Reference: Kuss & Rasmussen (2005); Nickisch & Rasmussen (2008)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from kernels.kernels import kernel_rbf
from methods.laplace import LaplaceGPC
from methods.EP import EPGPC
from methods.VI import VIGPC
from methods.MCMC import MCMCGPC
from utils.evaluate_small_data import evaluate_model

np.random.seed(42)

# HELPER: median pairwise distance → log length-scale


def get_log_l(X_train):
    """
    Compute log of median pairwise Euclidean distance.
    This is the standard data-driven heuristic for fixing the RBF
    length-scale without hyperparameter optimisation.
    Garreau et al. (2017); Scholkopf & Smola (2002).
    """
    dists = pdist(X_train, metric='euclidean')
    median_dist = np.median(dists)
    return np.log(max(median_dist, 1e-6))




# MAIN EXPERIMENT


def run_small_data_experiment(X_full, y_full,
                               train_sizes=[20, 50, 100],
                               n_repeats=10,
                               dataset_name='Dataset'):
    """
    For each training size:
      1. Randomly sample n training points (stratified)
      2. Use remaining points as test set
      3. Compute median pairwise distance on training set → log_l
      4. Fit all four methods with that log_l, log_sf=0.0
      5. Evaluate NLL and ECE on test set
      6. Repeat n_repeats times and average

    Parameters
    ----------
    X_full, y_full  : full standardised dataset
    train_sizes     : list of training set sizes to test
    n_repeats       : number of random splits per size
    dataset_name    : string for plot titles
    """

    # store results: results[method][n] = list of (nll, ece) across repeats
    methods = ['Laplace', 'EP', 'VI', 'MCMC']
    results = {m: {n: [] for n in train_sizes} for m in methods}

    for n_train in train_sizes:
        print(f"\n  Training size n = {n_train}")
        print(f"  {'Method':<10} {'NLL':>8} {'ECE':>8}")
        print(f"  {'-'*28}")

        for repeat in range(n_repeats):
            # ── stratified split ──
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_full, y_full,
                train_size=n_train,
                stratify=y_full,
                random_state=repeat    # different split each repeat
            )

            # ── compute length-scale from THIS training set ──
            log_l = get_log_l(X_tr)    # median pairwise distance heuristic

            # ── define classifiers with this log_l ──
            # signal variance fixed at sf=1.0 (log_sf=0.0) throughout
            params = {'log_l': log_l, 'log_sf': 0.0}

            clfs = {
                'Laplace': LaplaceGPC(kernel_rbf, params),
                'EP':      EPGPC(kernel_rbf,      params),
                'VI':      VIGPC(kernel_rbf,       params),
                'MCMC':    MCMCGPC(kernel_rbf,     params,
                                   n_samples=100, n_burnin=50, thin=2),
            }

            # ── fit and evaluate each method ──
            for mname, clf in clfs.items():
                try:
                    clf.fit(X_tr, y_tr)
                    nll, ece = evaluate_model(clf, X_te, y_te)
                    results[mname][n_train].append((nll, ece))
                except Exception as e:
                    # if a method fails on very small n, skip that repeat
                    print(f"    {mname} failed at n={n_train}: {e}")

        # ── print average across repeats ──
        for mname in methods:
            vals = results[mname][n_train]
            if len(vals) == 0:
                continue
            avg_nll = np.mean([v[0] for v in vals])
            avg_ece = np.mean([v[1] for v in vals])
            print(f"  {mname:<10} {avg_nll:>8.3f} {avg_ece:>8.3f}")

    return results

