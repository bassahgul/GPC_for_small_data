
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
import numpy as np
from utils.evaluate_small_data import evaluate_model
from kernels.kernels import kernel_rbf

# Import all method classes
from optimized_methods.optimised_laplace import LaplaceGPC
from optimized_methods.optimised_EP import EPGPC
from optimized_methods.optimised_VI import VIGPC
from optimized_methods.optimised_MCMC import MCMCGPC

def run_small_data_experiment_optimised(X_full, y_full, train_sizes, n_repeats, dataset_name):
    methods = ['Laplace', 'EP', 'VI', 'MCMC']
    results = {m: {n: [] for n in train_sizes} for m in methods}

    for n_train in train_sizes:
        print(f"\n  Training size n = {n_train}")
        print(f"  {'Method':<10} {'NLL':>8} {'ECE':>8}")
        print(f"  {'-'*28}")

        for repeat in range(n_repeats):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_full, y_full,
                train_size=n_train,
                stratify=y_full,
                random_state=repeat
            )

            # Median heuristic for initial length scale (used as starting point for optimisation)
            dists = pdist(X_tr, metric='euclidean')
            med = np.median(dists)
            log_l_init = np.log(max(med, 1e-6))
            params_init = {'log_l': log_l_init, 'log_sf': 0.0}

            # ----- Laplace (has .optimise() method) -----
            laplace = LaplaceGPC(kernel_rbf, params_init)
            laplace.optimise(X_tr, y_tr)
            laplace.fit(X_tr, y_tr)

            # ----- EP (optimise flag in constructor) -----
            ep = EPGPC(kernel_rbf, params_init, optimise=True)   # will optimise inside fit
            ep.fit(X_tr, y_tr)

            # ----- VI (has .optimise() method) -----
            vi = VIGPC(kernel_rbf, params_init)
            vi.optimise(X_tr, y_tr)
            vi.fit(X_tr, y_tr)

            # ----- MCMC (has .optimise() method) -----
            mcmc = MCMCGPC(kernel_rbf, params_init, n_samples=100, n_burnin=50, thin=2)
            mcmc.optimise(X_tr, y_tr)
            mcmc.fit(X_tr, y_tr)

            # Evaluate all four
            for name, clf in zip(methods, [laplace, ep, vi, mcmc]):
                try:
                    nll, ece = evaluate_model(clf, X_te, y_te)
                    results[name][n_train].append((nll, ece))
                except Exception as e:
                    print(f"    {name} evaluation failed at n={n_train}: {e}")

        # Print averages for this training size
        for mname in methods:
            vals = results[mname][n_train]
            if len(vals) == 0:
                continue
            avg_nll = np.mean([v[0] for v in vals])
            avg_ece = np.mean([v[1] for v in vals])
            print(f"  {mname:<10} {avg_nll:>8.3f} {avg_ece:>8.3f}")

    return results