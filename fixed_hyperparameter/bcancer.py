
from fixed_hyperparameters import run_small_data_experiment
from utils.small_data_plot import plot_small_data_results
from dataset.datasets import load_breast_cancer_data
from sklearn.preprocessing import StandardScaler



if __name__ == '__main__':
    print("=" * 60)
    print("  Small Data Experiment")
    print("  Training sizes : n = 20, 50, 100")
    print("  Length-scale   : median pairwise distance (per split)")
    print("  Signal variance: fixed sf = 1.0")
    print("  Repeats        : 10 per training size")
    print("=" * 60)

    # ── load and standardise once ──
    print("\n[1] Loading dataset...")
    X_full, y_full = load_breast_cancer_data()

    # standardise before experiment — important for distance computation
    sc     = StandardScaler()
    X_full = sc.fit_transform(X_full)

    print(f"    n={len(X_full)}, d={X_full.shape[1]}")
    print(f"    Class 0: {(y_full==0).sum()}, Class 1: {(y_full==1).sum()}")

    # ── run experiment ──
    print("\n[2] Running small data experiment...")
    results = run_small_data_experiment(
        X_full, y_full,
        train_sizes=[20, 50, 100],
        n_repeats=10,
        dataset_name='Breast Cancer'
    )

    # ── plot ──
    print("\n[3] Generating figures...")
    plot_small_data_results(results, [20, 50, 100], 'Breast Cancer')

    print("\nDone.")