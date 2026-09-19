
import matplotlib.pyplot as plt
import numpy as np


def plot_small_data_results(results, train_sizes, dataset_name='Dataset'):
    """
    Two figures:
      Fig A — NLL vs training size for all methods
      Fig B — ECE vs training size for all methods
    """
    methods = list(results.keys())
    colors  = {'Laplace': '#e41a1c', 'EP': '#377eb8',
                'VI': '#4daf4a',     'MCMC': '#984ea3'}
    markers = {'Laplace': 'o', 'EP': 's', 'VI': '^', 'MCMC': 'D'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Small Data Experiment — {dataset_name}',
                 fontsize=13, fontweight='bold')

    for ax, metric, label, idx in zip(
            axes,
            ['nll', 'ece'],
            ['Negative Log-Likelihood (lower = better)',
             'Expected Calibration Error (lower = better)'],
            [0, 1]):

        for mname in methods:
            means, stds = [], []
            for n in train_sizes:
                vals = results[mname][n]
                if len(vals) == 0:
                    means.append(np.nan); stds.append(0); continue
                # extract nll (idx=0) or ece (idx=1)
                v = [x[idx] for x in vals]
                means.append(np.mean(v))
                stds.append(np.std(v))

            means = np.array(means)
            stds  = np.array(stds)

            ax.plot(train_sizes, means,
                    color=colors[mname], marker=markers[mname],
                    linewidth=2, markersize=7, label=mname)
            ax.fill_between(train_sizes,
                            means - stds, means + stds,
                            alpha=0.15, color=colors[mname])

        ax.set_xlabel('Training Set Size (n)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(train_sizes)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fname = f'fig_small_data_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {fname}")
