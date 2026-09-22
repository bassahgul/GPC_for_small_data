
# Small-Data Gaussian Process Classification

Code for the paper "Small-Data Gaussian Process Classification: Comparing Laplace, EP, VI, and MCMC".

## Install

```bash
pip install numpy scipy scikit-learn matplotlib
```

## Fixed hyperparameters
To reproduce Table 1 of the paper, run the script for each dataset:
```bash
python experiments/fixed_hyperparameter/banana.py
python experiments/fixed_hyperparameter/bcancer.py
python experiments/fixed_hyperparameter/ionosphere.py
python experiments/fixed_hyperparameter/pima.py
python experiments/fixed_hyperparameter/sonar.py
```

## Optimised hyperparameters
To reproduce Table 2 of the paper, run the script for each dataset:
```bash
python experiments/optimized_hyperparameters/banana.py
python experiments/optimized_hyperparameters/bcancer.py
python experiments/optimized_hyperparameters/ionosphere.py
python experiments/optimized_hyperparameters/pima.py
python experiments/optimized_hyperparameters/Sonar.py
```

Each script prints the NLL and ECE values reported in the paper.
