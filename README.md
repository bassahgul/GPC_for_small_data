
# Small-Data Gaussian Process Classification

Code for the paper "Small-Data Gaussian Process Classification: Comparing Laplace, EP, VI, and MCMC".

## Install

```bash
pip install numpy scipy scikit-learn matplotlib
```

## Fixed hyperparameters
To reproduce Table 1 of the paper, run the script for each dataset:
```bash
python fixed_hyperparameter/banana.py
python fixed_hyperparameter/bcancer.py
python fixed_hyperparameter/ionosphere.py
python fixed_hyperparameter/pima.py
python fixed_hyperparameter/sonar.py
```

## Optimised hyperparameters
To reproduce Table 2 of the paper, run the script for each dataset:
```bash
python optimized_hyperparameters/banana.py
python optimized_hyperparameters/bcancer.py
python optimized_hyperparameters/ionosphere.py
python optimized_hyperparameters/pima.py
python optimized_hyperparameters/Sonar.py
```

Each script prints the NLL and ECE values reported in the paper.
