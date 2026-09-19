
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_breast_cancer
import numpy as np

def load_banana_openml():
    data = fetch_openml(data_id=1460, as_frame=False)
    X = data.data
    y = data.target.astype(float)
    y = np.where(y == 1, 0.0, 1.0)   # remap: 1→0, 2→1
    return X, y.astype(int)

def load_breast_cancer_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target   # already {0, 1}
    return X, y

def load_ionosphere():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=59, as_frame=False)
    X = data.data.astype(float)
    y = (data.target == 'g').astype(int)  # 'g' = good, 'b' = bad
    return X, y

def load_pima():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=37, as_frame=False)
    X = data.data.astype(float)
    y = (data.target == 'tested_positive').astype(int)
    return X, y

def load_sonar():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=40, as_frame=False)
    X = data.data.astype(float)
    y = (data.target == 'Mine').astype(int)  # 'Mine' vs 'Rock'
    return X, y