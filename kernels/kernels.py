
import numpy as np

def kernel_rbf(X1, X2, log_l=0.0, log_sf=0.0):
    l = np.exp(log_l)
    sf = np.exp(log_sf)
    sq_dist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return sf ** 2 * np.exp(-0.5 * sq_dist / l ** 2)

def kernel_matern52(X1, X2, log_l=0.0, log_sf=0.0):
    l = np.exp(log_l)
    sf = np.exp(log_sf)
    r = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
    r_l = np.sqrt(5) * r / l
    return sf ** 2 * (1 + r_l + r_l ** 2 / 3) * np.exp(-r_l)

def kernel_rq(X1, X2, log_l=0.0, log_sf=0.0, log_alpha=0.0):
    l = np.exp(log_l)
    sf = np.exp(log_sf)
    alpha = np.exp(log_alpha)
    sq_dist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return sf ** 2 * (1 + sq_dist / (2 * alpha * l ** 2)) ** (-alpha)

KERNELS = {'RBF': kernel_rbf, 'Matern52': kernel_matern52, 'RQ': kernel_rq}