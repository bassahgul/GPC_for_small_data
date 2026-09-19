
import numpy as np
from sklearn.metrics import log_loss


# HELPER: Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    ECE measures how well predicted probabilities match true frequencies.
    A perfectly calibrated model has ECE = 0.
    Guo et al. (2017) 'On Calibration of Modern Neural Networks'.

    Formula: ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    for i in range(n_bins):
        # find samples whose predicted probability falls in this bin
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        # average confidence and accuracy in this bin
        conf = y_prob[mask].mean()
        acc  = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return ece




# HELPER: evaluate one trained model on test set
def evaluate_model(clf, X_test, y_test):
    """
    Returns NLL and ECE for a fitted classifier.
    NLL  = negative log-likelihood (lower is better)
    ECE  = expected calibration error (lower is better)
    """
    proba  = np.clip(clf.predict_proba(X_test), 1e-7, 1 - 1e-7)
    p_pos  = proba[:, 1]
    nll    = log_loss(y_test, proba)
    ece    = expected_calibration_error(y_test, p_pos)
    return nll, ece


