# src.regression

"""
Describe algorithms within fields of regression, includes:

"""

import numpy as np

def compute_mse(y_true, y_pred) -> float:
    """
    Compute MSE (to minimise the loss function)
    Args:
        y_true: vector real true label
        y_pred: vector predicted label
    
    Returns:
        float: MSE (non-negative real number)
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = len(y_true) # shape (N,)
    return np.sum((y_true - y_pred) ** 2) / N

