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

def gradient_descent(X, y, lr: float, epochs: int, save_history: bool = False):

    w = np.zeros(X.shape[1])
    loss_history = [] if save_history else None
    N = len(X)

    for epoch in range(epochs):
        y_pred = X @ w
        error = y_pred - y
    
        gradient = (2 / N) * (X.T @ error)
        w = w - lr * gradient
        
        if save_history:
            mse = np.mean(error ** 2)
            loss_history.append(mse)
        
        if epoch % max(1, epochs // 5) == 0:
            current_mse = np.mean(error ** 2)
            print(f"Epoch {epoch:4d} | MSE: {current_mse:.4f} | w0: {w[0]:.2f} | w1: {w[1]:.4f}")

    print("-" * 50)
    print(f"Final result: y = {w[0]:.3f} + {w[1]:.4f}x")
    
    if save_history:
        return w, loss_history
    else:
        return w

