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
    """
    Fit linear regression by gradient descent
    Args:
        X: matrix of features (N, D)
        y: vector of true labels (N,)
        lr: learning rate
        epochs: number of training iterations
        save_history: if True, save MSE at each epoch
    
    Returns:
        tuple: (weights, loss_history) if save_history is True
        ndarray: weights otherwise
    """
    w = np.zeros(X.shape[1]) # shape (D,)
    loss_history = [] if save_history else None
    N = len(X) # shape (N,)

    for epoch in range(epochs):
        y_pred = X @ w # shape (N,)
        error = y_pred - y # shape (N,)
        
        if save_history:
            mse = np.mean(error ** 2)
            loss_history.append(mse)

        gradient = (2 / N) * (X.T @ error) # shape (D,)
        # Update weights with gradient descent
        w = w - lr * gradient
        
        
        if epoch % max(1, epochs // 5) == 0:
            current_mse = np.mean(error ** 2)
            print(f"Epoch {epoch:4d} | MSE: {current_mse:.4f} | w0: {w[0]:.2f} | w1: {w[1]:.4f}")

    print("-" * 50)
    print(f"Final result: y = {w[0]:.3f} + {w[1]:.4f}x")
    
    if save_history:
        return w, loss_history
    else:
        return w

def scale_zerone(X):
    """
    Scale features to [0, 1] by min-max normalization
    Args:
        X: matrix of features (N, D)
    
    Returns:
        tuple: (X_scaled, X_min, range_X)
    """
    # Compute min and max for each feature
    X_min = np.min(X, axis=0) # shape (D,)
    X_max = np.max(X, axis=0) # shape (D,)
    
    # Avoiding divide to zero
    range_X = X_max - X_min # shape (D,)
    range_X[range_X == 0] = 1 
    
    # Scale features to [0, 1]
    X_scaled = (X - X_min) / range_X
    return X_scaled, X_min, range_X