# src.logistic

import numpy as np

def sigmoid(z) -> float:
    """
    Compute the sigmoid function
    Args:
        z: input value (or array) of linear combination

    Returns:
        float: sigmoid output, in (0, 1)
    """
    # Apply sigmoid: 1 / (1 + e^(-z))
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat, epsilon = 1e-9):
    """
    Compute binary cross-entropy loss (to minimise the loss function)
    Args:
        y: vector of true labels (0 or 1)
        y_hat: vector of predicted probabilities
        epsilon: small value to avoid log(0)

    Returns:
        float: binary cross-entropy loss (non-negative real number)
    """
    # Compute -mean(y*log(y_hat) + (1-y)*log(1-y_hat)), epsilon avoids log(0)
    return -np.mean(y * np.log(y_hat + epsilon) 
                    + (1 - y) * np.log(1 - y_hat + epsilon))

def _train_bilogistic(X, y, lr = 0.01, epochs = 1000):
    """
    Fit binary logistic regression by gradient descent
    Args:
        X: matrix of features (N, D)
        y: vector of true labels (0 or 1), shape (N,)
        lr: learning rate
        epochs: number of training iterations

    Returns:
        tuple: (weights, bias) after training
    """
    # Get number of samples (N) and features (D)
    n, d = X.shape
    # Initialise weights to zeros, shape (D,)
    w = np.zeros(d)
    # Initial start bias
    b = 0.0

    for epoch in range(epochs):
        # Compute linear combination of features and weights
        z = X @ w + b # shape (N,)
        # Apply sigmoid to get predicted probabilities
        y_hat = sigmoid(z) # shape (N,)
        
        # Compute error between predictions and true labels
        error = y_hat - y # shape (N,)
        # Compute gradient of weights, shape (D,)
        dw = (X.T @ error) / n
        # Compute gradient of bias, scalar
        db = np.mean(error)
        
        # Update weights with gradient descent
        w -= lr * dw
        # Update bias with gradient descent
        b -= lr * db

    return w, b

def log_sample(n: int, random_state: int = None, ranging: int = 1):
    """
    Create sample of n-number of dots
    True label: 1 if y > -log(x), else 0
    Args:
        n: number of sample points
        random_state: seed for reproducibility
        ranging: upper bound of the uniform range

    Returns:
        tuple: (X, y_labels) where X has shape (n, 2) and y_labels has shape (n,)
    """
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Small value to avoid log(0)
    eps = 1e-7
    # Generate first feature uniformly in (eps, ranging)
    x = np.random.uniform(eps, ranging, size=n) # shape (n,)
    # Generate second feature uniformly in (0, ranging)
    y = np.random.uniform(0.0, ranging, size=n) # shape (n,)

    # Stack features into a matrix, shape (n, 2)
    X = np.column_stack((x, y))
    # Compute decision boundary: -log(x)
    boundary = -np.log(x) # shape (n,)
    # Label 1 if y is above the boundary, else 0
    y_labels = (y > boundary).astype(int) # shape (n,)

    return X, y_labels
