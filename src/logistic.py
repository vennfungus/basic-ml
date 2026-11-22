# src.logistic

import numpy as np

def sigmoid(z) -> float:
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat, epsilon = 1e-9):
    return -np.mean(y * np.log(y_hat + epsilon) 
                    + (1 - y) * np.log(1 - y_hat + epsilon))

def _train_bilogistic(X, y, lr = 0.01, epochs = 1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0 # Initial start bias
    for epoch in range(epochs):
        z = X @ w + b
        y_hat = sigmoid(z)
        
        error = y_hat - y
        dw = (X.T @ error) / n
        db = np.mean(error)
        
        w -= lr * dw
        b -= lr * db
    return w, b

def log_sample(n: int, random_state: int = None):
    """
    Create sample of n-number of dots
    True label: 
    """

    if random_state is not None:
        np.random.seed(random_state)

    eps = 1e-7
    x = np.random.uniform(eps, 1.0, size=n)
    y = np.random.uniform(0.0, 1.0, size=n)

    X = np.column_stack((x, y))
    boundary = -np.log(x)
    y_labels = (y > boundary).astype(int)

    return X, y_labels
