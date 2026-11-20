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
