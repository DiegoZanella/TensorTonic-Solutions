import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    D = X.shape[1]
    w = np.array([0 for i in range(D)])
    b = 0

    for i in range(steps):
        z = np.matmul(X, w) + b
        p = _sigmoid(z)
    
        grad_w = X.T @ (p-y)/len(y)
        grad_b = np.mean(p-y)

        w = w - lr*grad_w
        b = b - lr*grad_b

    return (w, b)