import cupy as np
from typing import Any
from numpy import floating

class LossFunction:

    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def absolute_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
        return np.mean(np.abs(y_true - y_pred) ** 2)

    @staticmethod
    def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta : float = 1) -> floating[Any]:
        abs_diff = np.abs(y_true - y_pred)
        loss = np.clip(abs_diff, 0, delta) ** 2 * 0.5 + (abs_diff - delta) * delta * (abs_diff > delta)
        return np.sum(loss, dtype=np.float64)

    @staticmethod
    def logistic_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.number[Any]:
        epsilon = 1e-15  # To prevent log(0) issues
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0) errors
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    @staticmethod
    def categorical_crossentropy(y_true, y_pred, epsilon=1e-12):

        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)  # Prevent log(0)
        return -np.einsum('ij,ij->', y_true, np.log1p(y_pred - 1)) / y_true.shape[1]

