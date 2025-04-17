"""Defines model evaluation functions."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from constants import FINGER_LABELS


def correlation_metric(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Implements the cosine similarity metric.

    Parameters:
        x (torch.Tensor):
            Data to compare against.
        y (torch.Tensor):
            Data to compare against.

    Returns:
        torch.Tensor:
            Cosine similarity between the data values.
    """

    cos_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)
    return torch.mean(cos_metric(x, y))


def pearson_correlation_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Implements the Pearson correlation coefficient metric.

    Parameters:
        y_pred (torch.Tensor):
            Predicted values to compare against.
        y_true (torch.Tensor):
            Actual values to compare against.

    Returns:
        torch.Tensor:
            Pearson correlation coefficient between the predicted and actual values.
    """

    x = y_pred - torch.mean(y_pred)
    y = y_true - torch.mean(y_true)

    numerator = torch.sum(x * y)
    denominator = torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2))

    return numerator / (denominator + 1e-6)


def pearson_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Implements the Pearson correlation loss metric.

    Parameters:
        y_pred (torch.Tensor):
            Predicted values to compare against.
        y_true (torch.Tensor):
            Actual values to compare against.

    Returns:
        torch.Tensor:
            Pearson correlation loss between the predicted and actual values.
    """

    return 1 - pearson_correlation_metric(y_pred, y_true)


def plot_fingerwise_timeseries(
    actuals: np.ndarray,
    predictions: np.ndarray,
    time_axis: np.ndarray,
    finger_labels: list[str] = FINGER_LABELS,
) -> None:
    """Plot actual vs. predicted ECoG signals for each finger as separate time series subplots.

    Parameters:
        actuals (np.ndarray):
            Numpy array of shape `(timesteps, num_fingers)`.
        predictions (np.ndarray):
            Numpy array of shape `(timesteps, num_fingers)`.
        time_axis (np.ndarray):
            Numpy array of shape `(timesteps,)`, optional.
        finger_labels (list[int], default=FINGER_LABELS):
            List of string labels for fingers.
    """

    num_fingers = actuals.shape[1]

    plt.figure(figsize=(12, 8))

    for i in range(num_fingers):
        plt.subplot(num_fingers, 1, i+1)
        plt.plot(time_axis, actuals[:, i], label='True', color='blue', alpha=0.7)
        plt.plot(time_axis, predictions[:, i], label='Prediction', color='green', alpha=0.7)

        plt.ylabel(finger_labels[i])

        plt.grid(True)

        if i == 0:
            plt.legend()

        if i == num_fingers - 1:
            plt.xlabel("Time (s)")

    plt.suptitle("Actual vs Predicted ECoG Time Series for Each Finger")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
