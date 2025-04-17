"""Defines model evaluation functions."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
from torch.utils.data import DataLoader

from constants import FINGER_LABELS


def cosine_correlation_metric(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    sample: int | None = None,
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
        sample (Optional[int], default=None):
            Selected sample to plot, or all if unspecified.
    """

    num_fingers = actuals.shape[1]
    samples = actuals.shape[2] if len(actuals.shape) > 2 else 1

    plt.figure(figsize=(12, 8))

    for i in range(num_fingers):
        plt.subplot(num_fingers, 1, i+1)

        # Plot all actual flexion data
        actual = actuals[:, i]
        for j in range(samples):
            if sample is not None and j != sample:
                continue
            plt.plot(
                time_axis,
                actual[:, j],
                label='True' if sample is not None or j == 0 else '',
                color='blue',
                alpha=0.7,
            )

        # Plot all prediction flexion data
        pred = predictions[:, i]
        for j in range(samples):
            if sample is not None and j != sample:
                continue
            plt.plot(
                time_axis,
                pred[:, j],
                label='Prediction' if sample is not None or j == 0 else '',
                color='green',
                alpha=0.7,
            )

        plt.ylabel(finger_labels[i])

        plt.grid(True)

        if i == 0:
            plt.legend()

        if i == num_fingers - 1:
            plt.xlabel("Time (s)")

    plt.suptitle(f"Actual vs Predicted ECoG Time Series {'(Collated) ' if sample is None else ''}for Each Finger")
    plt.tight_layout()
    plt.show()


def test_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates an instance of a trained model.

    Parameters:
        model (nn.Module):
            The model instance to train.
        val_loader (DataLoader):
            Dataloader of the test dataset.
        device (torch.device, default=DEVICE):
            Desired torch device to use for computation.

    Returns:
        The actual and predicted values.
    """

    model.to(device)
    model.eval()

    all_actuals, all_predictions = [], []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            pred = model(x)

            all_actuals.append(y.cpu().numpy())
            all_predictions.append(pred.cpu().numpy())

    all_actuals = np.concatenate(all_actuals, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    all_predictions = gaussian_filter1d(all_predictions, sigma=2, axis=0)

    # Plot the entire time series
    return np.array(all_actuals), np.array(all_predictions)


if __name__ == '__main__':
    pass
