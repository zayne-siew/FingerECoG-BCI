"""Defines storing and handling of preprocessed data."""

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class EcogDataset(Dataset):
    """Custom dataset for storing and handling (preprocessed) data."""

    def __init__(self, spectrogram: np.ndarray, finger_flex: np.ndarray, data_length: int = 256):
        """Loads preprocessed ECoG (spectrogram) and finger flexion data.

        Parameters:
            spectrogram (np.ndarray):
                Preprocessed ECoG data.
            finger_flex (np.ndarray):
                Preprocessed finger flexion data.
            data_length (int):
                Length of the recording window of each data point.
        """

        self.spectrogram = spectrogram.astype('float32')
        self.finger_flex = finger_flex.astype('float32')
        self.data_length = data_length

    def __len__(self):
        return self.spectrogram.shape[2] - self.data_length

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        spectrogram_crop = self.spectrogram[...,index:index+self.data_length]
        finger_flex_crop = self.finger_flex[...,index:index+self.data_length]
        return spectrogram_crop, finger_flex_crop


def plot_ecog_data(train_dataset: EcogDataset) -> None:
    """Extracts a few samples to visualise the ECoG + finger flexion data.

    Parameters:
        train_dataset (EcogDataset):
            Custom dataset that stores the preprocessed data.
    """

    # Create a figure with subplots
    _, axs = plt.subplots(2, 1, figsize=(15, 10))

    # Get a few samples (let's take the first 3)
    for i in range(min(3, len(train_dataset))):
        x, y = train_dataset[i]

        # Plot input data (x)
        axs[0].set_title('Input ECoG Data (x) - Shape: ' + str(x.shape))
        # Assuming x is (62, 40, timeseries)
        # We'll plot a few channels/slices
        im = axs[0].imshow(x[0, :, :], aspect='auto', cmap='viridis')

        # Plot target data (y)
        plt.title('Target Data (5 Fingers) - Time Series')
        plt.plot(y.T)
        plt.xlabel('Time Steps')
        plt.ylabel('Finger Movement/Neural Activity')
        plt.legend([f'Finger {i+1}' for i in range(y.shape[0])])
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Additional statistical analysis
    x, y = train_dataset[0]
    print("Input (x) Data Statistics:")
    print(f"Shape: {x.shape}")
    print(f"Mean: {x.mean()}")
    print(f"Std: {x.std()}")
    print(f"Min: {x.min()}")
    print(f"Max: {x.max()}")

    print("\nTarget (y) Data Statistics:")
    print(f"Shape: {y.shape}")
    print(f"Mean: {y.mean()}")
    print(f"Std: {y.std()}")
    print(f"Min: {y.min()}")
    print(f"Max: {y.max()}")


if __name__ == '__main__':
    pass
