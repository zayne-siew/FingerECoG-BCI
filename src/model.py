"""Defines models to train on the ECoG data."""

import torch
from torch import nn


class EncoderDecoder(nn.Module):
    """Vanilla convolutional autoencoder (CAE)."""

    def __init__(self, input_channels: int, output_channels: int, dropout_rate: float = 0.3):
        """Constructs an `EncoderDecoder` instance.

        Parameters:
            input_channels (int):
                Number of input channels.
            output_channels (int):
                Number of output channels.
            dropout_rate (float, default=0.3):
                Dropout rate, for the dropout layers.
        """

        super(EncoderDecoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(64, output_channels, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through the encoder and decoder."""

        batch, elec, freq, time = x.shape
        x = x.reshape(batch, -1, time)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


if __name__ == '__main__':
    pass
