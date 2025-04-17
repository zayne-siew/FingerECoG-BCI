"""Defines models to train on the ECoG data."""

import torch
from torch import nn


class EncoderDecoderVanilla(nn.Module):
    """Vanilla convolutional autoencoder (CAE)."""

    def __init__(self, input_channels: int, output_channels: int, dropout_rate: float = 0.3):
        """Constructs an `EncoderDecoderVanilla` instance.

        Parameters:
            input_channels (int):
                Number of input channels.
            output_channels (int):
                Number of output channels.
            dropout_rate (float, default=0.3):
                Dropout rate, for the dropout layers.
        """

        super(EncoderDecoderVanilla, self).__init__()

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

class LSTM_flatten(nn.LSTM):
    """LSTM with built-in parameter flattening."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool, batch_first: bool):
        """Constructs a `LSTM_flatten` instance.

        All parameters are based off `nn.LSTM` parameters.
        """

        super(LSTM_flatten, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Flatten parameters and perform forward pass through LSTM."""

        self.lstm.flatten_parameters()
        output, hidden = self.lstm(x)
        return output, hidden


class ResidualBlock(nn.Module):
    """Custom encoding/decoding block with skip on a nested conv + batchnorm + dropout layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout_rate: float = 0.4):
        """Constructs a `ResidualBlock` instance.

        Parameters:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            kernel_size (int, default=3):
                Kernel size of the convolutional layer(s).
            dropout_rate (float, default=0.4):
                Dropout rate for the dropout layer.
        """

        super(ResidualBlock, self).__init__()

        padding = kernel_size // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels)
        )

        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the convolutional block and computes the result with skip."""

        identity = self.skip(x)
        out = self.conv_block(x)
        out += identity
        return self.activation(out)


class EncoderDecoder(nn.Module):
    """CAE-LSTM hybrid model with `ResidualBlock` encoder/decoder blocks and `LSTM_flatten` LSTM."""

    def __init__(self, input_channels: int, output_channels: int, dropout_rate: float = 0.4):
        """Constructs an `EncoderDecoder` instance.

        Parameters:
            input_channels (int):
                Number of input channels.
            output_channels (int):
                Number of output channels.
            dropout_rate (float, default=0.4):
                Dropout rate, for the dropout layers.
        """

        super(EncoderDecoder, self).__init__()

        # Encoder
        self.enc_block1 = ResidualBlock(input_channels, 32, kernel_size=3, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc_block2 = ResidualBlock(32, 64, kernel_size=3, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc_block3 = ResidualBlock(64, 128, kernel_size=3, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM with gradient clipping built in
        self.lstm = LSTM_flatten(input_size=128, hidden_size=128, num_layers=1,
                                    bidirectional=True, batch_first=True)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_block1 = ResidualBlock(256, 128, kernel_size=3, dropout_rate=dropout_rate)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_block2 = ResidualBlock(128 + 64, 64, kernel_size=3, dropout_rate=dropout_rate)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_block3 = ResidualBlock(64 + 32, 32, kernel_size=3, dropout_rate=dropout_rate)

        self.final_conv = nn.Conv1d(32, output_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through the encoder, LSTM, and decoder."""

        batch, elec, freq, time = x.shape
        x = x.reshape(batch, -1, time)

        # Encoder
        e1 = self.enc_block1(x)
        p1 = self.pool1(e1)

        e2 = self.enc_block2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc_block3(p2)
        p3 = self.pool3(e3)

        # LSTM
        p3_permuted = p3.permute(0, 2, 1)
        lstm_out, _ = self.lstm(p3_permuted)
        lstm_out = lstm_out.permute(0, 2, 1)

        # Decoder
        d1 = self.up1(lstm_out)
        d1 = self.dec_block1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec_block2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec_block3(d3)

        output = self.final_conv(d3)

        return output


if __name__ == '__main__':
    pass
