from torch import nn

_CNN_CHANNELS = (16, 32)
_CNN_HIDDEN_SIZE = 16


class CNNModel(nn.Module):
    def __init__(self, input_dimension):

        super().__init__()

        # Convolutional feature extraction.
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=_CNN_CHANNELS[0], kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=_CNN_CHANNELS[0],
                out_channels=_CNN_CHANNELS[1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        # Compress the feature dimension.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected regression head.
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_CNN_CHANNELS[1], _CNN_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(_CNN_HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)

        x = self.global_pool(x)

        x = self.regression_head(x)

        return x

