import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from neural_network import CNNModel

_CNN_LEARNING_RATE = 0.001
_CNN_WEIGHT_DECAY = 0.001
_CNN_EPOCHS = 500


def fit_cnn_surrogate(X, y, seed: int):
    if len(y) < max(8, X.shape[1] + 2):
        return None, None

    if np.std(y) < 1e-12:
        return None, None

    torch.manual_seed(seed)

    np.random.seed(seed)

    # Standardise inputs and outputs.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)

    y_scaled = y_scaler.fit_transform(np.asarray(y).reshape(-1, 1))

    # CNN input format:
    # (samples, channels, sequence_length)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)

    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # Build CNN.
    model = CNNModel(input_dimension=X.shape[1])

    # Adam optimiser with L2 regularisation.
    optimiser = optim.Adam(
        model.parameters(), lr=_CNN_LEARNING_RATE, weight_decay=_CNN_WEIGHT_DECAY
    )

    loss_function = nn.MSELoss()

    # Training loop.
    model.train()

    for epoch in range(_CNN_EPOCHS):
        optimiser.zero_grad()

        predictions = model(X_tensor)

        loss = loss_function(predictions, y_tensor)

        loss.backward()

        optimiser.step()

    model.eval()

    return model, (x_scaler, y_scaler)
