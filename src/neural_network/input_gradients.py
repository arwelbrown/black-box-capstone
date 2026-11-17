import numpy as np
import torch

_GRADIENT_STEPS = 30
_GRADIENT_STEP_SIZE = 0.03


def _cnn_input_gradient(cnn_result, X):
    model, scalers = cnn_result

    if model is None:
        return None

    x_scaler, y_scaler = scalers

    X = np.atleast_2d(np.asarray(X, dtype=float))

    X_scaled = x_scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled, dtype=torch.float32, requires_grad=True
    ).unsqueeze(1)

    model.eval()

    predictions = model(X_tensor)

    gradients = torch.autograd.grad(predictions.sum(), X_tensor)[0]

    # Remove the channel dimension.
    gradients = gradients.squeeze(1).detach().numpy()

    # Convert gradients to original scales.
    gradients *= float(y_scaler.scale_[0])

    gradients /= x_scaler.scale_.reshape(1, -1)

    return gradients


def generate_gradient_candidates(cnn_result, X, y, n_candidates, rng):
    dimension = X.shape[1]

    if cnn_result[0] is None or n_candidates <= 0:
        return np.empty((0, dimension))

    n_starts = min(max(4, int(np.sqrt(n_candidates))), len(X))

    # Start from the best observed points.
    start_indices = np.argsort(y)[:n_starts]

    starts = X[start_indices].copy()

    points = []

    for start in starts:
        x = start.copy()

        for _ in range(_GRADIENT_STEPS):
            gradient = _cnn_input_gradient(cnn_result, x.reshape(1, -1))

            if gradient is None:
                break

            if not np.all(np.isfinite(gradient)):
                break

            gradient = gradient[0]

            gradient_norm = np.linalg.norm(gradient)

            if gradient_norm < 1e-10:
                break

            # Follow the negative gradient.
            x = np.clip(x - _GRADIENT_STEP_SIZE * gradient / gradient_norm, 0.0, 1.0)

            points.append(x.copy())

            if len(points) >= n_candidates:
                break

        if len(points) >= n_candidates:
            break

    # Supplement with random candidates.
    remaining = n_candidates - len(points)

    if remaining > 0:
        supplement = rng.uniform(0.0, 1.0, size=(remaining, dimension))

        points.extend(supplement)

    return np.asarray(points[:n_candidates])


def generate_candidates(n_candidates, dimension, rng):
    return rng.uniform(0.0, 1.0, size=(n_candidates, dimension))
