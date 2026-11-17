import torch


def cnn_predict(cnn_result, X):
    model, scalers = cnn_result

    if model is None:
        return None

    x_scaler, y_scaler = scalers

    X_scaled = x_scaler.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)

    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    predictions = y_scaler.inverse_transform(predictions).ravel()

    return predictions

