from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path("../../initial_data")
_HISTORY_DIR = Path("../../history")


def _load_function(function_id):
    path = _DATA_DIR / f"function_{function_id}"

    X = np.asarray(np.load(path / "initial_inputs.npy"), dtype=float)

    y = np.asarray(np.load(path / "initial_outputs.npy"), dtype=float).reshape(-1)

    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Invalid input/output shapes.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Inputs and outputs must be finite.")

    if np.any(X < 0) or np.any(X > 1):
        raise ValueError("Inputs outside [0, 1].")

    return X, y


def _load_history(function_id):
    path = _HISTORY_DIR / f"function_{function_id}_history.csv"

    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    x_columns = [column for column in df.columns if column.startswith("x")]

    if "y" not in df.columns or not x_columns:
        raise ValueError(f"Invalid history file: {path}")

    def column_order(name):
        suffix = name[1:]

        if suffix.isdigit():
            return (0, int(suffix))

        return (1, name)

    x_columns = sorted(x_columns, key=column_order)

    X = df[x_columns].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite history values.")

    if np.any(X < 0) or np.any(X > 1):
        raise ValueError("History inputs outside [0, 1].")

    return X, y


def load_all_data(function_id):

    X, y = _load_function(function_id)

    X_history, y_history = _load_history(function_id)

    if X_history is None:
        return X, y

    if X_history.shape[1] != X.shape[1]:
        raise ValueError("History dimension mismatch.")

    X = np.vstack([X, X_history])
    y = np.concatenate([y, y_history])

    # Average repeated observations at identical inputs.
    df = pd.DataFrame(X)
    df["_y"] = y

    grouped = df.groupby(list(range(X.shape[1])), sort=False, as_index=False)[
        "_y"
    ].mean()

    X = grouped.iloc[:, : X.shape[1]].to_numpy(dtype=float)
    y = grouped["_y"].to_numpy(dtype=float)

    return X, y
