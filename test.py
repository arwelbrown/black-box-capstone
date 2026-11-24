import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import norm
from scipy.spatial import cKDTree

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, balanced_accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = Path("./initial_data")
HISTORY_DIR = Path("./history")

EXPECTED_OBSERVATIONS = 17

# Randomised cross-validated hyperparameter search.
TUNING_FOLDS = 4
GP_TUNING_TRIALS = 12
SVM_TUNING_TRIALS = 12
CNN_TUNING_TRIALS = 4
TUNING_SEED = 2026

N_CANDIDATES = 100_000
SEED = 42

# False for minimisation; True for maximisation.
MAXIMISE = False

PROMISING_FRACTION = 0.30

# Hybrid acquisition weights.
WEIGHT_EI = 0.50
WEIGHT_LINEAR = 0.05
WEIGHT_SVM = 0.20
WEIGHT_CNN = 0.25

# CNN configuration.
CNN_CHANNELS = (16, 32)
CNN_HIDDEN_SIZE = 16

CNN_EPOCHS = 500
CNN_LEARNING_RATE = 0.001
CNN_WEIGHT_DECAY = 0.001
CNN_SEED = 42

# Gradient-guided search.
GRADIENT_FRACTION = 0.30
GRADIENT_STEPS = 30
GRADIENT_STEP_SIZE = 0.03


# ==========================================
# DATA LOADING
# ==========================================

def load_function(function_id):
    path = DATA_DIR / f"function_{function_id}"

    X = np.asarray(
        np.load(path / "initial_inputs.npy"),
        dtype=float
    )

    y = np.asarray(
        np.load(path / "initial_outputs.npy"),
        dtype=float
    ).reshape(-1)

    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Invalid input/output shapes.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Inputs and outputs must be finite.")

    if np.any(X < 0) or np.any(X > 1):
        raise ValueError("Inputs outside [0, 1].")

    return X, y


def load_history(function_id):
    path = HISTORY_DIR / f"function_{function_id}_history.csv"

    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    x_columns = [
        column for column in df.columns
        if column.startswith("x")
    ]

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
    X, y = load_function(function_id)

    X_history, y_history = load_history(function_id)

    if X_history is None:
        return X, y

    if X_history.shape[1] != X.shape[1]:
        raise ValueError("History dimension mismatch.")

    X = np.vstack([X, X_history])
    y = np.concatenate([y, y_history])

    # Average repeated observations at identical inputs.
    df = pd.DataFrame(X)
    df["_y"] = y

    grouped = df.groupby(
        list(range(X.shape[1])),
        sort=False,
        as_index=False
    )["_y"].mean()

    X = grouped.iloc[:, :X.shape[1]].to_numpy(dtype=float)
    y = grouped["_y"].to_numpy(dtype=float)

    return X, y


# ==========================================
# GAUSSIAN PROCESS
# ==========================================

def fit_gp(X, y, length_scale=1.0, noise_level=1e-5, nu=2.5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dimension = X.shape[1]

    kernel = (
        ConstantKernel(
            1.0,
            constant_value_bounds="fixed"
        )
        * Matern(
            length_scale=np.full(dimension, length_scale),
            length_scale_bounds="fixed",
            nu=nu
        )
        + WhiteKernel(
            noise_level=noise_level,
            noise_level_bounds="fixed"
        )
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        optimizer=None,
        random_state=SEED
    )

    model.fit(X_scaled, y)

    return model, scaler


def expected_improvement(
    X_candidates,
    model,
    scaler,
    y_best
):
    X_scaled = scaler.transform(X_candidates)

    mean, std = model.predict(
        X_scaled,
        return_std=True
    )

    std = np.maximum(std, 1e-12)

    improvement = y_best - mean

    z = improvement / std

    ei = (
        improvement * norm.cdf(z)
        + std * norm.pdf(z)
    )

    return np.maximum(ei, 0.0)


# ==========================================
# LINEAR REGRESSION
# ==========================================

def fit_linear_model(X, y):
    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    model.fit(X, y)

    return model


# ==========================================
# SVM CLASSIFIER
# ==========================================

def fit_svm_classifier(X, y, C=1.0, gamma="scale"):
    threshold = np.quantile(
        y,
        PROMISING_FRACTION
    )

    labels = (y <= threshold).astype(int)

    if len(np.unique(labels)) < 2:
        return None

    class_counts = np.bincount(
        labels,
        minlength=2
    )

    use_probability = class_counts.min() >= 5

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight="balanced",
            probability=use_probability,
            random_state=SEED
        )
    )

    model.fit(X, labels)

    return model, use_probability


def svm_promising_score(svm_result, candidates):
    if svm_result is None:
        return np.full(len(candidates), 0.5)

    model, use_probability = svm_result

    if use_probability:
        classes = model.named_steps["svc"].classes_

        positive_column = int(
            np.flatnonzero(classes == 1)[0]
        )

        return model.predict_proba(
            candidates
        )[:, positive_column]

    decision = model.decision_function(candidates)

    decision = np.clip(decision, -30.0, 30.0)

    return 1.0 / (1.0 + np.exp(-decision))


# ==========================================
# PYTORCH 1D CNN ARCHITECTURE
# ==========================================

class CNNModel(nn.Module):
    def __init__(self, input_dimension, hidden_size=CNN_HIDDEN_SIZE):
        super().__init__()

        # Convolutional feature extraction.
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=CNN_CHANNELS[0],
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=CNN_CHANNELS[0],
                out_channels=CNN_CHANNELS[1],
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        # Compress the feature dimension.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected regression head.
        self.regression_head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                CNN_CHANNELS[1],
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                1
            )
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.global_pool(x)
        x = self.regression_head(x)

        return x


# ==========================================
# CNN TRAINING
# ==========================================

def fit_cnn_surrogate(
    X, y, seed=CNN_SEED, epochs=CNN_EPOCHS,
    learning_rate=CNN_LEARNING_RATE,
    weight_decay=CNN_WEIGHT_DECAY,
    hidden_size=CNN_HIDDEN_SIZE
):
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

    y_scaled = y_scaler.fit_transform(
        np.asarray(y).reshape(-1, 1)
    )

    # CNN input format:
    # (samples, channels, sequence_length)
    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    ).unsqueeze(1)

    y_tensor = torch.tensor(
        y_scaled,
        dtype=torch.float32
    )

    # Build CNN.
    model = CNNModel(
        input_dimension=X.shape[1],
        hidden_size=hidden_size
    )

    # Adam optimiser with L2 regularisation.
    optimiser = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    loss_function = nn.MSELoss()

    # Training loop.
    model.train()

    for epoch in range(epochs):
        optimiser.zero_grad()

        predictions = model(X_tensor)

        loss = loss_function(
            predictions,
            y_tensor
        )

        loss.backward()

        optimiser.step()

    model.eval()

    return model, (x_scaler, y_scaler)


# ==========================================
# CNN PREDICTION
# ==========================================

def cnn_predict(cnn_result, X):
    model, scalers = cnn_result

    if model is None:
        return None

    x_scaler, y_scaler = scalers

    X_scaled = x_scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    ).unsqueeze(1)

    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    predictions = y_scaler.inverse_transform(
        predictions
    ).ravel()

    return predictions


# ==========================================
# CNN INPUT GRADIENTS
# ==========================================

def cnn_input_gradient(cnn_result, X):
    model, scalers = cnn_result

    if model is None:
        return None

    x_scaler, y_scaler = scalers

    X = np.atleast_2d(
        np.asarray(X, dtype=float)
    )

    X_scaled = x_scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        requires_grad=True
    ).unsqueeze(1)

    model.eval()

    predictions = model(X_tensor)

    gradients = torch.autograd.grad(
        predictions.sum(),
        X_tensor
    )[0]

    # Remove the channel dimension.
    gradients = gradients.squeeze(1).detach().numpy()

    # Convert gradients to original scales.
    gradients *= float(y_scaler.scale_[0])

    gradients /= x_scaler.scale_.reshape(1, -1)

    return gradients


# ==========================================
# GRADIENT-GUIDED CANDIDATES
# ==========================================

def generate_gradient_candidates(
    cnn_result,
    X,
    y,
    n_candidates,
    rng
):
    dimension = X.shape[1]

    if cnn_result[0] is None or n_candidates <= 0:
        return np.empty((0, dimension))

    n_starts = min(
        max(4, int(np.sqrt(n_candidates))),
        len(X)
    )

    # Start from the best observed points.
    start_indices = np.argsort(y)[:n_starts]

    starts = X[start_indices].copy()

    points = []

    for start in starts:
        x = start.copy()

        for _ in range(GRADIENT_STEPS):
            gradient = cnn_input_gradient(
                cnn_result,
                x.reshape(1, -1)
            )

            if gradient is None:
                break

            if not np.all(np.isfinite(gradient)):
                break

            gradient = gradient[0]

            gradient_norm = np.linalg.norm(gradient)

            if gradient_norm < 1e-10:
                break

            # Follow the negative gradient.
            x = np.clip(
                x - GRADIENT_STEP_SIZE
                * gradient / gradient_norm,
                0.0,
                1.0
            )

            points.append(x.copy())

            if len(points) >= n_candidates:
                break

        if len(points) >= n_candidates:
            break

    # Supplement with random candidates.
    remaining = n_candidates - len(points)

    if remaining > 0:
        supplement = rng.uniform(
            0.0,
            1.0,
            size=(remaining, dimension)
        )

        points.extend(supplement)

    return np.asarray(points[:n_candidates])


def generate_candidates(n_candidates, dimension, rng):
    return rng.uniform(
        0.0,
        1.0,
        size=(n_candidates, dimension)
    )


# ==========================================
# ACQUISITION SCALING
# ==========================================

def minmax_scale(values):
    values = np.asarray(values, dtype=float)

    low = np.min(values)
    high = np.max(values)

    if high - low < 1e-12:
        return np.zeros_like(values)

    return (values - low) / (high - low)



# ==========================================
# RANDOMISED HYPERPARAMETER TUNING
# ==========================================

def tune_hyperparameters(X, y, seed=TUNING_SEED):
    """
    Select surrogate hyperparameters using shuffled K-fold CV.
    Each model is scored only on held-out folds; the selected settings
    are subsequently refitted on the complete available data.
    """
    rng = np.random.default_rng(seed)
    n_splits = min(TUNING_FOLDS, len(y))

    if n_splits < 2:
        raise ValueError("At least two observations are required for tuning.")

    cv = list(KFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    ).split(X))

    # Gaussian Process: minimise held-out mean squared error.
    gp_space = [
        (length_scale, noise, nu)
        for length_scale in (0.25, 0.5, 1.0, 2.0)
        for noise in (1e-6, 1e-4, 1e-3)
        for nu in (1.5, 2.5)
    ]

    gp_indices = rng.choice(
        len(gp_space), size=min(GP_TUNING_TRIALS, len(gp_space)),
        replace=False
    )

    best_gp, best_gp_loss = None, np.inf

    for index in gp_indices:
        params = gp_space[int(index)]
        fold_losses = []

        for train_idx, valid_idx in cv:
            model, scaler = fit_gp(
                X[train_idx], y[train_idx],
                length_scale=params[0],
                noise_level=params[1],
                nu=params[2]
            )

            predictions = model.predict(scaler.transform(X[valid_idx]))
            fold_losses.append(mean_squared_error(y[valid_idx], predictions))
        
        score = float(np.mean(fold_losses))

        if score < best_gp_loss:
            best_gp_loss, best_gp = score, params

    # SVM: classify promising observations and maximise held-out
    # balanced accuracy. Skip tuning if a fold cannot contain both classes.
    threshold = np.quantile(y, PROMISING_FRACTION)
    labels = (y <= threshold).astype(int)

    svm_space = [
        (C, gamma)
        for C in (0.1, 1.0, 10.0, 100.0)
        for gamma in ("scale", 0.1, 1.0)
    ]

    svm_indices = rng.choice(
        len(svm_space), size=min(SVM_TUNING_TRIALS, len(svm_space)),
        replace=False
    )

    best_svm, best_svm_score = (1.0, "scale"), -np.inf
    class_counts = np.bincount(labels, minlength=2)
    svm_cv = None

    if class_counts.min() >= 2:
        svm_splits = min(TUNING_FOLDS, int(class_counts.min()))
        svm_cv = list(StratifiedKFold(
            n_splits=svm_splits, shuffle=True, random_state=seed
        ).split(X, labels))

    if svm_cv is not None:
        for index in svm_indices:
            params = svm_space[int(index)]
            fold_scores = []

            for train_idx, valid_idx in svm_cv:
                model = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel="rbf", C=params[0], gamma=params[1],
                        class_weight="balanced"
                    )
                )

                model.fit(X[train_idx], labels[train_idx])
                fold_scores.append(balanced_accuracy_score(
                    labels[valid_idx], model.predict(X[valid_idx])
                ))

            score = float(np.mean(fold_scores))

            if score > best_svm_score:
                best_svm_score, best_svm = score, params

    # CNN: a deliberately small search to limit computation with 16 points.
    # Select settings by held-out MSE, using fewer epochs during CV.
    cnn_space = [
        (0.0005, 0.001, 8),
        (0.001, 0.001, 16),
        (0.003, 0.0001, 16),
        (0.001, 0.01, 32)
    ]

    cnn_indices = rng.choice(
        len(cnn_space), size=min(CNN_TUNING_TRIALS, len(cnn_space)),
        replace=False
    )

    best_cnn, best_cnn_loss = (
        (CNN_LEARNING_RATE, CNN_WEIGHT_DECAY, CNN_HIDDEN_SIZE), np.inf
    )

    for index in cnn_indices:
        params = cnn_space[int(index)]
        fold_losses = []

        for fold_number, (train_idx, valid_idx) in enumerate(cv):
            model, scalers = fit_cnn_surrogate(
                X[train_idx], y[train_idx],
                seed=seed + fold_number,
                epochs=150,
                learning_rate=params[0],
                weight_decay=params[1],
                hidden_size=params[2]
            )

            if model is None:
                fold_losses = []
                break
           
            predictions = cnn_predict((model, scalers), X[valid_idx])
            fold_losses.append(mean_squared_error(y[valid_idx], predictions))
       
        if fold_losses:
            score = float(np.mean(fold_losses))
            
            if score < best_cnn_loss:
                best_cnn_loss, best_cnn = score, params

    return {
        "gp": best_gp,
        "svm": best_svm,
        "cnn": best_cnn,
        "cv_scores": {
            "gp_mse": best_gp_loss,
            "svm_balanced_accuracy": (
                best_svm_score if np.isfinite(best_svm_score) else None
            ),
            "cnn_mse": (
                best_cnn_loss if np.isfinite(best_cnn_loss) else None
            )
        }
    }


# ==========================================
# HYBRID BAYESIAN OPTIMISATION
# ==========================================

def propose_next(
    X,
    y,
    maximise=False,
    n_candidates=N_CANDIDATES,
    seed=SEED
):
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X and y have incompatible shapes.")

    if len(y) < 2:
        raise ValueError("At least two observations are required.")

    if n_candidates < 1:
        raise ValueError("n_candidates must be positive.")

    rng = np.random.default_rng(seed)

    # Convert maximisation into minimisation.
    target = (
        -np.asarray(y, dtype=float)
        if maximise
        else np.asarray(y, dtype=float)
    )

    # Tune model hyperparameters using cross-validation, then refit
    # each surrogate on all observations using the selected settings.
    tuning = tune_hyperparameters(X, target, seed=seed + TUNING_SEED)

    gp_params = tuning["gp"]
    gp, gp_scaler = fit_gp(
        X, target,
        length_scale=gp_params[0],
        noise_level=gp_params[1],
        nu=gp_params[2]
    )

    linear_model = fit_linear_model(X, target)

    svm_params = tuning["svm"]
    svm_result = fit_svm_classifier(
        X, target, C=svm_params[0], gamma=svm_params[1]
    )

    cnn_params = tuning["cnn"]
    cnn_result = fit_cnn_surrogate(
        X, target,
        seed=seed,
        learning_rate=cnn_params[0],
        weight_decay=cnn_params[1],
        hidden_size=cnn_params[2]
    )

    # Generate mixed candidate pool.
    n_gradient = (
        int(n_candidates * GRADIENT_FRACTION)
        if cnn_result[0] is not None
        else 0
    )

    n_random = n_candidates - n_gradient

    random_candidates = generate_candidates(
        n_random,
        X.shape[1],
        rng
    )

    if n_gradient > 0:
        gradient_candidates = generate_gradient_candidates(
            cnn_result,
            X,
            target,
            n_gradient,
            rng
        )

        candidates = np.vstack([
            random_candidates,
            gradient_candidates
        ])

    else:
        candidates = random_candidates

    # Remove previously evaluated candidates.
    tree = cKDTree(X)

    distances, _ = tree.query(
        candidates,
        k=1
    )

    candidates = candidates[distances > 1e-10]

    if len(candidates) == 0:
        raise RuntimeError("No candidates remain.")

    # 1. Gaussian Process Expected Improvement.
    y_best = np.min(target)

    ei = expected_improvement(
        candidates,
        gp,
        gp_scaler,
        y_best
    )

    ei_score = minmax_scale(ei)

    # 2. Linear regression.
    linear_prediction = linear_model.predict(candidates)

    linear_score = minmax_scale(
        np.max(linear_prediction) - linear_prediction
    )

    # 3. SVM promising-region score.
    svm_score = svm_promising_score(
        svm_result,
        candidates
    )

    # 4. CNN predicted objective.
    cnn_prediction = cnn_predict(
        cnn_result,
        candidates
    )

    if cnn_prediction is None:
        cnn_score = np.zeros(len(candidates))

        cnn_weight = 0.0

    else:
        cnn_score = minmax_scale(
            np.max(cnn_prediction) - cnn_prediction
        )

        cnn_weight = WEIGHT_CNN

    # Normalise active acquisition weights.
    weights = np.array([
        WEIGHT_EI,
        WEIGHT_LINEAR,
        WEIGHT_SVM,
        cnn_weight
    ], dtype=float)

    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid acquisition weights.")

    weights /= weights.sum()

    # Combined acquisition score.
    score = (
        weights[0] * ei_score
        + weights[1] * linear_score
        + weights[2] * svm_score
        + weights[3] * cnn_score
    )

    best_idx = int(np.argmax(score))

    return (
        candidates[best_idx],
        float(score[best_idx]),
        svm_result,
        cnn_result,
        tuning
    )


# ==========================================
# QUERY FORMATTING
# ==========================================

def format_query(x):
    return "-".join(
        f"{value:.6f}"
        for value in x
    )


# ==========================================
# MAIN EXECUTION
# ==========================================

query_results = []

for i in range(1, 9):
    print("\n" + "=" * 50)
    print(f"FUNCTION {i}")
    print("=" * 50)

    try:
        X, y = load_all_data(i)

        print(f"Number of observations: {len(y)}")

        if len(y) != EXPECTED_OBSERVATIONS:
            print(
                f"WARNING: Expected "
                f"{EXPECTED_OBSERVATIONS} observations, "
                f"but found {len(y)}."
            )

        # Identify current best observation.
        if MAXIMISE:
            best_idx = np.argmax(y)
        else:
            best_idx = np.argmin(y)

        best_x = X[best_idx]
        best_y = y[best_idx]

        print(f"Current best output: {best_y:.6f}")

        # Generate next query.
        x_next, score, svm_result, cnn_result, tuning = propose_next(
            X,
            y,
            maximise=MAXIMISE,
            n_candidates=N_CANDIDATES,
            seed=SEED + i
        )

        query = format_query(x_next)

        print("\nCurrent best input:")
        print(best_x)

        print("\nRecommended next input:")
        print(x_next)

        print("\nFormatted query:")
        print(query)

        print(f"\nAcquisition score: {score:.6f}")

        print("\nSelected hyperparameters:")
        print(f"  GP (length scale, noise, nu): {tuning['gp']}")
        print(f"  SVM (C, gamma): {tuning['svm']}")
        print(f"  CNN (learning rate, weight decay, hidden size): {tuning['cnn']}")
        print(f"  Cross-validation scores: {tuning['cv_scores']}")

        print(
            "SVM:",
            "Fitted" if svm_result is not None
            else "Unavailable"
        )

        print(
            "PyTorch CNN:",
            "Fitted" if cnn_result[0] is not None
            else "Unavailable"
        )

        query_results.append({
            "function": i,
            "observations": len(y),
            "best_output": best_y,
            "query": query,
            "acquisition_score": score,
            "gp_hyperparameters": str(tuning["gp"]),
            "svm_hyperparameters": str(tuning["svm"]),
            "cnn_hyperparameters": str(tuning["cnn"]),
            "cv_scores": str(tuning["cv_scores"])
        })

    except Exception as exc:
        print(f"ERROR: Function {i} failed: {exc}")


# ==========================================
# SAVE AND DISPLAY RESULTS
# ==========================================

if query_results:
    results_df = pd.DataFrame(query_results)
    print("\n" + "=" * 50)
    print("EIGHTH ROUND QUERY SUMMARY")
    print("=" * 50)

    print(
        results_df[
            [
                "function",
                "observations",
                "best_output",
                "query"
            ]
        ].to_string(index=False)
    )

    results_df.to_csv(
        "eighth_round_queries.csv",
        index=False
    )

    print("\nResults saved to eighth_round_queries.csv")

else:
    print("No queries were successfully generated.")
