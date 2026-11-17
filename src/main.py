import numpy as np
from scipy.spatial import cKDTree

import acquisition
import data
import gaussian_process
import linear_regression
import svm_model
from neural_network import input_gradients, prediction, training

_EXPECTED_OBSERVATIONS = 15
_GRADIENT_FRACTION = 0.30
_N_CANDIDATES = 100000
_SEED = 42

_MAXIMISE = False

_WEIGHT_EI = 0.50
_WEIGHT_SVM = 0.20
_WEIGHT_LINEAR = 0.05
_WEIGHT_CNN = 0.25


def format_query(x):
    return "-".join(f"{value:.6f}" for value in x)


def propose_next(
    X, y, maximise: bool = False, n_candidates: int = _N_CANDIDATES, seed: int = _SEED
):
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X and y have incompatible shapes.")

    if len(y) < 2:
        raise ValueError("At least two observations are required.")

    if n_candidates < 1:
        raise ValueError("n_candidates must be positive.")

    rng = np.random.default_rng(seed)

    # Convert maximisation into minimisation.
    target = -np.asarray(y, dtype=float) if maximise else np.asarray(y, dtype=float)

    # Fit all surrogate models.
    gp, gp_scaler = gaussian_process.fit_gp(X, target, seed=seed)

    linear_model = linear_regression.fit_linear_model(X, target)

    svm_result = svm_model.fit_svm_classifier(X, target, seed=seed)

    cnn_result = training.fit_cnn_surrogate(X, target, seed=seed)

    # Generate mixed candidate pool.
    n_gradient = (
        int(n_candidates * _GRADIENT_FRACTION) if cnn_result[0] is not None else 0
    )

    n_random = n_candidates - n_gradient

    random_candidates = input_gradients.generate_candidates(n_random, X.shape[1], rng)

    if n_gradient > 0:
        gradient_candidates = input_gradients.generate_gradient_candidates(
            cnn_result, X, target, n_gradient, rng
        )

        candidates = np.vstack([random_candidates, gradient_candidates])

    else:
        candidates = random_candidates

    # Remove previously evaluated candidates.
    tree = cKDTree(X)

    distances, _ = tree.query(candidates, k=1)

    candidates = candidates[distances > 1e-10]

    if len(candidates) == 0:
        raise RuntimeError("No candidates remain.")

    # 1. Gaussian Process Expected Improvement.
    y_best = np.min(target)

    ei = gaussian_process.expected_improvement(candidates, gp, gp_scaler, y_best)

    ei_score = acquisition.minmax_scale(ei)

    # 2. Linear regression.
    linear_prediction = linear_model.predict(candidates)

    linear_score = acquisition.minmax_scale(
        np.max(linear_prediction) - linear_prediction
    )

    # 3. SVM promising-region score.
    svm_score = svm_model.svm_promising_score(svm_result, candidates)

    # 4. CNN predicted objective.
    cnn_prediction = prediction.cnn_predict(cnn_result, candidates)

    if cnn_prediction is None:
        cnn_score = np.zeros(len(candidates))

        cnn_weight = 0.0

    else:
        cnn_score = acquisition.minmax_scale(np.max(cnn_prediction) - cnn_prediction)

        cnn_weight = _WEIGHT_CNN

    # Normalise active acquisition weights.
    weights = np.array(
        [_WEIGHT_EI, _WEIGHT_LINEAR, _WEIGHT_SVM, cnn_weight], dtype=float
    )

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

    return (candidates[best_idx], float(score[best_idx]), svm_result, cnn_result)


# ==========================================
# MAIN EXECUTION
# ==========================================

query_results = []

for i in range(1, 9):
    print("\n" + "=" * 50)
    print(f"FUNCTION {i}")
    print("=" * 50)

    try:
        X, y = data.load_all_data(i)

        print(f"Number of observations: {len(y)}")

        if len(y) != _EXPECTED_OBSERVATIONS:
            print(
                f"WARNING: Expected "
                f"{_EXPECTED_OBSERVATIONS} observations, "
                f"but found {len(y)}."
            )

        # Identify current best observation.
        if _MAXIMISE:
            best_idx = np.argmax(y)

        else:
            best_idx = np.argmin(y)

        best_x = X[best_idx]
        best_y = y[best_idx]

        print(f"Current best output: {best_y:.6f}")

        # Generate next query.
        x_next, score, svm_result, cnn_result = propose_next(
            X, y, maximise=_MAXIMISE, n_candidates=_N_CANDIDATES, seed=_SEED + i
        )

        query = format_query(x_next)

        print("\nCurrent best input:")
        print(best_x)

        print("\nRecommended next input:")
        print(x_next)

        print("\nFormatted query:")
        print(query)

        print(f"\nAcquisition score: {score:.6f}")

        print("SVM:", "Fitted" if svm_result is not None else "Unavailable")

        print("PyTorch CNN:", "Fitted" if cnn_result[0] is not None else "Unavailable")

        query_results.append(
            {
                "function": i,
                "observations": len(y),
                "best_output": best_y,
                "query": query,
                "acquisition_score": score,
            }
        )

    except Exception as exc:
        print(f"ERROR: Function {i} failed: {exc}")
