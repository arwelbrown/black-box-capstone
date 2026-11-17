import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


def fit_gp(X, y, seed: int):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dimension = X.shape[1]

    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        length_scale=np.ones(dimension), length_scale_bounds="fixed", nu=2.5
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")

    model = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, optimizer=None, random_state=seed
    )

    model.fit(X_scaled, y)

    return model, scaler


def expected_improvement(X_candidates, model, scaler, y_best):
    X_scaled = scaler.transform(X_candidates)

    mean, std = model.predict(X_scaled, return_std=True)

    std = np.maximum(std, 1e-12)

    improvement = y_best - mean

    z = improvement / std

    ei = improvement * norm.cdf(z) + std * norm.pdf(z)

    return np.maximum(ei, 0.0)
