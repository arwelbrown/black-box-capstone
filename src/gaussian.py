def fit_gp(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dimension = X.shape[1]

    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(
            length_scale=np.ones(dimension),
            length_scale_bounds="fixed",
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=1e-5,
            noise_level_bounds="fixed",
        )
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        optimizer=None,
        random_state=SEED,
    )

    model.fit(X_scaled, y)

    return model, scaler