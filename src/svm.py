def fit_svm_classifier(X, y):
    threshold = np.quantile(y, PROMISING_FRACTION)
    labels = (y <= threshold).astype(int)

    if len(np.unique(labels)) < 2:
        return None

    class_counts = np.bincount(labels, minlength=2)

    use_probability = class_counts.min() >= 5

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=use_probability,
            random_state=SEED,
        ),
    )

    model.fit(X, labels)

    return model, use_probability

def svm_promising_score(svm_result, candidates):
    if svm_result is None:
        return np.full(len(candidates), 0.5)

    model, use_probability = svm_result

    if use_probability:
        classes = model.named_steps["svc"].classes_
        positive_column = int(np.flatnonzero(classes == 1)[0])

        return model.predict_proba(candidates)[:, positive_column]

    decision = model.decision_function(candidates)
    decision = np.clip(decision, -30.0, 30.0)

    return 1.0 / (1.0 + np.exp(-decision))