from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_linear_model(X, y):
    model = make_pipeline(StandardScaler(), LinearRegression())

    model.fit(X, y)

    return model

