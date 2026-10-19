from sklearn.pipeline import Pipeline

from src.evaluation import regression_metrics
from src.features import build_preprocessor, TARGET


def fit_and_score(name, model, train, test, numeric, categorical):
    """Fit one model and return (fitted pipeline, a row of train/test metrics)."""
    cols = numeric + categorical
    pipe = Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("model", model),
    ])
    pipe.fit(train[cols], train[TARGET])
    tr = regression_metrics(train[TARGET], pipe.predict(train[cols]))
    te = regression_metrics(test[TARGET], pipe.predict(test[cols]))
    row = {
        "model": name,
        "train_MAE": tr["MAE"], "test_MAE": te["MAE"],
        "train_RMSE": tr["RMSE"], "test_RMSE": te["RMSE"],
        "train_R2": tr["R2"], "test_R2": te["R2"],
    }
    return pipe, row