import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.evaluation import regression_metrics
from src.features import (
    build_preprocessor, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

experiments = {
    "1. Predict the mean": (DummyRegressor(strategy="mean"),
                            NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    "2. LinReg: ship mode only": (LinearRegression(), [], ["Ship Mode"]),
    "3. LinReg: all 9 features": (LinearRegression(),
                                  NUMERIC_FEATURES, CATEGORICAL_FEATURES),
}

rows = []
for name, (model, num, cat) in experiments.items():
    pipe = Pipeline([("prep", build_preprocessor(num, cat)), ("model", model)])
    pipe.fit(train[num + cat], train[TARGET])
    tr = regression_metrics(train[TARGET], pipe.predict(train[num + cat]))
    te = regression_metrics(test[TARGET], pipe.predict(test[num + cat]))
    rows.append({"model": name,
                 "train_MAE": tr["MAE"], "test_MAE": te["MAE"],
                 "train_RMSE": tr["RMSE"], "test_RMSE": te["RMSE"],
                 "train_R2": tr["R2"], "test_R2": te["R2"]})

print(pd.DataFrame(rows).round(3).to_string(index=False))
print("target std on test set:", round(test[TARGET].std(), 3))