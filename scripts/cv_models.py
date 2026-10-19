import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features import (
    build_preprocessor, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

train = pd.read_csv("data/processed/train.csv")
X = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = train[TARGET]
groups = train["Order ID"]          # keeps each order inside one fold
cv = GroupKFold(n_splits=5)

models = {
    "LinReg": LinearRegression(),
    "RandomForest leaf=5": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "RandomForest leaf=20": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=-1),
    "GradBoost": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
}

rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
    res = cross_validate(
        pipe, X, y, groups=groups, cv=cv,
        scoring={"mae": "neg_mean_absolute_error",
                 "rmse": "neg_root_mean_squared_error"},
    )
    rows.append({
        "model": name,
        "cv_MAE_mean": -res["test_mae"].mean(),
        "cv_MAE_std": res["test_mae"].std(),
        "cv_RMSE_mean": -res["test_rmse"].mean(),
        "cv_RMSE_std": res["test_rmse"].std(),
    })

print(pd.DataFrame(rows).round(3).to_string(index=False))