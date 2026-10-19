from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from src.train import fit_and_score

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
)
pipe, row = fit_and_score(
    "Gradient Boosting (final)", model, train, test,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
print(pd.DataFrame([row]).round(3).to_string(index=False))

# Save the model together with the feature lists it expects
Path("models").mkdir(exist_ok=True)
path = "models/lead_time_gb.joblib"
joblib.dump(
    {"pipeline": pipe, "numeric": NUMERIC_FEATURES,
     "categorical": CATEGORICAL_FEATURES},
    path,
)
loaded = joblib.load(path)["pipeline"]
assert np.allclose(loaded.predict(test[cols].head(5)),
                   pipe.predict(test[cols].head(5)))
print("saved", path, "- reloads and predicts identically")

# Permutation importance on the test set (shuffles the ORIGINAL columns)
result = permutation_importance(
    pipe, test[cols], test[TARGET],
    scoring="neg_mean_absolute_error",
    n_repeats=10, random_state=42, n_jobs=-1,
)
imp = pd.DataFrame({
    "feature": cols,
    "MAE_increase_days": result.importances_mean,
    "std": result.importances_std,
}).sort_values("MAE_increase_days", ascending=False)
print(imp.round(4).to_string(index=False))