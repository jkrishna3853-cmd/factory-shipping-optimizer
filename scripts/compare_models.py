import time
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.train import fit_and_score

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

experiments = [
    ("LinReg (ship mode only)", LinearRegression(), [], ["Ship Mode"]),
    ("LinReg (9 features)", LinearRegression(),
     NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    ("Random Forest (9 features)",
     RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                           random_state=42, n_jobs=-1),
     NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    ("Gradient Boosting (9 features)",
     GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                               max_depth=3, random_state=42),
     NUMERIC_FEATURES, CATEGORICAL_FEATURES),
]

rows = []
for name, model, num, cat in experiments:
    start = time.time()
    _, row = fit_and_score(name, model, train, test, num, cat)
    row["seconds"] = time.time() - start
    rows.append(row)

result = pd.DataFrame(rows).round(3)
print(result.to_string(index=False))
print("best test MAE:", result.loc[result["test_MAE"].idxmin(), "model"])