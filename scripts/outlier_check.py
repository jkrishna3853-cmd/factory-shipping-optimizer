import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.train import fit_and_score

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

CHECK_COLS = ["lead_adj", "Sales", "Units", "Cost", "Gross Profit"]


def outlier_mask(df, ref, k):
    """True for rows outside k*IQR in ANY checked column (bounds from `ref`)."""
    mask = pd.Series(False, index=df.index)
    for c in CHECK_COLS:
        q1, q3 = ref[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask |= (df[c] < q1 - k * iqr) | (df[c] > q3 + k * iqr)
    return mask


variants = {"keep all rows": None, "drop beyond 3.0 x IQR": 3.0,
            "drop beyond 1.5 x IQR": 1.5}

rows = []
for name, k in variants.items():
    sub = train if k is None else train[~outlier_mask(train, train, k)]
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    _, row = fit_and_score(name, model, sub, test,
                           NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    row["train_rows"] = len(sub)
    rows.append(row)

out = pd.DataFrame(rows)[["model", "train_rows", "test_MAE", "test_RMSE", "test_R2"]]
print(out.round(3).to_string(index=False))