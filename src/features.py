import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit

TARGET = "lead_adj"

NUMERIC_FEATURES = [
    "Units", "Sales", "Cost", "distance_km", "order_month", "order_dayofweek",
]
CATEGORICAL_FEATURES = ["Ship Mode", "Region", "Product Name"]

# Never allowed as model inputs (they contain or reveal the target)
LEAKY_COLUMNS = ["Ship Date", "lead_raw", "year_shift", "ship_date_adj", "lead_adj"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month and day-of-week of the order (known at order time)."""
    out = df.copy()
    out["Order Date"] = pd.to_datetime(out["Order Date"])
    out["order_month"] = out["Order Date"].dt.month
    out["order_dayofweek"] = out["Order Date"].dt.dayofweek
    return out


def split_by_order(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split so that every order lands entirely in train or entirely in test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["Order ID"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

def build_preprocessor(numeric=None, categorical=None) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical ones."""
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
