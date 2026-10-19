import pandas as pd
from src.features import (
    add_time_features, split_by_order,
    TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES, LEAKY_COLUMNS,
)

df = pd.read_csv("data/processed/orders_geo.csv")
df = add_time_features(df)
train, test = split_by_order(df)

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
print("train rows:", len(train), " test rows:", len(test))
print("orders in BOTH train and test:", len(set(train["Order ID"]) & set(test["Order ID"])))
print("target mean  train / test:", round(train[TARGET].mean(), 3), "/", round(test[TARGET].mean(), 3))
print("ship mode share:")
print(pd.concat(
    [train["Ship Mode"].value_counts(normalize=True),
     test["Ship Mode"].value_counts(normalize=True)],
    axis=1, keys=["train", "test"]).round(3))
print("features used:", features)
print("leaky columns excluded:", [c for c in LEAKY_COLUMNS if c not in features])
print("missing feature values:", int(train[features].isna().sum().sum()))