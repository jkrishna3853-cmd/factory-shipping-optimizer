import numpy as np
import pandas as pd

from src.clustering import add_excess

df = add_excess(pd.read_csv("data/processed/orders_geo.csv"))
g = (
    df.groupby(["Factory", "Product Name", "Region"])
    .agg(rows=("excess", "size"), mean_excess=("excess", "mean"),
         sd=("excess", "std"))
    .reset_index()
)
g = g[g["rows"] >= 30].copy()
g["std_error"] = g["sd"] / np.sqrt(g["rows"])
g["z"] = g["mean_excess"] / g["std_error"]

THRESHOLD = 3.1  # about 5% overall risk of a false alarm across 25 groups
g["significant"] = g["z"].abs() > THRESHOLD
g.to_csv("data/processed/route_significance.csv", index=False)

n = len(g)
print("groups tested:", n)
print("groups with |z| > 2:", int((g["z"].abs() > 2).sum()),
      " (expected by pure chance:", round(n * 0.0455, 1), ")")
print("groups with |z| >", THRESHOLD, ":", int(g["significant"].sum()))
print(g.reindex(g["z"].abs().sort_values(ascending=False).index)
        .head(6)[["Product Name", "Region", "rows", "mean_excess", "z"]]
        .round(2).to_string(index=False))