import numpy as np
import pandas as pd

from src.clustering import (
    add_excess, build_group_features, silhouette_by_k, fit_clusters,
)

df = add_excess(pd.read_csv("data/processed/orders_geo.csv"))
g = build_group_features(df, min_rows=30)
print("groups used:", len(g), "(of 53, those with at least 30 rows)")

real = silhouette_by_k(g)
print("silhouette by k:", {k: round(v, 3) for k, v in real.items()})
best_k = max(real, key=real.get)
best_real = real[best_k]
print("best k:", best_k, " silhouette:", round(best_real, 3))

# Honesty test: shuffle the excess values across rows and repeat
rng = np.random.default_rng(42)
null_best = []
for _ in range(20):
    shuffled = df.copy()
    shuffled["excess"] = rng.permutation(shuffled["excess"].values)
    gs = build_group_features(shuffled, min_rows=30)
    null_best.append(max(silhouette_by_k(gs).values()))
print("shuffled data, best silhouette: mean",
      round(float(np.mean(null_best)), 3), " max", round(float(np.max(null_best)), 3))
print("real silhouette beats all 20 shuffles:", best_real > max(null_best))

clustered = fit_clusters(g, best_k)
clustered.to_csv("data/processed/route_clusters.csv", index=False)

print("\nCluster summary:")
print(clustered.groupby("cluster").agg(
    groups=("rows", "size"), rows=("rows", "sum"),
    mean_excess=("mean_excess", "mean"), std_excess=("std_excess", "mean"),
).round(3))

print("\nGroups by cluster:")
print(clustered.sort_values(["cluster", "mean_excess"])[
    ["cluster", "Product Name", "Region", "rows", "mean_excess", "std_excess"]
].round(3).to_string(index=False))