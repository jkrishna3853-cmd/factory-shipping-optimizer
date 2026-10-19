import pandas as pd

df = pd.read_csv("data/processed/orders_geo.csv")

# Split distance into 5 equal-sized groups (quintiles)
df["dist_bin"] = pd.qcut(df["distance_km"], 5, labels=False)

print("Distance range of each group (km):")
print(df.groupby("dist_bin")["distance_km"].agg(["count", "min", "max"]).round(0))

print("\nMean lead time per distance group:")
print(df.groupby("dist_bin")[["lead_adj", "lead_raw"]].mean().round(2))

print("\nShare of each whole-year shift per distance group:")
print(pd.crosstab(df["dist_bin"], df["year_shift"], normalize="index").round(3))

print("\nMean adjusted lead time: distance group x ship mode:")
print(df.pivot_table(index="dist_bin", columns="Ship Mode",
                     values="lead_adj", aggfunc="mean").round(2))

print("\nSpearman (catches curved trends) distance vs lead_adj:",
      round(df["distance_km"].corr(df["lead_adj"], method="spearman"), 4))
print("Spearman distance vs lead_raw:",
      round(df["distance_km"].corr(df["lead_raw"], method="spearman"), 4))