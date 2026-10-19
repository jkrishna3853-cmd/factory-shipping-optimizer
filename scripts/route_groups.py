import pandas as pd

df = pd.read_csv("data/processed/orders_geo.csv")

# What ship mode alone would predict, and how far each row is from it
df["expected"] = df.groupby("Ship Mode")["lead_adj"].transform("mean")
df["excess"] = df["lead_adj"] - df["expected"]

g = (
    df.groupby(["Factory", "Product Name", "Region"])
    .agg(
        rows=("lead_adj", "size"),
        orders=("Order ID", "nunique"),
        mean_lead=("lead_adj", "mean"),
        std_lead=("lead_adj", "std"),
        mean_excess=("excess", "mean"),
        share_standard=("Ship Mode", lambda s: (s == "Standard Class").mean()),
        mean_km=("distance_km", "mean"),
        mean_units=("Units", "mean"),
    )
    .reset_index()
)

g.to_csv("data/processed/route_groups.csv", index=False)

print("groups:", len(g), " smallest group (rows):", g["rows"].min())
print("groups with fewer than 30 rows:", int((g["rows"] < 30).sum()))
print("spread of mean_excess:", g["mean_excess"].describe().round(3).to_dict())

cols = ["Product Name", "Region", "rows", "mean_lead", "mean_excess", "share_standard"]
big = g[g["rows"] >= 30].sort_values("mean_excess")
print("\nFASTEST groups relative to their ship-mode mix:")
print(big.head(6)[cols].round(2).to_string(index=False))
print("\nSLOWEST groups relative to their ship-mode mix:")
print(big.tail(6)[cols].round(2).to_string(index=False))