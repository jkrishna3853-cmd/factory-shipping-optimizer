import pandas as pd

df = pd.read_csv("data/raw/Nassau Candy Distributor.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y")
df["lead_raw"] = (df["Ship Date"] - df["Order Date"]).dt.days

# whole-year band, found in the previous step
df["band"] = ((df["lead_raw"] - 178) / 365.25).round().astype(int)

# subtract the whole calendar years from each Ship Date
df["ship_adj"] = df["Ship Date"]
for b in sorted(df["band"].unique()):
    mask = df["band"] == b
    df.loc[mask, "ship_adj"] = df.loc[mask, "Ship Date"] - pd.DateOffset(years=int(b))
df["lead_adj"] = (df["ship_adj"] - df["Order Date"]).dt.days

print(df["lead_adj"].describe())
print(df["lead_adj"].value_counts().sort_index())
print(df.groupby("Ship Mode")["lead_adj"].agg(["count", "mean", "min", "max"]))
print(pd.crosstab(df["Ship Mode"], df["band"]))