import pandas as pd
from src.geo import load_postal_coordinates, attach_customer_coordinates

df = pd.read_csv("data/processed/orders_clean.csv", dtype={"Postal Code": str})
coords = load_postal_coordinates(
    "data/external/geonames/US.txt", "data/external/geonames/CA.txt"
)
out = attach_customer_coordinates(df, coords)

print("rows before:", len(df), " rows after:", len(out))
missing = out["cust_lat"].isna()
print("rows without coordinates:", int(missing.sum()),
      f"({missing.mean():.2%})")
print(out.groupby("Country/Region")["cust_lat"].apply(lambda s: s.isna().sum()))
print("unmatched keys:",
      out.loc[missing, ["Country/Region", "State/Province", "postal_key"]]
         .drop_duplicates().head(20).to_string())
print(out[["cust_lat", "cust_lon"]].describe())