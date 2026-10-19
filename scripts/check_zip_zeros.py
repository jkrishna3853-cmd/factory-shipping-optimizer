import pandas as pd

df = pd.read_csv("data/processed/orders_clean.csv", dtype={"Postal Code": str})
us = df[df["Country/Region"] == "United States"]
short = us[us["Postal Code"].str.len() == 4]

print("rows with a 4-digit ZIP:", len(short))
print(short["State/Province"].value_counts())
print("sample:", short["Postal Code"].drop_duplicates().head(8).tolist())