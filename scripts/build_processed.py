from src.data_processing import (
    load_raw, parse_dates, add_lead_times, add_factory,
)

RAW = "data/raw/Nassau Candy Distributor.csv"
MAPPING = "data/external/product_factory.csv"
OUT = "data/processed/orders_clean.csv"

df = load_raw(RAW)
df = parse_dates(df)
df = add_lead_times(df)
df = add_factory(df, MAPPING)

df.to_csv(OUT, index=False)

print("rows:", len(df), " columns:", df.shape[1])
print(df[["Order Date", "Ship Date", "lead_raw", "year_shift",
          "lead_adj", "Factory"]].head())
print(df["lead_adj"].describe())
print(df["Factory"].value_counts())