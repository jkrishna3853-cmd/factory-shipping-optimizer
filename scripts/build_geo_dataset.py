import pandas as pd
from src.geo import (
    load_postal_coordinates, attach_customer_coordinates,
    add_factory_distance, haversine_km,
)

# --- Sanity check: New York -> Los Angeles is about 3,936 km ---
d = float(haversine_km(40.7128, -74.0060, 34.0522, -118.2437))
print("NYC -> LA:", round(d, 1), "km")
assert abs(d - 3936) < 15, "haversine formula looks wrong"
print("sanity check passed")

# --- Build the dataset ---
df = pd.read_csv("data/processed/orders_clean.csv", dtype={"Postal Code": str})
coords = load_postal_coordinates(
    "data/external/geonames/US.txt", "data/external/geonames/CA.txt"
)
factories = pd.read_csv("data/external/factories.csv")

df = attach_customer_coordinates(df, coords)
df = add_factory_distance(df, factories)
df.to_csv("data/processed/orders_geo.csv", index=False)
print("saved orders_geo.csv:", df.shape)

# --- Does distance relate to lead time? ---
print(df["distance_km"].describe().round(1))
print(df.groupby("Factory")["distance_km"].agg(["count", "mean", "min", "max"]).round(0))
print("overall correlation, distance vs lead_adj:",
      round(df["distance_km"].corr(df["lead_adj"]), 4))
print("correlation within each ship mode:")
print(df.groupby("Ship Mode").apply(
    lambda g: round(g["distance_km"].corr(g["lead_adj"]), 4)
))