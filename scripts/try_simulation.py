import joblib
import pandas as pd

from src.features import add_time_features
from src.simulation import simulate_reassignment

bundle = joblib.load("models/lead_time_gb.joblib")
pipe = bundle["pipeline"]
features = bundle["numeric"] + bundle["categorical"]

df = add_time_features(pd.read_csv("data/processed/orders_geo.csv"))
factories = pd.read_csv("data/external/factories.csv")

pd.set_option("display.width", 200)
cols = ["factory", "is_current", "rows", "pred_current", "pred_new",
        "delta_days", "reduction_pct", "delta_distance_km", "share_in_seen_range"]

for product, region, mode in [
    ("Wonka Bar - Milk Chocolate", "Interior", "Standard Class"),
    ("Wonka Bar - Nutty Crunch Surprise", "Pacific", "Standard Class"),
    ("Wonka Gum", "Atlantic", "Standard Class"),
]:
    res = simulate_reassignment(df, pipe, features, factories, product, region, mode)
    print(f"\n{product} | {region} | {mode}  (observed mean lead: "
          f"{res['observed_mean_lead'].iloc[0]:.2f} days)")
    print(res[cols].round(3).to_string(index=False))