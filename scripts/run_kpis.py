import joblib
import pandas as pd

from src.features import add_time_features
from src.kpis import best_options, break_even_cost, compute_kpis, profit_sweep

bundle = joblib.load("models/lead_time_gb.joblib")
pipe = bundle["pipeline"]
features = bundle["numeric"] + bundle["categorical"]

df = add_time_features(pd.read_csv("data/processed/orders_geo.csv"))
factories = pd.read_csv("data/external/factories.csv")
pd.set_option("display.width", 250)

opts = best_options(df, pipe, features, factories)
opts["break_even_cost"] = break_even_cost(opts)
sweep = profit_sweep(opts)

cols = ["product", "best_alt", "orders", "days_saved",
        "confidence", "recommended", "break_even_cost"]
print(opts[cols].round(5).to_string(index=False))
print("\nProfit sweep (illustrative costs):")
print(sweep.to_string(index=False))
print("\nKPIs:")
for name, value in compute_kpis(opts, sweep).items():
    print(f"  {name}: {value:.4f}")