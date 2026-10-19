import joblib
import pandas as pd

from src.features import add_time_features
from src.recommendations import top_recommendations

bundle = joblib.load("models/lead_time_gb.joblib")
pipe = bundle["pipeline"]
features = bundle["numeric"] + bundle["categorical"]

df = add_time_features(pd.read_csv("data/processed/orders_geo.csv"))
factories = pd.read_csv("data/external/factories.csv")
pd.set_option("display.width", 250)

rows = []
for product in sorted(df["Product Name"].unique()):
    scored, _ = top_recommendations(df, pipe, features, factories, product)
    best = scored[~scored["is_current"]].iloc[0]   # best-scoring alternative
    rows.append({
        "product": product,
        "current": scored.loc[scored["is_current"], "factory"].iloc[0],
        "orders": int(best["rows"]),
        "best_alt": best["factory"],
        "days_saved": best["days_saved"],
        "saved_pct": best["reduction_pct"],
        "in_range": best["share_in_seen_range"],
        "confidence": best["confidence"],
        "recommended": bool(best["recommend"]),
    })

res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

print("\nCoverage = share of products with a confident recommendation")
for t in [0.0, 0.05, 0.10, 0.25, 0.50]:
    ok = (res["days_saved"] >= t) & (res["confidence"] >= 0.5)
    print(f"  minimum saving {t:>4} days -> {ok.mean():.0%} ({int(ok.sum())} of {len(res)})")