import pandas as pd

from src.geo import haversine_km


def simulate_reassignment(
    df, pipeline, features, factories, product, region,
    ship_mode=None, cost_per_unit_km=0.0,
):
    """Predict lead time if `product` were made at each candidate factory.

    Labels: observed_* = from data, pred_* = model output,
    profit_impact_assumed = needs cost_per_unit_km (an ASSUMPTION, 0 = not estimated).
    """
    rows = df[df["Product Name"] == product]
    if region is not None:
        rows = rows[rows["Region"] == region]
    if ship_mode is not None:
        rows = rows[rows["Ship Mode"] == ship_mode]
    if rows.empty:
        raise ValueError("No orders match this product / region / ship mode")

    current = rows["Factory"].iloc[0]
    seen = df.loc[df["Product Name"] == product, "distance_km"]
    seen_min, seen_max = seen.min(), seen.max()

    base = rows[features].copy()
    pred_current = pipeline.predict(base)
    dist_current = rows["distance_km"].to_numpy()
    units = rows["Units"].to_numpy()

    out = []
    for _, fac in factories.iterrows():
        new_dist = haversine_km(
            fac["latitude"], fac["longitude"], rows["cust_lat"], rows["cust_lon"]
        ).to_numpy()
        changed = base.copy()
        changed["distance_km"] = new_dist
        pred_new = pipeline.predict(changed)

        delta = pred_new.mean() - pred_current.mean()
        out.append({
            "factory": fac["factory"],
            "is_current": fac["factory"] == current,
            "rows": len(rows),
            "observed_mean_lead": rows["lead_adj"].mean(),
            "pred_current": pred_current.mean(),
            "pred_new": pred_new.mean(),
            "delta_days": delta,
            "reduction_pct": -delta / pred_current.mean() * 100,
            "mean_distance_km": new_dist.mean(),
            "delta_distance_km": (new_dist - dist_current).mean(),
            "share_in_seen_range": ((new_dist >= seen_min) & (new_dist <= seen_max)).mean(),
            "observed_gross_profit": rows["Gross Profit"].sum(),
            "unit_km_change": (units * (new_dist - dist_current)).sum(),
            "profit_impact_assumed": -cost_per_unit_km * (units * (new_dist - dist_current)).sum(),
        })
    return pd.DataFrame(out).sort_values("pred_new").reset_index(drop=True)