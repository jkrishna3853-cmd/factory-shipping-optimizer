import numpy as np
import pandas as pd

from src.recommendations import top_recommendations

# ILLUSTRATIVE grid (currency per unit per km): NOT from the data
COST_GRID = [0.0, 0.0001, 0.0002, 0.0005, 0.001]
PROFIT_TOLERANCE = 0.01  # a move may cost at most 1% of the product's gross profit


def best_options(df, pipeline, features, factories, priority=0.0, cfg=None):
    """For every product: its best-scoring alternative factory."""
    rows = []
    for product in sorted(df["Product Name"].unique()):
        scored, _ = top_recommendations(
            df, pipeline, features, factories, product,
            priority=priority, cfg=cfg)
        best = scored[~scored["is_current"]].iloc[0]
        rows.append({
            "product": product,
            "best_alt": best["factory"],
            "orders": int(best["rows"]),
            "days_saved": best["days_saved"],
            "reduction_pct": best["reduction_pct"],
            "confidence": best["confidence"],
            "recommended": bool(best["recommend"]),
            "unit_km_change": best["unit_km_change"],
            "gross_profit": best["observed_gross_profit"],
        })
    return pd.DataFrame(rows)


def profit_sweep(opts, grid=COST_GRID, tol=PROFIT_TOLERANCE):
    """Share of products staying within tolerance, for each assumed cost."""
    rows = []
    for c in grid:
        impact = -c * opts["unit_km_change"]
        ok = impact >= -tol * opts["gross_profit"]
        rows.append({"cost_per_unit_km": c, "share_within_tolerance": ok.mean()})
    return pd.DataFrame(rows)


def break_even_cost(opts, tol=PROFIT_TOLERANCE):
    """Cost per unit-km at which a move loses `tol` of gross profit (inf if no extra distance)."""
    extra = opts["unit_km_change"].clip(lower=0)
    safe = extra.replace(0, np.nan)
    return np.where(extra > 0, tol * opts["gross_profit"] / safe, np.inf)


def compute_kpis(opts, sweep):
    w = opts["orders"]
    rec = opts[opts["recommended"]]
    return {
        "Lead time reduction % (recommended moves)": (
            np.average(rec["reduction_pct"], weights=rec["orders"])
            if len(rec) else float("nan")),
        "Lead time reduction % (best available, NOT confident)":
            np.average(opts["reduction_pct"], weights=w),
        "Profit impact stability": sweep["share_within_tolerance"].mean(),
        "Scenario confidence (order-weighted)":
            np.average(opts["confidence"], weights=w),
        "Recommendation coverage": opts["recommended"].mean(),
    }