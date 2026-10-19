import numpy as np

from src.simulation import simulate_reassignment

DEFAULTS = {
    "scale_days": 0.89,       # model MAE: saving this much -> speed score 1
    "residual_sd": 1.085,     # model test RMSE: typical size of one miss
    "risk_weight": 0.2,
    "min_days_saved": 0.25,   # PLACEHOLDER: a business decision
    "min_confidence": 0.5,
}


def get_weights(priority: float, risk_weight: float) -> dict:
    """priority 0 = all speed, 1 = all profit; weights always sum to 1."""
    if not 0 <= priority <= 1:
        raise ValueError("priority must be between 0 and 1")
    rest = 1 - risk_weight
    return {"speed": rest * (1 - priority), "profit": rest * priority,
            "risk": risk_weight}


def score_options(sim, priority=0.0, cfg=None):
    """Add scores, confidence and a recommend flag to a simulation table."""
    cfg = {**DEFAULTS, **(cfg or {})}
    w = get_weights(priority, cfg["risk_weight"])
    out = sim.copy()

    out["days_saved"] = -out["delta_days"]
    out["speed_score"] = (out["days_saved"] / cfg["scale_days"]).clip(-1, 1)
    out["risk_score"] = -(1 - out["share_in_seen_range"])
    out["profit_score"] = (
        out["profit_impact_assumed"] / out["observed_gross_profit"]
    ).clip(-1, 1)
    out["score"] = (w["speed"] * out["speed_score"]
                    + w["risk"] * out["risk_score"]
                    + w["profit"] * out["profit_score"])

    std_err = cfg["residual_sd"] / np.sqrt(out["rows"])
    z = out["days_saved"] / std_err
    out["confidence"] = out["share_in_seen_range"] * (z / 2).clip(0, 1)

    out["recommend"] = (
        (~out["is_current"])
        & (out["days_saved"] >= cfg["min_days_saved"])
        & (out["confidence"] >= cfg["min_confidence"])
        & (out["score"] > 0)
    )
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def top_recommendations(df, pipeline, features, factories, product,
                        region=None, ship_mode=None, priority=0.0, n=3,
                        cost_per_unit_km=0.0, cfg=None):
    """Return (all scored options, the top-n that pass every threshold)."""
    sim = simulate_reassignment(df, pipeline, features, factories, product,
                                region, ship_mode, cost_per_unit_km)
    scored = score_options(sim, priority, cfg)
    return scored, scored[scored["recommend"]].head(n)