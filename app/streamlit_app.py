import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # lets "from src..." work inside Streamlit

import joblib
import pandas as pd
import streamlit as st

from src.features import add_time_features
from src.recommendations import top_recommendations

st.set_page_config(page_title="Factory Optimization Simulator", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(ROOT / "models" / "lead_time_gb.joblib")
    return bundle["pipeline"], bundle["numeric"] + bundle["categorical"]


@st.cache_data
def load_data():
    df = add_time_features(
        pd.read_csv(ROOT / "data" / "processed" / "orders_geo.csv"))
    factories = pd.read_csv(ROOT / "data" / "external" / "factories.csv")
    return df, factories


pipe, features = load_model()
df, factories = load_data()

# ---------------- Sidebar: optimization controls ----------------
st.sidebar.header("Optimization controls")
product = st.sidebar.selectbox("Product", sorted(df["Product Name"].unique()))
region_choice = st.sidebar.selectbox(
    "Region", ["All regions"] + sorted(df["Region"].unique()))
mode_choice = st.sidebar.selectbox(
    "Ship mode", ["All ship modes"] + sorted(df["Ship Mode"].unique()))
priority = st.sidebar.slider("Priority: Speed (0) to Profit (1)", 0.0, 1.0, 0.0, 0.05)
cost = st.sidebar.number_input(
    "Assumed shipping cost per unit-km (ASSUMPTION; 0 = not estimated)",
    min_value=0.0, value=0.0, step=0.0001, format="%.4f")

region = None if region_choice == "All regions" else region_choice
mode = None if mode_choice == "All ship modes" else mode_choice

# ---------------- Module 1: Factory Optimization Simulator ----------------
st.title("Factory Optimization Simulator")
st.caption(
    "Observed = from the dataset. Predicted = model output. "
    "Simulated = the model re-run with another factory's distance. "
    "Assumed = a value you set; it is not in the data.")

try:
    scored, top = top_recommendations(
        df, pipe, features, factories, product, region, mode,
        priority=priority, n=3, cost_per_unit_km=cost)
except ValueError as err:
    st.warning(f"{err}. Try a different region or ship mode.")
    st.stop()

current = scored.loc[scored["is_current"], "factory"].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Current factory", current)
c2.metric("Orders in this scenario", int(scored["rows"].iloc[0]))
c3.metric("Observed mean lead time (days)",
          f"{scored['observed_mean_lead'].iloc[0]:.2f}")

if top.empty:
    st.info("No confident recommendation: keep the current factory. "
            "The best alternative is shown below with the reason it did not pass.")
else:
    st.success(f"Top option: {top.iloc[0]['factory']} "
               f"(saves {top.iloc[0]['days_saved']:.3f} days)")

table = scored[[
    "factory", "is_current", "pred_current", "pred_new", "days_saved",
    "reduction_pct", "delta_distance_km", "share_in_seen_range",
    "confidence", "score", "recommend",
]].rename(columns={
    "pred_current": "predicted lead now (days)",
    "pred_new": "predicted lead if moved (days)",
    "days_saved": "days saved",
    "reduction_pct": "reduction %",
    "delta_distance_km": "distance change (km)",
    "share_in_seen_range": "share in observed range",
})
st.subheader("Predicted performance across factories")
st.dataframe(table.round(3), hide_index=True)

st.subheader("Predicted change in lead time vs. the current factory (days)")
st.bar_chart(scored.set_index("factory")["delta_days"])

# ---------------- Module 2: What-If Scenario Analysis ----------------
st.divider()
st.header("What-If Scenario Analysis")

alternatives = scored[~scored["is_current"]]
alt = st.selectbox(
    "Compare the current factory with:",
    alternatives["factory"].tolist(),   # best-scoring alternative first
)
row = scored[scored["factory"] == alt].iloc[0]

w1, w2, w3, w4 = st.columns(4)
w1.metric("Predicted lead now (days)", f"{row['pred_current']:.3f}")
w2.metric("Predicted lead if moved (days)", f"{row['pred_new']:.3f}",
          delta=f"{-row['days_saved']:.3f}", delta_color="inverse")
w3.metric("Reduction", f"{row['reduction_pct']:.3f} %")
w4.metric("Average distance change (km)", f"{row['delta_distance_km']:+.0f}")

std_err = 1.085 / (row["rows"] ** 0.5)   # 1.085 = model's typical test error (RMSE)
st.write(
    f"Predicted saving: **{row['days_saved']:.3f} days**. Normal noise for an "
    f"average over {int(row['rows'])} orders is about **±{std_err:.3f} days**. "
    "A saving smaller than the noise cannot be told apart from chance."
)

st.subheader("Financial impact")
f1, f2 = st.columns(2)
f1.metric("Gross profit in this scenario (observed)",
          f"{row['observed_gross_profit']:,.2f}")
if cost > 0:
    f2.metric("Profit impact (ASSUMED shipping cost)",
              f"{row['profit_impact_assumed']:,.2f}")
else:
    f2.metric("Profit impact", "Not estimated")
    st.caption(
        "The dataset has no shipping or production costs, so profit impact is "
        "not estimated. Enter an assumed cost per unit-km in the sidebar to "
        "explore profit sensitivity."
    )

# ---------------- Module 4: Risk & Impact Panel ----------------
st.divider()
st.header("Risk & Impact Panel")
st.caption(f"Checks for: {current} -> {alt}")

flags = 0
if row["days_saved"] < 0:
    st.error(f"This move is predicted to be SLOWER by {-row['days_saved']:.3f} days.")
    flags += 1
if row["rows"] < 30:
    st.warning(f"Small sample: only {int(row['rows'])} orders in this scenario.")
    flags += 1
if row["share_in_seen_range"] < 0.8:
    st.warning(
        f"Extrapolation: only {row['share_in_seen_range']:.0%} of the new distances "
        "fall inside the range this product has actually shipped over."
    )
    flags += 1
if row["confidence"] < 0.5:
    st.warning(f"Low scenario confidence ({row['confidence']:.2f}; the bar is 0.50).")
    flags += 1
if 0 <= row["days_saved"] < 0.25:
    st.info(
        f"Predicted saving ({row['days_saved']:.3f} days) is below the minimum "
        "worthwhile saving of 0.25 days (a placeholder setting; a business decision)."
    )
    flags += 1
if cost > 0 and row["profit_impact_assumed"] < -0.01 * row["observed_gross_profit"]:
    st.error(
        f"Profit alert: at the assumed cost, this move loses "
        f"{-row['profit_impact_assumed']:.2f}, more than 1% of the gross profit "
        f"({row['observed_gross_profit']:.2f})."
    )
    flags += 1
if flags == 0:
    st.success("No risk flags for this option.")

# ---------------- Module 3: Recommendation Dashboard ----------------
from src.kpis import best_options, break_even_cost, compute_kpis, profit_sweep


@st.cache_data
def compute_dashboard(priority, _pipe, _df, _factories, _features):
    opts = best_options(_df, _pipe, _features, _factories, priority=priority)
    opts["break_even_cost"] = break_even_cost(opts)
    sweep = profit_sweep(opts)
    return opts, sweep, compute_kpis(opts, sweep)


st.divider()
st.header("Recommendation Dashboard")
opts, sweep, kpis = compute_dashboard(priority, pipe, df, factories, features)

rec_key = "Lead time reduction % (recommended moves)"
best_key = "Lead time reduction % (best available, NOT confident)"

k1, k2, k3, k4 = st.columns(4)
if pd.isna(kpis[rec_key]):
    k1.metric("Lead time reduction", "N/A",
              help="No move passed the confidence and minimum-saving checks.")
else:
    k1.metric("Lead time reduction", f"{kpis[rec_key]:.3f} %")
k1.caption(f"Best available (not confident): {kpis[best_key]:.4f} %")
k2.metric("Profit impact stability", f"{kpis['Profit impact stability']:.0%}")
k2.caption("Share of ASSUMED cost levels where the best move stays within 1% of profit.")
k3.metric("Scenario confidence",
          f"{kpis['Scenario confidence (order-weighted)']:.2f}")
k3.caption("Order-weighted average; 1.00 would be fully reliable.")
k4.metric("Recommendation coverage", f"{kpis['Recommendation coverage']:.0%}")
k4.caption("Share of products with a confident recommendation.")

st.subheader("Ranked reassignment suggestions (all products)")
n_show = st.slider("Products to show", 1, len(opts), 5)

ranked = opts.copy()
ranked["current factory"] = ranked["product"].map(
    df.groupby("Product Name")["Factory"].first())
ranked["verdict"] = ranked["recommended"].map(
    {True: "Recommend move", False: "Keep current (not confident)"})
ranked = ranked.sort_values(
    ["recommended", "confidence"], ascending=False).head(n_show)

st.dataframe(
    ranked[["product", "current factory", "best_alt", "orders", "days_saved",
            "reduction_pct", "confidence", "verdict"]]
    .rename(columns={"best_alt": "best alternative",
                     "days_saved": "days saved",
                     "reduction_pct": "reduction %"})
    .round(3),
    hide_index=True,
)

st.subheader("Profit sensitivity (illustrative cost levels, not from the data)")
st.line_chart(sweep.set_index("cost_per_unit_km")["share_within_tolerance"])
st.caption(
    "Each point shows what share of products keep their best move within "
    "1% of gross profit if shipping cost per unit-km were at that level. "
    "The cost levels are assumptions chosen to show sensitivity."
)