from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/processed/orders_clean.csv")
out = Path("reports/figures")
out.mkdir(parents=True, exist_ok=True)

# ---------- Numbers first: outliers and consistency ----------
for col in ["lead_adj", "Sales", "Units", "Gross Profit", "Cost"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"{col}: {n} rows outside 1.5*IQR ({n / len(df):.1%})")

print("rows with negative Gross Profit:", int((df["Gross Profit"] < 0).sum()))
print("rows with zero or negative Sales:", int((df["Sales"] <= 0).sum()))
print("max |Sales - Cost - Gross Profit|:",
      round((df["Sales"] - df["Cost"] - df["Gross Profit"]).abs().max(), 4))
print(df[["Sales", "Units", "Gross Profit", "Cost"]].describe())

# ---------- Chart 1: lead time distribution ----------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["lead_adj"], bins=range(172, 187), edgecolor="black")
ax.set_xlabel("Adjusted lead time (days)")
ax.set_ylabel("Number of rows")
ax.set_title("Distribution of adjusted lead time")
fig.tight_layout()
fig.savefig(out / "01_lead_time_hist.png", dpi=120)
plt.close(fig)

# ---------- Chart 2: lead time by ship mode ----------
order = ["Same Day", "First Class", "Second Class", "Standard Class"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([df.loc[df["Ship Mode"] == m, "lead_adj"] for m in order],
           tick_labels=order)
ax.set_ylabel("Adjusted lead time (days)")
ax.set_title("Adjusted lead time by ship mode")
fig.tight_layout()
fig.savefig(out / "02_lead_by_ship_mode.png", dpi=120)
plt.close(fig)

# ---------- Chart 3: rows per factory ----------
counts = df["Factory"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(counts.index, counts.values)
ax.set_xlabel("Number of order rows")
ax.set_title("Order rows per current factory")
fig.tight_layout()
fig.savefig(out / "03_rows_per_factory.png", dpi=120)
plt.close(fig)

# ---------- Chart 4: mean lead time per factory ----------
means = df.groupby("Factory")["lead_adj"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(means.index, means.values)
ax.set_xlim(170, 182)
ax.set_xlabel("Mean adjusted lead time (days)")
ax.set_title("Mean adjusted lead time per current factory")
fig.tight_layout()
fig.savefig(out / "04_mean_lead_per_factory.png", dpi=120)
plt.close(fig)

print("Saved 4 charts to", out)