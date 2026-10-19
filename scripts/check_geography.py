import pandas as pd

# Read Postal Code as text so leading zeros (e.g. 02134) are not lost
df = pd.read_csv("data/processed/orders_clean.csv", dtype={"Postal Code": str})

print(df["Country/Region"].value_counts())
print("States/provinces per country:")
print(df.groupby("Country/Region")["State/Province"].nunique())
print("Unique postal codes per country:")
print(df.groupby("Country/Region")["Postal Code"].nunique())

for c in df["Country/Region"].unique():
    sub = df[df["Country/Region"] == c]
    print("\n==", c, "==")
    print("States:", sorted(sub["State/Province"].unique()))
    print("Sample postal codes:", sub["Postal Code"].drop_duplicates().head(10).tolist())
    print("Postal code lengths:", sub["Postal Code"].str.len().value_counts().to_dict())