import numpy as np
import pandas as pd

COORD_COLUMNS = [
    "country", "postal_code", "place", "state_name", "state_code",
    "admin2_name", "admin2_code", "admin3_name", "admin3_code",
    "latitude", "longitude", "accuracy",
]


def normalize_postal_code(code, country: str) -> str:
    """Turn a dataset postal code into the lookup key used by GeoNames."""
    code = str(code).strip().upper().replace(" ", "")
    if country == "United States":
        return code.zfill(5)   # restore leading zeros lost upstream
    if country == "Canada":
        return code[:3]        # GeoNames uses the 3-character area code
    return code


def load_postal_coordinates(us_path: str, ca_path: str) -> pd.DataFrame:
    """Read the GeoNames files into one table: country, postal_code, lat, lon."""
    frames = []
    for path, country in ((us_path, "United States"), (ca_path, "Canada")):
        raw = pd.read_csv(
            path, sep="\t", header=None, names=COORD_COLUMNS,
            dtype={"postal_code": str},
        )
        raw["country"] = country
        raw["postal_code"] = raw["postal_code"].str.strip().str.upper()
        if country == "Canada":
            raw["postal_code"] = raw["postal_code"].str[:3]
        frames.append(raw[["country", "postal_code", "latitude", "longitude"]])
    coords = pd.concat(frames, ignore_index=True)
    # If a key appears more than once, use the average position
    return coords.groupby(["country", "postal_code"], as_index=False)[
        ["latitude", "longitude"]
    ].mean()


def attach_customer_coordinates(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Add cust_lat / cust_lon to each order (blank if no match is found)."""
    out = df.copy()
    out["postal_key"] = [
        normalize_postal_code(c, k)
        for c, k in zip(out["Postal Code"], out["Country/Region"])
    ]
    lookup = coords.rename(columns={
        "country": "Country/Region", "postal_code": "postal_key",
        "latitude": "cust_lat", "longitude": "cust_lon",
    })
    return out.merge(
        lookup, on=["Country/Region", "postal_key"],
        how="left", validate="many_to_one",
    )

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points (arrays OK)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_factory_distance(df: pd.DataFrame, factories: pd.DataFrame) -> pd.DataFrame:
    """Add distance_km from each order's current factory to its customer."""
    fac = factories.rename(columns={
        "factory": "Factory", "latitude": "factory_lat", "longitude": "factory_lon",
    })
    out = df.merge(fac, on="Factory", how="left", validate="many_to_one")
    if out["factory_lat"].isna().any():
        raise ValueError("Some factories have no coordinates")
    out["distance_km"] = haversine_km(
        out["factory_lat"], out["factory_lon"], out["cust_lat"], out["cust_lon"]
    )
    return out