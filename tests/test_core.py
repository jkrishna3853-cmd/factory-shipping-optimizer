import pandas as pd
import pytest

from src.data_processing import add_lead_times
from src.features import (
    CATEGORICAL_FEATURES, LEAKY_COLUMNS, NUMERIC_FEATURES, split_by_order,
)
from src.geo import haversine_km, normalize_postal_code
from src.recommendations import get_weights


# ---------- geography ----------
def test_haversine_nyc_to_la():
    d = float(haversine_km(40.7128, -74.0060, 34.0522, -118.2437))
    assert abs(d - 3936) < 15


def test_haversine_same_point_is_zero():
    assert float(haversine_km(35.0, -90.0, 35.0, -90.0)) == 0.0


def test_us_zip_gets_leading_zero_back():
    assert normalize_postal_code("2151", "United States") == "02151"


def test_canada_postal_code_uses_first_three_characters():
    assert normalize_postal_code("m7a 1b2", "Canada") == "M7A"


# ---------- lead time ----------
def test_year_shift_is_removed():
    df = pd.DataFrame({
        "Order Date": pd.to_datetime(["2024-01-03"]),
        "Ship Date": pd.to_datetime(["2026-06-30"]),
    })
    out = add_lead_times(df)
    assert out["year_shift"].iloc[0] == 2
    assert out["lead_adj"].iloc[0] == 179


# ---------- leakage safety ----------
def test_split_never_shares_an_order():
    df = pd.DataFrame({
        "Order ID": [f"O{i // 3}" for i in range(300)],
        "x": range(300),
    })
    train, test = split_by_order(df, test_size=0.2, seed=1)
    assert set(train["Order ID"]).isdisjoint(set(test["Order ID"]))
    assert len(train) + len(test) == len(df)


def test_no_leaky_column_is_used_as_a_feature():
    used = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert used.isdisjoint(set(LEAKY_COLUMNS))


# ---------- recommendation weights ----------
def test_weights_sum_to_one():
    for p in (0, 0.3, 1):
        assert sum(get_weights(p, 0.2).values()) == pytest.approx(1.0)


def test_priority_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        get_weights(1.5, 0.2)