import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV without changing anything."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}")
    df = pd.read_csv(p)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Return basic data-quality facts as a dictionary."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum()[lambda s: s > 0].to_dict(),
        "full_row_duplicates": int(df.duplicated().sum()),
        "duplicate_row_ids": int(df["Row ID"].duplicated().sum())
        if "Row ID" in df
        else None,
    }

DATE_FORMAT = "%d-%m-%Y"
SHIFT_OFFSET_DAYS = 178  # typical leftover days; only used to find the year shift


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Order Date / Ship Date from text to datetime."""
    out = df.copy()
    for col in ("Order Date", "Ship Date"):
        out[col] = pd.to_datetime(out[col], format=DATE_FORMAT, errors="raise")
    return out


def add_lead_times(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw lead time, the whole-year shift, and the adjusted lead time.

    ASSUMPTION: Ship Dates appear shifted by whole calendar years.
    We remove that shift; the original columns are kept unchanged.
    """
    out = df.copy()
    out["lead_raw"] = (out["Ship Date"] - out["Order Date"]).dt.days
    out["year_shift"] = (
        ((out["lead_raw"] - SHIFT_OFFSET_DAYS) / 365.25).round().astype(int)
    )
    out["ship_date_adj"] = [
        d - pd.DateOffset(years=int(y))
        for d, y in zip(out["Ship Date"], out["year_shift"])
    ]
    out["lead_adj"] = (out["ship_date_adj"] - out["Order Date"]).dt.days
    return out


def add_factory(df: pd.DataFrame, mapping_path: str) -> pd.DataFrame:
    """Attach each product's current factory from the project mapping file."""
    mapping = pd.read_csv(mapping_path).rename(
        columns={"product_name": "Product Name", "factory": "Factory"}
    )
    out = df.merge(mapping, on="Product Name", how="left", validate="many_to_one")
    if out["Factory"].isna().any():
        missing = out.loc[out["Factory"].isna(), "Product Name"].unique()
        raise ValueError(f"Products with no factory: {list(missing)}")
    return out