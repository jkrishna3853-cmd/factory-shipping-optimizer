import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

GROUP_KEYS = ["Factory", "Product Name", "Region"]
CLUSTER_FEATURES = ["mean_excess", "std_excess"]


def add_excess(df: pd.DataFrame) -> pd.DataFrame:
    """excess = actual lead time minus the average for that ship mode."""
    out = df.copy()
    out["expected"] = out.groupby("Ship Mode")["lead_adj"].transform("mean")
    out["excess"] = out["lead_adj"] - out["expected"]
    return out


def build_group_features(df: pd.DataFrame, min_rows: int = 30) -> pd.DataFrame:
    """One row per factory x product x region, keeping only groups with enough rows."""
    g = (
        df.groupby(GROUP_KEYS)
        .agg(
            rows=("excess", "size"),
            mean_excess=("excess", "mean"),
            std_excess=("excess", "std"),
        )
        .reset_index()
    )
    return g[g["rows"] >= min_rows].reset_index(drop=True)


def _scaled(g: pd.DataFrame):
    return StandardScaler().fit_transform(g[CLUSTER_FEATURES])


def silhouette_by_k(g: pd.DataFrame, ks=range(2, 6), seed: int = 42) -> dict:
    """Silhouette score for each candidate number of clusters."""
    X = _scaled(g)
    scores = {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return scores


def fit_clusters(g: pd.DataFrame, k: int, seed: int = 42) -> pd.DataFrame:
    """Return the group table with a 'cluster' column."""
    out = g.copy()
    out["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
        _scaled(g)
    )
    return out