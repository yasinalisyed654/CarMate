
import re
import pandas as pd

TEXT_COLS = ["name", "company", "fuel_type"]
NUM_COLS = ["year", "kms_driven"]
ALL_COLS = TEXT_COLS + NUM_COLS + ["Price"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Minimal cleaning
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Ensure required columns exist
    missing = set(ALL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    # Normalize text
    for c in TEXT_COLS:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def apply_filters(df: pd.DataFrame,
                  min_price: int = None, max_price: int = None,
                  company: str = "All", fuel_type: str = "All",
                  min_year: int = None, max_year: int = None,
                  max_kms: int = None) -> pd.DataFrame:
    out = df.copy()
    if min_price is not None:
        out = out[out["Price"] >= min_price]
    if max_price is not None:
        out = out[out["Price"] <= max_price]
    if company and company != "All":
        out = out[out["company"] == company.lower()]
    if fuel_type and fuel_type != "All":
        out = out[out["fuel_type"] == fuel_type.lower()]
    if min_year is not None:
        out = out[out["year"] >= min_year]
    if max_year is not None:
        out = out[out["year"] <= max_year]
    if max_kms is not None:
        out = out[out["kms_driven"] <= max_kms]
    return out

def build_search_text(df: pd.DataFrame) -> pd.Series:
    # A simple combined text field for TF‑IDF
    return (df["name"].fillna("") + " " + df["company"].fillna("") + " " + df["fuel_type"].fillna(""))
