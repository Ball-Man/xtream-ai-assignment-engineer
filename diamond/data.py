"""Data pipeline components."""
import pandas as pd


def load_raw(file_or_path) -> pd.DataFrame:
    """Load raw dataset from file (csv only, so far)."""
    return pd.read_csv(file_or_path)
