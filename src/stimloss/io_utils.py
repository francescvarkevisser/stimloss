from __future__ import annotations

from typing import List

import pandas as pd


def read_table(path: str, fmt: str) -> pd.DataFrame:
    fmt = fmt.lower()
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {fmt}")


def peek_columns(path: str, fmt: str) -> List[str]:
    df = read_table(path, fmt)
    return list(df.columns)
