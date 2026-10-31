"""Load BanglaBook-style CSV files (columns: Review, label)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .preprocess import normalize

LABELS = {0: "negative", 1: "neutral", 2: "positive"}
TEXT_COLUMN = "Review"
LABEL_COLUMN = "label"


class DataError(Exception):
    """Raised when a data file is missing, malformed or empty."""


def load_split(
    path: str | Path,
    max_rows: int = 0,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Read one CSV split and return (texts, labels).

    Reviews are normalised. Rows whose review is empty after cleaning, or whose label
    is not 0/1/2, are dropped. ``max_rows`` > 0 takes a random sample (for quick runs).
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise DataError(f"Data file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path, usecols=[TEXT_COLUMN, LABEL_COLUMN])
    except ValueError as exc:
        raise DataError(
            f"{csv_path.name} must contain the columns '{TEXT_COLUMN}' and '{LABEL_COLUMN}'"
        ) from exc
    except (OSError, pd.errors.ParserError) as exc:
        raise DataError(f"Could not read {csv_path.name}: {exc}") from exc

    df = df[df[LABEL_COLUMN].isin(LABELS)]
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    texts = df[TEXT_COLUMN].map(normalize)
    keep = texts.str.len() > 0
    texts, labels = texts[keep], df.loc[keep, LABEL_COLUMN].astype(int)
    if texts.empty:
        raise DataError(f"{csv_path.name} has no usable rows")
    return texts.tolist(), labels.tolist()


def find_split(data_dir: str | Path, name: str) -> Path:
    """Locate ``train.csv`` / ``validation.csv`` / ``test.csv`` inside ``data_dir``."""
    path = Path(data_dir) / f"{name}.csv"
    if not path.is_file():
        raise DataError(
            f"Expected {path}. Download BanglaBook (see README) and point --data-dir "
            "at the folder that contains train.csv, validation.csv and test.csv."
        )
    return path
