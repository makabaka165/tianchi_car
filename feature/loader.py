import csv
from pathlib import Path

import pandas as pd


MISSING_TOKENS = {"", "-"}


def load_space_separated_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter=" "))
    if not rows:
        raise ValueError(f"Empty file: {path}")
    header = rows[0]
    body = rows[1:]
    df = pd.DataFrame(body, columns=header)
    return df.replace({token: pd.NA for token in MISSING_TOKENS})
