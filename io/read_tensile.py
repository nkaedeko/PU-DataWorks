# io/read_tensile.py
from pathlib import Path
from io import StringIO

import pandas as pd


def read_tensile(filepath):
    """
    Read raw tensile data exported by MTS from a .txt file.

    The function searches for the line that contains the column header
    (e.g. 'Crosshead' and 'Load') and uses that as the start of the data
    table.

    Parameters
    ----------
    filepath : str or Path
        Path to the tensile .txt file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with at least the columns:
        - 'crosshead' : extension (mm)
        - 'load'      : load (N)
        - 'time'      : time (s)  (if available)
    """
    path = Path(filepath)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Crosshead" in line and "Load" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Could not find data header line in {filepath!s} "
            "(expected a line containing 'Crosshead' and 'Load')."
        )

    data_str = "".join(lines[header_idx:])
    df = pd.read_csv(
        StringIO(data_str),
        delim_whitespace=True,
        engine="python"
    )

    df.columns = [c.strip().lower() for c in df.columns]

    possible_cols = ["crosshead", "load", "time"]
    keep = [c for c in possible_cols if c in df.columns]

    return df[keep]
