import pandas as pd
import re

def read_tensile(filepath):
    """
    Reads an Instron tensile .txt file with metadata header and a table
    beginning at the 'Extension' row.
    """

    header_line = None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("extension"):
                header_line = i
                break

    if header_line is None:
        raise ValueError("Could not find header row starting with 'Extension'.")

    df = pd.read_csv(
        filepath,
        delim_whitespace=True,
        skiprows=header_line,
        engine="python"
    )

    df.columns = df.columns.str.strip()
    return df
