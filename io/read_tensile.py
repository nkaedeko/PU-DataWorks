import pandas as pd

def read_tensile(filepath):
    df = pd.read_csv(filepath, sep="\t", engine="python")
    df.columns = df.columns.str.strip()
    return df
