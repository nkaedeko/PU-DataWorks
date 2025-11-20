import pandas as pd

def read_tga(filepath):
    df = pd.read_csv(filepath)
    return df
