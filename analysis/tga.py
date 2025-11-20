import numpy as np

def compute_weight_percent(df):
    w0 = df["Weight"].iloc[0]
    df["weight_percent"] = df["Weight"] / w0 * 100
    return df

def compute_dtg(df):
    df["dtg"] = -np.gradient(df["weight_percent"], df["Temperature"])
    return df

def get_tga_properties(df):
    T5 = df.loc[df["weight_percent"] <= 95, "Temperature"].iloc[0]
    T50 = df.loc[df["weight_percent"] <= 50, "Temperature"].iloc[0]
    Tmax = df.loc[df["dtg"].idxmax(), "Temperature"]
    return T5, T50, Tmax
