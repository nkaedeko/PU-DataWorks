import numpy as np

def compute_stress_strain(df, area, initial_length):
    df["stress_MPa"] = df["Load"] / area
    df["strain"] = df["Extension"] / initial_length
    return df

def youngs_modulus(df, strain_max=0.02):
    mask = df["strain"] <= strain_max
    x = df.loc[mask, "strain"]
    y = df.loc[mask, "stress_MPa"]
    slope = np.polyfit(x, y, 1)[0]
    return slope

def tensile_properties(df):
    uts = df["stress_MPa"].max()
    break_strain = df["strain"].max()
    toughness = np.trapz(df["stress_MPa"], df["strain"])
    return uts, break_strain, toughness
