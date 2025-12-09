# analysis/tensile.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TensileResult:
    """
    Store processed tensile results and key properties.
    """
    data: pd.DataFrame
    modulus: float
    uts: float
    break_strain: float
    toughness: float  # area under stress–strain curve


def compute_stress_strain(df: pd.DataFrame,
                          area_mm2: float,
                          initial_length_mm: float) -> pd.DataFrame:
    """
    Add stress (MPa) and strain columns to raw tensile data.

    This follows the same formulas you use in Excel/Origin:
      stress (MPa)  = Load / Area
      strain        = Crosshead / Initial length

    Parameters
    ----------
    df : DataFrame
        Must contain 'crosshead' (mm) and 'load' (N).
    area_mm2 : float
        Cross-section area in mm^2.
    initial_length_mm : float
        Gauge length in mm.

    Returns
    -------
    DataFrame
        Copy of df with new columns:
        - 'strain'
        - 'stress_mpa'
    """
    df = df.copy()

    if "crosshead" not in df or "load" not in df:
        raise KeyError("DataFrame must contain 'crosshead' and 'load' columns.")

    df["strain"] = df["crosshead"] / float(initial_length_mm)
    # N / mm^2 = MPa
    df["stress_mpa"] = df["load"] / float(area_mm2)

    return df


def youngs_modulus(df: pd.DataFrame, max_strain: float = 0.02) -> float:
    """
    Estimate Young's modulus (slope of stress–strain in the initial linear region).

    Use 0–2% strain data to do linear fit。
    """
    if "strain" not in df or "stress_mpa" not in df:
        raise KeyError("DataFrame must contain 'strain' and 'stress_mpa'.")

    mask = (df["strain"] >= 0) & (df["strain"] <= max_strain)
    linear = df.loc[mask]

    if len(linear) < 2:
        raise ValueError("Not enough points in the linear region to estimate modulus.")

    coeffs = np.polyfit(linear["strain"], linear["stress_mpa"], 1)
    slope = float(coeffs[0])
    return slope


def tensile_properties(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Compute UTS, break strain, and toughness from a stress–strain curve.

    - UTS: max stress
    - Break strain: strain at the last positive load
    - Toughness: integral of stress over strain up to break
    """
    if "strain" not in df or "stress_mpa" not in df or "load" not in df:
        raise KeyError(
            "DataFrame must contain 'strain', 'stress_mpa', and 'load' columns."
        )

    # UTS
    uts = float(df["stress_mpa"].max())

    # Find last load>0 point
    positive = df[df["load"] > 0]
    if positive.empty:
        break_strain = float(df["strain"].iloc[-1])
        last_for_area = df
    else:
        last_idx = positive.index[-1]
        break_strain = float(df.loc[last_idx, "strain"])
        last_for_area = df.loc[df.index.min(): last_idx]

    # Toughness = ∫σ dε （Unit：MPa -> MJ/m^3）
    toughness = float(
        np.trapz(last_for_area["stress_mpa"], last_for_area["strain"])
    )

    return uts, break_strain, toughness


def analyze_tensile(df_raw: pd.DataFrame,
                    area_mm2: float,
                    initial_length_mm: float,
                    max_strain_for_modulus: float = 0.02) -> TensileResult:
    """
    Convenience wrapper: from raw data get result
    """
    df = compute_stress_strain(df_raw, area_mm2, initial_length_mm)
    E = youngs_modulus(df, max_strain=max_strain_for_modulus)
    uts, break_strain, toughness = tensile_properties(df)

    return TensileResult(
        data=df,
        modulus=E,
        uts=uts,
        break_strain=break_strain,
        toughness=toughness,
    )
