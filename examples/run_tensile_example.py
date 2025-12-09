# examples/run_tensile_example.py

from pathlib import Path

from io.read_tensile import read_tensile
from analysis.tensile import analyze_tensile
from plots.tensile_plot import plot_stress_strain


def main():
    # Directory of this script
    here = Path(__file__).resolve().parent

    # Tensile data file (make sure the file exists in the examples folder)
    data_file = here / "12_5_Tensile_MEK_1.25%_Bulk.txt"

    # Experimental parameters (modify according to your sample dimensions)
    area_mm2 = 2.23          # Cross-sectional area in mm^2
    initial_length_mm = 30.0 # Gauge length in mm

    print(">>> Reading file:", data_file)

    # 1. Load raw tensile data
    df_raw = read_tensile(data_file)
    print("Raw columns:", list(df_raw.columns))
    print(df_raw.head())

    # 2. Perform complete tensile analysis
    result = analyze_tensile(df_raw, area_mm2, initial_length_mm)

    # 3. Print results
    print("\n=== Tensile Properties ===")
    print(f"Young's modulus (MPa): {result.modulus:.2f}")
    print(f"UTS (MPa):            {result.uts:.2f}")
    print(f"Break strain (%):     {result.break_strain * 100:.2f}")
    print(f"Toughness (MJ/m^3):   {result.toughness:.3f}")

    # 4. Plot the stress–strain curve
    plot_stress_strain(result.data, label='MEK 1.25% Bulk')


if __name__ == "__main__":
    main()
