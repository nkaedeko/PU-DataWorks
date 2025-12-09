# examples/run_tensile_example.py
from pathlib import Path

from io.read_tensile import read_tensile
from analysis.tensile import analyze_tensile
from plots.tensile_plot import plot_stress_strain


def main():
    here = Path(__file__).resolve().parent
    data_file = here / "12_5_Tensile_MEK_1.25%_Bulk.txt"

    # Change the number using real data：
    area_mm2 = 2.23      # mm^2
    initial_length_mm = 30.0  # mm

    df_raw = read_tensile(data_file)
    result = analyze_tensile(df_raw, area_mm2, initial_length_mm)

    print("=== Tensile properties ===")
    print(f"Young's modulus (MPa): {result.modulus:.2f}")
    print(f"UTS (MPa):            {result.uts:.2f}")
    print(f"Break strain (%):     {result.break_strain * 100:.2f}")
    print(f"Toughness (MJ/m^3):   {result.toughness:.3f}")

    plot_stress_strain(result.data, label="MEK 1.25% Bulk")


if __name__ == "__main__":
    main()

