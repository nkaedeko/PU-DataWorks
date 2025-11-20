from io.read_tensile import read_tensile
from analysis.tensile import compute_stress_strain, youngs_modulus, tensile_properties
from plots.tensile_plot import plot_stress_strain
import matplotlib.pyplot as plt

area = 2.23      # mm^2
initial_length = 30.0  # mm

df = read_tensile("10_8_2025_MEK_5%_Fabric.txt")

df = compute_stress_strain(df, area, initial_length)

E = youngs_modulus(df)
uts, break_strain, toughness = tensile_properties(df)

print("Modulus:", E)
print("UTS:", uts)
print("Break strain:", break_strain)
print("Toughness:", toughness)

plot_stress_strain(df, label="MEK 5% Fabric")
plt.legend()
plt.show()
