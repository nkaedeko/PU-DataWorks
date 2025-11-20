import matplotlib.pyplot as plt
from .style import set_plot_style

def plot_stress_strain(df, label=None):
    set_plot_style()
    plt.plot(df["strain"], df["stress_MPa"], linewidth=2, label=label)
