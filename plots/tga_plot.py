import matplotlib.pyplot as plt
from .style import set_plot_style

def plot_tga(df):
    set_plot_style()
    plt.plot(df["Temperature"], df["weight_percent"], label="TGA")

def plot_dtg(df):
    set_plot_style()
    plt.plot(df["Temperature"], df["dtg"], label="DTG")
