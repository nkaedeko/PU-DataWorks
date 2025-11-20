import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "font.size": 12,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2,
        "axes.labelpad": 10,
        "xtick.major.size": 5,
        "ytick.major.size": 5
    })
