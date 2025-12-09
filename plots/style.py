# plots/style.py
import matplotlib.pyplot as plt


def apply_default_style():
    """
    Apply a simple, clean plotting style for all figures.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
        }
    )
