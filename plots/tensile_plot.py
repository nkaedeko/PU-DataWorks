# plots/tensile_plot.py
import matplotlib.pyplot as plt

from plots.style import apply_default_style


def plot_stress_strain(df, label=None, show=True, ax=None):
    """
    Plot a stress–strain curve.

    Parameters
    ----------
    df : DataFrame
        Must contain 'strain' and 'stress_mpa'.
    label : str, optional
        Legend label.
    show : bool, default True
        Whether to call plt.show().
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    apply_default_style()

    if ax is None:
        fig, ax = plt.subplots()

    x = df["strain"] * 100  # 转成 %
    y = df["stress_mpa"]

    ax.plot(x, y, label=label)
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (MPa)")

    if label is not None:
        ax.legend()

    if show:
        plt.show()

    return ax
