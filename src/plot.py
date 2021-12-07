import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

sns.set()


def signif(x, p):
    """Returns p significant digits of x

    Parameters
    ----------
        x : np.array
            The values to round
        p : int
            The number of significant digits

    Returns
    -------
        float
            The rounded number
    """
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def surface_plot(
    title, x, y, z, subtitles="", xlabel="x", ylabel="y", zlabel="z", filename=""
):
    """Plot values in a surface plot

    Parameters
    ----------
        title : str
            The title of the plots
        x : np.array
            The x values for which to plot
        y : np.array
            The y values for which to plot
        z_array : np.array
            The z values for which to plot
        subtitles : str
            Subtitles for the plot
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    fig = plt.figure()

    z_np_arr = np.array(z)
    vmin = np.min(z_np_arr)
    vmax = np.max(z_np_arr)

    if type(z) != list:
        z = [[z]]

    nrows, ncols = len(z[0]), len(z)

    axes = []
    for row in range(nrows):
        for col in range(ncols):
            ax = fig.add_subplot(ncols, nrows, ncols * col + row + 1, projection="3d")
            axes.append(ax)
            surf = ax.plot_surface(
                x,
                y,
                z[col][row],
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=False,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)

            if subtitles:
                ax.set_title(subtitles[col][row])

    cax, kw = matplotlib.colorbar.make_axes([ax for ax in axes])
    plt.colorbar(surf, cax=cax, **kw)

    fig.suptitle(title)

    if filename:
        plt.savefig(f"output/{filename.replace(' ', '_')}")
    plt.show()


def heat_plot(
    title,
    table_values,
    xticklabels,
    yticklabels,
    x_label,
    y_label,
    selected_idx=None,
    show=True,
    filename="",
):
    """Plots the heat plot

    Parameters
    ----------
        title : str
            The title of the plots
        table_values : float[][]
            The values of the values for which to plot in the heat plot
        xticklabels : str
            The labels for the ticks for the x-axis
        yticklabels : str
            The labels for the ticks for the y-axis
        x_label : str
            The label for the x-axis
        y_label : str
            The label for the y-axis
        selected_idx : tuple[int, int]
            The index for which to give an extra mark
        show : bool
            Whether to show the plot
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    g = sns.heatmap(
        table_values,
        xticklabels=signif(xticklabels, 2),
        yticklabels=signif(yticklabels, 2),
        annot=True,
    )
    from matplotlib.patches import Rectangle

    if selected_idx:
        g.add_patch(Rectangle(selected_idx, 1, 1, fill=False, edgecolor="blue", lw=3))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if filename:
        plt.savefig(f"output/{filename.replace(' ', '_')}")
    if show:
        plt.show()


def line_plot(
    title,
    x_datas,
    y_datas,
    data_labels,
    x_label,
    y_label,
    x_log=False,
    y_log=False,
    filename="",
    show=True,
):
    """Plots a line plot

    Parameters
    ----------
        title : str
            The title of the plots
        x_datas : float[]
            The x data for the plot
        y_datas : float[]
            The y data for the plot
        data_labels : str[]
            The labels for the plot
        x_label : str
            The label for the x-axis
        y_label : str
            The label for the y-axis
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    plt.title(title)
    for x_data, y_data, label in zip(x_datas, y_datas, data_labels):
        sns.lineplot(x=x_data, y=y_data, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_log:
        plt.xscale("log")
    if y_log:
        plt.yscale("log")
    if filename:
        plt.savefig(f"output/{filename.replace(' ', '_')}")
    if show:
        plt.show()
