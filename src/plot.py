import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_3d_contour(x, y, z, x_label, y_label, z_label, title, filename):
    ax = plt.axes(projection="3d")
    ax.contour3D(x, y, z)
    ax.set_xlabel(x_label)
    ax.set_ylabel()
    ax.set_zlabel()
    plt.show()
