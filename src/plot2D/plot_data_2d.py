__author__="huziy"
__date__ ="$1 fevr. 2011 16:52:23$"

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

from plot2D.map_parameters import polar_stereographic
from util.plot_utils import draw_meridians_and_parallels

import numpy as np

from readers.read_infocell import plot_basin_boundaries_from_shape
import matplotlib as mpl

import data.data_select as data_select


m = polar_stereographic.basemap
xs = polar_stereographic.xs
ys = polar_stereographic.ys


def plot_data(data, i_array, j_array, name='AEX', title = None, digits = 1,
                                      color_map = mpl.cm.get_cmap('RdBu'),
                                      minmax = (None, None),
                                      units = '%',
                                      colorbar_orientation = 'vertical'
                                      ):



    if name != None:
        plt.figure()

    to_plot = np.ma.masked_all(xs.shape)
    for index, i, j in zip( range(len(data)), i_array, j_array):
        to_plot[i, j] = data[index]


    print np.ma.min(data), np.ma.max(data)

  #  m.pcolor(xs, ys, to_plot, cmap = mpl.cm.get_cmap('RdBu_r'))

    extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]
    plt.imshow(to_plot.transpose().copy(), interpolation = 'nearest' ,
                                    extent = extent,
                                    origin = 'lower',
                                    cmap = color_map,
                                    vmin = minmax[0],
                                    vmax = minmax[1]
                                    )


    plot_basin_boundaries_from_shape(m, linewidth = 1)
    m.drawrivers()
    m.drawcoastlines()
    draw_meridians_and_parallels(m, step_degrees = 30)

    int_ticker = LinearLocator()
    cb = plt.colorbar(ticks = int_ticker, orientation = colorbar_orientation)
    cb.ax.set_ylabel(units)

    override = {'fontsize': 20,
                  'verticalalignment': 'baseline',
                  'horizontalalignment': 'center'}


    plt.title(title if title != None else name, override)

    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.12 * (ymax - ymin), ymax * 0.32)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.65, 0.85*xmax)

    if name != None:
        plt.savefig(name + '.png', bbox_inches = 'tight')


if __name__ == "__main__":
    i_indices, j_indices = data_select.get_indices_from_file()
    data = data_select.get_field_from_file('data/test_data/divided.nc', 'water_discharge')
    data = np.max(data, axis = 0)
    plot_data(data, i_indices, j_indices, name = 'discharge_scaling', title = 'Discharge response \n to runoff doubling')
    print "Hello World"
