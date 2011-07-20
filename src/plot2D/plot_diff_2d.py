__author__="huziy"
__date__ ="$18 nov. 2010 10:51:54$"


from plot2D.map_parameters import polar_stereographic
from mpl_toolkits.basemap import Basemap, NetCDFFile
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.pyplot as plt

from plot2D.map_parameters import zoom_on_quebec
import application_properties
application_properties.set_current_directory()

m = polar_stereographic.basemap
xs = polar_stereographic.xs
ys = polar_stereographic.ys

VAR_NAME = 'water_discharge'


import pylab
from math import *

inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 2000 * inches_per_pt          # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]

params = {
        'axes.labelsize': 14,
        'font.size':18,
        'text.fontsize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.figsize': fig_size
        }

pylab.rcParams.update(params)



def plot_diff_time_averaged(file1, file2):
    '''
    file2 - file1
    '''
    plt.figure()
    nc1 = NetCDFFile(file1)
    i_array, j_array = nc1.variables['x-index'].data, nc1.variables['y-index'].data

    data1 = nc1.variables[VAR_NAME].data

    nc2 = NetCDFFile(file2)
    data2 = nc2.variables[VAR_NAME].data

    diff = ma.masked_all(xs.shape)

    for i, j, pos in zip(i_array, j_array, range(len(i_array))):
        diff[i, j] = np.mean(data2[:,pos] - data1[:, pos])

    m.pcolormesh(xs, ys, diff, shading='gouraud', cmap = mpl.cm.get_cmap('RdBu_r'))

    plt.colorbar()
    zoom_on_quebec(plt)
    plt.savefig('diff.png')


def plot_file(file):
    plt.figure()
    nc1 = NetCDFFile(file)
    i_array, j_array = nc1.variables['x-index'].data, nc1.variables['y-index'].data

    data1 = nc1.variables[VAR_NAME].data
    diff = ma.masked_all(xs.shape)
    for i, j, pos in zip(i_array, j_array, range(len(i_array))):
        diff[i, j] = np.mean(data1[:, pos])

    m.pcolormesh(xs, ys, diff, shading='gouraud', cmap = mpl.cm.get_cmap('RdBu_r'))

    plt.colorbar()
    zoom_on_quebec(plt)
    plt.savefig('file.png')








if __name__ == "__main__":
    f1 = 'data/streamflows/changed_iter_stop_condition/discharge_1970_01_01_00_00.nc'
    f2 = 'data/streamflows/VplusF_newmask1/aex_discharge_1970_01_01_00_00.nc'
    plot_diff_time_averaged(f1, f2)
    plot_file(f1)
    print "Hello World"
