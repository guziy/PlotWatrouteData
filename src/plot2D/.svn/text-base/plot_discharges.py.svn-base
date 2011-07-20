__author__="huziy"
__date__ ="$1 juin 2010 14:44:04$"

from util.lat_lon_holder import LatLonHolder
import application_properties
application_properties.set_current_directory()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, NetCDFFile

import pylab
from math import *

PATH_TO_NETCDF_FILE = 'data/simulation_results/discharge_1961_01_01_00_00_waterloo_without_init.nc'
#PATH_TO_NETCDF_FILE = 'data/simulation_results/discharge_1961_01_01_00_00_Canada.nc'

def get_str_from_char_array(char_array):
    result = ''
    for c in char_array:
        result += c
    return result


def adjust_for_qc():
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max * 0.3)

    x_min, x_max = plt.xlim()
    plt.xlim(x_max * 0.4, x_max * 0.7 )

def adjust_for_ca():
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max * 0.5)

    x_min, x_max = plt.xlim()
    plt.xlim(x_max * 0.1, x_max * 0.9 )


def plot_data(data, longitudes, latitudes, time, x_indices, y_indices, adjust_function):
    lon_tmp = []
    lat_tmp = []
    for i in range(len(x_indices)):
        ix = x_indices[i]
        iy = y_indices[i]

        lon_tmp.append(longitudes[ix, iy])
        lat_tmp.append(latitudes[ix, iy])

    lon_min = np.min(lon_tmp)
    lon_max = np.max(lon_tmp)
    lat_min = np.min(lat_tmp)
    lat_max = np.max(lat_tmp) 


    m =  Basemap(projection = 'npstere',
                 resolution = 'c',
                 lon_0 = (lon_min + lon_max) / 2.0, lat_0 = (lat_min + lat_max) / 2.0,
                 boundinglat = 40,
                 lat_ts = 60)


    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = 1500 * inches_per_pt           # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, fig_height]

    params = {'axes.labelsize': 14,
        'text.fontsize': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.figsize': fig_size}

    pylab.rcParams.update(params)

    projected_lons, projected_lats = m(longitudes, latitudes )


    nx, ny = longitudes.shape

    for i in range(len(time)):
        if i % 1000 != 0:
            continue

        t_str = get_str_from_char_array(time[i,:])
        the_data = np.zeros(longitudes.shape)
        the_data[:, :] = None
        for index in range(len(x_indices)):
            ix = x_indices[index]
            iy = y_indices[index]
            the_data[ix, iy] = data[i,index]
                 
        m.bluemarble()
        m.contourf(projected_lons, projected_lats, the_data, alpha = 0.3 )
        plt.colorbar()
        #m.drawcoastlines()

        

        plt.title(t_str)
        
        adjust_function()
        plt.savefig(t_str + '.png')
        plt.clf()
    pass


def main():
    # read in data from netCDF file.
    fpin = NetCDFFile(PATH_TO_NETCDF_FILE)
    #print fpin.variables
    #dims: time, cell_index
    discharge = fpin.variables['water discharge'].data

    x_indices = fpin.variables['x-index'].data
    y_indices = fpin.variables['y-index'].data

    time = fpin.variables['time'].data

    holder = LatLonHolder()

    nx = holder.get_num_cells_along_x()
    ny = holder.get_num_cells_along_y()

    longitudes = np.zeros((nx, ny))
    latitudes = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            #+1 because index starts from 1 in the holder
            lon, lat = holder.get_lon_lat(i + 1, j + 1)
            longitudes[i,j] = lon
            latitudes[i,j] = lat
        
    plot_data(discharge, longitudes, latitudes, time, x_indices, y_indices, adjust_for_qc)
    pass


if __name__ == "__main__":
    main()
