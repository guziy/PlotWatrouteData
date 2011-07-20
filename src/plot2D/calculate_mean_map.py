__author__="huziy"
__date__ ="$6 oct. 2010 21:06:01$"

import application_properties
from math import isnan
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

from mpl_toolkits.basemap import NetCDFFile
from math import *

from datetime import datetime

import pylab
#from util.geo.ps_and_latlon import *
from math import *
from plot2D.plot_utils import draw_meridians_and_parallels
import matplotlib as mpl

from readers.read_infocell import *

from scipy.stats import ttest_ind

import data.data_select as data_select

inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 1000 * inches_per_pt          # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, 2 * fig_height]

font_size = 20
params = {
        'axes.labelsize': font_size,
        'font.size':font_size,
        'text.fontsize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize': fig_size
        }

pylab.rcParams.update(params)


#set current directory to the root directory of the project
application_properties.set_current_directory()

from plot2D.map_parameters import polar_stereographic
from matplotlib.ticker import LinearLocator

n_cols = polar_stereographic.n_cols
n_rows = polar_stereographic.n_rows
xs = polar_stereographic.xs
ys = polar_stereographic.ys
m = polar_stereographic.basemap

def zoom_to_qc():
    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.05 * (ymax - ymin) , ymax * 0.25)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.55, 0.72*xmax)



def plot_data(data, i_array, j_array, name='AEX', title = None, digits = 1,
                                      color_map = mpl.cm.get_cmap('RdBu_r'), 
                                      minmax = (None, None),
                                      units = '%',
                                      colorbar_orientation = 'vertical',
                                      draw_colorbar = True
                                      ):



    if name != None:
        plt.figure()

    to_plot = ma.masked_all((n_cols, n_rows))
    for index, i, j in zip( range(len(data)), i_array, j_array):
        to_plot[i, j] = data[index]


    print np.ma.min(data), np.ma.max(data)
    
  #  m.pcolor(xs, ys, to_plot, cmap = mpl.cm.get_cmap('RdBu_r'))


    m.pcolormesh(xs, ys, to_plot.copy(), cmap = color_map,
                                          vmin = minmax[0],
                                          vmax = minmax[1],)



    #ads to m fields basins and basins_info which contain shapes and information
 #   m.readshapefile('data/shape/contour_bv_MRCC/Bassins_MRCC_utm18', 'basins')
 #   m.scatter(xs, ys, c=to_plot)
    plot_basin_boundaries_from_shape(m, linewidth = 1)
    m.drawrivers(linewidth = 0.5)
    m.drawcoastlines(linewidth = 0.5)
    draw_meridians_and_parallels(m, step_degrees = 30)

    if draw_colorbar:
        int_ticker = LinearLocator()
        cb = plt.colorbar(ticks = int_ticker, orientation = colorbar_orientation, format = '%d')
        cb.ax.set_ylabel(units)

    override = {'fontsize': 25,
                  'verticalalignment': 'baseline',
                  'horizontalalignment': 'center'}


    plt.title(title if title != None else name, override)

    zoom_to_qc()

    if name != None:
        plt.savefig(name + '.png')




def get_meanof_means_and_stds_from_files(files):
    mean = None
    stdevs = None
    for path in files:
        data = get_data_from_file(path)
        if mean == None:
            mean = np.zeros(data.shape[1])
            stdevs = np.zeros(data.shape[1])
            
        mean += np.mean(data, axis = 0)
        stdevs += np.std(data, axis = 0)


    mean = mean / float(len(files))
    stdevs = stdevs / float(len(files))

    print 'max deviation: ', np.max(stdevs)
    assert mean.shape[0] == data.shape[1]
    return mean, stdevs

def get_dispersion_between_members(files):
    datas = []
    for path in files:
        data = get_data_from_file(path)
        datas.append(data)

    nt, ncell = datas[0].shape
    nmembers = len(datas)
    all_data = np.zeros((nmembers, nt, ncell))
    for i, the_data in enumerate(datas):
        all_data[i, :, :] = the_data[:,:]

    return np.mean(np.std(all_data, axis = 0), axis = 0)




def get_indices(folder):
    for file in os.listdir(folder):
        filename = file
        if filename.startswith('.'): continue
        break

    path = os.path.join( folder, filename)
    print path
    ncfile = NetCDFFile(path)
    return ncfile.variables['x-index'].data, ncfile.variables['y-index'].data



def plot_diff_between_files(file1, file2, i_array, j_array):
    data1 = data_select.get_data_from_file(file1)
    data2 = data_select.get_data_from_file(file2)
    the_diff = np.mean(data2 - data1, axis = 0)

    plot_data(the_diff, i_array, j_array, name = 'the_diff', title='AEX, difference between \n %s \n and \n %s' % (file2, file1))

    pass

def plot_diff(folder = 'data/streamflows/VplusFmask_newton',
              current_start_date = datetime(1970,1,1,0,0,0),
              current_end_date   = datetime(1999,12,31,0,0,0),
              future_start_date  = datetime(2041,1,1,0,0,0),
              future_end_date    = datetime(2070,12,31,0,0,0),
              plot_f_and_c_means_separately = False):



    '''
    Plot difference between the means for future and current climate
    '''
    current = []
    future = []
    
    for file in os.listdir(folder):
        if file.startswith('.'): continue

        path = folder + os.path.sep + file
        if '2041' in file:
            future.append(path)
        elif 'aex' not in file:
            current.append(path)

    i_array, j_array = get_indices(folder)

    n_grid_cells = len(i_array)

    #save data for current climate in one array
    current_array = None
    for path in current:
        print path
        #read data from file
        data, times, x_indices, y_indices = data_select.get_data_from_file(path)
        #calculate annual means for each grid point
        the_means = data_select.get_annual_means(data, times,
                                    start_date = current_start_date,
                                    end_date = current_end_date)

        print len(the_means)

        for year, the_mean in the_means.iteritems():
            if current_array == None:
                current_array = the_mean
            else:
                current_array = np.append(current_array, the_mean)

    current_array = np.reshape(current_array, (-1, n_grid_cells))


    #save data for futuure climate in one array
    future_array = None
    for path in future:
        print path
        #read data from file
        data, times, x_indices, y_indices = data_select.get_data_from_file(path)
        #calculate annual means for each grid point
        the_means = data_select.get_annual_means(data, times,
                                    start_date = future_start_date,
                                    end_date = future_end_date)
        print len(the_means)
        if len(the_means) < 30:
            print sorted(the_means.keys())

        for year, the_mean in the_means.iteritems():
            if future_array == None:
                future_array = the_mean
            else:
                future_array = np.append(future_array, the_mean)

    future_array = np.reshape(future_array, (-1, n_grid_cells))

    ttest, p = ttest_ind(current_array, future_array, axis = 0)

    to_plot_5 = np.ma.masked_all((n_cols, n_rows))
    for pi, i, j in  zip(p, i_array, j_array):
        if pi >= 0.05:
            to_plot_5[i, j] = 1



    to_plot_1 = np.ma.masked_all((n_cols, n_rows))
    for pi, i, j in  zip(p, i_array, j_array):
        if pi >= 0.01:
            to_plot_1[i, j] = 1



    print p

    print 'future:(nt, npos) = (%d, %d)' % future_array.shape
    print 'current:(nt, npos) = (%d, %d)' % current_array.shape


    future_mean = future_array.mean(axis = 0)
    current_mean = current_array.mean(axis = 0)


    #start plotting (f-c)/c * 100

    plt.subplots_adjust(hspace = 0.2)

    #significance level 5%
    plt.subplot(2,1,1)
    plot_data(  (future_mean - current_mean)/current_mean * 100.0,
                i_array, j_array, name = None,
                color_map = mpl.cm.get_cmap('RdBu', 10), minmax = (-40, 40),
                title = 'significance level: 5%'
                )


    m.pcolormesh(xs, ys, 0.5 * to_plot_5, cmap = 'gray', vmin = 0, vmax = 1)

    #significance level 1%
    plt.subplot(2,1,2)
    plot_data(  (future_mean - current_mean)/current_mean * 100.0,
                i_array, j_array, name = None,
                color_map = mpl.cm.get_cmap('RdBu', 10), minmax = (-40, 40),
                title = 'significance level: 1%'
                )

   
    m.pcolormesh(xs, ys, 0.5 * to_plot_1, cmap = 'gray', vmin = 0, vmax = 1)
    plt.savefig('future-current(sign).png', bbox_inches = 'tight')


    if plot_f_and_c_means_separately:
        plt.figure()

        print 'plotting means'

        plt.subplot(2,1,1)
        plot_data(  current_mean,
                    i_array, j_array, title = 'current mean river discharge (${\\rm m^3/s}$)', name = None,
                    color_map = mpl.cm.get_cmap('RdBu', 20), minmax = (0, None), colorbar_orientation = 'vertical')


        plt.subplot(2,1,2)
        plot_data(  future_mean,
                    i_array, j_array, title = 'future mean river discharge (${\\rm m^3/s}$)', name = None,
                    color_map = mpl.cm.get_cmap('RdBu', 20), minmax = (0, None), colorbar_orientation = 'vertical'
                    )


        plt.savefig('future_and_current_means.png', bbox_inches='tight')
    pass



def plot_maximums(data_folder = 'data/streamflows/hydrosheds_euler3'):
    k = 1
    plt.figure()
    i_array, j_array = get_indices(data_folder)
    for f in os.listdir(data_folder):
        if f.startswith('.'):
            continue

        file = NetCDFFile(os.path.join(data_folder, f))
        streamflow = file.variables['water_discharge'].data
        maximum_field = np.max(streamflow, axis = 0)
        

        plt.subplot(6,2,k)
        plot_data(maximum_field,i_array, j_array, title = f , name = None,
                    color_map = mpl.cm.get_cmap('RdBu', 20), minmax = (None, None), colorbar_orientation = 'vertical')
        print k
        k += 1
    plt.savefig('maximums.png', bbox_inches = 'tight')
    pass



if __name__ == "__main__":

#    data = get_data_from_file('data/streamflows/fdirv1/aex_discharge_1970_01_01_00_00.nc')
#    i_array, j_array = get_indices()
#    plot_data(np.std(data, axis = 0) / np.mean(data, axis = 0) * 100, i_array , j_array, name = 'aex_temp_variability',
#    title = 'AEX (std/mean * 100 %)')

    print os.getcwd()
    plot_diff(folder = 'data/streamflows/hydrosheds_euler9')
#    plot_maximums(data_folder = 'data/streamflows/hydrosheds_euler7')
    print "Hello World"
