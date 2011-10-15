__author__="huziy"
__date__ ="$18 nov. 2010 10:49:18$"


from util.plot_utils import draw_meridians_and_parallels
import numpy as np
from numpy import ma

from mpl_toolkits.basemap import NetCDFFile
import os.path
import os
from math import *
#from plot2D.calculate_mean_map import plot_data
import pylab
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt

from readers.read_infocell import plot_basin_boundaries_from_shape
import application_properties
application_properties.set_current_directory()


from plot2D.map_parameters import polar_stereographic

xs = polar_stereographic.xs
ys = polar_stereographic.ys

lons = polar_stereographic.lons
lats = polar_stereographic.lats

basemap = polar_stereographic.basemap


inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 1000 * inches_per_pt          # width in inches
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





def get_indices_from_file(path):
    ncFile = NetCDFFile(path)
    return ncFile.variables['x-index'], ncFile.variables['y-index']



def test():
    y = np.zeros((100,100))
    for i in range(1,7):
        ax = plt.subplot(3, 2, i)
        plt.title('%d' % i)
        y[5 + 10*i, 5 + 10*i] = 80
        h = ax.imshow(y.copy(), vmin=0, vmax=100)
        plt.colorbar(h)
    plt.savefig("test_subplots.png")

from matplotlib.ticker import MaxNLocator
def plot_forcing_error(errors_map, i_list, j_list):
    pylab.rcParams.update(params)
    to_plot = ma.masked_all(xs.shape)

    err = None
    for id, error in errors_map.iteritems():
        if err == None:
            err = np.zeros(error.shape)
        err += error

    err = err / len(errors_map)

    for i, j, er in zip(i_list, j_list, err):
        to_plot[i,j] = er



    n_levels = 20
    plt.imshow(to_plot.transpose().copy(), extent = [np.min(xs), np.max(xs), ys.min(), ys.max()],
                                        origin = 'lower',
                                        interpolation = 'bilinear' ,
                                        cmap = mpl.cm.get_cmap('RdBu_r', n_levels),
                                        vmin = -30, vmax = 30)

    int_ticker = MaxNLocator(nbins=n_levels, integer=True)
    cb = plt.colorbar(ticks = int_ticker)
    cb.ax.set_ylabel('%')

    basemap.drawcoastlines()
    print np.min(lons), np.max(lons), lats.min(), lats.max()

    plot_basin_boundaries_from_shape(basemap, linewidth = 0.5)

    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.12 * (ymax - ymin), ymax * 0.32)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.65, 0.85*xmax)
    draw_meridians_and_parallels(basemap, 10)
    plt.savefig("forcing_errors.png")


    

def plot_forcing_errors(errors_map, i_list, j_list):
    pylab.rcParams.update(params)
    num_plots = len(errors_map)
    nrows = int(round(num_plots / 2.0))
    k = 1
    to_plot = ma.masked_all(xs.shape)


    plt.subplots_adjust(left = 0., right = 0.98, wspace = 0, hspace = 0.2)
    for id, errors in errors_map.iteritems():
        print '------------------------------'
        print id
        print k


        plt.subplot(2, nrows, k)
        plt.title(id.upper())
        for i, j, er in zip(i_list, j_list, errors):
            to_plot[i,j] = er

#        dx = 45000.0
#        basemap.pcolor(xs - dx / 2.0 , ys - dx / 2.0, to_plot, shading='gouraud',
#                          lw = 0,
#                          vmin = -30, vmax = 30,
#                          cmap = mpl.cm.get_cmap('RdYlBu_r'))



      
        
        plt.imshow(to_plot.transpose().copy(), extent = [np.min(xs), np.max(xs), ys.min(), ys.max()],
                                        origin = 'lower',
                                        interpolation = 'bilinear' ,
                                        cmap = mpl.cm.get_cmap('RdBu_r'),
                                        vmin = -30, vmax = 30)



        int_ticker = MaxNLocator(nbins=6, integer=True)


        cb = plt.colorbar(ticks = int_ticker)
        cb.ax.set_ylabel('%')

        basemap.drawcoastlines()
        print np.min(lons), np.max(lons), lats.min(), lats.max()
        
        

        plot_basin_boundaries_from_shape(basemap, linewidth = 0.5)
        
        ymin, ymax = plt.ylim()
        plt.ylim(ymin + 0.12 * (ymax - ymin), ymax * 0.32)



        xmin, xmax = plt.xlim()
        plt.xlim(xmin + (xmax - xmin) * 0.65, 0.85*xmax)



       

        
  #      plt.ylabel('( %s - aex ) /(0.5(%s + aex))' % (id,id) + '  (%)', size = 12)


        k += 1

    
    plt.savefig("forcing_errors.png")
    

def delete_feb29(start_date, end_date, data):
    date = start_date
    dt = timedelta(days = 1)
    i = 0
    while date <= end_date:
        if date.month == 2 and date.day == 29:
            data = np.delete(data, i, axis = 0)
            i -= 1
        i += 1
        date += dt
    return data

    


def calculate_and_plot(path_to_folder):
    pylab.rcParams.update(params)
    errors = {}
    files = os.listdir(path_to_folder)

    for file in files:
        if 'aex' in file:
            path = os.path.join(path_to_folder, file)
            data_aex, time, i_list, j_list = get_data_from_file(path)
            #i_list, j_list = get_indices_from_file(path)
            break


    data_aex = delete_feb29(datetime(1970,1,1,0), datetime(2000, 2,1,0),data_aex)
    print data_aex.shape

 #   plot_data(np.mean(data_aex, axis = 0), i_list, j_list)

    for file in files:
        if not file.endswith('.nc'):
            continue

        if file.startswith('.'):
            continue


 
        #check whether the file is future
        if '2041' in file or 'aex' in file:
            continue

        id = file.split("_",1)[0]

        data, times, x_ind, y_ind = get_data_from_file(os.path.join(path_to_folder, file))
        print data.shape
        errors[id] =  100 * np.mean( data - data_aex , axis = 0) / np.mean(data_aex, axis = 0)
#        if id == 'aet':
#            plot_data(np.mean((data - data_aex) / (data_aex + data), axis = 0),
#                      i_list, j_list, name = 'aet')
        print id
        print file


    plot_forcing_error(errors, i_list, j_list)
#    plot_forcing_errors(errors, i_list, j_list)
    pass





if __name__ == "__main__":
    calculate_and_plot('data/streamflows/VplusF_newmask1')
  #  test()
    print "Hello World"
