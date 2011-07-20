import os
__author__="huziy"
__date__ ="$May 23, 2011 4:09:58 PM$"

from animate.OccurrencesDB import OccurrencesDB
from plot2D.map_parameters import polar_stereographic
import numpy as np
import application_properties
application_properties.set_current_directory()
from shape.read_shape_file import *
from shape.basin_boundaries import *
import pickle
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator



def zoom_to_qc():
    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.05 * (ymax - ymin) , ymax * 0.25)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.55, 0.72*xmax)



##Compare the dates of high flow for the current and future climate

def main():
    basemap = polar_stereographic.basemap
    xs = polar_stereographic.xs
    ys = polar_stereographic.ys

    basin_patches = get_features_from_shape(basemap,linewidth = 0.5)

    cache_file = 'cache_high_dates.bin'
    if os.path.isfile(cache_file):
        data = pickle.load(open(cache_file))
        to_plot_current = data[0]
        to_plot_future = data[1]
    else:
        db = OccurrencesDB()

        current_days = db.get_mean_dates_of_maximum_annual_flow(current = True)
        print 'got current dates'

        future_days = db.get_mean_dates_of_maximum_annual_flow(current = False)
        print 'got future dates'

        print 'got data, plotting ...'

        to_plot_current = np.ma.masked_all(xs.shape)
        to_plot_future = np.ma.masked_all(xs.shape)
        for i, j, c_day, f_day in zip(db.i_indices, db.j_indices, current_days, future_days):
            to_plot_current[i, j] = c_day
            to_plot_future[i, j] = f_day
        data = [to_plot_current, to_plot_future]
        pickle.dump(data, open(cache_file, 'wb'))


    plt.figure()
    plt.subplot(1, 2, 1)
    basemap.pcolormesh(xs, ys, to_plot_current, vmin=1, vmax=300, cmap=mpl.cm.get_cmap('jet', 10))
    basemap.drawcoastlines(linewidth=0.3)
    # plot_basin_boundaries_from_shape(basemap, plotter = plt, linewidth = 0.5)
    plot_patches(plt, get_copies(basin_patches))
    zoom_to_qc()
    cb = plt.colorbar(ticks = MaxNLocator(nbins = 10))
    plt.title('current mean')

    plt.subplot(1, 2, 2)
    basemap.pcolormesh(xs, ys, to_plot_future, vmin=1, vmax=300, cmap=mpl.cm.get_cmap('jet', 10))
    basemap.drawcoastlines(linewidth=0.3)
    zoom_to_qc()
    plot_patches(plt, get_copies(basin_patches))
    # plot_basin_boundaries_from_shape(basemap, plotter = plt)
    plt.title('future mean')

    plt.colorbar(ticks = MaxNLocator(nbins = 10))

    plt.savefig('high_dates.png')




if __name__ == "__main__":
    main()
    print "Hello World"
