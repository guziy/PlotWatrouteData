from animate.OccurrencesDB import OccurrencesDB
from animate.OccurrencesDB import YearPositionObject
from animate.OccurrencesDB import ModelDataObject
from shape.read_shape_file import get_copies
import os.path
import os
__author__="huziy"
__date__ ="$20-May-2011 1:04:23 AM$"


from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from plot2D.map_parameters import polar_stereographic

import numpy as np

import application_properties
application_properties.set_current_directory()

from shape.basin_boundaries import plot_basin_boundaries_from_shape
from shape.basin_boundaries import plot_patches
from shape.read_shape_file import *

def zoom_to_qc():
    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.05 * (ymax - ymin) , ymax * 0.25)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.55, 0.72*xmax)


def measure_of_time_to_event(current_time, event_time):
    dt = current_time - event_time
    return 100.0 / (1.0 + dt.days ** 2)


def main_for_one_member():

    #define animation period
    current_start_date = datetime(1970,2,1,0,0,0)
    current_end_date = datetime(1970,8,1,0,0,0)

    future_start_date = datetime(2041,2,1,0,0,0)
    future_end_date = datetime(2070,8,1,0,0,0)

    simulation_step = timedelta(days = 5)

    folder_path = 'data/streamflows/hydrosheds_euler9'
    current_file = 'aet_discharge_1970_01_01_00_00.nc'
    future_file = 'aeu_discharge_2041_01_01_00_00.nc'
    


    current_path = os.path.join(folder_path, current_file)
    future_path = os.path.join(folder_path, future_file)


    current_data = ModelDataObject(member_id = 'aet')
    current_data.init_from_path(current_path)

   
    future_data = ModelDataObject(member_id = 'aeu')
    future_data.init_from_path(future_path)



    #domain properties
    xs = polar_stereographic.xs
    ys = polar_stereographic.ys
    basemap = polar_stereographic.basemap

    positions = current_data.get_all_positions()
    i_indices = current_data.i_indices
    j_indices = current_data.j_indices
    index_zip = zip(i_indices, j_indices, positions)

    basin_patches = get_features_from_shape(basemap,linewidth = 0.5)


    

    #plot plots
    i = 0
    current_time = current_start_date
    future_time = future_start_date

    while current_time < current_end_date:
        to_plot_current = np.ma.masked_all(xs.shape)
        to_plot_future = np.ma.masked_all(xs.shape)
        for x, y, position in index_zip:
            current_year_pos = YearPositionObject(current_time.year, position)
            future_year_pos = YearPositionObject(future_time.year, position)

            current_high_date = current_data.select_high_date_for_year_and_position(current_year_pos)
            future_high_date = future_data.select_high_date_for_year_and_position(future_year_pos)

            to_plot_current[x, y] = measure_of_time_to_event(current_time, current_high_date)
            to_plot_future[x, y] = measure_of_time_to_event(future_time, future_high_date)



        #actual plotting
        plt.figure()
        plt.subplot(1,2,1)
        basemap.pcolormesh(xs, ys, to_plot_current, vmin = 0, vmax = 1, cmap = 'Blues')
        basemap.drawcoastlines(linewidth = 0.3)
       # plot_basin_boundaries_from_shape(basemap, plotter = plt, linewidth = 0.5)
        plot_patches(plt, get_copies(basin_patches))
        zoom_to_qc()
        plt.title(current_time.strftime('current: %Y-%m-%d'))

        plt.subplot(1,2,2)
        basemap.pcolormesh(xs, ys, to_plot_future, vmin = 0, vmax = 1, cmap = 'Blues', linewidth = 0.5)
        basemap.drawcoastlines(linewidth = 0.3)
        zoom_to_qc()
        plot_patches(plt, get_copies(basin_patches))
       # plot_basin_boundaries_from_shape(basemap, plotter = plt)
        plt.title(future_time.strftime('future: %Y-%m-%d'))
 
        plt.savefig('frame_%010d.png' % i)

        i += 1
        current_time += simulation_step
        future_time += simulation_step

        if current_time.month > current_end_date.month:
            current_time = datetime(current_time.year + 1, current_start_date.month, current_start_date.day)
            future_time = datetime(future_time.year + 1, future_start_date.month, future_start_date.day)

    pass


def main():
    db = OccurrencesDB()

    #define animation period
    current_start_date = datetime(1996,3,1,0,0,0)
    current_end_date = datetime(1999,8,1,0,0,0)

    future_start_date = datetime(2067,3,1,0,0,0)
    future_end_date = datetime(2070,8,1,0,0,0)

    simulation_step = timedelta(days = 5)


    #domain properties
    xs = polar_stereographic.xs
    ys = polar_stereographic.ys
    basemap = polar_stereographic.basemap

    positions = db.get_all_positions()
    i_indices = db.i_indices
    j_indices = db.j_indices
    index_zip = zip(i_indices, j_indices, positions)

    basin_patches = get_features_from_shape(basemap,linewidth = 0.5)

    #plot plots
    i = 0
    current_time = current_start_date
    future_time = future_start_date

    while current_time < current_end_date:
        to_plot_current = np.ma.masked_all(xs.shape)
        to_plot_future = np.ma.masked_all(xs.shape)
        for x, y, position in index_zip:
            current_year_pos = YearPositionObject(current_time.year, position)
            future_year_pos = YearPositionObject(future_time.year, position)

            to_plot_current[x, y] = \
                db.get_measure_of_distance_to_high_event(current_year_pos, current_time, current = True)
            to_plot_future[x, y] = \
                db.get_measure_of_distance_to_high_event(future_year_pos, future_time, current = False)



        #actual plotting
        plt.figure()
        plt.subplot(1,2,1)
        basemap.pcolormesh(xs, ys, to_plot_current, vmin = 0, vmax = 1, cmap = 'Blues')
        basemap.drawcoastlines(linewidth = 0.3)
       # plot_basin_boundaries_from_shape(basemap, plotter = plt, linewidth = 0.5)
        plot_patches(plt, get_copies(basin_patches))
        zoom_to_qc()
        plt.title(current_time.strftime('current: %Y-%m-%d'))

        plt.subplot(1,2,2)
        basemap.pcolormesh(xs, ys, to_plot_future, vmin = 0, vmax = 1, cmap = 'Blues', linewidth = 0.5)
        basemap.drawcoastlines(linewidth = 0.3)
        zoom_to_qc()
        plot_patches(plt, get_copies(basin_patches))
       # plot_basin_boundaries_from_shape(basemap, plotter = plt)
        plt.title(future_time.strftime('future: %Y-%m-%d'))

        plt.savefig('frame_%010d.png' % i)

        i += 1
        current_time += simulation_step
        future_time += simulation_step

        if current_time.month > current_end_date.month:
            current_time = datetime(current_time.year + 1, current_start_date.month, current_start_date.day)
            future_time = datetime(future_time.year + 1, future_start_date.month, future_start_date.day)


    pass

if __name__ == "__main__":
    main()
    print "Hello World"
