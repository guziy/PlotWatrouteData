from station_manager import StationManager
from data.modelpoint import ModelPoint
from util.lat_lon_holder import LatLonHolder
from math import *

from copy import *
from datetime import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pylab
from data.station import *
import os

import application_properties
application_properties.set_current_directory()



PATH_TO_MODEL_DATA = 'data/dstflow_aex 1.txt'
MODEL_DATA_FOLDER = 'data/simulation_results'
DATE_TIME_FORMAT = '%d/%m/%Y %Hh%S'

###In this module are the functions which read text files created by route model
###
###

def read_file(model_points, path = PATH_TO_MODEL_DATA):
    '''
    read dstflow file to fill in the list model_points
    '''
    f = open(path)
    header = f.readline()

    if 'dstflow' in path:
        tokens = header.split()
        for i in range(0, len(tokens) - 1, 2):
            model_points.append(ModelPoint(int(tokens[i].strip()), int(tokens[i + 1].strip())))
    else:
        tokens = header.split()
        for token in tokens:
            model_points.append(ModelPoint(id = token))

    for line in f:
        tokens = line.split()
        if len(tokens) == 0:
            continue
        date = datetime.strptime(tokens[1] + ' ' + tokens[2], DATE_TIME_FORMAT)
        for i in range(len(model_points)):
            model_points[i].add_timeseries_data(date, float(tokens[i + 3]))


def set_axes_font_size(axis, fontsize):
# update the font size of the x and y axes
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
     


def plot_monthly_means(model_points):
    plt.cla()
    years    = mdates.YearLocator(5, month=1, day=1)   # every year
    months   = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')


    fig = plt.figure()
    num_rows = len(model_points) / 2 + len(model_points) % 2
    num_cols = 2
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = 100 * inches_per_pt           # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, fig_height]

    params = {'axes.labelsize': 10,
        'text.fontsize': 5,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': fig_size}

    pylab.rcParams.update(params)


    for i, station in enumerate(model_points):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        print 'ax = ', ax
        
        # @type station Station
        dates = station.get_monthly_dates_sorted()
        values = station.get_monthly_means_sorted_by_date()
        #ax.plot(model_dates, values, label=station.__str__())

        plt.title(station)

        plt.bar(dates, values)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)

        # ax.xaxis.set_minor_locator(months)


    fig.savefig('water_means.png')
    



        
def plot_data(model_points, save_to_file = True, figure_obj = None,
                            num_rows = None, num_cols = None, sub_plot_index = None):
    years    = mdates.YearLocator(5, month=1, day=1)   # every year
    months   = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')


    dates = model_points[0].get_dates_sorted();


    if figure_obj == None:
        figure_obj = plt.figure()
        num_cols = 2
        num_rows = len(model_points) / num_cols
        if len(model_points) % num_cols > 0:
            num_rows += 1
        

    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = 100 * inches_per_pt           # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, fig_height]

    params = {'axes.labelsize': 10,
        'text.fontsize': 5,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': fig_size}
    
    pylab.rcParams.update(params)


    
    for i, model_point in enumerate(model_points):
        sub_index = sub_plot_index if sub_plot_index != None else i + 1
        ax = figure_obj.add_subplot(num_rows, num_cols, sub_index)
        values = model_point.get_values_sorted_by_date()
        print values[5]
        ax.plot(dates, values, label=model_point.__str__())
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)



        # ax.xaxis.set_minor_locator(months)

    if save_to_file:
        fig.savefig('water.png')



def plot_monthly_means_with_measurements(model_points):
    years    = mdates.YearLocator(5, month=1, day=1)   # every year
    months   = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    plt.cla()
    
    ##holder of longitudes and latitudes of the model grid points
    holder = LatLonHolder()

    #Manager of the measurement data
    station_manager = StationManager()
    station_manager.read_stations_from_files_rivdis()
    station_manager.read_data_from_files_hydat()

    print station_manager.get_station_by_id('02NG005')

    measure_stations = []
    selected_model_points = []
    for model_point in model_points:
        lon, lat = holder.get_lon_lat(model_point.ix, model_point.iy) 
        the_station = station_manager.get_station_closest_to(lon, lat)
        if the_station != None:
            measure_stations.append(the_station)
            model_point.set_longitude(lon)
            model_point.set_latitude(lat)
            selected_model_points.append(model_point)

        

    fig = plt.figure(figsize = (18, 18))


    num_cols = 2
    num_rows = len(selected_model_points) / num_cols
    if len(selected_model_points) % num_cols > 0:
        num_rows += 1


    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = 100.0 * inches_per_pt           # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, 10 * fig_height]

    params = {'axes.labelsize': 14,
        'text.fontsize': 14,
        'font.size': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
       }

    print 'fig_size = ', fig_size

    pylab.rcParams.update(params)


    assert len(selected_model_points) > 0
    model_dates = selected_model_points[0].get_monthly_dates_sorted();
    
    
    for i, model_point in enumerate(selected_model_points):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        #plot model data
        values = model_point.get_monthly_means_sorted_by_date() 
        ax.plot(model_dates, values, label = 'model' )

        #plot measurement data
        values = measure_stations[i].get_values_for_dates(model_dates)
        ax.plot(model_dates, values, '-', label = 'measurements', lw = 3 )

        plt.title(model_point.__str__() + '\n' + measure_stations[i].__str__())

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)


        ax.legend()
        # ax.xaxis.set_minor_locator(months)


    fig.subplots_adjust(wspace = 0.4)
    #fig.autofmt_xdate()
    fig.savefig('comparison.png')
    
    
    pass



def get_run_type(file_name):
    '''
    returns the type of the model run aet, aex , etc. from the file name
    '''
    return file_name[-7:-4]


def plot_files(prefix = 'dstflow', column = 0):
    
    files = os.listdir(MODEL_DATA_FOLDER)

    files_to_process = []
    for file in files:
        if file.startswith(prefix):
            the_path = MODEL_DATA_FOLDER + os.sep + file
            files_to_process.append(the_path)

    figure_obj = plt.figure()
    num_cols = 2
    n_files_to_process = len( files_to_process )

    num_rows = n_files_to_process / num_cols
    if n_files_to_process % num_cols > 0:
        num_rows += 1


    for i, path in  enumerate(files_to_process):
        model_points = []
        read_file( model_points, path )
        print len(model_points)

        print column
        plot_data([model_points[column]], save_to_file = False, figure_obj = figure_obj,
                                          num_cols = num_cols, num_rows = num_rows,
                                          sub_plot_index = i + 1)
        plt.title(get_run_type(path))
    plt.subplots_adjust(hspace = 0.5)
    plt.savefig("scenario_comparisons.png")



def plot_core_comparison(model_points1, model_points2, label1 = "set 1", label2 = "set 2"):
    plt.cla()
    years    = mdates.YearLocator(5, month=1, day=1)   # every year
    months   = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')


    

    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = 100 * inches_per_pt           # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, fig_height]

    params = {'axes.labelsize': 10,
        'text.fontsize': 5,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': fig_size}

    pylab.rcParams.update(params)

    i = 0
    dates = model_points1[0].get_dates_sorted()
    for point1 in model_points1:
        for point2 in model_points2:
            # @type point1 ModelPoint
            if point1.ix == point2.ix and point1.iy == point2.iy:
                plt.cla()

                plt.title(point1.__str__())

                ax = plt.subplot(111)
                values = point1.get_values_for_dates(dates)
                ax.plot(dates, values,  label = label1)

                values = point2.get_values_for_dates(dates)
                ax.plot(dates, values, 'o', markersize = 2.0 , label = label2)

                # format the ticks
                ax.xaxis.set_major_locator(years)
                ax.xaxis.set_major_formatter(yearsFmt)


                ax.legend()
                plt.savefig('compare_%d.png' % i)
                print 'i = ', i
                i += 1
                break
    pass


if __name__ == '__main__':
   print 'backend: ', plt.get_backend() 
   # plot_files(prefix = 'lakes', column = 3)
   model_points_uqam = []
   model_points_waterloo = []

   read_file(model_points_uqam, 'data/route_core_comparison/dstflow_aez_UQAM.txt')
   read_file(model_points_waterloo, 'data/route_core_comparison/dstflow_aez_Waterloo11.txt')

   plot_core_comparison(model_points_uqam, model_points_waterloo, 'UQAM', 'Waterloo')

    
