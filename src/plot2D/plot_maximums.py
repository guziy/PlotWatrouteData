
__author__="huziy"
__date__ ="$1 fevr. 2011 14:27:40$"


from plot2D.MaximumReader import VincentMaximumsReader
from datetime import datetime
import data.data_select as data_select
import plot2D.plot_data_2d as plot_data_2d
import matplotlib as mpl

import numpy as np
from datetime import timedelta
import application_properties
application_properties.set_current_directory()

from math import sqrt
import matplotlib.pyplot as plt
import pylab


inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 1500 * inches_per_pt          # width in inches
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



def compare_simulations(path1, path2, label1 = '1', label2 = '2', field_name = 'water_discharge'):
    data1, time1, i_indices, j_indices =  data_select.get_data_from_file(path1, field_name)
    data2, time2, i_indices, j_indices =  data_select.get_data_from_file(path2, field_name)

    the_mins1 = np.min(data1, axis = 0)
    the_mins2 = np.min(data2, axis = 0)


    the_maxs1 = np.max(data1, axis = 0)
    the_maxs2 = np.max(data2, axis = 0)

    the_means1 = np.mean(data1, axis = 0)
    the_means2 = np.mean(data2, axis = 0)

    #scatter plot for means
    plt.subplots_adjust(hspace = 0.5)

    plt.subplot(2,2,1)
    plt.title('means', override)
    plt.scatter( the_means1 , the_means2, linewidth = 0)
    plt.xlabel(label1)
    plt.ylabel(label2)

    x = plt.xlim()
    plt.plot(x,x, color = 'k')
    plt.grid(True)

    #scatter plot for minimums
    plt.subplot(2,2,2)
    plt.title('minimums', override)

    plt.scatter( the_mins1 , the_mins2, linewidth = 0)
    plt.xlabel(label1)
    plt.ylabel(label2)

    x = plt.xlim()
    plt.plot(x,x, color = 'k')
    plt.grid(True)

    #scatter plot for minimums
    plt.subplot(2,2,3)
    plt.title('maximums', override)
    plt.scatter( the_maxs1 , the_maxs2, linewidth = 0)
    plt.xlabel(label1)
    plt.ylabel(label2)

    x = plt.xlim()
    plt.plot(x,x, color = 'k')
    plt.grid(True)
    plt.savefig('{0}_{1}_scatter.png'.format(label1, label2), bbox_inches = 'tight')



    pass


#max over t
def plot_annual_extremes(data_path = 'data/streamflows/VplusFmask_newton/aex_discharge_1970_01_01_00_00.nc',
                         start_date = datetime(1970, 1,1, 0,0,0),
                         end_date = datetime(2000, 1,1, 0,0,0),
                         ):

    streamflows, times, i_array, j_array = data_select.get_data_from_file(data_path)

    period_start_month = 1
    period_end_month = 12
    the_minima = data_select.get_minimums_for_domain(streamflows, times,
                                             start_date = start_date, end_date = end_date,
                                             start_month = period_start_month,
                                             end_month = period_end_month,
                                             duration_days = 1)

    plot_data_2d.plot_data(the_minima, i_array, j_array, name = "minima", title = "min",
                            digits = 1,
                            color_map = mpl.cm.get_cmap("OrRd", 10),
                            minmax = (None, None),
                            units = "m**3/s")



    period_start_month = 4
    period_end_month = 6
    the_maximums = data_select.get_maximums_for_domain(streamflows, times,
                                             start_date = start_date, end_date = end_date,
                                             start_month = period_start_month,
                                             end_month = period_end_month,
                                             duration_days = 7)

    plot_data_2d.plot_data(the_maximums, i_array, j_array, name = "maxima", title = "max",
                            digits = 1,
                            color_map = mpl.cm.get_cmap("OrRd", 10),
                            minmax = (None, None),
                            units = "m**3/s")



    pass


override = {'fontsize': 20}

def compare_means(member = 'aet', my_data_path = '',
                  start_date = datetime(1961, 1, 1, 0, 0),
                  end_date = datetime(1990, 12, 31, 0, 0)):
    streamflows, times, i_array, j_array = data_select.get_data_from_file(my_data_path)

    
    event_duration = timedelta(days = 1)

    my_data = data_select.get_list_of_annual_maximums_for_domain(streamflows, times,
                                    start_date = start_date, end_date = end_date,
                                    start_month = 1, end_month = 12,
                                    event_duration = event_duration)


    data_path = 'data/streamflows/Vincent_annual_max/mapHIGH_{0}.txt'.format(member)
    v = VincentMaximumsReader(data_path = data_path)

    the_format = '{0}: i = {1}, j = {2}, min = {3}, max = {4}, mean = {5}'
    vmeans = []
    vmins = []
    vmaxs = []
   # my_data = 500 * np.ones((10,547))
    for i, j, the_index in zip(i_array, j_array, range(my_data.shape[1])):
        data = my_data[:, the_index]
        print the_format.format('Sasha', i, j, np.min(data), np.max(data), np.mean(data))
        data = v.get_data_at(i + 1, j + 1)
        print the_format.format('Vincent', i, j, np.min(data), np.max(data), np.mean(data))
        vmeans.append(np.mean(data))
        vmins.append(np.min(data))
        vmaxs.append(np.max(data))
        print '=' * 30


    #scatter plot for means
    plt.subplots_adjust(hspace = 0.5)

    plt.subplot(2,2,1)
    plt.title('annual maximums, \n average for each grid point', override)
    plt.scatter( vmeans , np.mean(my_data, axis = 0), linewidth = 0)
    plt.xlabel('Vincent')
    plt.ylabel('Sasha')

    x = plt.xlim()
    plt.plot(x,x, color = 'k')
    plt.grid(True)
    
    #scatter plot for minimums
    plt.subplot(2,2,2)
    plt.title('annual maximums, \n minimum for each grid point', override)

    plt.scatter( vmins , np.min(my_data, axis = 0), linewidth = 0)
    plt.xlabel('Vincent')
    plt.ylabel('Sasha')

    x = plt.xlim()
    plt.plot(x,x, color = 'k')
    plt.grid(True)
    
    #scatter plot for minimums
    plt.subplot(2,2,3)
    plt.title('annual maximums, \n maximum for each grid point', override)
    plt.scatter( vmaxs , np.max(my_data, axis = 0), linewidth = 0)
    plt.xlabel('Vincent')
    plt.ylabel('Sasha')

    x = plt.xlim()
    plt.plot(x,x, color = 'k')
    plt.grid(True)
    plt.savefig('{0}_scatter_max.png'.format(member), bbox_inches = 'tight')




def main():

    data_path = 'data/streamflow/to_compare_with_Vincent/aet_discharge_1970_01_01_00_00.nc'
    plot_annual_extremes(data_path,
                         start_date = datetime(1961, 1,1, 0,0,0),
                         end_date = datetime(1990,12, 31 , 0,0,0)
                         )

if __name__ == "__main__":
    #main()

    current_members = ['aet', 'aev']
    future_members = ['aeu', 'aew']

    current_start_date = datetime(1961, 1, 1, 0, 0)
    current_end_date = datetime(1990, 12, 31, 0, 0)

    future_start_date = datetime(2041, 1, 1, 0, 0)
    future_end_date = datetime(2070, 12, 31, 0, 0)


    for member in current_members:
        print member
        data_path = 'data/streamflows/to_compare_with_Vincent/{0}_discharge_1961_01_01_00_00.nc'.format(member)
        compare_means(member, my_data_path = data_path, start_date = current_start_date, end_date = current_end_date)

    for member in future_members:
        print member
        data_path = 'data/streamflows/to_compare_with_Vincent/{0}_discharge_2041_01_01_00_00.nc'.format(member)
        compare_means(member, my_data_path = data_path, start_date = future_start_date, end_date = future_end_date)

    print "Hello World"
