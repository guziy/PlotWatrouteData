from readers.read_infocell import Basin
import os.path

__author__="huziy"
__date__ ="$Mar 17, 2011 3:37:50 PM$"

import os
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import plot2D.calculate_performance_errors as pe
import data.members as members
import lowflow

import readers.read_infocell as infocell

import data.data_select as data_select

import application_properties
application_properties.set_current_directory()




def plot_mean_hydrograph_with_gw_ouflow():
#TODO: Implement
    basins = infocell.get_basins_with_cells_connected_using_hydrosheds_data()


    data_path = 'data/streamflows/hydrosheds_euler12/aew_discharge_2041_01_01_00_00.nc'

    basinName = 'RDO'
    theBasin = None
    for basin in basins:
        # @type basin Basin
        if basin.name == basinName:
            theBasin = basin
            break

    data = data_select.get_data_from_file(path = data_path, field_name = 'gw_outflow')
    gw_outflow = data[0]
    times = data[1]
    x_index = data[2]
    y_index = data[3]

    data = data_select.get_data_from_file(path = data_path, field_name = 'surface_runoff')
    surface_runoff = data[0]

    for t in times:
        pass
    





#compares mean hydrographs
def compare_means(member_id = 'aex' ,data_folder1 = '', label1 = '', data_folder2 = '', label2 = ''):
    basin_path = 'data/infocell/amno180x172_basins.nc'
    basin_indices = lowflow.read_basin_indices(basin_path)


    for f in os.listdir(data_folder1):
        if f.lower().startswith(member_id):
            path1 = os.path.join(data_folder1, f)

    for f in os.listdir(data_folder2):
        if f.lower().startswith(member_id):
            path2 = os.path.join(data_folder2, f)


    discharge_1, times1, i_list, j_list = data_select.get_data_from_file(path1, 'water_discharge')
    discharge_2, times2, i_list, j_list = data_select.get_data_from_file(path2, 'water_discharge')



    discharge_values_1 = []
    discharge_values_2 = []
   

    for pos in range(discharge_1.shape[1]):
        dates, discharge_tmp = pe.average_for_each_day_of_year(times1, discharge_1[:, pos], year = 2000)
        discharge_values_1.append(np.array(discharge_tmp))

        dates, discharge_tmp = pe.average_for_each_day_of_year(times2, discharge_2[:, pos], year = 2000)
        discharge_values_2.append(np.array(discharge_tmp))


    basin_to_discharge_1 = {}
    basin_to_discharge_2 = {}

    the_zip = zip(i_list, j_list, discharge_values_1, discharge_values_2)

    for basin in basin_indices:
        for i, j, d_1, d_2 in the_zip:
            if basin.mask[i, j] == 1:
                if basin_to_discharge_1.has_key(basin):
                    basin_to_discharge_1[basin] += d_1
                    basin_to_discharge_2[basin] += d_2
                else:
                    basin_to_discharge_1[basin] = d_1
                    basin_to_discharge_2[basin] = d_2

    for basin in basin_to_discharge_1.keys():
        n = float(basin.get_number_of_cells())
        basin_to_discharge_1[basin] /= n
        basin_to_discharge_2[basin] /= n


    plt.figure()
    n = 1
    for basin, d in basin_to_discharge_1.iteritems():
        plt.subplot(7, 3, n)
        plt.title(basin.name)
        dicharge_line_1 = plt.plot(dates, d, linewidth = 2, color = 'b')
        discharge_line_2 = plt.plot(dates, basin_to_discharge_2[basin],
                                    linewidth = 2, color = 'r')

        #runoff_line = plt.plot(dates, basin_to_runoff[basin])

        ax = plt.gca()
        ax.xaxis.set_major_locator(
            mpl.dates.MonthLocator(bymonth = range(2,13,2))
        )


        ax.xaxis.set_major_formatter(
            mpl.dates.DateFormatter('%b')
        )
        n += 1

    plt.figlegend([dicharge_line_1, discharge_line_2], [label1, label2], 'upper right')
    plt.savefig('{0}_hydrographs.png'.format(member_id), bbox_inches = 'tight')

    pass

def plot_basin_mean_hydrograph(current_id = 'aex', future_id = None,
                                data_folder = 'data/streamflows/hydrosheds_euler7',
                                current_start_date = None, current_end_date = None,
                                future_start_date = None, future_end_date = None):



    basin_path = 'data/infocell/amno180x172_basins.nc'
    basin_indices = lowflow.read_basin_indices(basin_path)


    for f in os.listdir(data_folder):
        if f.lower().startswith(current_id):
            path_current = os.path.join(data_folder, f)
        if f.lower().startswith(future_id):
            path_future = os.path.join(data_folder, f)


    discharge_current, times_current, i_list, j_list = data_select.get_data_from_file(path_current, 'water_discharge')
    discharge_future, times_future, i_list, j_list = data_select.get_data_from_file(path_future, 'water_discharge')



    discharge_values_current = []
    discharge_values_future = []

    for pos in range(discharge_current.shape[1]):
        dates, discharge1 = pe.average_for_each_day_of_year(times_current, discharge_current[:, pos],
                                   start_date = current_start_date,
                                   end_date = current_end_date, year = 2000)
        discharge_values_current.append(np.array(discharge1))


        dates, discharge1 = pe.average_for_each_day_of_year(times_future, discharge_future[:, pos],
                                   start_date = future_start_date,
                                   end_date = future_end_date, year = 2000)
        discharge_values_future.append(np.array(discharge1))

    
    basin_to_discharge_current = {}
    basin_to_discharge_future = {}
    
    the_zip = zip(i_list, j_list, discharge_values_current, discharge_values_future)

    for basin in basin_indices:
        for i, j, d_current, d_future in the_zip:
            if basin.mask[i, j] == 1:
                if basin_to_discharge_current.has_key(basin):
                    basin_to_discharge_current[basin] += d_current
                    basin_to_discharge_future[basin] += d_future
                else:
                    basin_to_discharge_current[basin] = d_current
                    basin_to_discharge_future[basin] = d_future

    for basin in basin_to_discharge_current.keys():
        n = float(basin.get_number_of_cells())
        basin_to_discharge_current[basin] /= n
        basin_to_discharge_future[basin] /= n


    plt.figure()
    n = 1
    plt.subplots_adjust(hspace = 0.5)
    for basin, d in basin_to_discharge_current.iteritems():
        plt.subplot(7, 3, n)
        plt.title(basin.name)

        dicharge_line_current = plt.plot(dates, d, linewidth = 2, color = 'b')
        discharge_line_future = plt.plot(dates, basin_to_discharge_future[basin], linewidth = 2,
                                            color = 'r')

        plt.ylabel('${\\rm m^3/s}$')
        #runoff_line = plt.plot(dates, basin_to_runoff[basin])

        ax = plt.gca()
        ax.xaxis.set_major_locator(
            mpl.dates.MonthLocator(bymonth = range(2,13,2))
        )


        ax.xaxis.set_major_formatter(
            mpl.dates.DateFormatter('%b')
        )
        n += 1

    plt.figlegend([dicharge_line_current, discharge_line_future], ['current', 'future'], 'upper right')
    plt.savefig('{0}_{1}_hydrographs.png'.format(current_id, future_id), bbox_inches = 'tight')

    pass


def main():
    data_folder = 'data/streamflows/hydrosheds_euler10_spinup100yrs'
    current_start_date = datetime(1970, 1, 1, 0, 0)
    current_end_date = datetime(1999, 12, 31, 0, 0)

    future_start_date = datetime(2041, 1, 1, 0, 0)
    future_end_date = datetime(2070, 12, 31, 0, 0)

    data_folder2 = 'data/streamflows/hydrosheds_euler7_without_gw'

    for current_id in members.current_ids:
        future_id = members.current2future[current_id]
        plot_basin_mean_hydrograph(current_id = current_id, future_id = future_id,
                        data_folder = data_folder,
                        current_start_date = current_start_date, current_end_date = current_end_date,
                        future_start_date = future_start_date, future_end_date = future_end_date
                        )
        print current_id
        break
#        compare_means(current_id, data_folder1 = data_folder, label1 = 'with GW' ,
#                                  data_folder2 = data_folder2, label2 = 'without GW')
        



if __name__ == "__main__":
    main()

    print "Hello World"
