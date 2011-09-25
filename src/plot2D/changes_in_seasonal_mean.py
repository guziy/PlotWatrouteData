__author__="huziy"
__date__ ="$Aug 31, 2011 3:12:13 PM$"

from matplotlib.ticker import LinearLocator
import data.members as members
import data.data_select as data_select
import os
import numpy as np
import application_properties
from plot2D.map_parameters import polar_stereographic
import shape.basin_boundaries as bb
import matplotlib as mpl
import matplotlib.pyplot as plt
import gevfit.matplotlib_helpers.my_colormaps as my_cm

import util.plot_utils as plot_utils
from scipy import stats

#Plot a panel plot with 4 subplots
#DJF, ,,, (winter, spring, summer and autumn)

def calculate_seasonal_means(times, data, month_indices = range(1,13)):
    """
    Calculate seasonal means, for the specified months,
    means for the whole domain.
    month_indices is a list containing zero based indices of months
    Jan = 1,..., Dec = 12
    """
    result = {}
    for i, t in enumerate(times):
        if t.month not in month_indices:
            continue

        if result.has_key(t.year):
            result[t.year].append(data[i, :])
        else:
            result[t.year] = [data[i,:]]

    result1 = []
    for data in result.values():
        data1 = np.array(data)
        result1.append(np.mean(data1, axis = 0)) #mean over a winter of each year for all points

    return result1


def get_member_to_path_mapping(path_to_folder = 'data/streamflows/hydrosheds_euler9'):
    member_to_path = {}

    file_names = os.listdir(path_to_folder)
    for current_id in members.current_ids:
        future_id = members.current2future[current_id]
        for fName in file_names:
            the_path = os.path.join(path_to_folder, fName)
            if current_id in fName:
                member_to_path[current_id] = the_path
            if future_id in fName:
                member_to_path[future_id] = the_path
            if members.control_id in fName:
                member_to_path[members.control_id] = the_path

    return member_to_path


def calculate_seasonal_changes_in_mean_stfl_and_plot(folder_path = 'data/streamflows/hydrosheds_euler9',
                                                     months = None, subplot_dims = None, subplot_count = 1, label = ""):

    if months is None:
        print "please specify months"
        return

    member_to_path = get_member_to_path_mapping(path_to_folder = folder_path)

    current_data = []
    future_data = []
    i_indices = None
    j_indices = None
    for current_id in members.current_ids:
        future_id = members.current2future[current_id]

        path_c = member_to_path[current_id]
        path_f = member_to_path[future_id]

        stfl_c, times_c, i_indices, j_indices = data_select.get_data_from_file(path_c)
        stfl_f, times_f, i_indices, j_indices = data_select.get_data_from_file(path_f)

        current_data.extend(calculate_seasonal_means(times_c, stfl_c, month_indices = months))
        future_data.extend(calculate_seasonal_means(times_f, stfl_f, month_indices = months))


    current_data = np.array(current_data)
    future_data = np.array(future_data)

    c_mean = np.mean(current_data, axis = 0)
    f_mean = np.mean(future_data, axis = 0)
    print c_mean.shape

    t_value, p_value = stats.ttest_ind(current_data, future_data, axis = 0)

    change = (f_mean - c_mean) / c_mean * 100

    xs = polar_stereographic.xs
    ys = polar_stereographic.ys
    basemap = polar_stereographic.basemap



        #zoom to domain
    to_plot = np.ma.masked_all(xs.shape)
    for i, j, v_mean in zip(i_indices, j_indices, c_mean):
        to_plot[i,j] = v_mean


    selected_x = xs[~to_plot.mask]
    selected_y = ys[~to_plot.mask]
    marginx = abs(np.min(selected_x) * 5.0e-2)
    marginy = abs(np.min(selected_y) * 5.0e-2)


    #plot mean change
    plt.subplot(subplot_dims[0], subplot_dims[1], subplot_count)
    plt.title('Mean flow change, {0}'.format(label))
    to_plot = np.ma.masked_all(xs.shape)
    for i, j, the_change in zip(i_indices, j_indices, change):
        to_plot[i, j] = the_change

    basemap.pcolormesh(xs, ys, to_plot, cmap = my_cm.get_red_blue_colormap(ncolors = 6),
                       vmax = 150, vmin = -150)
    basemap.drawcoastlines()
    plt.colorbar(ticks = LinearLocator(numticks = 7), format = '%d')
    bb.plot_basin_boundaries_from_shape(basemap, plotter = plt, linewidth = 1, edge_color = 'k')
    plt.xlim(np.min(selected_x) - marginx, np.max(selected_x) + marginx)
    plt.ylim(np.min(selected_y) - marginy, np.max(selected_y) + marginy)


    #plot p-value
    plt.subplot(subplot_dims[0], subplot_dims[1], subplot_count + 1)
    plt.title('p-value, {0}'.format(label))
    to_plot = np.ma.masked_all(xs.shape)
    for i, j, pv in zip(i_indices, j_indices, p_value):
        to_plot[i, j] = pv

    basemap.pcolormesh(xs, ys, to_plot, cmap = mpl.cm.get_cmap('jet', 5), vmax = 0.2)
    basemap.drawcoastlines()
    plt.colorbar(ticks = LinearLocator(numticks = 6), format = '%.2f')
    bb.plot_basin_boundaries_from_shape(basemap, plotter = plt, linewidth = 1, edge_color = 'k')
    plt.xlim(np.min(selected_x) - marginx, np.max(selected_x) + marginx)
    plt.ylim(np.min(selected_y) - marginy, np.max(selected_y) + marginy)
    



    print current_data.shape
    
    pass


def calculate_mean_change_and_plot_for_all_seasons():

    seasons = ["DJF", "MAM", "JJA", "SON"]
    months = [(12,1,2), (3,4,5), (6,7,8), (9,10,11)]
    folder_path = 'data/streamflows/hydrosheds_euler9'

    subplot_count = 1
    plot_utils.apply_plot_params(aspect_ratio = 2)
    for s, m in zip(seasons, months):
        calculate_seasonal_changes_in_mean_stfl_and_plot(folder_path = folder_path,
            months = m,  subplot_dims = (4,2), subplot_count = subplot_count, label = s
        )
        subplot_count += 2

    plt.savefig('mean_seasonal_change.pdf', bbox_inches = 'tight')
    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    #calculate_seasonal_changes_in_mean_stfl_and_plot()
    calculate_mean_change_and_plot_for_all_seasons()
    print "Hello World"
