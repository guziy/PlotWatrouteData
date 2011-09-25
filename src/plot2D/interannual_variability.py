__author__="huziy"
__date__ ="$Aug 31, 2011 11:22:55 AM$"

import matplotlib.pyplot as plt
import numpy as np
import data.members as members
import os
import matplotlib as mpl
import data.data_select as data_select

from plot2D.map_parameters import polar_stereographic
import application_properties

import util.plot_utils as plot_utils
from matplotlib.ticker import LinearLocator
import shape.basin_boundaries as bb

def calculate_annual_means(times, data):
    """
    Calculate annual means for the domain
    """
    result = {}
    for i, t in enumerate(times):
        if result.has_key(t.year):
            result[t.year].append(data[i, :])
        else:
            result[t.year] = [data[i,:]]
    
    the_mean_list = []
    for year, data in result.iteritems():
        data1 = np.array(data)
        the_mean_list.append(np.mean(data1, axis = 0))

    the_mean_list = np.array(the_mean_list)
    return the_mean_list


def plot_subplot(i_indices, j_indices, data_1d, mark = ''):
    basemap = polar_stereographic.basemap
    xs = polar_stereographic.xs
    ys = polar_stereographic.ys

    to_plot = np.ma.masked_all(xs.shape)
    for i, j, f_v in zip(i_indices, j_indices, data_1d):
        to_plot[i,j] = f_v

    basemap.pcolormesh(xs, ys, to_plot, cmap = mpl.cm.get_cmap('jet', 7) )
    basemap.drawcoastlines()
    plt.colorbar(ticks = LinearLocator(numticks = 8), format = '%.1f')
    bb.plot_basin_boundaries_from_shape(basemap, plotter = plt, linewidth = 2)

    #zoom to domain
    selected_x = xs[~to_plot.mask]
    selected_y = ys[~to_plot.mask]
    marginx = abs(np.min(selected_x) * 5.0e-2)
    marginy = abs(np.min(selected_y) * 5.0e-2)

    plt.xlim(np.min(selected_x) - marginx, np.max(selected_x) + marginx)
    plt.ylim(np.min(selected_y) - marginy, np.max(selected_y) + marginy)
    title = '(%s)' % mark
    plt.title(title)



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



def compare_and_plot(path_to_folder = 'data/streamflows/hydrosheds_euler9'):
    """
    Calculates interannual variability (standard ceviations) for each pair of members
    and plots their ratios

    create annual mean matrices -> calculate standard deviations for future
    and current climate, plot ratios of variations for memebers and
    the std for the control run,
    """

    member_to_path = get_member_to_path_mapping(path_to_folder)
    plot_utils.apply_plot_params(aspect_ratio = 1.5)
    plt.figure()

    plot_marks = ['a', 'b', 'c', 'd', 'e']
    subplot_count = 1
    for current_id, plot_mark in zip(members.current_ids, plot_marks):
        future_id = members.current2future[current_id]

        path_c = member_to_path[current_id]
        path_f = member_to_path[future_id]

        stfl_c, times_c, i_indices, j_indices = data_select.get_data_from_file(path_c)
        stfl_f, times_f, i_indices, j_indices = data_select.get_data_from_file(path_f)

        means_c = calculate_annual_means(times_c, stfl_c)
        means_f = calculate_annual_means(times_f, stfl_f)

        std_c = np.std(means_c, axis = 0)
        std_f = np.std(means_f, axis = 0)

        f_values = std_f / std_c
        plt.subplot(3, 2, subplot_count)
        plot_subplot(i_indices, j_indices, f_values, mark = plot_mark)

        subplot_count += 1

    #plot variance for the control simulation
    plt.subplot(3,2, subplot_count)
    stfl_c, times_c, i_indices, j_indices = data_select.get_data_from_file(path_c)
    means_c = calculate_annual_means(times_c, stfl_c)
    std_c = np.std(means_c, axis = 0)
    plot_subplot(i_indices, j_indices, std_c, mark = 'f')

    super_title = 'a-e: Changes in interannual variability ($\\sigma_{\\rm future}/ \\sigma_{\\rm current}$). \n'
    super_title += 'f: Interannual variability of the control simulation'
    plt.suptitle(super_title)



    plt.show()

    

def plot_variability_1d(path_to_folder = 'data/streamflows/hydrosheds_euler9'):
    '''
    Compare current and future interannual variability taking into account
    intermember dispersion, for all the points of the domain.
    '''



    pass









if __name__ == "__main__":
    application_properties.set_current_directory()
    #compare_and_plot()
    print "Hello World"
