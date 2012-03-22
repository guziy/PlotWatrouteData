from multiprocessing.pool import Pool
import os
import pickle
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.ticker import LinearLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from ccc import ccc_analysis
from data import members
from diagnose_ccc.plot_seasonal_mean_biases import _get_routing_indices
from plot2D.map_parameters import polar_stereographic
from shape.basin_boundaries import plot_basin_boundaries_from_shape
from util import plot_utils
import matplotlib.pyplot as plt
import gevfit.matplotlib_helpers.my_colormaps as my_cm
import matplotlib as mpl
from matplotlib import colors

from scipy import stats
__author__ = 'huziy'

import numpy as np


def _get_annual_means_for_year_range_p(args):
    """
    function to be used in multiprocessing Pool
    """
    m_folder, year_range, months = args
    return ccc_analysis.get_seasonal_means_for_year_range(m_folder, year_range, months=months)

def _get_data(data_folder = "data/crcm4_data", v_name = "pcp",
              member_list = None, year_range = None, months = None):
    """
    returns seasonal means of each year for all members in the list
    Note!: uses caching
    """
    year_range = list(year_range)
    cache_file = "_".join(member_list) + "_" + "_".join(map(str, months)) + \
                 "_{0}_from_{1}_to_{2}_cache.bin".format(v_name, year_range[0], year_range[-1])



    if os.path.isfile(cache_file):
        return pickle.load(open(cache_file))

    p = Pool(processes=len(member_list))

    #prepare input for the parallel processes
    m_folders = map(lambda x: os.path.join(data_folder,"{0}_p1{1}".format(x, v_name)), member_list)
    year_ranges = [year_range] * len(member_list)
    months_for_p = [months] * len(member_list)
    #calculate means
    result = p.map(_get_annual_means_for_year_range_p, zip(m_folders, year_ranges, months_for_p))

    result = np.concatenate(result, axis = 0) #shape = (n_members * len(year_range)) x nx x ny
    print result.shape

    pickle.dump(result, open(cache_file, "w"))
    return result


def plot_temp():
    season_to_months = {
        "DJF" : [12, 1, 2],
        "MAM" : [3, 4, 5],
        "JJA" : [6, 7, 8],
        "SON" : [9, 10, 11],
        "Annual": range(1, 13)
    }
    var_name = "st"
    year_range_c = xrange(1970,2000)
    year_range_f = xrange(2041,2071)
    x, y = polar_stereographic.xs, polar_stereographic.ys
    i_array, j_array = _get_routing_indices()
    x_min, x_max, y_min, y_max = plot_utils.get_ranges(x[i_array, j_array], y[i_array, j_array])



    plot_utils.apply_plot_params(width_pt= None, font_size=9, aspect_ratio=2.5)
    fig = plt.figure()
    assert isinstance(fig, Figure)

    basemap = polar_stereographic.basemap
    assert isinstance(basemap, Basemap)
    gs = gridspec.GridSpec(3,4, height_ratios=[1,1,1], width_ratios=[1,1,1,1])

    #color_map = my_cm.get_red_blue_colormap(ncolors = 20, reversed=True)
    color_map = mpl.cm.get_cmap(name="jet", lut=10)
    clevels = xrange(-8, 9, 2)
    all_plot_axes = []
    for i, season in enumerate(season_to_months.keys()):
        if not i:
            ax = fig.add_subplot(gs[0,1:3])
        else:
            row, col = (i - 1)  // 2 + 1, i % 2
            ax = fig.add_subplot(gs[row, col * 2 : col * 2 + 2 ])
        all_plot_axes.append(ax)


        current = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.current_ids, year_range=year_range_c)
        future = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.future_ids, year_range=year_range_f)


        t, p = stats.ttest_ind(current, future, axis=0)

        significant = np.array(p <= 0.05)

        assert not np.all(~significant)
        assert not np.all(significant)

        current_m = np.mean(current, axis=0)
        future_m = np.mean(future, axis= 0)

        delta = (future_m - current_m)

        assert isinstance(ax, Axes)

        #delta = maskoceans(polar_stereographic.lons, polar_stereographic.lats, delta)
        save = delta[i_array, j_array]
        delta = np.ma.masked_all(delta.shape)
        delta[i_array, j_array] = save
        d_min = np.floor( np.min(save) )
        d_max = np.ceil( np.max(save) )

        img = basemap.pcolormesh(x, y, delta, cmap = color_map, vmin = d_min, vmax = d_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "8%", pad="3%")
        assert isinstance(cax, Axes)

        int_ticker = LinearLocator(numticks = color_map.N + 1)
        cb = fig.colorbar(img, cax = cax, ticks = int_ticker)




        where_significant = significant
        significant = np.ma.masked_all(significant.shape)

        significant[~where_significant] = 0
        basemap.pcolormesh( x, y, significant , cmap = mpl.cm.get_cmap(name = "gray", lut = 3),
                           vmin = -1, vmax = 1, ax = ax)

        ax.set_title(season)

    for the_ax in all_plot_axes:
        the_ax.set_xlim(x_min, x_max)
        the_ax.set_ylim(y_min, y_max)
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.1)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=0.4)


    gs.update(wspace=0.5)
    fig.suptitle("Projected changes, T(2m), degrees, CRCM4")
    fig.savefig("proj_change_{0}_ccc.png".format(var_name))

def plot_precip():
    season_to_months = {
        "DJF" : [12, 1, 2],
        "MAM" : [3, 4, 5],
        "JJA" : [6, 7, 8],
        "SON" : [9, 10, 11],
        "Annual": range(1, 13)
    }
    var_name = "pcp"
    year_range_c = xrange(1970,2000)
    year_range_f = xrange(2041,2071)
    x, y = polar_stereographic.xs, polar_stereographic.ys
    i_array, j_array = _get_routing_indices()
    x_min, x_max, y_min, y_max = plot_utils.get_ranges(x[i_array, j_array], y[i_array, j_array])



    plot_utils.apply_plot_params(width_pt= None, font_size=9, aspect_ratio=2.5)
    fig = plt.figure()
    assert isinstance(fig, Figure)

    basemap = polar_stereographic.basemap
    assert isinstance(basemap, Basemap)
    gs = gridspec.GridSpec(3,4, height_ratios=[1,1,1], width_ratios=[1,1,1,1])

    #color_map = my_cm.get_red_blue_colormap(ncolors = 20, reversed=True)
    color_map = mpl.cm.get_cmap(name="jet_r", lut=10)
    clevels = xrange(-8, 9, 2)
    all_plot_axes = []
    for i, season in enumerate(season_to_months.keys()):
        if not i:
            ax = fig.add_subplot(gs[0,1:3])
        else:
            row, col = (i - 1)  // 2 + 1, i % 2
            ax = fig.add_subplot(gs[row, col * 2 : col * 2 + 2 ])
        all_plot_axes.append(ax)


        current = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.current_ids, year_range=year_range_c)
        future = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.future_ids, year_range=year_range_f)


        t, p = stats.ttest_ind(current, future, axis=0)

        significant = np.array(p <= 0.05)

        assert not np.all(~significant)
        assert not np.all(significant)

        current_m = np.mean(current, axis=0)
        future_m = np.mean(future, axis= 0)

        seconds_per_day = 24 * 60 * 60
        delta = (future_m - current_m) * seconds_per_day
        delta = np.array(delta)

        assert isinstance(ax, Axes)

        #delta = maskoceans(polar_stereographic.lons, polar_stereographic.lats, delta)
        save = delta[i_array, j_array]
        delta = np.ma.masked_all(delta.shape)
        delta[i_array, j_array] = save
        d_min = np.floor( np.min(save) * 10 ) / 10.0
        d_max = np.ceil( np.max(save) *10 ) / 10.0


        img = basemap.pcolormesh(x, y, delta, cmap = color_map, vmin = d_min, vmax = d_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "8%", pad="3%")
        assert isinstance(cax, Axes)

        int_ticker = LinearLocator(numticks = color_map.N + 1)

        cb = fig.colorbar(img, cax = cax, ticks = int_ticker)





        where_significant = significant
        significant = np.ma.masked_all(significant.shape)

        significant[(~where_significant)] = 0
        save = significant[i_array, j_array]
        significant = np.ma.masked_all(significant.shape)
        significant[i_array, j_array] = save

        basemap.pcolormesh( x, y, significant , cmap = mpl.cm.get_cmap(name = "gray", lut = 3),
                           vmin = -1, vmax = 1, ax = ax)

        ax.set_title(season)

    for the_ax in all_plot_axes:
        the_ax.set_xlim(x_min, x_max)
        the_ax.set_ylim(y_min, y_max)
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.1)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=0.4)


    gs.update(wspace=0.5)

    #gs.tight_layout(fig)
    fig.suptitle("Projected changes, total precip (mm/day), CRCM4")
    fig.savefig("proj_change_{0}_ccc.png".format(var_name))

    pass



def main():
    plot_precip()
    plot_temp()
    pass

if __name__ == "__main__":
    import time
    import application_properties

    t0 = time.clock()
    application_properties.set_current_directory()
    main()
    print("Execution time: {0} seconds".format(time.clock() - t0))
    print "Hello world"
  