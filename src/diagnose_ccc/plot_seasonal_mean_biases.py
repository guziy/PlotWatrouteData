from datetime import datetime, timedelta
import os
import pickle
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LinearLocator, MultipleLocator
from mpl_toolkits.basemap import Basemap, maskoceans
import application_properties
from ccc import ccc_analysis
from data import data_select
from data.cru_data_reader import CruReader
from util import plot_utils

__author__ = 'huziy'
from shape.basin_boundaries import plot_basin_boundaries_from_shape
import numpy as np
import matplotlib.pyplot as plt
from plot2D.map_parameters import polar_stereographic
import gevfit.matplotlib_helpers.my_colormaps as my_cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data import members

def _get_routing_indices():
    """
    Used for the plot domain centering
    """
    i_indices, j_indices = data_select.get_indices_from_file(path = "data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc")
    return i_indices, j_indices



def _get_comparison_data(    start_date = datetime(1970,1,1),
    end_date = datetime(1999, 12, 31),
    crcm4_data_folder = "data/ccc_data/aex/aex_p1st",
    cru_data_path = "data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.tmp.dat.nc",
    cru_var_name = "tmp", season_to_months = None, crcm4_id = "aex"):



    cache_file = "seasonal_bias_{0}.bin".format(cru_var_name)

    if not os.path.isfile(cache_file):
        #get crcm4 data
        crcm4_data_store = {}
        day_seconds = 24.0 * 60.0 * 60.0
        dt = members.id_to_step[crcm4_id]
        crcm4_coef = day_seconds
        for season, month_list in season_to_months.iteritems():
            data = ccc_analysis.get_seasonal_mean(crcm4_data_folder, start_date=start_date,
                end_date=end_date, months=month_list
            )

            if cru_var_name == "pre":
                data *= crcm4_coef

            crcm4_data_store[season] = data

        #get cru data
        cru_data_store = {}
        cru = CruReader(path=cru_data_path, var_name=cru_var_name)
        lons2d_amno, lats2d_amno = polar_stereographic.lons, polar_stereographic.lats
        dt = cru.get_time_step()
        cru_coef = day_seconds / (dt.seconds + dt.days * day_seconds)
        for season, month_list in season_to_months.iteritems():
            data = cru.get_seasonal_mean_field(months=month_list, start_date=start_date, end_date=end_date)

            nneighbors = 1 if cru_var_name == "pre" else 4

            data = cru.interpolate_data_to(data, lons2d_amno, lats2d_amno, nneighbors=nneighbors)

            if cru_var_name == "pre":
                data *= cru_coef

            cru_data_store[season] = data

        data_list = [cru_data_store, crcm4_data_store]
        pickle.dump(data_list, open(cache_file, "w"))
    else:
        data_list = pickle.load(open(cache_file))

    cru_data_store, crcm4_data_store = data_list
    return cru_data_store, crcm4_data_store


def plot_precip_biases():
    season_to_months = {
        "DJF" : [12, 1, 2],
        "MAM" : [3, 4, 5],
        "JJA" : [6, 7, 8],
        "SON" : [9, 10, 11],
        "Annual": range(1, 13)
    }
    cru_var_name = "pre"
    cru_data_store, crcm4_data_store = _get_comparison_data(
        crcm4_data_folder="data/ccc_data/aex/aex_p1pcp",
        cru_data_path = "data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc",
        cru_var_name=cru_var_name, season_to_months=season_to_months
    )

    x, y = polar_stereographic.xs, polar_stereographic.ys
    i_array, j_array = _get_routing_indices()
    x_min, x_max, y_min, y_max = plot_utils.get_ranges(x[i_array, j_array], y[i_array, j_array])



    plot_utils.apply_plot_params(width_pt= None, font_size=9, aspect_ratio=2.5)

    fig = plt.figure()
    assert isinstance(fig, Figure)

    basemap = polar_stereographic.basemap
    assert isinstance(basemap, Basemap)
    gs = gridspec.GridSpec(3,2, width_ratios=[1,1], height_ratios=[1, 1, 1])

    color_map = my_cm.get_red_blue_colormap(ncolors = 16, reversed=False)
    color_map.set_over("k")
    color_map.set_under("k")
    all_plot_axes = []
    img = None
    for i, season in enumerate(season_to_months.keys()):
        if not i:
            ax = fig.add_subplot(gs[0,:])
        else:
            row, col = (i - 1)  // 2 + 1, i % 2
            ax = fig.add_subplot(gs[row, col])
        all_plot_axes.append(ax)
        assert isinstance(ax, Axes)

        delta = crcm4_data_store[season] - cru_data_store[season]
        save = delta[i_array, j_array]
        delta[:, :] = np.ma.masked
        delta[i_array, j_array] = save
        img = basemap.pcolormesh(x, y, delta, cmap = color_map, vmin = -2, vmax = 2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "8%", pad="3%")
        int_ticker = LinearLocator(numticks = color_map.N + 1)
        fig.colorbar(img, cax = cax, ticks = MultipleLocator(base = 0.5))
        ax.set_title(season)

    for the_ax in all_plot_axes:
        assert isinstance(the_ax, Axes)
        the_ax.set_xlim(x_min, x_max)
        the_ax.set_ylim(y_min, y_max)
        the_ax.set_xmargin(0)
        the_ax.set_ymargin(0)
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.5)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=1)

#    ax = fig.add_subplot(gs[3,:])
#    assert isinstance(ax, Axes)
#    fig.colorbar(img, cax = ax, orientation = "horizontal", ticks = MultipleLocator(base = 0.5))

    #gs.tight_layout(fig)
    fig.suptitle("Total precip, mm/day, CRCM4 - CRU")
    fig.savefig("seasonal_{0}_ccc.png".format(cru_var_name))

def plot_temperature_biases():

    season_to_months = {
        "DJF" : [12, 1, 2],
        "MAM" : [3, 4, 5],
        "JJA" : [6, 7, 8],
        "SON" : [9, 10, 11],
        "Annual": range(1, 13)
    }
    cru_var_name = "tmp"
    cru_data_store, crcm4_data_store = _get_comparison_data(cru_var_name= cru_var_name,
        season_to_months=season_to_months)

    x, y = polar_stereographic.xs, polar_stereographic.ys
    i_array, j_array = _get_routing_indices()
    x_min, x_max, y_min, y_max = plot_utils.get_ranges(x[i_array, j_array], y[i_array, j_array])



    plot_utils.apply_plot_params(width_pt= None, font_size=9, aspect_ratio=2.5)
    fig = plt.figure()
    assert isinstance(fig, Figure)

    basemap = polar_stereographic.basemap
    assert isinstance(basemap, Basemap)
    gs = gridspec.GridSpec(3,2)

    color_map = my_cm.get_red_blue_colormap(ncolors = 14, reversed=True)
    clevels = xrange(-8, 9, 2)
    all_plot_axes = []
    for i, season in enumerate(season_to_months.keys()):
        if not i:
            ax = fig.add_subplot(gs[0,:])
        else:
            row, col = (i - 1)  // 2 + 1, i % 2
            ax = fig.add_subplot(gs[row, col])
        all_plot_axes.append(ax)
        assert isinstance(ax, Axes)
        delta = crcm4_data_store[season] - cru_data_store[season]
        if cru_var_name == "tmp": delta -= 273.15
        #delta = maskoceans(polar_stereographic.lons, polar_stereographic.lats, delta)
        save = delta[i_array, j_array]
        delta[:, :] = np.ma.masked
        delta[i_array, j_array] = save
        img = basemap.pcolormesh(x, y, delta, cmap = color_map, vmin = -7, vmax = 7)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "8%", pad="3%")
        int_ticker = LinearLocator(numticks = color_map.N + 1)
        fig.colorbar(img, cax = cax, ticks = MultipleLocator(base = 2))
        ax.set_title(season)

    for the_ax in all_plot_axes:
        the_ax.set_xlim(x_min, x_max)
        the_ax.set_ylim(y_min, y_max)
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.5)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=1)


    #gs.tight_layout(fig)
    fig.suptitle("T(2m), degrees, CRCM4 - CRU")
    fig.savefig("seasonal_{0}_ccc.png".format(cru_var_name))


def main():
   # plot_temperature_biases()
    plot_precip_biases()

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  