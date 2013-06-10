from datetime import datetime, timedelta
import os
import pickle
import itertools
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LinearLocator, MultipleLocator
from mpl_toolkits.basemap import Basemap, maskoceans
import application_properties
from ccc import ccc_analysis
from data import data_select
from data.cehq_station import read_station_data
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


selected_station_ids = [
    #"104001", "103715", "093806", "093801", "092715", "081006",
    "061502", #"080718",
    #"040830"
]


#selected_station_ids = [
#    "104001", "103715", "093801", "090613", "081006", "090602", "080718", "040830"
#]

stations = None

def put_selected_stations(ax, the_basemap, i_list, j_list):
    """
    :type the_basemap: Basemap
    """

    stations_dump = 'stations_dump.bin'
    global stations

    if stations is not None:
        pass
    elif os.path.isfile(stations_dump):
        print 'unpickling'
        stations = pickle.load(open(stations_dump))
    else:
        stations = read_station_data()
        pickle.dump(stations, open(stations_dump, 'w'))

    #get selected stations
    sel_stations = list( itertools.ifilter( lambda x: x.id in selected_station_ids, stations ) )

    xs, ys = polar_stereographic.xs, polar_stereographic.ys

    dx = 0.01 * ( xs[i_list, j_list].max() - xs[i_list, j_list].min() )
    dy = 0.01 * ( ys[i_list, j_list].max() - ys[i_list, j_list].min() )



    the_xs = []
    the_ys = []
    for station in sel_stations:
        x, y = the_basemap(station.longitude, station.latitude)
        the_xs.append(x)
        the_ys.append(y)

        xtext = 1.005 * x
        ytext = y
        if station.id in ['061906']:
            xtext = 1.00 * x
            ytext = 0.97 * y

        if station.id in ['103603', '081002']:
            ytext = 0.98 * y

        if station.id in ['081007']:
            xtext = 0.97 * x

        if station.id in ["090602"]:
            ytext -= 7 * dy
            xtext -= 5 * dx

        if station.id in ["090613"]:
            ytext += 4 * dy
            xtext -= 6 * dx

#        the_id = station.id
#
#        xtext = None
#        ytext = None
#        if the_id.startswith("0807"):
#            xtext, ytext = 0.1, 0.2
#
#        if the_id.startswith("081006"):
#            xtext, ytext = 0.1, 0.3
#
#        if the_id.startswith("093801"):
#            xtext, ytext = 0.1, 0.5
#
#        if the_id.startswith("093806"):
#            xtext, ytext = 0.1, 0.6
#
#        if the_id.startswith("103715"):
#            xtext, ytext = 0.1, 0.95
#
#        if the_id.startswith("104001"):
#            xtext, ytext = 0.4, 0.95
#
#        if the_id.startswith("061502"):
#            xtext, ytext = 0.8, 0.4
#
#        if the_id.startswith("040830"):
#            xtext, ytext = 0.8, 0.2
#
#        if the_id.startswith("092715"):
#            xtext, ytext = 0.1, 0.4




#        ax.annotate(station.id, xy = (x, y), xytext = (xtext, ytext), #textcoords = "axes fraction",
#                             bbox = dict(facecolor = 'white'), weight = "bold",
                             #arrowprops=dict(facecolor='black', width = 1, headwidth = 1.5)
#                             )

    the_basemap.scatter(the_xs,the_ys, c = 'c', s = 60, marker='^', linewidth = 0.5, alpha = 1,
        zorder = 5, ax = ax)


    pass


def _get_comparison_data_swe(    start_date = datetime(1980,1,1),
                                 end_date = datetime(1997, 12, 31),
    crcm4_data_folder = "/home/huziy/skynet1_rech3/crcm4_data/aex_p1sno",
    swe_obs_data_path = "data/swe_ross_brown/swe.nc",
    swe_var_name = "swe", season_to_months = None, crcm4_id = "aex"):

    cache_file = "seasonal_bias_{0}.bin".format(swe_var_name)

    if not os.path.isfile(cache_file):
        #get crcm4 data
        crcm4_data_store = {}
        for season, month_list in season_to_months.iteritems():
            data = ccc_analysis.get_seasonal_mean(crcm4_data_folder, start_date=start_date,
                end_date=end_date, months=month_list
            )

            crcm4_data_store[season] = data

        #get cru data
        cru_data_store = {}
        cru = CruReader(path=swe_obs_data_path, var_name=swe_var_name, transpose_xy_dimensions=False)
        lons2d_amno, lats2d_amno = polar_stereographic.lons, polar_stereographic.lats
        for season, month_list in season_to_months.iteritems():
            data = cru.get_seasonal_mean_field(months=month_list, start_date=start_date, end_date=end_date)

            nneighbors = 1 if swe_var_name in ["pre", "swe"] else 4
            data = cru.interpolate_data_to(data, lons2d_amno, lats2d_amno, nneighbors=nneighbors)
            cru_data_store[season] = data
        data_list = [cru_data_store, crcm4_data_store]
        pickle.dump(data_list, open(cache_file, "w"))
    else:
        data_list = pickle.load(open(cache_file))

    cru_data_store, crcm4_data_store = data_list
    return cru_data_store, crcm4_data_store





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


def plot_swe_and_temp_on_one_plot():
    seasons = ["(a) Annual", " (b) Winter (DJF)", "(c) Spring (MAM)", "(d) Summer (JJA)", "(e) Fall (SON)"]
    months =  [range(1, 13), [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]  ]
    season_to_months = dict(zip(seasons, months))
    cru_var_name = "tmp"

    temp_obs_data_store, temp_crcm4_data_store = _get_comparison_data(cru_var_name= cru_var_name,
        season_to_months=season_to_months)

    x, y = polar_stereographic.xs, polar_stereographic.ys
    i_array, j_array = _get_routing_indices()
    x_min, x_max, y_min, y_max = plot_utils.get_ranges(x[i_array, j_array], y[i_array, j_array])

    #get swe data
    swe_obs_name = "swe"
    swe_obs_data_store, swe_crcm4_data_store = _get_comparison_data_swe(swe_var_name= swe_obs_name,
        season_to_months=season_to_months, start_date=datetime(1980,1,1), end_date=datetime(1997, 1, 1))



    plot_utils.apply_plot_params(width_pt= None, font_size=9, aspect_ratio=2.5)

    fig = plt.figure()
    assert isinstance(fig, Figure)
    gs = gridspec.GridSpec(2, 1)

    basemap = polar_stereographic.basemap
    swe_season = seasons[1]
    temp_season = seasons[2]

    var_names = [ "swe", cru_var_name]
    the_seasons = [ swe_season, temp_season ]

    labels = [ "(a) Winter (DJF)", "(b) Sping (MAM)"]
    units = [ "mm", "$^{\\circ}{\\rm C}$" ]

    data_stores = [
        [swe_obs_data_store, swe_crcm4_data_store],
        [temp_obs_data_store, temp_crcm4_data_store]
    ]



    all_plot_axes = []
    for i, season, var_name, store, label, unit in zip(xrange(len(seasons)), the_seasons, var_names, data_stores,
            labels, units):
        ax = fig.add_subplot(gs[i, 0])
        all_plot_axes.append(ax)
        assert isinstance(ax, Axes)

        crcm4 = store[1][season]
        obs = store[0][season]

        delta = crcm4 - obs
        if var_name == "tmp": delta -= 273.15

        if var_name == "swe":
            ax.annotate("(1979-1997)", (0.1, 0.1), xycoords = "axes fraction",
                font_properties = FontProperties(weight = "bold"))
        elif var_name == "tmp":
            ax.annotate("(1970-1999)", (0.1, 0.1), xycoords = "axes fraction",
                font_properties = FontProperties(weight = "bold"))


        color_map = my_cm.get_red_blue_colormap(ncolors = 16, reversed=(var_name == "tmp"))
        color_map.set_over("k")
        color_map.set_under("k")



        #delta = maskoceans(polar_stereographic.lons, polar_stereographic.lats, delta)
        save = delta[i_array, j_array]
        delta = np.ma.masked_all(delta.shape)
        delta[i_array, j_array] = save



        vmin = np.floor( np.min(save)  )
        vmax = np.ceil( np.max(save)  )

        decimals = 0 if var_name == "swe" else 1
        round_func = lambda x: np.round(x, decimals= decimals)
        bounds = plot_utils.get_boundaries_for_colobar(vmin, vmax, color_map.N, round_func= round_func)
        bn = BoundaryNorm( bounds, color_map.N )

        img = basemap.pcolormesh(x, y, delta, cmap = color_map, norm = bn)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "8%", pad="3%")
        fig.colorbar(img, cax = cax, boundaries = bounds, ticks = bounds)
        ax.set_title(label)
        cax.set_title(unit)




    for the_ax in all_plot_axes:
        the_ax.set_xlim(x_min, x_max)
        the_ax.set_ylim(y_min, y_max)
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.1)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=0.4)
        put_selected_stations(the_ax, basemap, i_array, j_array)

    fig.tight_layout()
    fig.savefig("swe_temp_biases.png")





def plot_precip_biases():
    seasons = ["(a) Annual", " (b) Winter (DJF)", "(c) Spring (MAM)", "(d) Summer (JJA)", "(e) Fall (SON)"]
    months =  [range(1, 13), [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]  ]
    season_to_months = dict(zip(seasons, months))
    cru_var_name = "pre"
    cru_data_store, crcm4_data_store = _get_comparison_data(
        crcm4_data_folder="/home/huziy/skynet1_rech3/crcm4_data/aex_p1pcp",
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
    for i, season in enumerate(seasons):
        if not i:
            ax = fig.add_subplot(gs[0,:])
        else:
            row, col = (i - 1)  // 2 + 1, (i - 1) % 2
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
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.1)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=0.4)
#        put_selected_stations(the_ax, basemap, i_array, j_array)

#    ax = fig.add_subplot(gs[3,:])
#    assert isinstance(ax, Axes)
#    fig.colorbar(img, cax = ax, orientation = "horizontal", ticks = MultipleLocator(base = 0.5))

    #gs.tight_layout(fig)
    fig.suptitle("Total precip, mm/day, CRCM4 - CRU")
    fig.savefig("seasonal_{0}_ccc.png".format(cru_var_name))

def plot_temperature_biases():

    seasons = ["(a) Annual", " (b) Winter (DJF)", "(c) Spring (MAM)", "(d) Summer (JJA)", "(e) Fall (SON)"]
    months =  [range(1, 13), [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]  ]
    season_to_months = dict(zip(seasons, months))

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
    for i, season in enumerate(seasons):
        if not i:
            ax = fig.add_subplot(gs[0,:])
        else:
            row, col = (i - 1)  // 2 + 1, (i - 1) % 2
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
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.1)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=0.4)
        put_selected_stations(the_ax, basemap, i_array, j_array)

    #gs.tight_layout(fig)
    fig.suptitle("T(2m), degrees, CRCM4 - CRU")
    fig.savefig("seasonal_{0}_ccc.png".format(cru_var_name))

def plot_swe_biases():

    seasons = ["(a) Annual", " (b) Winter (DJF)", "(c) Spring (MAM)", "(d) Summer (JJA)", "(e) Fall (SON)"]
    months =  [range(1, 13), [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]  ]
    season_to_months = dict(zip(seasons, months))

    swe_obs_name = "swe"
    cru_data_store, crcm4_data_store = _get_comparison_data_swe(swe_var_name= swe_obs_name,
        season_to_months=season_to_months, start_date=datetime(1980,1,1), end_date=datetime(1996, 12, 31))

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
    for i, season in enumerate(seasons):
        if not i:
            ax = fig.add_subplot(gs[0,:])
        else:
            row, col = (i - 1)  // 2 + 1, (i -1) % 2
            ax = fig.add_subplot(gs[row, col])
        all_plot_axes.append(ax)
        assert isinstance(ax, Axes)
        delta = crcm4_data_store[season] - cru_data_store[season]

        #delta = maskoceans(polar_stereographic.lons, polar_stereographic.lats, delta)
        save = delta[i_array, j_array]
        delta = np.ma.masked_all(delta.shape)
        delta[i_array, j_array] = save


        vmax = np.ceil( np.max(save) / 10.0) * 10
        vmin = np.floor( np.min(save) / 10.0) * 10

        bounds = plot_utils.get_boundaries_for_colobar(vmin, vmax, color_map.N, lambda x: np.round(x, decimals = 1))
        bn = BoundaryNorm(bounds, color_map.N)
        img = basemap.pcolormesh(x, y, delta, cmap = color_map, vmin = vmin, vmax = vmax, norm = bn)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "8%", pad="3%")
        int_ticker = LinearLocator(numticks = color_map.N + 1)





        fig.colorbar(img, cax = cax, ticks = bounds, boundaries = bounds)
        ax.set_title(season)

    for the_ax in all_plot_axes:
        the_ax.set_xlim(x_min, x_max)
        the_ax.set_ylim(y_min, y_max)
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.1)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=0.4)
#        put_selected_stations(the_ax, basemap, i_array, j_array)


    #gs.tight_layout(fig)
    fig.suptitle("SWE (mm), CRCM4 - Ross Brown dataset (1981-1997)")
    fig.savefig("seasonal_{0}_ccc.png".format(swe_obs_name))


def main():
    plot_temperature_biases()
#    plot_precip_biases()
#    plot_swe_biases()
#    plot_swe_and_temp_on_one_plot()

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  