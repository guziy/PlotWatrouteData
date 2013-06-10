from multiprocessing.pool import Pool
import os
import pickle
import re
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm
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
import matplotlib_helpers.my_colormaps as my_cm
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
    seasons = ["(a) Annual", " (b) Winter (DJF)", "(c) Spring (MAM)", "(d) Summer (JJA)", "(e) Fall (SON)"]
    months =  [range(1, 13), [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]  ]
    season_to_months = dict(zip(seasons, months))

    #take out annual
    seasons.pop(0)
    #put new numbering for the subplots
    new_numbering = ["a", "b", "c", "d"]


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
    gs = gridspec.GridSpec(3,2)

    #color_map = mpl.cm.get_cmap(name="jet", lut=10)
    clevels = xrange(-8, 9, 2)
    all_plot_axes = []
    
    
    
    #determine min an max for color scales
    min_val = np.inf
    max_val = -np.inf
    for season in seasons:
        current = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.current_ids, year_range=year_range_c)
        future = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.future_ids, year_range=year_range_f)

        current_m = np.mean(current, axis=0)
        future_m = np.mean(future, axis= 0)


        delta = future_m[i_array, j_array] - current_m[i_array, j_array]

        the_min = delta.min()
        the_max = delta.max()

        min_val = min(min_val, the_min)
        max_val = max(max_val, the_max)

    min_val = np.floor(min_val)
    max_val = np.ceil(max_val)

    color_map = my_cm.get_red_blue_colormap(ncolors = 10, reversed=True)
    if min_val >= 0:
        color_map = my_cm.get_red_colormap(ncolors=10)
    
    for i, season in enumerate(seasons):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col ])
        all_plot_axes.append(ax)


        current = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.current_ids, year_range=year_range_c)
        future = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.future_ids, year_range=year_range_f)


#        t, p = stats.ttest_ind(current, future, axis=0)

#        significant = np.array(p <= 0.05)

        significant = calculate_significance_using_bootstrap(current, future)
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

        img = basemap.pcolormesh(x, y, delta, cmap = color_map, vmin = min_val, vmax = max_val)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "8%", pad="3%")
        assert isinstance(cax, Axes)

        int_ticker = LinearLocator(numticks = color_map.N + 1)
        cb = fig.colorbar(img, cax = cax, ticks = int_ticker)
        cax.set_title("$^{\\circ}{\\rm C}$")



        where_significant = significant
        significant = np.ma.masked_all(significant.shape)

        significant[~where_significant] = 0
        basemap.pcolormesh( x, y, significant , cmap = mpl.cm.get_cmap(name = "gray", lut = 3),
                           vmin = -1, vmax = 1, ax = ax)
        ax.set_title( season.replace( re.findall( "([a-z])", season)[0], new_numbering[i]) )

    for the_ax in all_plot_axes:
        the_ax.set_xlim(x_min, x_max)
        the_ax.set_ylim(y_min, y_max)
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.1)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=0.4)


    #gs.update(wspace=0.5)
    fig.tight_layout()
    #fig.suptitle("Projected changes, T(2m), degrees, CRCM4")
    fig.savefig("proj_change_{0}_ccc.png".format(var_name))


def calculate_significance_using_bootstrap(current_data, future_data,
                                           nsamples = 1000, n_members = 5):
    """
    current_data.shape == (nyears * nmembers, nx, ny)
    for sign testing takes mean of standard deviation of each member.
    returns boolean numpy array of shape (nx, ny), with True where the changes are significant to the
    5% significance level
    """
    n_data_per_member = current_data.shape[0] / n_members

    print("n_data_per_member = {0}".format(n_data_per_member))
    print(current_data.shape)

    fut_stds = []
    cur_stds = []

    for i in xrange(0, n_members * n_data_per_member, n_data_per_member):
        c_data = current_data[i:(i+n_data_per_member),:,:]
        f_data = future_data[i:(i+n_data_per_member),:,:]

        fut_indices = np.random.randint(0, n_data_per_member, size=(n_data_per_member, nsamples))
        cur_indices = np.random.randint(0, n_data_per_member, size=(n_data_per_member, nsamples))

        res_cur_means = np.mean( c_data[cur_indices,:,:] , axis=0)
        res_fut_means = np.mean( f_data[fut_indices,:,:], axis = 0)

        print(res_cur_means.shape)

        cur_stds.append( np.std(res_cur_means, axis = 0) )
        fut_stds.append( np.std(res_fut_means, axis = 0 ) )


    std_cur = np.mean(cur_stds, axis=0)
    std_fut = np.mean(fut_stds, axis=0)

    mean_cur = np.mean(current_data, axis = 0)
    mean_fut = np.mean(future_data, axis = 0)

    return np.abs(mean_cur - mean_fut) >= 1.96 * (std_cur + std_fut)



def plot_precip(data_path = "/home/huziy/skynet1_rech3/crcm4_data"):
    seasons = ["(a) Annual", " (b) Winter (DJF)", "(c) Spring (MAM)", "(d) Summer (JJA)", "(e) Fall (SON)"]
    months =  [range(1, 13), [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]  ]
    season_to_months = dict(zip(seasons, months))


    #remove annual
    seasons.pop(0)
    #put new numbering for the subplots
    new_numbering = ["a", "b", "c", "d"]


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
    gs = gridspec.GridSpec(3,2, height_ratios=[1,1,1], width_ratios=[1,1])



    #determine min an max for color scales
    min_val = np.inf
    max_val = -np.inf
    for season in seasons:
        current = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.current_ids, year_range=year_range_c)
        future = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.future_ids, year_range=year_range_f)

        current_m = np.mean(current, axis=0)
        future_m = np.mean(future, axis= 0)


        delta = future_m[i_array, j_array] - current_m[i_array, j_array]

        the_min = delta.min()
        the_max = delta.max()

        min_val = min(min_val, the_min)
        max_val = max(max_val, the_max)

    min_val = np.floor(min_val)
    max_val = np.ceil(max_val)




    color_map = my_cm.get_red_blue_colormap(ncolors = 10, reversed=False)
    #color_map = mpl.cm.get_cmap(name="jet_r", lut=10)
    clevels = xrange(-8, 9, 2)
    all_plot_axes = []
    for i, season in enumerate(seasons):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col ])
        all_plot_axes.append(ax)


        current = _get_data(data_folder=data_path, v_name=var_name, months = season_to_months[season],
                    member_list=members.current_ids, year_range=year_range_c)
        future = _get_data(data_folder=data_path, v_name=var_name, months = season_to_months[season],
                    member_list=members.future_ids, year_range=year_range_f)


        #t, p = stats.ttest_ind(current, future, axis=0)

        #significant = np.array(p <= 0.05)
        significant = calculate_significance_using_bootstrap(current, future)
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
        d_min = np.floor( min_val * 10 ) / 10.0
        d_max = np.ceil( max_val *10 ) / 10.0

        if d_min  > 0: color_map = my_cm.get_blue_colormap(ncolors=10)

        img = basemap.pcolormesh(x, y, delta, cmap = color_map, vmin = d_min, vmax = d_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "8%", pad="3%")
        assert isinstance(cax, Axes)

        int_ticker = LinearLocator(numticks = color_map.N + 1)

        cb = fig.colorbar(img, cax = cax, ticks = int_ticker)
        cax.set_title("mm/d")

        where_significant = significant
        significant = np.ma.masked_all(significant.shape)

        significant[(~where_significant)] = 0
        save = significant[i_array, j_array]
        significant = np.ma.masked_all(significant.shape)
        significant[i_array, j_array] = save

        basemap.pcolormesh( x, y, significant , cmap = mpl.cm.get_cmap(name = "gray", lut = 3),
                           vmin = -1, vmax = 1, ax = ax)

        ax.set_title( season.replace( re.findall( "([a-z])", season)[0], new_numbering[i]) )


    #plot djf swe change
    season = " (b) Winter (DJF)"
    ax = fig.add_subplot(gs[2, : ])
    all_plot_axes.append(ax)
    var_name = "sno"

    current = _get_data(data_folder=data_path, v_name=var_name, months = season_to_months[season],
                    member_list=members.current_ids, year_range=year_range_c)
    future = _get_data(data_folder=data_path, v_name=var_name, months = season_to_months[season],
                    member_list=members.future_ids, year_range=year_range_f)


    #t, p = stats.ttest_ind(current, future, axis=0)

    #significant = np.array(p <= 0.05)
    significant = calculate_significance_using_bootstrap(current, future)

    assert not np.all(~significant)
    assert not np.all(significant)

    current_m = np.mean(current, axis=0)
    future_m = np.mean(future, axis= 0)


    delta = future_m - current_m
    delta = np.array(delta)

    assert isinstance(ax, Axes)

    #delta = maskoceans(polar_stereographic.lons, polar_stereographic.lats, delta)
    save = delta[i_array, j_array]
    delta = np.ma.masked_all(delta.shape)
    delta[i_array, j_array] = save
    d_min = np.floor( np.min(save) * 10 ) / 10.0
    d_max = np.ceil( np.max(save) *10 ) / 10.0

    if d_min >= 0: color_map = my_cm.get_blue_colormap(ncolors=10)

    bounds = plot_utils.get_boundaries_for_colobar(d_min, d_max, color_map.N, lambda x: np.round(x, decimals=0))

    bn = BoundaryNorm(bounds, color_map.N)


    img = basemap.pcolormesh(x, y, delta, cmap = color_map, vmin = bounds[0], vmax = bounds[-1], norm = bn)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "8%", pad="3%")
    assert isinstance(cax, Axes)

    int_ticker = LinearLocator(numticks = color_map.N + 1)

    cb = fig.colorbar(img, cax = cax, ticks = bounds)
    cax.set_title("mm")

    where_significant = significant
    significant = np.ma.masked_all(significant.shape)

    significant[(~where_significant)] = 0
    save = significant[i_array, j_array]
    significant = np.ma.masked_all(significant.shape)
    significant[i_array, j_array] = save

    basemap.pcolormesh( x, y, significant , cmap = mpl.cm.get_cmap(name = "gray", lut = 3),
                        vmin = -1, vmax = 1, ax = ax)

    ax.set_title( season.replace( re.findall( "([a-z])", season)[0], "e"))
    #finish swe djf



    for the_ax in all_plot_axes:
        the_ax.set_xlim(x_min, x_max)
        the_ax.set_ylim(y_min, y_max)
        basemap.drawcoastlines(ax = the_ax, linewidth = 0.1)
        plot_utils.draw_meridians_and_parallels(basemap, step_degrees=30.0, ax = the_ax)
        plot_basin_boundaries_from_shape(basemap, axes = the_ax, linewidth=0.4)


    #gs.update(wspace=0.5)

    #gs.tight_layout(fig)
    #fig.suptitle("Projected changes, total precip (mm/day), CRCM4")
    fig.tight_layout()
    fig.savefig("proj_change_{0}_ccc.png".format(var_name))

    pass

def plot_swe():
    seasons = ["(a) Annual", " (b) Winter (DJF)", "(c) Spring (MAM)", "(d) Summer (JJA)", "(e) Fall (SON)"]
    months =  [range(1, 13), [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]  ]
    season_to_months = dict(zip(seasons, months))

    var_name = "sno"
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


    #color_map = my_cm.get_
    color_map = my_cm.get_red_blue_colormap(ncolors = 10, reversed=True)
    #color_map = mpl.cm.get_cmap(name="jet_r", lut=10)
    clevels = xrange(-8, 9, 2)
    all_plot_axes = []
    for i, season in enumerate(seasons):
        if not i:
            ax = fig.add_subplot(gs[0,1:3])
        else:
            row, col = (i - 1)  // 2 + 1, (i - 1) % 2
            ax = fig.add_subplot(gs[row, col * 2 : col * 2 + 2 ])
        all_plot_axes.append(ax)


        current = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.current_ids, year_range=year_range_c)
        future = _get_data(v_name=var_name, months = season_to_months[season],
                    member_list=members.future_ids, year_range=year_range_f)


        t, p = stats.ttest_ind(current, future, axis=0)
        #TODO: change it back to p <= 0.05 wheen doing real sign test
        significant = np.array(p <= 1)

        assert not np.all(~significant)
        assert not np.all(significant)

        current_m = np.mean(current, axis=0)
        future_m = np.mean(future, axis= 0)


        delta = (future_m - current_m)
        delta = np.array(delta)

        assert isinstance(ax, Axes)

        #delta = maskoceans(polar_stereographic.lons, polar_stereographic.lats, delta)
        save = delta[i_array, j_array]
        delta = np.ma.masked_all(delta.shape)
        delta[i_array, j_array] = save
        d_min = np.floor( np.min(save)  )
        d_max = np.ceil( np.max(save)  )


        bounds = plot_utils.get_boundaries_for_colobar(d_min, d_max, color_map.N, lambda x: np.round(x, decimals=10))
        print bounds

        bn = BoundaryNorm(bounds, color_map.N)

        d = np.max( np.abs([d_min, d_max]) )

        print season, np.min(delta), np.max(delta)

        #delta = np.ma.masked_where(delta < 0, delta )
        img = basemap.pcolormesh(x, y, delta, cmap = color_map, norm = bn, vmin = bounds[0], vmax = bounds[-1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "8%", pad="3%")
        assert isinstance(cax, Axes)

        int_ticker = LinearLocator(numticks = color_map.N + 1)

        cb = fig.colorbar(img, cax = cax, ticks = bounds, boundaries = bounds)





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
    fig.suptitle("Projected changes, SWE, mm, CRCM4")
    fig.savefig("proj_change_{0}_ccc.png".format(var_name))

    pass


def main():
#    plot_temp()
#    plot_precip()
    plot_swe()
    pass

if __name__ == "__main__":
    import time
    import application_properties

    t0 = time.clock()
    application_properties.set_current_directory()
    main()
    print("Execution time: {0} seconds".format(time.clock() - t0))
    print "Hello world"
  