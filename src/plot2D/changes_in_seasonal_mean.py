import bootstrap_for_mean, bootstrap_for_mean_merged
import ttest_for_mean_of_merged

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
import calculate_mean_map

#Plot a panel plot with 4 subplots
#DJF, ,,, (winter, spring, summer and autumn)

import matplotlib.gridspec as gridspec




def calculate_seasonal_changes_in_mean_stfl_and_plot(folder_path = 'data/streamflows/hydrosheds_euler9',
                                                     months = None, label = "",
                                                     cb_axes = None,
                                                     plot_axes = None,
                                                     impose_lower_limit = None,
                                                     upper_limited = False,
                                                     minmax = None
                                                     ):

    """

    """
    if months is None:
        print "please specify months"
        return

    fileName = None
    for fName in os.listdir(folder_path):
        if fName.startswith( "aex" ):
            fileName = fName


    i_indices, j_indices = data_select.get_indices_from_file(path = os.path.join(folder_path, fileName))

    xs = polar_stereographic.xs
    ys = polar_stereographic.ys
    basemap = polar_stereographic.basemap

    #get mean changes along with its significance
    #ensemble mean
    significance, change = bootstrap_for_mean.get_significance_for_change_in_mean_over_months(months=months)
    #significance, change = ttest_for_mean_of_merged.get_significance_and_changes_for_months(months=months)
    #merged
    #significance, change = bootstrap_for_mean_merged.get_significance_for_change_in_mean_of_merged_over_months(months= months)



    #zoom to domain
    selected_x = xs[i_indices, j_indices]
    selected_y = ys[i_indices, j_indices]
    x_min, x_max, y_min, y_max = plot_utils.get_ranges(selected_x, selected_y)

    #plot mean change

    print "plotting"
    plot_axes.set_title('{0}'.format(label))

    calculate_mean_map.plot_data(change, i_indices, j_indices, minmax = minmax, title=label, name = None,
                                 color_map = my_cm.get_red_blue_colormap(ncolors = 10),
                                 draw_colorbar=True, basemap=basemap, axes=plot_axes,
                                 impose_lower_limit = impose_lower_limit,
                                 upper_limited = upper_limited
                                 )



    plot_significance = True
    if plot_significance:
        to_plot = np.ma.masked_all(xs.shape)
        significance = significance.astype(int)
        significance = np.ma.masked_where(significance == 1, significance)
        for the_significance, i, j in zip(significance, i_indices, j_indices):
            to_plot[i, j] = the_significance

        basemap.pcolormesh( xs, ys, to_plot.copy(), cmap = mpl.cm.get_cmap(name = "gray", lut = 3),
                           vmin = -1, vmax = 1, ax = plot_axes)

#    plt.xlim(x_min, x_max)
#    plt.ylim(y_min, y_max)







def put_upper_limited_label(the_colorbar):
    """
    the_colorbar - colorbar object
    appends the \geq symbol to the upper limit label
    """
    ticks = the_colorbar.ax.get_yticklabels()
    ticks = map(lambda x: x.get_text(), ticks)
    ticks[-1] = "$\\geq$ {0}".format(ticks[-1])
    the_colorbar.ax.set_yticklabels(ticks)


from mpl_toolkits.axes_grid1 import make_axes_locatable
def calculate_mean_change_and_plot_for_all_seasons():
    """
    calculate changes in seasonal means
    """

    seasons = ["(b) Winter (DJF)",  "(c) Spring (MAM)", "(d) Summer (JJA)", "(e) Fall (SON)"]
    months = [(12,1,2), (3,4,5), (6,7,8), (9,10,11)]
    folder_path = 'data/streamflows/hydrosheds_euler9'


    gs = gridspec.GridSpec(3,2)
    #gs.update()
    plot_utils.apply_plot_params(width_pt= 800, font_size=15, aspect_ratio=2)
    #plt.subplots_adjust( wspace = 0.0)
    i = 0
    for  row in xrange(1,3):
        for col in xrange(2):
            m = months[i]
            s = seasons[i]
            ax_plot = plt.subplot(gs[row, col])
            calculate_seasonal_changes_in_mean_stfl_and_plot(folder_path = folder_path,
                months = m, label = s, plot_axes = ax_plot, impose_lower_limit=-60.0, upper_limited=True,
                minmax=(-100,100)
            )
            i += 1
            print i, row, col

    ax_plot = plt.subplot(gs[0, :])
    calculate_seasonal_changes_in_mean_stfl_and_plot(folder_path = folder_path,
                months = range(1,13), label = "(a) Annual", plot_axes = ax_plot, minmax=(-40, 40)
            )


#    divider = make_axes_locatable(ax_plot)
#    ax_cb = divider.append_axes("right", "5%", pad="3%")
#    plt.colorbar(cax=ax_cb)
    print "laying out"
    ax_plot.figure.tight_layout( w_pad=0)
    print "saving"
    ax_plot.figure.savefig('mean_seasonal_change.png')
    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    #calculate_seasonal_changes_in_mean_stfl_and_plot()
    calculate_mean_change_and_plot_for_all_seasons()
    print "Hello World"
