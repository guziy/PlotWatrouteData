from plot2D import bootstrap_for_mean, bootstrap_for_mean_merged
from shape.basin_boundaries import plot_basin_boundaries_from_shape

__author__="huziy"
__date__ ="$6 oct. 2010 21:06:01$"

import application_properties
import matplotlib.pyplot as plt
import data.data_select as data_select
import pylab
import matplotlib as mpl
import os
import numpy as np
from util import plot_utils
import gevfit.matplotlib_helpers.my_colormaps as my_cm







#set current directory to the root directory of the project


from plot2D.map_parameters import polar_stereographic
from matplotlib.ticker import LinearLocator

n_cols = polar_stereographic.n_cols
n_rows = polar_stereographic.n_rows
xs = polar_stereographic.xs
ys = polar_stereographic.ys
m = polar_stereographic.basemap



def plot_data(data, i_array, j_array, name='AEX', title = None, digits = 1,
                                      color_map = mpl.cm.get_cmap('RdBu_r'), 
                                      minmax = (None, None),
                                      units = '%',
                                      colorbar_orientation = 'vertical',
                                      draw_colorbar = True,
                                      basemap = None, axes = None,
                                      impose_lower_limit = None, upper_limited = False

                                      ):



    if name is not None:
        plt.figure()

    to_plot = np.ma.masked_all((n_cols, n_rows))
    for index, i, j in zip( range(len(data)), i_array, j_array):
        to_plot[i, j] = data[index]


    print np.ma.min(data), np.ma.max(data)
    
  #  m.pcolor(xs, ys, to_plot, cmap = mpl.cm.get_cmap('RdBu_r'))

    if basemap is None:
        the_basemap = m
    else:
        the_basemap = basemap

    image = the_basemap.pcolormesh(xs, ys, to_plot.copy(), cmap = color_map,
                                          vmin = minmax[0],
                                          vmax = minmax[1], ax = axes)



    #ads to m fields basins and basins_info which contain shapes and information
 #   m.readshapefile('data/shape/contour_bv_MRCC/Bassins_MRCC_utm18', 'basins')
 #   m.scatter(xs, ys, c=to_plot)
    plot_basin_boundaries_from_shape(m, axes=axes, linewidth = 2.1)
    #the_basemap.drawrivers(linewidth = 0.5, ax = axes)
    the_basemap.drawcoastlines(linewidth = 0.5, ax = axes)
    plot_utils.draw_meridians_and_parallels(the_basemap, step_degrees = 30)



    axes.set_title(title if title is not None else name)

    #zoom_to_qc()
    x_min, x_max, y_min, y_max = plot_utils.get_ranges(xs[i_array, j_array], ys[i_array, j_array])
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)


    if draw_colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", "8%", pad="3%")
        int_ticker = LinearLocator(numticks = color_map.N + 1)
        cb = axes.figure.colorbar(image, ticks = int_ticker,
                          orientation = colorbar_orientation,
                          format = '%d', cax = cax, ax=axes, drawedges = True)

        cb.ax.set_title(units)
        #cb.outline.remove()
        bottom, top = cb.ax.get_ylim()
        left, right = cb.ax.get_xlim()
        print bottom, top, left, right


        if impose_lower_limit is None:
            new_bottom = min( np.min(data), 0 )
        else:
            new_bottom = impose_lower_limit
        #new_bottom = np.floor( new_bottom / 10.0 ) * 10.0


        new_bottom = np.abs((new_bottom - minmax[0]) / (float(minmax[1] - minmax[0])))

        new_bottom = plot_utils.get_closest_tick_value(color_map.N + 1, new_bottom) - 1.0e-4



        print new_bottom
        cb.ax.set_ylim(bottom = new_bottom)
        left, right = cb.ax.get_xlim()
        bottom, top = cb.ax.get_ylim()

        #cb.ax.set_xlim(left = 0, right = 0.8)
        cb.outline.set_visible( False )

        if upper_limited:
            cl = cb.ax.get_yticklabels()
            labels = []
            for text in cl:
                labels.append(text.get_text())

            labels[-1] = '$\\geq$' + labels[-1]
            cb.ax.set_yticklabels(labels)


#        if not (impose_lower_limit is None):
#            cl = cb.ax.get_yticklabels()
#            labels = []
#            for text in cl:
#                labels.append(text.get_text())
#
#            select_index = -1
#            for index, the_label in enumerate(labels):
#                if impose_lower_limit == float(the_label):
#                    select_index = index
#                    break
#
#            if select_index >= 0:
#                labels[select_index] = '$\\leq$' + labels[select_index]
#            cb.ax.set_yticklabels(labels)
#

#        verts = np.zeros((4,2))
#        verts[0,0], verts[0,1] = left, bottom
#        verts[1,0], verts[1,1] = right, bottom
#        verts[2,0], verts[2,1] = right, top
#        verts[3,0], verts[3,1] = left, top
#
#        cb.ax.add_patch(Polygon(xy=verts, facecolor="none",edgecolor="k",linewidth=2))
#        #cb.draw_all()
        #cb.ax.set_xlim(left = 0, right = 1)
        #cb.ax.axhspan(0, 40)





def get_meanof_means_and_stds_from_files(files):
    mean = None
    stdevs = None

    if not len(files): return

    for path in files:
        data = data_select.get_data_from_file(path)
        if mean is None:
            mean = np.zeros(data.shape[1])
            stdevs = np.zeros(data.shape[1])
            
        mean += np.mean(data, axis = 0)
        stdevs += np.std(data, axis = 0)


    mean /= float(len(files))
    stdevs /= float(len(files))

    print 'max deviation: ', np.max(stdevs)
    assert mean.shape[0] == data.shape[1]
    return mean, stdevs

def get_dispersion_between_members(files):
    datas = []
    for path in files:
        data = data_select.get_data_from_file(path)

        datas.append(data)

    nt, ncell = datas[0].shape
    nmembers = len(datas)
    all_data = np.zeros((nmembers, nt, ncell))
    for i, the_data in enumerate(datas):
        all_data[i, :, :] = the_data[:,:]

    return np.mean(np.std(all_data, axis = 0), axis = 0)





def plot_diff_between_files(file1, file2, i_array, j_array):
    data1 = data_select.get_data_from_file(file1)
    data2 = data_select.get_data_from_file(file2)
    the_diff = np.mean(data2 - data1, axis = 0)

    plot_data(the_diff, i_array, j_array, name = 'the_diff',
              title='AEX, difference between \n %s \n and \n %s' % (file2, file1))

    pass


def print_means_to_file(current_array, future_array):
    """
    Write annual means to files
    one file per member, one line for 30 annual means
    """
    for i in xrange(current_array.shape[0]):
        f_c = open("c_%d.csv" % i, mode="w")
        f_f = open("f_%d.csv" % i, mode="w")
        for pos in xrange(current_array.shape[2]):
            f_c.write(",".join(map(str, current_array[i,:,pos])) + "\n")
            f_f.write(",".join(map(str, future_array[i,:,pos])) + "\n")

        f_c.close()
        f_f.close()



def plot_diff(folder = "data/streamflows/hydrosheds_euler9",
              plot_f_and_c_means_separately = False):
    """
    Plot difference between the means for future and current climate
    """


    file_name = None
    for f_name in os.listdir(folder):
        if f_name.startswith( "aex" ):
            file_name = f_name

    #get indices of interest
    x_indices, y_indices = data_select.get_indices_from_file(path=os.path.join(folder, file_name))

    ##get significance and percentage changes
    #signific_vector, change = bootstrap_for_mean.get_significance_for_change_in_mean_over_months()
    #signific_vector, change = ttest_for_mean_of_merged.get_significance_and_changes_for_months()
    signific_vector, change = bootstrap_for_mean_merged.get_significance_for_change_in_mean_of_merged_over_months()

    signific_vector = signific_vector.astype(int)

    #start plotting (f-c)/c * 100

    plt.subplots_adjust(hspace = 0.2)

    #significance level 5%
    plot_axes = plt.subplot(1,1,1)
    plot_data(  change,
                x_indices, y_indices, name = None,
                color_map = my_cm.get_red_blue_colormap(ncolors = 8),
                #mpl.cm.get_cmap('RdBu', 16),
                minmax = (-40, 40),
                title = '', axes=plot_axes
                )

    color_map = mpl.cm.get_cmap(name="gray", lut=3)

    signific_vector = np.ma.masked_where(signific_vector == 1, signific_vector)
    plot_data(signific_vector, x_indices, y_indices, name = None, title="",
            minmax = (-1, 1), color_map=color_map, draw_colorbar=False, axes=plot_axes)

    plt.tight_layout()

    plt.savefig('future-current(sign).png')





if __name__ == "__main__":
    pylab.rcParams.update(params)
#    data = get_data_from_file('data/streamflows/fdirv1/aex_discharge_1970_01_01_00_00.nc')
#    i_array, j_array = get_indices()
#    plot_data(np.std(data, axis = 0) / np.mean(data, axis = 0) * 100, i_array , j_array, name = 'aex_temp_variability',
#    title = 'AEX (std/mean * 100 %)')
    application_properties.set_current_directory()
    print os.getcwd()
    plot_diff(folder = 'data/streamflows/hydrosheds_euler9')
#    plot_maximums(data_folder = 'data/streamflows/hydrosheds_euler7')
    print "Hello World"
