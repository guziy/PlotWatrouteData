from datetime import timedelta
import os
from matplotlib.ticker import LinearLocator
from data import data_select
import matplotlib.pyplot as plt
import numpy as np
from plot2D.map_parameters import polar_stereographic
from matplotlib import mpl
import application_properties
from data import members
from util import plot_utils

__author__ = 'huziy'

# Plotting all kinds of means for the watroute driven by CRCM4 on AMNO 180x172 domain
# north polar stereographic projection
#

basemap = polar_stereographic.basemap
x = polar_stereographic.xs
y = polar_stereographic.ys



#np.seterr(all="raise")
def plot_cv_for_seasonal_mean(folder_path = "data/streamflows/hydrosheds_euler9",
                            member_ids = None,
                            file_name_pattern = "%s_discharge_2041_01_01_00_00.nc",
                            months = range(1,13),
                            out_file_name = "cv_for_annual_mean.png",
                            max_value = None
                            ):
    """
    calculate and plot cv for annual mean values
    """
    plt.figure()
    times = None
    i_indices = None
    j_indices = None
    x_min, x_max = None, None
    y_min, y_max = None, None
    seasonal_means = []
    for i, the_id in enumerate( member_ids ):
        fName = file_name_pattern % the_id
        fPath = os.path.join(folder_path, fName)
        if not i:
            data, times, i_indices, j_indices = data_select.get_data_from_file(fPath)
            interest_x = x[i_indices, j_indices]
            interest_y = y[i_indices, j_indices]
            x_min, x_max, y_min, y_max = _get_limits(interest_x = interest_x, interest_y = interest_y)
        else:
            data = data_select.get_field_from_file(path = fPath)
        assert data is not None, "i = %d " % i

        if len(months) == 12:
            the_seasonal_mean = np.mean(data, axis = 0)
        else:
            bool_vector = map(lambda t: t.month in months, times)
            indices = np.where(bool_vector)
            the_seasonal_mean = np.mean(data[indices[0],:], axis = 0)
        seasonal_means.append(the_seasonal_mean)


    seasonal_means = np.array( seasonal_means )
    mu = np.mean(seasonal_means, axis=0)
    sigma = np.std(seasonal_means,axis=0)
    cv = sigma / mu

    cMap = mpl.cm.get_cmap(name = "jet_r", lut = 30)
    cMap.set_over(color = "0.5")


    to_plot = np.ma.masked_all(x.shape)
    for the_index, i, j in zip( xrange(len(i_indices)), i_indices, j_indices):
        to_plot[i, j] = cv[the_index]


    basemap.pcolormesh(x, y, to_plot.copy(), cmap = cMap, vmin = 0, vmax = max_value)
    basemap.drawcoastlines(linewidth = 0.5)
    plt.colorbar(ticks = LinearLocator(numticks = 11), format = "%.1e")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.savefig(out_file_name, bbox_inches = "tight")




def plot_seasonal_mean_streamflows(folder_path = "data/streamflows/hydrosheds_euler9",
                                 member_ids = None,
                                 file_name_pattern = "%s_discharge_1970_01_01_00_00.nc",
                                 months = range(1,13),
                                 out_file_name = "annual_means.png"
                                 ):
    print months

    if member_ids is None:
        return

    i_indices = None
    j_indices = None
    times = None
    x_min, x_max = None, None
    y_min, y_max = None, None
    the_seasonal_mean_list = []
    for i, the_id in enumerate( member_ids ):
        fName = file_name_pattern % the_id
        fPath = os.path.join(folder_path, fName)
        print fPath
        data, times, i_indices, j_indices = data_select.get_data_from_file(fPath)
        if not i:
            interest_x = x[i_indices, j_indices]
            interest_y = y[i_indices, j_indices]
            x_min, x_max, y_min, y_max = _get_limits(interest_x = interest_x, interest_y = interest_y)

        assert data is not None, "i = %d " % i
        if len(months) == 12:
            the_seasonal_mean = np.mean(data, axis = 0)
        else:
            bool_vector = map(lambda t: t.month in months, times) # take only month of interest
            indices = np.where(bool_vector)
            print indices[0].shape
            print len(indices)

            the_seasonal_mean = np.mean(data[indices[0],:], axis = 0)
            print data.shape
            print "data = ", data[indices[0],:].shape
            print "mean = ", the_seasonal_mean.shape
            print sum(map(int, bool_vector))
        the_seasonal_mean_list.append(the_seasonal_mean)

    print np.array(the_seasonal_mean_list).shape

    plot_utils.apply_plot_params(aspect_ratio=0.8)
    plt.figure()
    plt.subplots_adjust(hspace = 0.1, wspace = 0.3)
    max_value = np.array(the_seasonal_mean_list).max()


    cMap = mpl.cm.get_cmap(name = "jet_r", lut = 18)

    for k, a_seasonal_mean in enumerate(the_seasonal_mean_list):
        plt.subplot( 2, len(member_ids) // 2 + 1 , k + 1)
        to_plot = np.ma.masked_all(x.shape)
        for the_index, i, j in zip( xrange(len(i_indices)), i_indices, j_indices):
            to_plot[i, j] = a_seasonal_mean[the_index]

        basemap.pcolormesh(x, y, to_plot.copy(), cmap = cMap,
                           vmin = 0, vmax = max_value)
        basemap.drawcoastlines(linewidth = 0.5)
        plt.colorbar(ticks = LinearLocator(numticks = 7), format = "%d")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        #plt.savefig(str(k+1)+"_"+out_file_name)

    plt.savefig(out_file_name)

    pass





def plot_mean_extreme_flow(folder_path = "data/streamflows/hydrosheds_euler9",
                                 member_ids = None,
                                 file_name_pattern = "%s_discharge_1970_01_01_00_00.nc",
                                 out_file_name = "annual_means.png",
                                 high = True,
                                 start_month = 1, end_month = 12
                                 ):
    """
    Plot mean extreme (1-day high or 15-day low) flow over time
    """
    if member_ids is None:
        return


    i_indices = None
    j_indices = None
    times = None
    x_min, x_max = None, None
    y_min, y_max = None, None
    the_extreme_list = []
    for i, the_id in enumerate( member_ids ):
        fName = file_name_pattern % the_id
        fPath = os.path.join(folder_path, fName)
        if not i:
            data, times, i_indices, j_indices = data_select.get_data_from_file(fPath)
            interest_x = x[i_indices, j_indices]
            interest_y = y[i_indices, j_indices]
            x_min, x_max, y_min, y_max = _get_limits(interest_x = interest_x, interest_y = interest_y)
        else:
            data = data_select.get_field_from_file(path = fPath)


        assert data is not None, "i = %d " % i


        if high:
            extremes = data_select.get_list_of_annual_maximums_for_domain(data, times,
                                                                          start_month = start_month,
                                                                          end_month = end_month)
        else:
            extremes = data_select.get_list_of_annual_minimums_for_domain(data, times,
                                                                          event_duration = timedelta(days = 15),
                                                                          start_month = start_month,
                                                                          end_month = end_month
                                                                          )

        the_extreme_list.append(np.mean(extremes, axis = 0))


    print "shape of extremes list ", np.array(the_extreme_list).shape

    plot_utils.apply_plot_params(aspect_ratio=0.8)
    plt.figure()
    plt.subplots_adjust(hspace = 0.1, wspace = 0.3)

    for k, the_extreme_mean in enumerate(the_extreme_list):
        plt.subplot( 2, len(member_ids) // 2 + 1 , k + 1)
        to_plot = np.ma.masked_all(x.shape)
        for the_index, i, j in zip( xrange(len(i_indices)), i_indices, j_indices):
            to_plot[i, j] = the_extreme_mean[the_index]

        basemap.pcolormesh(x, y, to_plot.copy(), cmap = mpl.cm.get_cmap(name = "jet_r", lut = 18), vmin = 0,
                           vmax = 1.5)
        basemap.drawcoastlines(linewidth = 0.5)
        plt.colorbar(ticks = LinearLocator(numticks = 7), format = "%.2e")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    plt.savefig(out_file_name)

    #plot cv for the extremes     (here for performance, no need to again fetch the extremes)
    max_value = 0.1
    plot_utils.apply_plot_params(width_pt=600)
    plt.figure()
    extreme_means = np.array( the_extreme_list )
    mu = np.mean(extreme_means, axis=0)
    sigma = np.std(extreme_means,axis=0)
    cv = sigma / mu

    to_plot = np.ma.masked_all(x.shape)
    for the_index, i, j in zip( xrange(len(i_indices)), i_indices, j_indices):
        to_plot[i, j] = cv[the_index]

    basemap.pcolormesh(x, y, to_plot.copy(), cmap = mpl.cm.get_cmap(name = "jet_r", lut = 30), vmin = 0,
                       vmax = max_value)
    basemap.drawcoastlines(linewidth = 0.5)
    plt.colorbar(ticks = LinearLocator(numticks = 11), format = "%.1e")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig("cv_" + out_file_name)
    pass



def _get_limits(interest_x = None, interest_y = None):
    """
    get limits of the region of interest for plotting
    """
    x_min, x_max = interest_x.min(), interest_x.max()
    y_min, y_max = interest_y.min(), interest_y.max()

    dx = x_max - x_min
    dx *= 0.1
    x_max += dx
    x_min -= dx

    dy = y_max - y_min
    dy *= 0.1
    y_max += dy
    y_min -= dy
    return x_min, x_max, y_min, y_max




def main():
    #

    high_to_start_month = {
        True : 3,
        False : 1
    }

    high_to_end_month = {
        True : 7,
        False: 5

    }

    season_to_months = {
        "djf" : [12,1,2],
        "mam" : [3,4,5],
        "jja" :  [6,7,8],
        "son" :  [9,10,11],
        "high_season" : [3,4,5,6,7],
        "low_season" : [1,2,3,4,5]
    }


    input_file_name_patterns = {
        "future" : "%s_discharge_2041_01_01_00_00.nc",
        "current" : "%s_discharge_1970_01_01_00_00.nc"
    }

#    plot_seasonal_mean_streamflows(member_ids = members.all_current,
#                              file_name_pattern=input_file_name_patterns["current"],
#                              out_file_name="current_annual_means.png"
#                              )
#
#    plot_seasonal_mean_streamflows(member_ids = members.all_future,
#                              file_name_pattern=input_file_name_patterns["future"],
#                              out_file_name="future_annual_means.png"
#                              )
    plot_cv_for_annual_means = False
    if plot_cv_for_annual_means:
        plot_cv_for_seasonal_mean(member_ids = members.all_future,
                                  file_name_pattern=input_file_name_patterns["future"],
                                  out_file_name="cv_for_future_annual_mean.png", max_value=0.1)
        plot_cv_for_seasonal_mean(member_ids = members.all_current,
                                  file_name_pattern=input_file_name_patterns["current"],
                                  out_file_name="cv_for_current_annual_mean.png", max_value=0.1
                                  )


    #high and low flow plots means and cv
    for high in [True, False]:
        high_low = "high" if high else "low"
        for period_name, in_file_pattern in input_file_name_patterns.iteritems():
            the_ids = members.all_current if period_name == "current" else members.all_future
            plot_mean_extreme_flow(member_ids=the_ids, out_file_name="mean_%s_%s.png" % (high_low, period_name),
                                   file_name_pattern = in_file_pattern, high = high,
                                   start_month=high_to_start_month[high],
                                   end_month=high_to_end_month[high]
                                   )





    plot_cv_of_seasonal_means = False

    if plot_cv_of_seasonal_means:
        for period_name, in_file_pattern in input_file_name_patterns.iteritems():
            for season, months in season_to_months.iteritems():
                the_ids = members.all_current if period_name == "current" else members.all_future
                plot_cv_for_seasonal_mean(member_ids = the_ids,
                                               months = months, out_file_name="cv_%s_%s.png" % (season, period_name),
                                               file_name_pattern=input_file_name_patterns[period_name]
                                               )
#            plot_seasonal_mean_streamflows(member_ids=the_ids,
#                                           months=months, out_file_name="seasonal_%s_%s.png" % (season, period_name) ,
#                                           file_name_pattern=input_file_name_patterns[period_name]
#                                           )

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()