from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter
from gevfit import gevfit

__author__="huziy"
__date__ ="$1-May-2011 2:35:39 PM$"

import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from data import data_select

import plot2D.calculate_performance_errors as pe_calc
import numpy as np

#compares low flows and high flows

import util.plot_utils as plot_utils

#return periods to colors and markers
colors = {2:'k', 5:'k', 10:'k', 30:'k'}
markers = {2: 'o', 5:'o', 10:'o', 30:'o' }
face_colors = {2: 'none', 5:'k', 10:'none', 30:'k' }


import application_properties


selected_station_ids = [
    "104001", "103715", "093806", "093801", "092715", "081006", "061502",
    #"080718",
    "040830"
]


#delete non-continuous parts of series
def delete_not_continuous(data):


    stations_to_remove = []
    model_points_to_remove = []
    for station, data_point in data.iteritems():
        #the_dates_to_retain = station.get_longest_continuous_series(data_step = timedelta(days = 1))


        print station.get_timeseries_length(), data_point.get_timeseries_length()
        if station.get_timeseries_length() == 0 or data_point.get_timeseries_length() == 0:
            station.remove_all_observations()
            data_point.clear_timeseries()

            stations_to_remove.append(station)
            model_points_to_remove.append(data_point)
            continue




        # @type data_point ModelPoint
        start_date = max(station.dates[0], data_point.get_sorted_dates()[0])
        end_date = min(station.dates[-1], data_point.get_sorted_dates()[-1])



        if start_date.day > 1 or start_date.month > 1:
            start_date = datetime(start_date.year + 1, 1, 1,0,0,0)

        if end_date.day < 31 or end_date.month < 12:
            end_date = datetime(end_date.year - 1, 12, 31,0,0,0)


        print start_date, end_date

        start_year = start_date.year
        end_year = end_date.year


        if end_date < start_date:
            print 'warning: start date ' + str(start_date) + ', end date ' + str(end_date)
            print 'clearing data for the station ' + station.id  + ' and corresponding point.'
            # @type station Station
            station.remove_all_observations()
            data_point.clear_timeseries()
            stations_to_remove.append(station)
            model_points_to_remove.append(data_point)

            continue

        # @type station Station
        station.delete_data_before_year(start_year)
        station.delete_data_after_year(end_year)

        data_point.delete_data_before_year(start_year)
        data_point.delete_data_after_year(end_year)




        for year in xrange(start_year, end_year + 1):
            # @type station Station
            series = station.get_continuous_dataseries_for_year(year)
            if len(series) < 365:
                station.delete_data_for_year(year)
                data_point.delete_data_for_year(year)

        if not data_point.get_timeseries_length():
            continue

        if not station.get_timeseries_length():
            continue
        print station.dates[0], data_point.get_sorted_dates()[0]
        print station.dates[-1], data_point.get_sorted_dates()[-1]
        print len(station.dates), len(data_point.get_sorted_dates())
        assert len(station.dates) == len(data_point.get_sorted_dates())

    #remove from consideration the stations with 0 relevan obs series
    for station in stations_to_remove:
       del data[station]

def get_minmax_for_plot(x_dict, y_dict):
    vals = []
    if type(x_dict) == type([]): #x_dict is a list
        vals.extend(x_dict)
        vals.extend(y_dict)
    else:
        vals.extend(x_dict.values())
        vals.extend(y_dict.values())
    return np.min(vals), np.max(vals)

def hide_even_pos_ax_labels(x, pos):
    if not pos % 2:
        return x
    return ""

    pass

def plot_scatter(x_dict, y_dict, xlabel = 'x', ylabel = 'y', title = '',
                 different_shapes_and_colors = True, new_figure = True
                 ):

    """
    ticklabel_modifier - takes ticklabel string and its pos as inut
    """
    if new_figure:
        plt.figure()

    x = []
    y = []
   

    handles = []
    labels = []
    if type(x_dict) == type([]):
        plt.scatter(x_dict, y_dict)
    else:
        for key in x_dict.keys():
            x = x_dict[key]
            if not y_dict.has_key(key):
                continue
            y = y_dict[key]

            if different_shapes_and_colors: #return levels {period => level}
                h = plt.scatter(x, y, color = colors[key],
                    marker = markers[key], s = 50, linewidths=1, facecolors = face_colors[key],
                    edgecolors = "k")
                handles.append(h)
                labels.append('{0}-year'.format(key))
            else: #extreme events
                plt.scatter(x, y, color = 'k', linewidths = 1)

    if different_shapes_and_colors:
        leg = plt.legend(handles, labels, 'upper left', scatterpoints = 1,
                                handletextpad = 0, borderpad = 0)
        leg.draw_frame(False)
        print leg.get_patches()


    z = get_minmax_for_plot(x_dict, y_dict)
    z = [0, z[1] * 1.1 ]

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


    plt.plot(z, z, color = 'k')
    plt.xlim(z)
    plt.ylim(z)



def date_time_to_day_of_year(d):
    return d.timetuple().tm_yday

##args - lists of start dates for the high and low flow events
def plot_dates_scatter(model_high_dates, station_high_dates, model_low_dates, station_low_dates):
    plt.figure()
    plt.xlabel('Observed (day of year)')
    plt.ylabel('Modelled (day of year)')

    plt.xlim(0,250)
    plt.ylim(0,250)
    model_high_days = map(date_time_to_day_of_year, model_high_dates)
    station_high_days = map(date_time_to_day_of_year, station_high_dates)
    model_low_days = map(date_time_to_day_of_year, model_low_dates)
    station_low_days = map(date_time_to_day_of_year, station_low_dates)

    h_high = plt.scatter(station_high_days, model_high_days, color = 'g', marker = '^')
    h_low = plt.scatter(station_low_days, model_low_days, color = 'r', marker = 'd')

    plt.plot([0,250], [0,250], color = 'k')
    plt.legend([h_low, h_high], ['low flow', 'high flow'], 'upper center')
    pass


def plot_boxplot(data, title = "", file_name = "", labels = None, point_ids = None):
    point_ids = [] if point_ids is None else point_ids
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.title(title)
    bp = plt.boxplot(data, bootstrap=5000)
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("Flow ${\\rm m^3/s}$")



    [y_bottom, y_top] = ax.get_ylim()
    if y_top == 600:
        y_top = 400
        ax.set_ylim(y_bottom, y_top)

    if len(point_ids) > 0:
        tick_locs = ax.xaxis.get_majorticklocs()

        assert len(point_ids) == len(tick_locs), "n_points = %d, nticks = %d" % (len(point_ids), len(tick_locs))

        for i, loc in enumerate(tick_locs):
            if i == tick_locs.shape[0] - 1:
                continue

            if not i % 2:
                xlabel = (tick_locs[i] + tick_locs[i + 1]) * 0.5
                plt.annotate(point_ids[i], xy = (xlabel, y_top - 0.02 * (y_top - y_bottom)),
                                            rotation = "vertical", va = "top")
            else:
                xline = (tick_locs[i] + tick_locs[i + 1]) * 0.5
                plt.plot([xline, xline], [y_bottom, y_top], color = "k", lw = 0.5)

    plt.setp(plt.xticks()[1], rotation=30)
    plt.tight_layout()
    plt.savefig(file_name)




def main():
    path = 'data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc'

    #path = "data/streamflows/hydrosheds_rk4_changed_partiotioning/aex_discharge_1970_01_01_00_00.nc"
    #path = 'data/streamflows/na/discharge_1990_01_01_00_00_na.nc'
    data = pe_calc.get_station_and_corresponding_model_data(path = path)
    delete_not_continuous(data)



    plot_utils.apply_plot_params(width_pt = None, height_cm=6, font_size = 9)

    high_return_periods = [10, 30]
    high_start_month = 3
    high_end_month = 7
    high_event_duration = timedelta(days = 1)


    low_return_periods = [2, 5]
    low_start_month = 1
    low_end_month = 5
    low_event_duration = timedelta(days = 15)


    #---------------------------high
    high_station_return_levels = {}
    high_model_return_levels = {}

    station_high_dates = []
    model_high_dates = []


    station_minima = []
    station_maxima = []

    model_minima = []
    model_maxima = []

    high_data = []
    labels = []

    point_ids = []

    for station, model_point in data.iteritems():
        if station.id not in selected_station_ids:
            continue


        # @type model_point ModelPoint
        if not model_point.get_timeseries_length():
            continue

        # @type station Station
        if not station.get_timeseries_length():
            continue
        
        high_values_station = data_select.get_period_maxima(station.values, station.dates,
                                      start_date = None, end_date = None,
                                      start_month = high_start_month,
                                      end_month = high_end_month,
                                      event_duration = high_event_duration)
        vals = np.array(high_values_station.values())
        high_data.append(high_values_station.values())
        labels.append("Observed")


        pars_station = gevfit.optimize_stationary_for_period(vals, high_flow = True)


        high_values_model = data_select.get_period_maxima(
                                      model_point.get_values_sorted_by_date(),
                                      model_point.get_sorted_dates(),
                                      start_date = None, end_date = None,
                                      start_month = high_start_month,
                                      end_month = high_end_month,
                                      event_duration = high_event_duration)

        high_data.append(high_values_model.values())
        labels.append("Modelled")
        vals = np.array(high_values_model.values())

        point_ids.append(station.id)
        point_ids.append(station.id)

        pars_model = gevfit.optimize_stationary_for_period(vals, high_flow = True)

        for ret_period in high_return_periods:
            if not high_station_return_levels.has_key(ret_period):
                high_station_return_levels[ret_period] = []
                high_model_return_levels[ret_period] = []
            high_station_return_levels[ret_period].append(gevfit.get_high_ret_level_stationary(pars_station, ret_period))
            high_model_return_levels[ret_period].append(gevfit.get_high_ret_level_stationary(pars_model, ret_period))
            

        #gather dates of the high flow events
        the_station_high_dates = data_select.get_period_maxima_dates(
                                    station.values, station.dates,
                                    start_date = None, end_date = None,
                                    start_month = high_start_month,
                                    end_month = high_end_month,
                                    event_duration = high_event_duration)
        station_high_dates.extend(the_station_high_dates.values())

        the_model_high_dates = data_select.get_period_maxima_dates(
                                    model_point.get_values_sorted_by_date(),
                                    # @type model_point ModelPoint
                                    model_point.get_sorted_dates(),
                                    start_date = None, end_date = None,
                                    start_month = high_start_month,
                                    end_month = high_end_month,
                                    event_duration = high_event_duration)
        model_high_dates.extend(the_model_high_dates.values())



        ##save corresponding maximums for the station and model
        current_station_maxima = []
        current_model_maxima = []
        for year, value in high_values_station.iteritems():
            if high_values_model.has_key(year):
                station_maxima.append(value)
                model_maxima.append(high_values_model[year])

                current_station_maxima.append(value)
                current_model_maxima.append(high_values_model[year])


        #plot scatter for low flow for each station separately
        plot_scatter( current_station_maxima, current_model_maxima, "observed (${\\rm m^3/s}$)",
                    "modelled (${\\rm m^3/s}$)",
                  "high flow values ({0})".format(station.id), different_shapes_and_colors = False)
        plt.savefig('high_values_scatter_{0}.png'.format( station.id ), bbox_inches = 'tight')

        print "%s: n(high_values) = %d;" % (station.id, len( current_station_maxima ))

    ##high flow values
    plot_scatter( station_maxima, model_maxima, "observed (${\\rm m^3/s}$)",
                    "modelled (${\\rm m^3/s}$)",
                    "high flow values", different_shapes_and_colors = False)

    plt.savefig('high_values_scatter.png', bbox_inches = 'tight')

    plot_boxplot([station_maxima, model_maxima],
            title="High flow amplitude",
            labels = ["Observed", "Modelled"], file_name="box_high.png")

    plot_boxplot(high_data, labels=labels, file_name="box_high_all_sep.png", point_ids=point_ids)



    #---------------------------low
    low_station_return_levels = {}
    low_model_return_levels = {}

    station_low_dates = []
    model_low_dates = []

    low_data = []
    for station, model_point in data.iteritems():
        if station.id not in selected_station_ids:
            continue

        #print "retained timeseries lenght of the station %s is %d " % (station.id, station.get_timeseries_length())

        # @type model_point ModelPoint
        if not model_point.get_timeseries_length():
            continue

        # @type station Station
        if not station.get_timeseries_length():
            continue


#        print 'station min: %f, max: %f' % (np.min(station.values), np.max(station.values))
#        print 'model min: %f, max: %f' % (
#                                            np.min(model_point.get_values_sorted_by_date()),
#                                            np.max(model_point.get_values_sorted_by_date())
#                                           )

        low_values_station = data_select.get_period_minima(station.values, station.dates,
                                      start_date = None, end_date = None,
                                      start_month = low_start_month,
                                      end_month = low_end_month,
                                      event_duration = low_event_duration)
        
        low_data.append(low_values_station.values())
        vals = np.array(low_values_station.values())
        pars_station = gevfit.optimize_stationary_for_period(vals, high_flow = False)

        low_values_model = data_select.get_period_minima(
                                      model_point.get_values_sorted_by_date(),
                                      model_point.get_sorted_dates(),
                                      start_date = None, end_date = None,
                                      start_month = low_start_month,
                                      end_month = low_end_month,
                                      event_duration = low_event_duration)
        vals = np.array(low_values_model.values())
        low_data.append(low_values_model.values())
        pars_model = gevfit.optimize_stationary_for_period(vals, high_flow = False)

        #gather dates of the low flow events
        the_station_low_dates = data_select.get_period_minima_dates(
                                    station.values, station.dates,
                                    start_date = None, end_date = None,
                                    start_month = low_start_month,
                                    end_month = low_end_month,
                                    event_duration = low_event_duration)
        station_low_dates.extend(the_station_low_dates.values())

        the_model_low_dates = data_select.get_period_minima_dates(
                                    model_point.get_values_sorted_by_date(),
                                    model_point.get_sorted_dates(),
                                    start_date = None, end_date = None,
                                    start_month = low_start_month,
                                    end_month = low_end_month,
                                    event_duration = low_event_duration)
        model_low_dates.extend(the_model_low_dates.values())



        for ret_period in low_return_periods:
            if not low_station_return_levels.has_key(ret_period):
                low_station_return_levels[ret_period] = []
                low_model_return_levels[ret_period] = []
            low_station_return_levels[ret_period].append(gevfit.get_low_ret_level_stationary(pars_station, ret_period))
            low_model_return_levels[ret_period].append(gevfit.get_low_ret_level_stationary(pars_model, ret_period))



        ##save corresponding maxima for the station and model
        current_station_minima = []
        current_model_minima = []
        for year, value in low_values_station.iteritems():
            if low_values_model.has_key(year):
                current_station_minima.append(value)
                current_model_minima.append(low_values_model[year])

                station_minima.append(value)
                model_minima.append(low_values_model[year])


        ##plot temporal dependency
        ts = sorted(low_values_station.keys())
        values_at_ts = []
        for the_t in ts:
            values_at_ts.append(low_values_station[the_t])
        plt.figure()
        plt.plot(ts, values_at_ts)
        plt.ylabel('low flow')
        plt.xlabel('time')
        plt.title(station.id)
        plt.savefig('lows_in_time_%s.png' % station.id)


        #plot scatter for low flow for each station separately
        plot_scatter( current_station_minima, current_model_minima, "observed (${\\rm m^3/s}$)",
                    "modelled (${\\rm m^3/s}$)",
                  "low flow values ({0})".format(station.id), different_shapes_and_colors = False)
        plt.savefig('low_values_scatter_{0}.png'.format( station.id ), bbox_inches = 'tight')

        print "%s: n(low_values) = %d;" % (station.id, len( current_station_minima ))



    #draw return levels
    fig = plt.figure()
    gs = gridspec.GridSpec(1,2)
    ax = fig.add_subplot(gs[0,0])
    plot_scatter(high_station_return_levels,
                 high_model_return_levels,
                 xlabel="Observed return levels",
                 ylabel="Modelled return levels", new_figure=False,
                 title="(a) High flow"

    )
    ax.xaxis.set_major_formatter(FuncFormatter(hide_even_pos_ax_labels))

    fig.add_subplot(gs[0,1])
    plot_scatter(low_station_return_levels,
                 low_model_return_levels,
                 xlabel="Observed return levels",
                 ylabel="Modelled  return levels",
        title="(b) Low flow",
        new_figure=False)


    plt.tight_layout()
    plt.savefig("rl_scatter.png")

    plot_scatter( station_minima, model_minima, "observed", "modelled",
                  "low flow values", different_shapes_and_colors = False)
    plt.savefig('low_values_scatter.png', bbox_inches = 'tight')

    plot_boxplot([station_minima, model_minima],
            title="Low flow amplitude",
            labels = ["Observed", "Modelled"], file_name="box_low.png")

    plot_boxplot(low_data, labels=labels, file_name="box_low_all_sep.png", point_ids= point_ids)

    #plot dates of the high and low flow occurences
    #before uncommenting figure out why the x and y arrays are of different sizes
#    plot_dates_scatter(model_high_dates, station_high_dates, model_low_dates, station_low_dates)
#    plt.savefig('occurences_scatter.png')

  #  plt.show()

    pass





if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello World"
