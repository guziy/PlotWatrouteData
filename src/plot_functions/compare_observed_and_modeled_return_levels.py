__author__="huziy"
__date__ ="$1-May-2011 2:35:39 PM$"

import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime

from gevfit import gevfit
from data import data_select

import plot2D.calculate_performance_errors as pe_calc
import numpy as np

#compares low flows and high flows

import util.plot_utils as plot_utils

#return periods to colors and markers
colors = {2:'green', 5:'red', 10:'m', 30:'k'}
markers = {2: '>', 5:'+', 10:'d', 30:'x' }


import application_properties



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

        if data_point.get_timeseries_length() == 0:
            continue

        if station.get_timeseries_length() == 0:
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


def plot_scatter(x_dict, y_dict, xlabel = 'x', ylabel = 'y', title = '',
                 different_shapes_and_colors = True):
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
                h = plt.scatter(x, y, color = colors[key], marker = markers[key], s = 300)
                handles.append(h)
                labels.append('{0}'.format(key))
            else: #extreme events
                plt.scatter(x, y, color = 'k')

    if different_shapes_and_colors:
        plt.legend(handles, labels, 'upper center')


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



def main():
    path = 'data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc'
    #path = 'data/streamflows/na/discharge_1990_01_01_00_00_na.nc'
    data = pe_calc.get_station_and_corresponding_model_data(path = path)
    print len(data)
    delete_not_continuous(data)



    plot_utils.apply_plot_params(width_pt = 500, font_size = 18)

    high_return_periods = [10, 30]
    high_start_month = 3
    high_end_month = 7
    high_event_duration = timedelta(days = 1)


    low_return_periods = [2, 5]
    low_start_month = 1
    low_end_month = 3
    low_event_duration = timedelta(days = 15)


    #---------------------------high
    station_return_levels = {}
    model_return_levels = {}

    station_high_dates = []
    model_high_dates = []


    station_minima = []
    station_maxima = []

    model_minima = []
    model_maxima = []


    for station, model_point in data.iteritems():
        # @type model_point ModelPoint
        if model_point.get_timeseries_length() == 0:
            continue

        # @type station Station
        if station.get_timeseries_length() == 0:
            continue
        
        high_values_station = data_select.get_period_maxima(station.values, station.dates,
                                      start_date = None, end_date = None,
                                      start_month = high_start_month,
                                      end_month = high_end_month,
                                      event_duration = high_event_duration)
        vals = np.array(high_values_station.values())

        pars_station = gevfit.optimize_stationary_for_period(vals, high_flow = True)


        high_values_model = data_select.get_period_maxima(
                                      model_point.get_values_sorted_by_date(),
                                      model_point.get_sorted_dates(),
                                      start_date = None, end_date = None,
                                      start_month = high_start_month,
                                      end_month = high_end_month,
                                      event_duration = high_event_duration)
        vals = np.array(high_values_model.values())
        pars_model = gevfit.optimize_stationary_for_period(vals, high_flow = True)

        for ret_period in high_return_periods:
            if not station_return_levels.has_key(ret_period):
                station_return_levels[ret_period] = []
                model_return_levels[ret_period] = []
            station_return_levels[ret_period].append(gevfit.get_high_ret_level_stationary(pars_station, ret_period))
            model_return_levels[ret_period].append(gevfit.get_high_ret_level_stationary(pars_model, ret_period))
            

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
        for year, value in high_values_station.iteritems():
            if high_values_model.has_key(year):
                station_maxima.append(value)
                model_maxima.append(high_values_model[year])




    #plot scatter plot (model vs station (high))
    #return levels

    print len(station_return_levels), len(model_return_levels)
    plot_scatter(station_return_levels, model_return_levels,
                'Observed return level (${\\rm m^3/s}$)',
                'Modelled return level (${\\rm m^3/s}$)'
                )
    plt.savefig('high_return_levels_scatter.pdf', bbox_inches = 'tight')

    ##high flow values
    plot_scatter( station_maxima, model_maxima, "observed (${\\rm m^3/s}$)",
                    "modelled (${\\rm m^3/s}$)",
                    "high flow values", different_shapes_and_colors = False)
    plt.savefig('high_values_scatter.pdf', bbox_inches = 'tight')

    #---------------------------low
    station_return_levels = {}
    model_return_levels = {}

    station_low_dates = []
    model_low_dates = []


    for station, model_point in data.iteritems():

        print "retained timeseries lenght of the station %s is %d " % (station.id, station.get_timeseries_length())

        # @type model_point ModelPoint
        if model_point.get_timeseries_length() == 0:
            continue

        # @type station Station
        if station.get_timeseries_length() == 0:
            continue


        print 'station min: %f, max: %f' % (np.min(station.values), np.max(station.values))
        print 'model min: %f, max: %f' % (
                                            np.min(model_point.get_values_sorted_by_date()),
                                            np.max(model_point.get_values_sorted_by_date())
                                           )

        low_values_station = data_select.get_period_minima(station.values, station.dates,
                                      start_date = None, end_date = None,
                                      start_month = low_start_month,
                                      end_month = low_end_month,
                                      event_duration = low_event_duration)
        

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
            if not station_return_levels.has_key(ret_period):
                station_return_levels[ret_period] = []
                model_return_levels[ret_period] = []
            station_return_levels[ret_period].append(gevfit.get_low_ret_level_stationary(pars_station, ret_period))
            model_return_levels[ret_period].append(gevfit.get_low_ret_level_stationary(pars_model, ret_period))



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
        plt.savefig('low_values_scatter_{0}.pdf'.format( station.id ), bbox_inches = 'tight')



    #plot scatter plot (model vs station (low))
    plot_scatter(station_return_levels, model_return_levels,
                'Observed return level (${\\rm m^3/s}$)',
                'Modelled return level (${\\rm m^3/s}$)' 
                )
    plt.savefig('low_return_levels_scatter.pdf', bbox_inches = 'tight')



    for sV, mV in zip(station_minima, model_minima):
        print sV, mV
    plot_scatter( station_minima, model_minima, "observed", "modelled",
                  "low flow values", different_shapes_and_colors = False)
    plt.savefig('low_values_scatter.pdf', bbox_inches = 'tight')




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
