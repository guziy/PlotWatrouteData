__author__="huziy"
__date__ ="$31.12.2010 4:42:33$"


from mpl_toolkits.basemap import NetCDFFile
import numpy as np
from datetime import timedelta
from datetime import datetime
#import netCDF4 as nc


def get_field_from_file(path = '', field_name = ''):
    fpin = NetCDFFile(path)
    the_field = fpin.variables[field_name][:,:]
    fpin.close()
    return the_field

def get_data_from_file(path, field_name = 'water_discharge'):
    date_format = '%Y_%m_%d_%H_%M'

    fpin = NetCDFFile(path)
    
    times = fpin.variables['time'][:]
    #dims: time, cell_index
    discharge = fpin.variables[field_name][:,:]
    print discharge.shape
    x_indices = fpin.variables['x-index'][:]
    y_indices = fpin.variables['y-index'][:]
    
    date_times = []
    for t in times:
        date_times.append( datetime.strptime( ''.join(t) , date_format ) )

    fpin.close()
    return discharge, date_times, x_indices, y_indices


#get i_indices and j_indices from a file
def get_indices_from_file(path = 'data/streamflows/hydrosheds_euler8/aex_discharge_1970_01_01_00_00.nc'):
    fpin = NetCDFFile(path)
    vars = fpin.variables

    x, y = vars['x-index'][:], vars['y-index'][:]
    fpin.close()
    return x, y



def get_period_minima_dates(streamflows, times,
                      start_date = None, end_date = None,
                      start_month = 1, end_month = 12, event_duration = timedelta(days = 7)):


    averaging_length = 2
    dt1 = times[1] - times[0] # data time step
    while (averaging_length - 1) * dt1 < event_duration:
        averaging_length += 1

    result = {}
    result_dates = {}
    for i, time in enumerate(times):
        time_plus_duration = time + event_duration
        #select by date
        if start_date != None and time < start_date:
            continue

        if end_date != None and time_plus_duration > end_date:
            break

        #select by month
        if time_plus_duration.month < start_month or time.month < start_month:
            continue

        if time_plus_duration.month > end_month or time.month > end_month:
            continue

        value = np.mean(streamflows[i : i + averaging_length])

        if not np.isfinite(value):
            print streamflows
        assert np.isfinite(value)


        the_year = time.year
        if result.has_key(the_year):
            if result[the_year] > value:
                result[the_year] = value
                result_dates[the_year] = time
        else:
            result[the_year] = value
            result_dates[the_year] = time

    return result_dates

##get dates of occurences of the minimums
#{year: date}
def get_period_maxima_dates(streamflows, times,
                      start_date = None, end_date = None,
                      start_month = 1, end_month = 12, event_duration = timedelta(days = 1)):
    averaging_length = 2
    dt1 = times[1] - times[0]
    while (averaging_length - 1) * dt1 < event_duration:
        averaging_length += 1

    result = {}
    result_dates = {}
    for i, time in enumerate(times):
        time_plus_duration = time + event_duration
        #select by date
        if start_date != None and time < start_date:
            continue

        if end_date != None and time_plus_duration > end_date:
            break

        #select by month
        if time_plus_duration.month < start_month or time.month < start_month:
            continue

        if time_plus_duration.month > end_month or time.month > end_month:
            continue

        if i + averaging_length > len(streamflows): break #in order to respect limits

        value = np.mean(streamflows[i : i + averaging_length])
        assert np.isfinite(value)

        the_year = time.year
        if result.has_key(the_year):
            if result[the_year] < value:
                result[the_year] = value
                result_dates[the_year] = time
        else:
            result[the_year] = value
            result_dates[the_year] = time

    return result_dates
    pass



# returns <>duration_days<> low flow over the period [start_month, end_month],
# considering data in the streamflow vector which correspond to
# times between start_date and end_date inclusive.
#streamflows - 1D array (time)
#returns dictionary {year => averaged min flow}
def get_period_minima(streamflows, times,
                      start_date = None, end_date = None,
                      start_month = 1, end_month = 12, event_duration = timedelta(days = 7)):

    
    averaging_length = 0
    for time in times:
        if time - times[0] < event_duration:
            averaging_length += 1
        else:
            break

    print 'av. length = %d' % averaging_length

    result = {}
    for i, time in enumerate(times):
        time_plus_duration = time + event_duration
        #select by date
        if start_date != None and time < start_date:
            continue

        if end_date != None and time_plus_duration > end_date:
            break

        #select by month
        if time_plus_duration.month < start_month or time.month < start_month:
            continue

        if time_plus_duration.month > end_month or time.month > end_month:
            continue

        if i + averaging_length > len(streamflows): break #in order to respect limits


        
        value = np.mean(streamflows[i : i + averaging_length])


        if not np.isfinite(value):
            print streamflows
        assert np.isfinite(value)


        the_year = time.year
        if result.has_key(the_year):
            if result[the_year] > value:
                result[the_year] = value
        else:
            result[the_year] = value

    return result

##get dates of occurences of the minimums
#{year: date}
def get_period_maxima_dates(streamflows, times,
                      start_date = None, end_date = None,
                      start_month = 1, end_month = 12, event_duration = timedelta(days = 1)):
    averaging_length = 0
    for time in times:
        if time - times[0] < event_duration:
            averaging_length += 1
        else:
            break

    result = {}
    result_dates = {}
    for i, time in enumerate(times):
        time_plus_duration = time + event_duration
        #select by date
        if start_date != None and time < start_date:
            continue

        if end_date != None and time_plus_duration > end_date:
            break

        #select by month
        if time_plus_duration.month < start_month or time.month < start_month:
            continue

        if time_plus_duration.month > end_month or time.month > end_month:
            continue

        value = np.mean(streamflows[i : i + averaging_length])
        assert np.isfinite(value)

        the_year = time.year
        if result.has_key(the_year):
            if result[the_year] < value:
                result[the_year] = value
                result_dates[the_year] = time
        else:
            result[the_year] = value
            result_dates[the_year] = time

    return result_dates
    pass

    
# returns <>duration_days<> high flow over the period [start_month, end_month],
# considering data in the streamflow vector which correspond to
# times between start_date and end_date inclusive.
#streamflows - 1D array (time)
#returns dictionary {year => averaged max flow}
def get_period_maxima(streamflows, times,
                      start_date = None, end_date = None,
                      start_month = 1, end_month = 12, event_duration = timedelta(days = 1)):

    
    averaging_length = 0
    for time in times:
        if time - times[0] < event_duration:
            averaging_length += 1
        else:
            break

    result = {}
    for i, time in enumerate(times):
        time_plus_duration = time + event_duration
        #select by date
        if start_date != None and time < start_date:
            continue

        if end_date != None and time_plus_duration > end_date:
            break

        #select by month
        if time_plus_duration.month < start_month or time.month < start_month:
            continue

        if time_plus_duration.month > end_month or time.month > end_month:
            continue

        value = np.mean(streamflows[i : i + averaging_length])
        assert np.isfinite(value)

        the_year = time.year
        if result.has_key(the_year):
            if result[the_year] < value:
                result[the_year] = value
        else:
            result[the_year] = value

    return result
    pass



def get_list_of_annual_maximums_for_domain(streamflows, times,
                      start_date = None, end_date = None,
                      start_month = 1,
                      end_month = 12, event_duration = timedelta(days = 1)):
    result = []
    for pos in range(streamflows.shape[1]):
        the_dict = get_period_maxima(streamflows[:, pos],
                                            times,
                                            start_date,
                                            end_date,
                                            start_month, end_month,
                                            event_duration = event_duration)

        sorted_vals = []
        sorted_keys = the_dict.keys()
        sorted_keys.sort()
        for key in sorted_keys:
            sorted_vals.append(the_dict[key])

        result.append(sorted_vals)
    return np.array(result).transpose() #(nyears, nx)







def get_list_of_annual_minimums_for_domain(streamflows, times,
                      start_date = None, end_date = None,
                      start_month = 1,
                      end_month = 12, event_duration = timedelta(days = 7)):
    result = []
    for pos in range(streamflows.shape[1]):
        the_dict = get_period_minima(streamflows[:, pos],
                                            times,
                                            start_date,
                                            end_date,
                                            start_month, end_month,
                                            event_duration = event_duration)
        result.append(the_dict.values())
    return result



#streamflow is of dimensions  (nt, n-grid-cells)
#return 1D array of size n-grid-cells


def get_maximums_for_domain(streamflows, times,
                      start_date = None, end_date = None,
                      start_month = 1,
                      end_month = 12, duration_days = timedelta(days = 1)):


    the_maxima = np.zeros((streamflows.shape[1],))
    for pos in range(streamflows.shape[1]):
        the_dict = get_period_maxima(streamflows[:, pos],
                                            times,
                                            start_date,
                                            end_date,
                                            start_month, end_month,
                                            event_duration = duration_days)
        the_maxima[pos] =  np.max(the_dict.values())

    return the_maxima



#streamflow is of dimensions  (nt, n-grid-cells)
#return 1D array of size n-grid-cells
def get_minimums_for_domain(streamflows, times,
                      start_date = None, end_date = None,
                      start_month = 1,
                      end_month = 12, duration_days = timedelta(days = 15)):

    result = np.zeros((streamflows.shape[1],))
    for pos in range(streamflows.shape[1]):
        the_dict = get_period_minima(streamflows[:, pos],
                                            times,
                                            start_date,
                                            end_date,
                                            start_month, end_month,
                                            event_duration = duration_days)
        result[pos] =  np.min(the_dict.values())
    return result




#selects annual means from data from the interval
# [start_date; end_date), all data are taken into consideration if
# start_date and end_date are None
#return dictionary {yeari : meani}
def get_annual_means(streamflows, times, start_date = None, end_date = None):
    result = {}
    counts = {}
    for i, time in enumerate(times):
        if time >= end_date:
            break
        if time < start_date:
            continue

        the_year = time.year
        if not result.has_key(the_year):
            result[the_year] = np.zeros(streamflows.shape[1])
            counts[the_year] = 0
            
        result[the_year] += streamflows[i, :]
        counts[the_year] += 1



    for the_year in result.keys():
        result[the_year] /= float( counts[the_year] )

    return result


def test_select():
    import application_properties
    application_properties.set_current_directory()
    data_file = 'data/streamflows/hydrosheds_euler8/aet_discharge_1970_01_01_00_00.nc'
    #get streamflow data
    streamflow, times, xs, ys = get_data_from_file(data_file)

    
    print streamflow.shape

    #test maxima selection
    maxs = get_period_maxima(streamflow[:, 10], times, start_date = datetime(1970,1,1,0,0,0),
                                                       end_date = datetime(1999,12, 31,0,0,0),
                                                       start_month = 4,
                     end_month = 6, event_duration = timedelta(days = 1))

    print maxs

    #test minima selection
    maxs = get_period_minima(streamflow[:, 10], times, start_date = datetime(1970,1,1,0,0,0),
                            end_date = datetime(1999,12,31,0,0,0), start_month = 3,
                            end_month = 4, event_duration = timedelta(days = 15))
    print maxs

    #test get means
    means = get_annual_means(streamflow, times, start_date = datetime(1970,1,1,0,0,0), 
                                                 end_date = datetime(1999,12, 31,0,0,0))

    print len(means)
    print means[1980]
    print means[1972].shape

if __name__ == "__main__":
    test_select()
    print "Hello World"
