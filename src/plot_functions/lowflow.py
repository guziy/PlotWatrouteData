from mpl_toolkits.basemap import NetCDFFile
import os.path
import sys
__author__="huziy"
__date__ ="$3 dec. 2010 11:21:58$"

import numpy as np
import application_properties
application_properties.set_current_directory()
from datetime import datetime, timedelta

from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt

from math import sqrt
import pylab

import matplotlib.dates as mdates
import data.members as members
import os

from matplotlib.ticker import FuncFormatter

import calendar

import pickle
import readers.read_infocell as infocell

#Plot number of occurences of high flow or low flow

TIME_FORMAT = '%Y_%m_%d_%H_%M'

inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 1000 * inches_per_pt          # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, 2*fig_height]


font_size = 16
params = {
        'axes.labelsize': font_size,
        'font.size': font_size,
        'text.fontsize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize': fig_size
}


title_font_size = font_size

        
pylab.rcParams.update(params)


class BasinIndices():
    def __init__(self, name, mask):
        self.mask = np.array(mask)
        self.name = name
        pass

    def get_number_of_cells(self):
        array = np.ma.masked_where(self.mask == 0, self.mask)
        return array.count()
        pass

    def get_i_indices(self):
        return np.where(self.mask == 1)[0]

    def get_j_indices(self):
        return np.where(self.mask == 1)[1]

    #override hashing methods to use in dictionary
    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if other is None:
            return False

        return self.name == other.name



def read_basin_indices(path, minimum_accumulated_cells = -1):
    result = []
    nc = NetCDFFile(path)

    for k, v in nc.variables.iteritems():
        result.append(BasinIndices(k, v.data))

    #select only cells with the number of previous cells bigger than minimum_accumulated_cells
    if minimum_accumulated_cells > 0:
        basins = infocell.get_basins_with_cells_connected_using_hydrosheds_data()

        for basin in basins:
            for basin_index in result:
                if basin.name == basin_index.name:
                    the_basin_index = basin_index
                    break

            for the_cell in basin.cells:
                # @type the_cell Cell
                acc_cells = the_cell.calculate_number_of_upstream_cells()
                if acc_cells < minimum_accumulated_cells:
                    the_basin_index.mask[the_cell.x, the_cell.y] = 0

    return result
    pass






def get_day_of_year(date_obj):
    return date_obj.timetuple().tm_yday

def get_low_flow(discharge_data, times, duration_days = timedelta(days = 7),
                   dates_of_stamp_year = None,
                   start_date = None, end_date = None,
                   stamp_year = 2000,
                   start_month_of_stamp_year = 7 ):

    assert len(dates_of_stamp_year) == len(times)
    
    averaging_length = 0
    while times[averaging_length] - times[0] < duration_days:
        averaging_length += 1

    dates_to_occurrences = {}

    min_value = sys.maxint
    
    the_min_date = None #date of the stamp year when the minimum occurred
    for i, the_time, the_date in zip(range(len(times)), times, dates_of_stamp_year):

        #check whether the_time inside the specified range
        if start_date is not None:
            if start_date > the_time:
                continue

        if end_date is not None:
            if end_date <= the_time + duration_days:
                break


        #skip months in the beginning year before start_month
        if the_time.year == times[0].year and the_time.month < start_month_of_stamp_year:
            continue

        #skip months at the end after start_month
        if the_time.year == times[-1].year and the_time.month > start_month_of_stamp_year:
            break



        
        #store day when the minimum occurred at the end of a year
        if the_time.month == start_month_of_stamp_year and the_time.day == 1:
            min_value = sys.maxint

            if the_min_date is None:
                the_min_date = the_date
            else:
                if dates_to_occurrences.has_key(the_min_date):
                    dates_to_occurrences[the_min_date] += 1
                else:
                    dates_to_occurrences[the_min_date] = 1


        if i + averaging_length >= len(discharge_data): break
        value = np.mean(discharge_data[i : i + averaging_length])
        if value < min_value:
            min_value = value
            the_min_date = the_date

    assert len(dates_to_occurrences) <= 366
    #fill the days where were not any occurrences with zeros
    for day in get_days_of_stamp_year(year = stamp_year, start_month_of_stamp_year = start_month_of_stamp_year):
        if not dates_to_occurrences.has_key(day):
            dates_to_occurrences[day] = 0
    

    assert len(dates_to_occurrences) <= 366 and len(dates_to_occurrences) >= 365, 'len = {0}'.format(len(dates_to_occurrences))
    return dates_to_occurrences


def get_high_flow(discharge_data, times, duration_days = timedelta(days = 7),
                   dates_of_stamp_year = None,
                   start_date = None, end_date = None , 
                   stamp_year = 2000, start_month_of_stamp_year = 1
                   ):


    #T = timedelta(days = duration_days)
    averaging_length = 0
    while times[averaging_length] - times[0] < duration_days:
        averaging_length += 1

    dates_to_occurrences = {}
    max_value = -sys.maxint
    the_max_date = None
    for i, the_time, the_date in zip(range(len(times)), times, dates_of_stamp_year):
        #check whether the_time inside the specified range
        if start_date is not None:
            if start_date > the_time:
                continue

        if end_date is not None:
            if end_date < the_time + duration_days:
                break
        #store day when the minimum occurred at the end of a year


        #skip months in the beginning year before start_month
        if the_time.year == times[0].year and the_time.month < start_month_of_stamp_year:
            continue

        #skip months at the end after start_month
        if the_time.year == times[-1].year and the_time.month > start_month_of_stamp_year:
            break



        if the_time.month == start_month_of_stamp_year and the_time.day == 1:
            max_value = -sys.maxint
            if the_max_date is None:
                the_max_date = the_date
            else:
                if dates_to_occurrences.has_key(the_max_date):
                    dates_to_occurrences[the_max_date] += 1
                else:
                    dates_to_occurrences[the_max_date] = 1


        value = np.mean(discharge_data[i : i + averaging_length])
        if value > max_value:
            max_value = value
            the_max_date = the_date

    
    #fill the days where were not any occurrences with zeros
    for day in get_days_of_stamp_year(year = stamp_year, start_month_of_stamp_year = start_month_of_stamp_year):
        if not dates_to_occurrences.has_key(day):
            dates_to_occurrences[day] = 0

    assert len(dates_to_occurrences) <= 366 and len(dates_to_occurrences) >= 365
    return dates_to_occurrences


def get_days_of_stamp_year(year = 2000, start_month_of_stamp_year = 1):
    dt = timedelta(days = 1)
    d = datetime(year, start_month_of_stamp_year, 1,0,0)
    n_days = 366
    return (d + i * dt for i in range(n_days))



def calculate_occurences_for_member(nc = None, bin_dates = [], high_flow = True,
                                    basin_indices = None,
                                    start_date = datetime(1970,1,1,0,0),
                                    end_date = datetime(1999,12,31,0,0),
                                    dates_of_stamp_year = [],
                                    i_indices = [], j_indices = [],
                                    event_duration = timedelta(days = 7),
                                    bin_interval_dt = timedelta(days = 10), times = [],
                                    stamp_year = 2000, start_month_of_stamp_year = 1
                                    ):

    discharge_data = nc.variables['water_discharge'].data[:,:]

#    discharge_data = np.zeros(discharge_data.shape) + 1000
#    for i, t in enumerate(times):
#        if t.month == 5 and t.day == 1:
#            discharge_data[i,:] = 0

    #sum occurences over each basin
    basin_to_data = {}



 
    for cell_index, i, j in zip(range(len(i_indices)), i_indices, j_indices):
        if not high_flow:
            the_map = get_low_flow(discharge_data[:,cell_index], times,
                                dates_of_stamp_year = dates_of_stamp_year,
                                duration_days = event_duration,
                                start_date = start_date,
                                end_date = end_date,
                                stamp_year = stamp_year,
                                start_month_of_stamp_year = start_month_of_stamp_year
                                )
        else:
            the_map = get_high_flow(discharge_data[:,cell_index], times,
                                dates_of_stamp_year = dates_of_stamp_year,
                                duration_days = event_duration,
                                start_date = start_date,
                                end_date = end_date,
                                stamp_year = stamp_year,
                                start_month_of_stamp_year = start_month_of_stamp_year
                                )

        the_basin = None

        for b in basin_indices:
            if b.mask[i, j] == 1:
                the_basin = b
                break

        if the_basin is None:
            continue

        if basin_to_data.has_key(the_basin):
            data = basin_to_data[the_basin]
            for k, v in the_map.iteritems():
                if data.has_key(k):
                    data[k] += v
                else:
                    data[k] = v
        else:
            basin_to_data[the_basin] = the_map


    day_start = datetime(stamp_year, start_month_of_stamp_year, 1)
    n_days = 364
    if calendar.isleap(stamp_year + 1):
        n_days += 1
    day_end = day_start + timedelta(days = n_days)

    #partition the occurences between the bins of width bin_interval_dt

    basin_to_occurences = {}
    for b, data  in  basin_to_data.iteritems():
        print b.name
        k = 0
        start_bin_date = bin_dates[k]
        occs = [0] * len(bin_dates)
        sorted_dates = data.keys()
        sorted_dates.sort()

        assert len(sorted_dates) == 366 , 'len(dates) = {0}'.format(len(sorted_dates))
        
        for the_date in sorted_dates:
            occ = data[the_date]
            if  the_date >= start_bin_date and (the_date < start_bin_date + bin_interval_dt):
                occs[k] += occ
            else:
                if start_bin_date + bin_interval_dt < day_end:
                    start_bin_date += bin_interval_dt
                    k += 1
                    occs[k] += occ
                else:
                    occs[k] += occ
            
        assert len(bin_dates) == len(occs)
        basin_to_occurences[b] = occs

    return basin_to_occurences


def calculate_mean_occurences_and_std(member2basin_and_occ = {}):
    """
        return mean occurence number and std for each day of the stamp year
    """
    basin_2_mean = {}
    basin_2_std = {}

    basin_2_2d_occs = {}

    for member, basin_2_occ in member2basin_and_occ.iteritems():
        for basin, occs in basin_2_occ.iteritems():
            if basin_2_2d_occs.has_key(basin):
                print np.array(occs).shape
                basin_2_2d_occs[basin].append(occs)
            else:
                basin_2_2d_occs[basin] = [occs]

    
    for basin, occs in basin_2_2d_occs.iteritems():
        occs_array = np.array(occs)
        basin_2_mean[basin] = np.mean(occs_array, axis = 0)
        basin_2_std[basin] = np.std(occs_array, axis = 0)
 
    return basin_2_mean, basin_2_std




def get_corresponding_date_of_stamp_year(the_date, stamp_year = 2000, start_month_of_stamp_year = 1):
    year = stamp_year if the_date.month >= start_month_of_stamp_year else stamp_year + 1
    return datetime(year, the_date.month, the_date.day, the_date.hour, the_date.minute, the_date.second)



def calculate_basin_means_and_stds( bin_dates = [],
                    current_start_date = datetime(1970, 1, 1, 0, 0),
                    current_end_date = datetime(1999, 12, 31, 0, 0),
                    future_start_date = datetime(2041, 1, 1, 0, 0),
                    future_end_date = datetime(2070, 12, 31, 0, 0),
                    stamp_year = 2000,
                    start_month_of_stamp_year = 7,
                    bin_interval_dt = timedelta(days = 10),
                    data_folder = '', event_duration = timedelta(days = 7),
                    high_flow = True
                    ):


    print 'data folder: {0}'.format(data_folder)
    path = os.path.join(data_folder, '%s_discharge_1970_01_01_00_00.nc' % members.current_ids[0])
    print 'stamp_year = ', stamp_year
    print 'start month of stamp year = ', start_month_of_stamp_year
    nc = NetCDFFile(path)
    i_ind = nc.variables['x-index'][:]
    j_ind = nc.variables['y-index'][:]


    current_member2basin_and_occurences = {}
    future_member2basin_and_occurences = {}

    basin_path = 'data/infocell/amno180x172_basins.nc'

    basin_indices = read_basin_indices(basin_path, minimum_accumulated_cells = 2)


    times_current = nc.variables['time'][:]
    times_current = map(''.join, times_current)
    print len(times_current), times_current[0], times_current[-1]
    times_current = map(lambda t: datetime.strptime(t, TIME_FORMAT), times_current)



    dates_of_stamp_year_current = map( lambda t :
                        get_corresponding_date_of_stamp_year(t, stamp_year, start_month_of_stamp_year) , times_current)



    
    filename = '%s_discharge_%s.nc' % (members.future_ids[0], future_start_date.strftime(TIME_FORMAT))
    path = os.path.join(data_folder, filename)
    nc = Dataset(path)
    times_future = nc.variables['time'][:]

    times_future = map(''.join, times_future)
    print len(times_future), times_future[0], times_future[-1]
    times_future = map(lambda t: datetime.strptime(t, TIME_FORMAT), times_future)
    dates_of_stamp_year_future = map( lambda t :
                        get_corresponding_date_of_stamp_year(t, stamp_year, start_month_of_stamp_year) , times_future)

    print members.current_ids[0], members.future_ids[0]
    print len(times_current), len(times_future)


    for current_member in members.current_ids:
        filename = '%s_discharge_%s.nc' % (current_member, current_start_date.strftime(TIME_FORMAT))
        path = os.path.join(data_folder, filename)
        nc_current = NetCDFFile(path)

        basin_to_occ_current = calculate_occurences_for_member(nc = nc_current, bin_dates = bin_dates,
                                                               high_flow = high_flow,
                                                               basin_indices = basin_indices,
                                                               start_date = current_start_date,
                                                               end_date = current_end_date,
                                                               dates_of_stamp_year = dates_of_stamp_year_current,
                                                               i_indices = i_ind, j_indices = j_ind,
                                                               event_duration = event_duration,
                                                               stamp_year = stamp_year,
                                                               start_month_of_stamp_year = start_month_of_stamp_year,
                                                               bin_interval_dt = bin_interval_dt, times = times_current)
        current_member2basin_and_occurences[current_member] = basin_to_occ_current
        nc_current.close()



        future_member = members.current2future[current_member]
        filename = '%s_discharge_%s.nc' % (future_member, future_start_date.strftime(TIME_FORMAT))
        path = os.path.join(data_folder, filename)

        nc_future = NetCDFFile(path)
        basin_to_occ_future = calculate_occurences_for_member(nc = nc_future, bin_dates = bin_dates,
                                                               high_flow = high_flow,
                                                               basin_indices = basin_indices,
                                                               start_date = future_start_date,
                                                               end_date = future_end_date,
                                                               dates_of_stamp_year = dates_of_stamp_year_future,
                                                               i_indices = i_ind, j_indices = j_ind,
                                                               event_duration = event_duration,
                                                               stamp_year = stamp_year,
                                                               start_month_of_stamp_year = start_month_of_stamp_year,
                                                               bin_interval_dt = bin_interval_dt, times = times_future)
        future_member2basin_and_occurences[future_member] = basin_to_occ_future
        nc_future.close()

    return calculate_mean_occurences_and_std(current_member2basin_and_occurences), \
           calculate_mean_occurences_and_std(future_member2basin_and_occurences)


def main(event_duration = timedelta(days = 7),
         prefix = '', data_folder = 'data/streamflows/VplusF_newmask1',
         start_month_of_stamp_year = 1, stamp_year = 2000, high_flow = True ):

    
    bin_interval_dt = timedelta(days = 10)

    current_start_date = datetime(1970, 1, 1, 0, 0)
    current_end_date = datetime(1999, 12, 31, 0, 0)
    future_start_date = datetime(2041, 1, 1, 0, 0)
    future_end_date = datetime(2070, 12, 31, 0, 0)



    bin_dates = []

    day_start = datetime(stamp_year, start_month_of_stamp_year, 1)
    day_end = day_start + timedelta(days = 365 if calendar.isleap(stamp_year + 1) else 364)

    print day_end
    d = day_start
    bin_widths = []
    bar_width_on_plot = mdates.date2num(day_start + bin_interval_dt) - mdates.date2num(day_start)
    while d < day_end:
        bin_dates.append(d)
        bin_widths.append(min(bar_width_on_plot, mdates.date2num(day_end) - mdates.date2num(d)))
        d += bin_interval_dt



    print 'Calculating mean and standard deviations ...'


    if os.path.isfile('%s_duration%d_current_basin2mean' % (prefix, event_duration.days)):
        current_basin2mean = pickle.load(open('%s_duration%d_current_basin2mean' % (prefix, event_duration.days)  ))
        current_basin2std = pickle.load(open('%s_duration%d_current_basin2std' % (prefix, event_duration.days)))
        future_basin2mean = pickle.load(open('%s_duration%d_future_basin2mean' % (prefix, event_duration.days)))
        future_basin2std = pickle.load(open('%s_duration%d_future_basin2std' % (prefix, event_duration.days)))
    else:
        current, future = \
                calculate_basin_means_and_stds( bin_dates = bin_dates, high_flow = high_flow,
                    current_start_date = current_start_date,
                    current_end_date = current_end_date,
                    future_start_date = future_start_date,
                    future_end_date = future_end_date,
                    stamp_year = stamp_year, start_month_of_stamp_year = start_month_of_stamp_year,
                    bin_interval_dt = bin_interval_dt, data_folder = data_folder,
                    event_duration = event_duration
                )


        current_basin2mean, current_basin2std = current
        future_basin2mean, future_basin2std = future

        pickle.dump( current_basin2mean, open('%s_duration%d_current_basin2mean' % (prefix, event_duration.days) ,'w'))
        pickle.dump( current_basin2std, open('%s_duration%d_current_basin2std' % (prefix, event_duration.days) ,'w'))
        pickle.dump( future_basin2mean, open('%s_duration%d_future_basin2mean' % (prefix, event_duration.days) ,'w'))
        pickle.dump( future_basin2std, open('%s_duration%d_future_basin2std' % (prefix, event_duration.days) ,'w'))




   


    print 'Starting to plot'
    pylab.rcParams.update(params)
    
    plt.figure()
    i = 1

    plt.subplots_adjust(hspace = 0.4, wspace = 0.0)

    start_date = datetime(current_start_date.year, start_month_of_stamp_year, 1, 0,0,0)
    end_date = datetime(current_end_date.year, start_month_of_stamp_year,1,0,0,0)
    n_years = (end_date - start_date).days / 365


    print len(bin_dates)
    basins_sorted = current_basin2mean.keys()
    basins_sorted.sort(key = (lambda basin : basin.name)) # sort by name
    
    ax_prev = None
    for basin in basins_sorted:
        print basin.name, basin.get_number_of_cells()
        ax = plt.subplot(6,4,i,sharey = ax_prev)
        
        if ax_prev is None:
            ax_prev = ax

        n_cells_and_years = float(basin.get_number_of_cells() * n_years)
        mean_current = current_basin2mean[basin]
        mean_current /= n_cells_and_years
        std_current = current_basin2std[basin] / n_cells_and_years
        std_current = std_current / np.sqrt(len(members.current_ids))

        mean_future = future_basin2mean[basin] / n_cells_and_years
        std_future = future_basin2std[basin] / n_cells_and_years
        std_future = std_future / np.sqrt(len(members.future_ids))

        print mean_current.shape

 #       plt.hist(occs, bins, rwidth = 0.8, histtype='bar')

        
        b_current = plt.bar(bin_dates, -mean_current , width = bin_widths, linewidth = 0.5,
                                         label = 'current climate', yerr = std_current, color = 'y')
        b_future = plt.bar(bin_dates, mean_future , width = bin_widths, linewidth = 0.5,
                                         label = 'future climate', yerr = std_future, color = 'r')
        plt.xlim(day_start, day_end)
        #plt.yticks([0, 185, 370])

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: abs(x))
        )


        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        ky = 0.85
#        if basin.name == 'RDO' and not high_flow:
#            ky = 1.1

        plt.text( x_min + 0.8 * (x_max - x_min), y_min + ky * (y_max - y_min), basin.name)
        #plt.title(basin.name, {'fontsize': title_font_size})

        ticks_list = [-0.3, -0.15, 0, 0.15, 0.3]
        if i % 4 == 1:
            plt.yticks(ticks_list)
        else:
            for label in ax.get_yticklabels():
                label.set_visible(False)

        #ax = plt.gca()
        ax.xaxis.set_major_locator(
            mpl.dates.MonthLocator(bymonth=xrange(2,13,3))
        )
        ax.xaxis.set_major_formatter(
            mpl.dates.DateFormatter('%b')
        )
#        plt.savefig(
#            '%s_basins/%s_%d_days_%s.png' % ( prefix, prefix, duration_days , b.name))
        i += 1

    plt.figlegend([b_current[0],b_future[0]],['Current climate', 'Future climate'], loc = (0.5, 0.1))
    plt.figtext(0.05, 0.6, "NORMALIZED FREQUENCY", rotation = 90)
    plt.savefig('%s_%d_panel.pdf' % (prefix, event_duration.days))
    pass

def plot_high_low_occ_together(high_event_duration = timedelta(days = 1),
         low_event_duration = timedelta(days = 15),
         data_folder = 'data/streamflows/VplusF_newmask1',
         start_month_of_stamp_year = 1, stamp_year = 2000):


    bin_interval_dt = timedelta(days = 10)

    current_start_date = datetime(1970, 1, 1, 0, 0)
    current_end_date = datetime(1999, 12, 31, 0, 0)
    future_start_date = datetime(2041, 1, 1, 0, 0)
    future_end_date = datetime(2070, 12, 31, 0, 0)



    bin_dates = []

    day_start = datetime(stamp_year, start_month_of_stamp_year, 1)
    day_end = day_start + timedelta(days = 365 if calendar.isleap(stamp_year + 1) else 364)

    print day_end
    d = day_start
    bin_widths = []
    bar_width_on_plot = mdates.date2num(day_start + bin_interval_dt) - mdates.date2num(day_start)
    while d < day_end:
        bin_dates.append(d)
        bin_widths.append(min(bar_width_on_plot, mdates.date2num(day_end) - mdates.date2num(d)))
        d += bin_interval_dt



    print 'Calculating mean and standard deviations ...'


    prefix = 'high'
    if os.path.isfile('%s_duration%d_current_basin2mean' % (prefix, high_event_duration.days)):
        prefix = 'high'
        high_basin2mean = pickle.load(open('%s_duration%d_current_basin2mean' % (prefix, high_event_duration.days)  ))
        high_basin2std = pickle.load(open('%s_duration%d_current_basin2std' % (prefix, high_event_duration.days)))

        prefix = 'low'
        low_basin2mean = pickle.load(open('%s_duration%d_current_basin2mean' % (prefix, low_event_duration.days)))
        low_basin2std = pickle.load(open('%s_duration%d_current_basin2std' % (prefix, low_event_duration.days)))
    else:
        the_high, future = \
                calculate_basin_means_and_stds( bin_dates = bin_dates, high_flow = True,
                    current_start_date = current_start_date,
                    current_end_date = current_end_date,
                    future_start_date = future_start_date,
                    future_end_date = future_end_date,
                    stamp_year = stamp_year, start_month_of_stamp_year = start_month_of_stamp_year,
                    bin_interval_dt = bin_interval_dt, data_folder = data_folder,
                    event_duration =high_event_duration
                )

        the_low, future = \
                calculate_basin_means_and_stds( bin_dates = bin_dates, high_flow = False,
                    current_start_date = current_start_date,
                    current_end_date = current_end_date,
                    future_start_date = future_start_date,
                    future_end_date = future_end_date,
                    stamp_year = stamp_year, start_month_of_stamp_year = start_month_of_stamp_year,
                    bin_interval_dt = bin_interval_dt, data_folder = data_folder,
                    event_duration = low_event_duration
                )


        high_basin2mean, high_basin2std = the_high
        low_basin2mean, low_basin2std = the_low

        prefix = 'high'
        pickle.dump( high_basin2mean, open('%s_duration%d_current_basin2mean' % (prefix, high_event_duration.days) ,'w'))
        pickle.dump( high_basin2std, open('%s_duration%d_current_basin2std' % (prefix, high_event_duration.days) ,'w'))

        prefix = 'low'
        pickle.dump( low_basin2mean, open('%s_duration%d_current_basin2mean' % (prefix, low_event_duration.days) ,'w'))
        pickle.dump( low_basin2std, open('%s_duration%d_current_basin2std' % (prefix, low_event_duration.days) ,'w'))







    print 'Starting to plot'
    pylab.rcParams.update(params)
    
    plt.figure()
    i = 1
    plt.subplots_adjust(hspace = 0.5, wspace = 0)


    start_date = datetime(current_start_date.year, start_month_of_stamp_year, 1, 0,0,0)
    end_date = datetime(current_end_date.year, start_month_of_stamp_year,1,0,0,0)
    n_years = (end_date - start_date).days / 365


    print len(bin_dates)
    basins_sorted = high_basin2mean.keys()
    basins_sorted.sort(key = (lambda basin : basin.name)) # sort by name
    
    ax_prev = None
    for basin in basins_sorted:
        print basin.name, basin.get_number_of_cells()
        ax = plt.subplot(6,4,i, sharey = ax_prev)
        
        if ax_prev == None:
            ax_prev = ax
        
        n_cells_and_years = float(basin.get_number_of_cells() * n_years)
        mean_high = high_basin2mean[basin]
        mean_high /= n_cells_and_years
        std_high = high_basin2std[basin] / n_cells_and_years
        std_high = std_high / np.sqrt(len(members.current_ids))

        mean_low = low_basin2mean[basin] / n_cells_and_years
        std_low = low_basin2std[basin] / n_cells_and_years
        std_low = std_low / np.sqrt(len(members.current_ids))

        print mean_high.shape

 #       plt.hist(occs, bins, rwidth = 0.8, histtype='bar')


        high_b = plt.bar(bin_dates, mean_high , width = bin_widths, linewidth = 0.5,
                                         label = 'high', yerr = std_high, color = 'b')
        low_b = plt.bar(bin_dates, -mean_low , width = bin_widths, linewidth = 0.5,
                                         label = 'low', yerr = std_low, color = 'r')
        plt.xlim(day_start, day_end)
        

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: abs(x))
        )


        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        ky = 0.8
        

        plt.text( x_min + 0.8 * (x_max - x_min), y_min + ky * (y_max - y_min), basin.name)
        #plt.title(basin.name, {'fontsize': title_font_size})

        
        ticks_list = [-0.3, -0.15, 0, 0.15, 0.3]
        plt.yticks(ticks_list)
 
        if i % 4 != 1:
             for label in ax.get_yticklabels():
                label.set_visible(False)

        #ax = plt.gca()
        ax.xaxis.set_major_locator(
            mpl.dates.MonthLocator(bymonth=xrange(2,13,2))
        )
        ax.xaxis.set_major_formatter(
            mpl.dates.DateFormatter('%b')
        )
#        plt.savefig(
#            '%s_basins/%s_%d_days_%s.png' % ( prefix, prefix, duration_days , b.name))
        i += 1

    plt.figlegend([high_b[0],low_b[0]],['Maximum flow', 'Minimum flow'], 'upper center')
    plt.savefig('current_occurences_panel.pdf')
    pass


if __name__ == "__main__":
    data_folder = 'data/streamflows/hydrosheds_euler9'
#compare current and future occurences
    main(data_folder = data_folder, event_duration = timedelta(days = 15), prefix = 'low', high_flow = False)
    main(data_folder = data_folder, event_duration = timedelta(days = 1), prefix = 'high')

#compare high and low occurences
    plot_high_low_occ_together(high_event_duration = timedelta(days = 1),
                               low_event_duration = timedelta(days = 15),
                               data_folder = data_folder,
                               start_month_of_stamp_year = 1, stamp_year = 2000)

    print "Hello World"
