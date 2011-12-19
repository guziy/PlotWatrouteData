from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D, Bbox
#from mpl_toolkits.basemap import NetCDFFile
import os.path
import sys
__author__="huziy"
__date__ ="$3 dec. 2010 11:21:58$"

import numpy as np
import application_properties
print sys.path

from datetime import datetime, timedelta

from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt


import matplotlib.dates as mdates
import data.members as members
import os

from matplotlib.ticker import FuncFormatter, FormatStrFormatter, NullFormatter
from util import plot_utils

import calendar

import pickle
import readers.read_infocell as infocell
from matplotlib import gridspec

#Plot number of occurences of high flow or low flow

TIME_FORMAT = '%Y_%m_%d_%H_%M'




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
    

    assert 366 >= len(dates_to_occurrences) >= 365, 'len = {0}'.format(len(dates_to_occurrences))
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

    assert 366 >= len(dates_to_occurrences) >= 365
    return dates_to_occurrences


def get_days_of_stamp_year(year = 2000, start_month_of_stamp_year = 1):
    dt = timedelta(days = 1)
    d = datetime(year, start_month_of_stamp_year, 1,0,0)
    n_days = 366
    return (d + i * dt for i in range(n_days))



def calculate_occurences_for_member(nc=None, bin_dates=None, high_flow=True, basin_indices=None,
                                    start_date=datetime(1970, 1, 1, 0, 0), end_date=datetime(1999, 12, 31, 0, 0),
                                    dates_of_stamp_year=None, i_indices=None, j_indices=None,
                                    event_duration=timedelta(days=7), bin_interval_dt=timedelta(days=10), times=None,
                                    stamp_year=2000, start_month_of_stamp_year=1):

    if not times: times = []
    if not j_indices: j_indices = []
    if not i_indices: i_indices = []
    if not dates_of_stamp_year: dates_of_stamp_year = []
    if not bin_dates: bin_dates = []
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


def calculate_mean_occurences_and_std(member2basin_and_occ = None):
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



def calculate_basin_means_and_stds(bin_dates=None, current_start_date=datetime(1970, 1, 1, 0, 0),
                                   current_end_date=datetime(1999, 12, 31, 0, 0),
                                   future_start_date=datetime(2041, 1, 1, 0, 0),
                                   future_end_date=datetime(2070, 12, 31, 0, 0), stamp_year=2000,
                                   start_month_of_stamp_year=7, bin_interval_dt=timedelta(days=10), data_folder='',
                                   event_duration=timedelta(days=7), high_flow=True):


    if not bin_dates: bin_dates = []
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
         prefix = '', data_folder = '',
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




#    plot_scatter(current_basin2mean = current_basin2mean, future_basin2mean = future_basin2mean,
#        bin_dates=bin_dates, prefix=prefix, start_date=day_start, end_date=day_end)

    plot_scatter_merge_for_months(current_basin2mean = current_basin2mean, future_basin2mean = future_basin2mean,
        bin_dates=bin_dates, prefix=prefix, start_date=day_start, end_date=day_end)

#    plot_changes_for_month_merged(current_basin2mean = current_basin2mean, future_basin2mean = future_basin2mean,
#        bin_dates=bin_dates, prefix=prefix, start_date=day_start, end_date=day_end)

    print 'Starting to plot'
#    plot_occ_old(bin_dates = bin_dates,  bin_widths=bin_widths,
#            current_start_date = current_start_date, current_end_date = current_end_date,
#            start_month_of_stamp_year = start_month_of_stamp_year,
#            current_basin2mean = current_basin2mean,
#            future_basin2mean = future_basin2mean,
#            current_basin2std = current_basin2std,  future_basin2std = future_basin2std,
#            event_duration = event_duration,
#            day_start = day_start,
#            day_end = day_end, prefix = prefix
#    )
    pass




basin_name2mark = {
    "ARN": "1",
    "FEU": "2",
    "MEL": "3",
    "CAN": "4",
    "PYR": "5",
    "GRB": "6",
    "BAL": "7",
    "GEO": "8",
    "CHU": "9",
    "LGR": "10",
    "NAT": "11",
    "ROM": "12",
    "MOI": "13",
    "MAN": "14",
    "RUP": "15",
    "BEL": "16",
    "STM": "17",
    "RDO": "18",
    "SAG": "19",
    "BOM": "20",
    "WAS": "21"
}

southern = ["RDO", "BEL", "STM", "WAS", "SAG", "BOM", "RUP"]
northern = ["ARN", "FEU", "MEL", "PYR", "GRB", "BAL", "GEO"]
central =  ["CAN", "CHU", "LGR", "NAT", "MAN", "MOI", "ROM"]


def plot_changes_for_month_merged(current_basin2mean = None,
                  future_basin2mean = None,
                  bin_dates = None, prefix = "",
                  start_date = None, end_date = None):
    n_years = 30.0
    font_props = FontProperties(weight="bold", size = 7)

    basins = current_basin2mean.keys()

    basin_groups = [[], [], []]
    #group_titles = [" Northern"]

    for b in basins:
        if b.name in northern:
            basin_groups[0].append(b)
        elif b.name in central:
            basin_groups[1].append(b)
        elif b.name in southern:
            basin_groups[2].append(b)

    plot_utils.apply_plot_params(width_pt=None, font_size=9, height_cm=60, width_cm=16)
    fig = plt.figure()
    gs  = gridspec.GridSpec(3,1)

    for i, basin_group in enumerate( basin_groups):

        host = fig.add_subplot(gs[i, 0])
        current_color = "k"


        for basin in basin_group:
            n_cells_and_years = float(basin.get_number_of_cells() * n_years)
            c_data = current_basin2mean[basin]
            f_data = future_basin2mean[basin]
    #        p_left = host.scatter(bin_dates, c_data / n_cells_and_years, marker="$%s$" % basin_name2mark[basin.name],
    #            color = current_color, linewidths = 0., s = 60
    #        )


            monthly_vals_current = [0 for i in xrange(12)]
            monthly_vals_future = [0 for i in xrange(12)]
            monthly_dates = [datetime(start_date.year, month, 15) for month in xrange(1,13)]

            for date, occ_cur, occ_fut in zip(bin_dates, c_data / n_cells_and_years, f_data / n_cells_and_years):
                monthly_vals_current[date.month - 1] += occ_cur
                monthly_vals_future[date.month - 1] += occ_fut

            change = np.array(monthly_vals_future) - np.array(monthly_vals_current)

            for date, occ in zip(monthly_dates, change):
                if np.abs(occ) < 0.01: continue #do not show the values <= 0.01
                host.annotate(basin_name2mark[basin.name], xy = (date, occ), font_properties = font_props,
                    color = current_color, ha = "right", zorder = 2)
            host.plot(monthly_dates, change, color = current_color,lw = 0.1)

    #        p_right = guest.scatter(bin_dates, f_data / n_cells_and_years, marker= "$%s$" % basin_name2mark[basin.name],
    #            color = future_color, linewidths = 0., s = 60)


        #position ticklabels between the ticks

        date_ticks = []
        current_date = datetime(start_date.year, 1,15)
        for months in xrange(1, 13):
            date_ticks.append(datetime(current_date.year, current_date.month, 1))
            date_ticks.append(current_date )

            if current_date.month + 1 == 13:
                month = 1
            else:
                month = current_date.month + 1
            current_date = datetime( current_date.year, month, current_date.day )

        host.set_ylim(-0.5, 0.5)
        host.xaxis.set_ticks(date_ticks)
        tls = host.xaxis.get_majorticklabels()
        major_ticks = host.xaxis.get_major_ticks()


        for i, mtl in enumerate(major_ticks):
            mtl.tick1line.set_visible(i % 2 == 0)
            mtl.tick2line.set_visible(i % 2 == 0)
            mtl.label1On = (i % 2 != 0)

        host.set_xlim(start_date, end_date)

            #    host.xaxis.set_major_locator(
            #        mpl.dates.MonthLocator(bymonth=xrange(2,13,3))
            #    )
        host.xaxis.set_major_formatter(
                        mpl.dates.DateFormatter('%b')
        )
    #plt.tight_layout()
    plt.savefig(prefix + "_occurrence_scatter_for_months_changes.png")


def plot_scatter_merge_for_months(current_basin2mean = None,
                  future_basin2mean = None,
                  bin_dates = None, prefix = "",
                  start_date = None, end_date = None):

    n_years = 30.0
    font_props = FontProperties(weight="normal", size = 6)

    basins = current_basin2mean.keys()

    basin_groups = [[], [], []]
    #group_titles = [" Northern"]

    tranformation = Affine2D()
    tranformation.translate(0.45, 0.04)

    for b in basins:
        if b.name in northern:
            basin_groups[0].append(b)
        elif b.name in central:
            basin_groups[1].append(b)
        elif b.name in southern:
            basin_groups[2].append(b)

    plot_utils.apply_plot_params(width_pt=None, font_size=9, height_cm=20, width_cm=16)
    fig = plt.figure()
    gs  = gridspec.GridSpec(3,1)

    main_subplots = []
    for i in xrange(len(basin_groups)):
        main_subplots.append(fig.add_subplot(gs[i, 0]))


    for i, basin_group in enumerate( basin_groups):

        host = main_subplots[i]
        guest = host.twinx()
        #invert axis
        guest.invert_yaxis()

        current_color = "b"
        future_color = "r"

        #host.yaxis.label.set_color(current_color)
        #host.set_ylabel("Current")

        #guest.yaxis.label.set_color(future_color)
        #guest.set_ylabel("Future")
        if i == 1:
            host.set_ylabel("Normalized frequency ( current climate )")
            guest.set_ylabel("Normalized frequency ( future climate )")


        guest.set_ylim(2, 0)
        host.set_ylim(0, 2)

        #tkw = dict(size=4, width=1.5)
        for ti in host.get_yticklabels():
            ti.set_color(current_color)

        for ti in guest.get_yticklabels():
            ti.set_color(future_color)

        #create inset axes
        inset_box = host.get_position()
        inset_box = inset_box.transformed(tranformation)
        inset_box = inset_box.shrunk(0.4, 0.6)
        axins = fig.add_axes(inset_box)


        for basin in basin_group:

            n_cells_and_years = float(basin.get_number_of_cells() * n_years)
            c_data = current_basin2mean[basin]
            f_data = future_basin2mean[basin]
    #        p_left = host.scatter(bin_dates, c_data / n_cells_and_years, marker="$%s$" % basin_name2mark[basin.name],
    #            color = current_color, linewidths = 0., s = 60
    #        )


            monthly_vals_current = [0 for i in xrange(12)]
            monthly_vals_future = [0 for i in xrange(12)]
            monthly_dates = [datetime(start_date.year, month, 15) for month in xrange(1,13)]

            for date, occ_cur, occ_fut in zip(bin_dates, c_data / n_cells_and_years, f_data / n_cells_and_years):
                monthly_vals_current[date.month - 1] += occ_cur
                monthly_vals_future[date.month - 1] += occ_fut



            for date, occ in zip(monthly_dates, monthly_vals_current):
                if occ < 0.01: continue #do not show the values <= 0.01
                host.annotate(basin_name2mark[basin.name], xy = (date, occ), font_properties = font_props,
                    color = current_color, ha = "right")
            host.plot(monthly_dates, monthly_vals_current, color = current_color,lw = 0.2)

    #        p_right = guest.scatter(bin_dates, f_data / n_cells_and_years, marker= "$%s$" % basin_name2mark[basin.name],
    #            color = future_color, linewidths = 0., s = 60)

            for date, occ in zip(monthly_dates, monthly_vals_future):
                if occ < 0.01: continue
                guest.annotate(basin_name2mark[basin.name], xy = (date, occ), font_properties = font_props,
                    va = "top", color = future_color, ha = "right")

            guest.plot(monthly_dates, monthly_vals_future, color = future_color, lw = 0.2)

            #plot changes as inset
            changes = np.array( monthly_vals_future ) - np.array(monthly_vals_current)
            for date, the_change in zip(monthly_dates, changes):
                if the_change < 0.01: continue
                axins.annotate(basin_name2mark[basin.name], xy = (date, the_change), font_properties = font_props,
                    va = "top", color = "k", ha = "right")
            axins.plot(monthly_dates, changes, color = "k", lw = 0.2)






        #position ticklabels between the ticks

        date_ticks = []
        current_date = datetime(start_date.year, 1,15)
        for months in xrange(1, 13):
            date_ticks.append(datetime(current_date.year, current_date.month, 1))
            date_ticks.append(current_date )

            if current_date.month + 1 == 13:
                month = 1
            else:
                month = current_date.month + 1
            current_date = datetime( current_date.year, month, current_date.day )

        #leave only month names and center the labels
        host.xaxis.set_ticks(date_ticks)
        major_ticks = host.xaxis.get_major_ticks()
        for i, mtl in enumerate(major_ticks):
            mtl.tick1line.set_visible(i % 2 == 0)
            mtl.tick2line.set_visible(i % 2 == 0)
            mtl.label1On = (i % 2 != 0)

        host.set_xlim(start_date, end_date)
        host.xaxis.set_major_formatter(
                        mpl.dates.DateFormatter('%b')
        )

        #leave only month names and center the labels
        axins.set_xlim(start_date, end_date)
        axins.xaxis.set_major_formatter(
                        mpl.dates.DateFormatter('%b')
        )


        axins.xaxis.set_ticks(date_ticks)
        axins.tick_params(labelsize = 6)
        major_ticks = axins.xaxis.get_major_ticks()
        for i, mtl in enumerate(major_ticks):
            mtl.tick1line.set_visible(i % 2 == 0)
            mtl.tick2line.set_visible(i % 2 == 0)
            mtl.label1On = (i % 4 == 1)


    #plt.tight_layout()
    plt.savefig(prefix + "_occurrence_scatter_for_months.png")

    pass


def plot_scatter( current_basin2mean = None,
                  future_basin2mean = None,
                  bin_dates = None, prefix = "",
                  start_date = None, end_date = None
                  ):


    n_years = 30.0
    font_props = FontProperties(weight="bold", size = 6)

    basins = current_basin2mean.keys()

    basin_groups = [[], [], []]
    #group_titles = [" Northern"]

    for b in basins:
        if b.name in northern:
            basin_groups[0].append(b)
        elif b.name in central:
            basin_groups[1].append(b)
        elif b.name in southern:
            basin_groups[2].append(b)

    plot_utils.apply_plot_params(width_pt=None, font_size=9, height_cm=20, width_cm=16)
    fig = plt.figure()
    gs  = gridspec.GridSpec(3,1)

    for i, basin_group in enumerate( basin_groups):

        host = fig.add_subplot(gs[i, 0])
        guest = host.twinx()
        #invert axis
        guest.invert_yaxis()

        current_color = "b"
        future_color = "r"

        host.yaxis.label.set_color(current_color)
        host.set_ylabel("Current")

        guest.yaxis.label.set_color(future_color)
        guest.set_ylabel("Future")

        guest.set_ylim(1, 0)
        host.set_ylim(0, 1)

        #tkw = dict(size=4, width=1.5)
        for ti in host.get_yticklabels():
            ti.set_color(current_color)

        for ti in guest.get_yticklabels():
            ti.set_color(future_color)




        for basin in basin_group:
            n_cells_and_years = float(basin.get_number_of_cells() * n_years)
            c_data = current_basin2mean[basin]
            f_data = future_basin2mean[basin]
    #        p_left = host.scatter(bin_dates, c_data / n_cells_and_years, marker="$%s$" % basin_name2mark[basin.name],
    #            color = current_color, linewidths = 0., s = 60
    #        )



            for date, occ in zip(bin_dates, c_data / n_cells_and_years):
                if occ < 0.01: continue
                host.annotate(basin_name2mark[basin.name], xy = (date, occ), font_properties = font_props,
                    color = current_color, ha = "right")
            host.plot(bin_dates, c_data / n_cells_and_years, color = current_color,lw = 0.2)

    #        p_right = guest.scatter(bin_dates, f_data / n_cells_and_years, marker= "$%s$" % basin_name2mark[basin.name],
    #            color = future_color, linewidths = 0., s = 60)

            for date, occ in zip(bin_dates, f_data / n_cells_and_years):
                if occ < 0.01: continue
                guest.annotate(basin_name2mark[basin.name], xy = (date, occ), font_properties = font_props,
                    va = "top", color = future_color, ha = "right")

            guest.plot(bin_dates, f_data / n_cells_and_years, color = future_color, lw = 0.2)

        #position ticklabels between the ticks

        date_ticks = []
        current_date = datetime(start_date.year, 1,15)
        for months in xrange(1, 13):
            date_ticks.append(datetime(current_date.year, current_date.month, 1))
            date_ticks.append(current_date )

            if current_date.month + 1 == 13:
                month = 1
            else:
                month = current_date.month + 1
            current_date = datetime( current_date.year, month, current_date.day )


        host.xaxis.set_ticks(date_ticks)
        tls = host.xaxis.get_majorticklabels()
        major_ticks = host.xaxis.get_major_ticks()

        for i, tl in enumerate(tls):
            tl.set_visible(i % 2 != 0 )

        for i, mtl in enumerate(major_ticks):
            mtl.tick1line.set_visible(i % 2 == 0)
            mtl.tick2line.set_visible(i % 2 == 0)

        host.set_xlim(start_date, end_date)

            #    host.xaxis.set_major_locator(
            #        mpl.dates.MonthLocator(bymonth=xrange(2,13,3))
            #    )
        host.xaxis.set_major_formatter(
                        mpl.dates.DateFormatter('%b')
        )
    #plt.tight_layout()
    plt.savefig(prefix + "_occurrence_scatter.png")



    pass


def plot_occ_old(bin_dates = None, bin_widths = None, current_start_date = None, current_end_date = None,
                 start_month_of_stamp_year = 1, current_basin2mean = None,
                 future_basin2mean = None,
                 current_basin2std = None,  future_basin2std = None,
                 event_duration = None,
                 day_start = None, day_end = None, prefix = ""
                 ):
    plt.figure()
    i = 1

    #plt.subplots_adjust(hspace = 0.4, wspace = 0.0)

    start_date = datetime(current_start_date.year, start_month_of_stamp_year, 1, 0,0,0)
    end_date = datetime(current_end_date.year, start_month_of_stamp_year,1,0,0,0) # not including
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
            FuncFormatter(lambda x, pos: "%.2f" % abs(x))
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
    plt.figtext(0.05, 0.6, "Normalized frequency", rotation = 90)
    plt.savefig('%s_%d_panel.png' % (prefix, event_duration.days))




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
    plt.savefig('current_occurences_panel.png')
    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    data_folder = 'data/streamflows/hydrosheds_euler9'
#compare current and future occurences
    main(data_folder = data_folder, event_duration = timedelta(days = 15), prefix = 'low', high_flow = False)
    main(data_folder = data_folder, event_duration = timedelta(days = 1), prefix = 'high')

#compare high and low occurences
#    plot_high_low_occ_together(high_event_duration = timedelta(days = 1),
#                               low_event_duration = timedelta(days = 15),
#                               data_folder = data_folder,
#                               start_month_of_stamp_year = 1, stamp_year = 2000)

    print "Hello World"
