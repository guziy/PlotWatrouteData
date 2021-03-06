from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from shape.basin_boundaries import plot_basin_boundaries_from_shape

__author__="huziy"
__date__ ="$8 dec. 2010 10:20:08$"

from plot2D.index_object import IndexObject
from data.modelpoint import ModelPoint
from data.cehq_station import Station
import os.path

import application_properties
#from readers.read_infocell import plot_basin_boundaries_from_shape
import os
import sys
from math import sqrt
import numpy as np
import pickle


from util.geo.lat_lon import get_distance_in_meters
from plot2D.map_parameters import polar_stereographic
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pylab
import matplotlib as mpl
from matplotlib import gridspec

import data.data_select as data_select
import util.plot_utils as plot_utils

from index_object import IndexObject
import netCDF4 as nc

from data.cell import Cell
from data import members, direction_and_value

import diagnose_ccc.compare_precip as compare_precip

MAXIMUM_DISTANCE_METERS = 45000.0 #m

TIME_FORMAT = '%Y_%m_%d_%H_%M'

lons = polar_stereographic.lons
lats = polar_stereographic.lats
xs = polar_stereographic.xs
ys = polar_stereographic.ys






#needed for calculation of mean of data for each day
def create_dates_of_year(date_list = None, year=2000):
    if not date_list: date_list = []
    result = []

    for d1 in date_list:
        try:
            result.append(datetime(year, d1.month, d1.day, d1.hour, d1.minute, d1.second))
        except ValueError:
            print 'warning in create_dates_of_year: day is out of range for month'
    return result





def objective_function(distance, da1, da2):
    """
    objective function that should be minimal
    at the station corresponding to the given cell
    """
    alpha = 1.0
    return distance / MAXIMUM_DISTANCE_METERS + np.abs(da2 - da1) / da1 * alpha


def get_neighbour_cell_with_closest_da(station):
    """
    returns a neighbour of station with the nearest drainage area
    """
    i, j = polar_stereographic.get_indices_of_the_closest_point_to(station.longitude, station.latitude)
    #TODO: finish this method
    pass


def get_corresponding_station(lon, lat, cell_drain_area_km2, station_list):
    """
    returns the closest station to (lon, lat),
    or None if it was not found
    """
    #find distance to the closest station
    objective = None
    result = None
    for station in station_list:
        station_drainage = station.drainage_km2
        distance = get_distance_in_meters(lon, lat, station.longitude, station.latitude)
        objective1 = objective_function(distance, station_drainage, cell_drain_area_km2)
        
        if objective is None:
            objective = objective1
            result = station
        else:
            if objective1 < objective:
                result = station
                objective = objective1
        
    
    if objective <= 1.0:
        return result
    else:
        return None




def read_station_data(folder = 'data/cehq_measure_data'):
    stations = []
    for file in os.listdir(folder):
        if not '.txt' in file:
            continue
        path = os.path.join(folder, file)
        s = Station()
        s.parse_from_cehq(path)
        stations.append(s)
    return stations



def average_for_each_day_of_year(times, data, start_date = None, end_date = None, year = 2000):
    """
    TODO: documentation
    """
    values = {}
    counts = {}

    assert len(times) == len(data)

    dates = create_dates_of_year(times, year = year)
    result_values = []
    result_dates = []

    for date, time, the_value in zip(dates, times, data):

        if start_date is not None and end_date is not None:
            if time < start_date:
                continue
            if time > end_date:
                break

        if values.has_key(date):
            values[date] += the_value
            counts[date] += 1
        else:
            values[date] = the_value
            counts[date] = 1


    dt = timedelta(days = 1)
    d = datetime(year, 1, 1, 0, 0)
    while d.year == year:
        if values.has_key(d):
            values[d] /= float(counts[d])
            result_values.append(values[d])
            result_dates.append(d)
        d += dt

    assert len(result_dates) == len(result_values)
    return result_dates, result_values
        
 
def calculate_skills(selected_stations=None, dates=None, selected_station_values=None, selected_model_values=None,
                     grid_drainages=None, grid_lons=None, grid_lats=None):



    if not grid_drainages: grid_drainages = []
    if not selected_model_values: selected_model_values = []
    if not selected_station_values: selected_station_values = []
    if not grid_lats: grid_lats = []
    if not grid_lons: grid_lons = []
    if not selected_stations: selected_stations = []
    for i in range(len(selected_stations)):
         station = selected_stations[i]

         model_values = np.array(selected_model_values[i])
         station_values = np.array(selected_station_values[i])

         grid_drainage  = grid_drainages[i]
         grid_lon = grid_lons[i]
         grid_lat = grid_lats[i]

#         values_without_gw = selected_values_without_gw[i]


         #calculate skill coefficient
         std1 = np.std(model_values)
         std2 = np.std(station_values)
         R = np.corrcoef(model_values, station_values)
         S = 2 * std1 * std2 / (std1 ** 2 + std2 ** 2) * R[0,1]


         #Calculate Nash-Sutcliff coefficient
         ns = 1.0 - np.sum((model_values - station_values) ** 2) / np.sum((station_values - np.mean(station_values)) ** 2)
         print("Nash-Sutcl. = {0}".format(ns))



         print 'Station:'
         print 'Id,Lon, Lat, DA: %s, %f, %f, %f' % ( station.id, station.longitude, station.latitude, station.drainage_km2 )
#         print 'Grid point'
#         print 'Lon, Lat, DA, Skill, dDA/DA : %f, %f, %f, %f, %f' % (grid_lon, grid_lat, grid_drainage,
#                                    S, (grid_drainage - station.drainage_km2) / station.drainage_km2 )

         print( 20 * "-" )

##returns a dictionary {station: modelpoint}
def get_station_and_corresponding_model_data(path = 'data/streamflows/hydrosheds_euler10_spinup100yrs/aex_discharge_1970_01_01_00_00.nc'):
    result = {}
    saved_selected_stations_file = 'selected_stations_and_model_data.bin'
    if os.path.isfile(saved_selected_stations_file):
        result = pickle.load(open(saved_selected_stations_file))
    else:
        print 'getting data from file ', path


        [data, times, i_list, j_list] = data_select.get_data_from_file(path)
        drainage_area = data_select.get_field_from_file(path, field_name = 'accumulation_area')

        if drainage_area is not None:
            lons = data_select.get_field_from_file(path, field_name = 'longitude')
            lats = data_select.get_field_from_file(path, field_name = 'latitude')
            da_2d = drainage_area
        else:
            drainage_area = data_select.get_field_from_file(path, field_name = 'drainage')
            da_2d = np.zeros(polar_stereographic.xs.shape)
            lons = polar_stereographic.lons
            lats = polar_stereographic.lats
            for index, i, j in zip( range(len(i_list)) , i_list, j_list):
                da_2d[i, j] = drainage_area[index]




        stations_dump = 'stations_dump.bin'
        if os.path.isfile(stations_dump):
            print 'unpickling'
            stations = pickle.load(open(stations_dump))
        else:
            stations = read_station_data()
            pickle.dump(stations, open(stations_dump, 'w'))

        reload(sys)
        sys.setdefaultencoding('iso-8859-1')

        selected_stations = []
        for index, i, j in zip( range(len(i_list)) , i_list, j_list):
            station = get_corresponding_station(lons[i, j], lats[i, j], da_2d[i, j], stations)
            if station is None or station in selected_stations:
                continue
            selected_stations.append(station)
            data_point = ModelPoint(times, data[:, index])
            result[station] = data_point

            print '=' * 20
            print station.get_timeseries_length() , station.id
            #found station plot data
            print station.name
            print station.id

        pickle.dump(result, open(saved_selected_stations_file,'wb'))

#    for station, point in result.iteritems():
#        plt.plot(station.dates, station.values, label = station.name)
#    plt.legend()
#    plt.show()
    assert len(result) > 0
    return result



def get_connected_cells(directions_path):
    """
    returns a 2d array of connected cells,
    caution: works with indices, so be careful whenever there is
    a subsetting
    """
    ds = nc.Dataset(directions_path)
    next_i = ds.variables['flow_direction_index0'][:]
    next_j = ds.variables['flow_direction_index1'][:]
    ds.close()


    [nx, ny] = next_i.shape

    cells = [[Cell(ix = i, jy = j) for j in xrange(ny)]
                for i in xrange(nx)]
    
    #connect cells
    for i in xrange(nx):
        for j in xrange(ny):
            iNext = next_i[i, j]
            jNext = next_j[i, j]
            theCell = cells[i][j]
            # @type theCell Cell
            if iNext >= 0 <= jNext:
                theCell.set_next(cells[iNext][jNext])

    return cells





def get_mask_for_station(directions_file = 'data/hydrosheds/directions_for_streamflow9.nc',
                          i_index = -1, j_index = -1):


    import diagnose_ccc.compare_swe as compare_swe

    cells = get_connected_cells(directions_file)

    nx = len(cells)
    ny = len(cells[0])

    the_mask = np.zeros((nx, ny))

    theCell = cells[i_index][j_index]
    the_mask[theCell.coords()] = 1 #of course include the current cell

    # @type theCell Cell
    upstream_cells = theCell.get_upstream_cells()
    for c in upstream_cells:
        the_mask[c.coords()] = 1

    domain_mask = compare_swe.get_domain_mask()

    the_mask *= domain_mask
    return the_mask




def plot_total_precip_for_upstream(directions_file = 'data/hydrosheds/directions_for_streamflow9.nc',
                          i_index = -1, j_index = -1, station_id = '', subplot_count = -1,
                          start_date = None, end_date = None):

    """
    Plot total precipitation integrated over upstream cells for model(ccc file) and observation(gpcc)
    """
    the_mask = get_mask_for_station(directions_file=directions_file, i_index=i_index, j_index=j_index)

    compare_precip.compare_precip(mask = the_mask, label = station_id,
                                  subplot_count=subplot_count,
                                  start_date = start_date, end_date=end_date
                                  )
    pass

def plot_swe_for_upstream(directions_file = 'data/hydrosheds/directions_for_streamflow9.nc',
                          i_index = -1, j_index = -1, station_id = ''):
    """
    plot sum of swe over all upstream cells for a given station, for model and observation
    data
    """

#    compare_swe.compare_daily_normals_mean_over_mask(the_mask,
#                                        start = datetime(1980,01,01,00),
#                                        end = datetime(1996, 12, 31,00),
#                                        label = station_id)
    import diagnose_ccc.compare_swe as compare_swe
    the_mask = get_mask_for_station(directions_file=directions_file, i_index=i_index, j_index=j_index)

    compare_swe.compare_daily_normals_integral_over_mask(the_mask,
                                        start = datetime(1980,01,01,00),
                                        end = datetime(1996, 12, 31,00),
                                        label = station_id)



    pass


def plot_precip_for_upstream(i_index, j_index, station_id):

    """
    plot precipitation timeseries averaged over
    upstream cells
    """
    pass

selected_station_ids = [
    "104001", "103715", "093806", "093801", "092715", "081006", "061502", #"080718",
    "040830"
]


def _get_monthly_means(dates, values):
    res = np.zeros((12,))
    for m in xrange(1,13):
        bool_vector = np.array(map(lambda x: x.month == m, dates))
        res[m - 1] = np.mean(values[bool_vector])
    return res


def get_unrouted_streamflow_for(selected_dates = None, all_dates = None , tot_runoff = None, cell_indices = None):

    """
    tot_runoff in m^3/s
    """
    bool_vector = np.array( map( lambda x: x in selected_dates, all_dates) )
    r_time_slice = tot_runoff[bool_vector, :]
    r_tx_slice = np.sum( r_time_slice[:, cell_indices], axis= 1)
    return r_tx_slice



def main():
    """

    """
    skip_ids = ['081007', '081002', "042607", "090605"]

    #comment to plot for all ensemble members
    members.current_ids = []


    #pylab.rcParams.update(params)
    path_format = 'data/streamflows/hydrosheds_euler9/%s_discharge_1970_01_01_00_00.nc'
    #path_format = "data/streamflows/hydrosheds_rk4_changed_partiotioning/%s_discharge_1970_01_01_00_00.nc"
    #path_format = "data/streamflows/piloted_by_ecmwf/ecmwf_nearest_neighbor_discharge_1970_01_01_00_00.nc"
    path_to_analysis_driven = path_format % members.control_id

    simIdToData = {}
    simIdToTimes = {}
    for the_id in members.current_ids:
        thePath = path_format % the_id
        [simIdToData[the_id], simIdToTimes[the_id], i_list, j_list] = data_select.get_data_from_file(thePath)


    old = True #in the old version drainage and lon,lats in the file are 1D


    [ data, times, i_list, j_list ] = data_select.get_data_from_file(path_to_analysis_driven)

    cell_list = []
    ij_to_cell = {}
    prev_cell_indices = []
    tot_rof = None
    if old:
        #surf_rof = data_select.get_data_from_file(path_format % ("aex",), field_name="")
        the_path = path_format % ("aex")
        static_data_path = "data/streamflows/hydrosheds_euler9/infocell9.nc"
        #ntimes x ncells
        tot_rof = data_select.get_field_from_file(the_path, field_name="total_runoff")
        cell_areas = data_select.get_field_from_file(static_data_path, field_name="AREA")

        #convert the runoff to m^3/s
        tot_rof *= 1.0e6 * cell_areas[i_list, j_list] / 1.0e3


        flow_dir_values = data_select.get_field_from_file(static_data_path,
            field_name="flow_direction_value")[i_list, j_list]

        cell_list = map(lambda i, j, the_id: Cell(id = the_id, ix = i, jy = j),
                                i_list, j_list, xrange(len(i_list)))


        ij_to_cell = dict( zip( zip(i_list, j_list), cell_list ))


        for ix, jy, aCell, dir_val in zip( i_list, j_list, cell_list, flow_dir_values):
            i_next, j_next = direction_and_value.to_indices(ix, jy, dir_val)
            the_key = (i_next, j_next)
            if ij_to_cell.has_key(the_key):
                next_cell = ij_to_cell[the_key]
            else:
                next_cell = None
            assert isinstance(aCell, Cell)
            aCell.set_next(next_cell)

        #determine list of indices of the previous cells for each cell
        #in this case they are equal to the ids

        for aCell in cell_list:
            assert isinstance(aCell, Cell)
            prev_cells = aCell.get_upstream_cells()
            prev_cell_indices.append(map(lambda c: c.id, prev_cells))
            prev_cell_indices[-1].append(aCell.id)



    if not old:
        da_2d = data_select.get_field_from_file(path_to_analysis_driven, 'accumulation_area')
        lons = data_select.get_field_from_file(path_to_analysis_driven, field_name = 'longitude')
        lats = data_select.get_field_from_file(path_to_analysis_driven, field_name = 'latitude')
    else:
        lons = polar_stereographic.lons
        lats = polar_stereographic.lats
        da_2d = np.zeros(lons.shape)
        drainage = data_select.get_field_from_file(path_to_analysis_driven, 'drainage')
        for i, j, theDa in zip(i_list, j_list, drainage):
            da_2d[i, j] = theDa




    data_step = timedelta(days = 1)


    stations_dump = 'stations_dump.bin'
    if os.path.isfile(stations_dump):
        print 'unpickling'
        stations = pickle.load(open(stations_dump))
    else:
        stations = read_station_data()
        pickle.dump(stations, open(stations_dump, 'w'))

#   Did this to solve text encoding issues
#    reload(sys)
#    sys.setdefaultencoding('iso-8859-1')


    selected_stations = []
    selected_model_values = []
    selected_station_values = []

    grid_drainages = []
    grid_lons = []
    grid_lats = []
    plot_utils.apply_plot_params(width_pt= None, font_size=9, aspect_ratio=2.5)
    #plot_utils.apply_plot_params(font_size=9, width_pt=None)
    ncols = 2
    gs = gridspec.GridSpec(5, ncols)
    fig = plt.figure()

    assert isinstance(fig, Figure)

    current_subplot = 0

    label1 = "modelled"
    label2 = "observed"
    line1 = None
    line2 = None
    lines_for_mems = None
    labels_for_mems = None
    #fig.subplots_adjust(hspace = 0.9, wspace = 0.4, top = 0.9)




    index_objects = []
    for index, i, j in zip( range(len(i_list)) , i_list, j_list):
        index_objects.append(IndexObject(positionIndex = index, i = i, j = j))

    #sort by latitude
    index_objects.sort( key = lambda x: x.j, reverse = True)

    #simulation id to continuous data map
    simIdToContData = {}
    for the_id in members.all_current:
        simIdToContData[the_id] = {}

    for indexObj in index_objects:
        i = indexObj.i
        j = indexObj.j
        # @type indexObj IndexObject
        index = indexObj.positionIndex
        station = get_corresponding_station(lons[i, j], lats[i, j], da_2d[i, j], stations)


        if station is None or station in selected_stations:
            continue

        #if you want to compare with stations add their ids to the selected
        if station.id not in selected_station_ids:
            continue


        #skip some stations
        if station.id in skip_ids:
            continue


        #try now to find the point with the closest drainage area
#        current_diff = np.abs(station.drainage_km2 - da_2d[i, j])
#        for di in xrange(-1,2):
#            for dj in xrange(-1,2):
#                the_diff = np.abs(station.drainage_km2 - da_2d[i + di, j + dj])
#                if the_diff < current_diff: #select different grid point
#                    current_diff = the_diff
#                    i = i + di
#                    j = j + dj
#                    indexObj.i = i
#                    indexObj.j = j




        #found station plot data
        print station.name


        start_date = max( np.min(times), np.min(station.dates))
        end_date = min( np.max(times),  np.max(station.dates))

        if start_date.day > 1 or start_date.month > 1:
            start_date = datetime(start_date.year + 1, 1, 1,0,0,0)

        if end_date.day < 31 or end_date.month < 12:
            end_date = datetime(end_date.year - 1, 12, 31,0,0,0)



        if end_date < start_date:
            continue


        #select data for years that do not have gaps
        start_year = start_date.year
        end_year = end_date.year
        continuous_station_data = {}
        continuous_model_data = {}
        num_of_continuous_years = 0
        for year in xrange(start_year, end_year + 1):
            # @type station Station
            station_data = station.get_continuous_dataseries_for_year(year)
            if len(station_data) >= 365:
                num_of_continuous_years += 1

                #save station data
                for d, v in station_data.iteritems():
                    continuous_station_data[d] = v

                #save model data
                for t_index, t in enumerate(times):
                    if t.year > year: break
                    if t.year < year: continue
                    continuous_model_data[t] = data[t_index, index]
                #fill the map sim id to cont model data
                for the_id in members.current_ids:
                    #save model data
                    for t_index, t in enumerate(simIdToTimes[the_id]):
                        if t.year > year: break
                        if t.year < year: continue
                        simIdToContData[the_id][t] = simIdToData[the_id][t_index, index]


        #if the length of continuous observation is less than 10 years, skip
        if len(continuous_station_data) < 3650: continue

        print 'Number of continuous years for station %s is %d ' % (station.id, num_of_continuous_years)

        #skip stations with less than 20 years of usable data
        #if num_of_continuous_years < 2:
        #    continue

        selected_stations.append(station)

#        plot_total_precip_for_upstream(i_index = i, j_index = j, station_id = station.id,
#                                        subplot_count = current_subplot,
#                                        start_date = datetime(1980,01,01,00),
#                                        end_date = datetime(1996,12,31,00)
#                                        )

        #tmp (if do not need to replot streamflow)
#        current_subplot += 1
#        continue

        ##Calculate means for each day of year,
        ##as a stamp year we use 2001, ignoring the leap year
        stamp_year = 2001
        start_day = datetime(stamp_year, 1, 1, 0, 0, 0)
        stamp_dates = []
        mean_data_model = []
        mean_data_station = []
        simIdToMeanModelData = {}
        for the_id in members.all_current:
            simIdToMeanModelData[the_id] = []

        for day_number in xrange(365):
            the_day = start_day + day_number * data_step
            stamp_dates.append(the_day)

            model_data_for_day = []
            station_data_for_day = []

            #select model data for each simulation, day
            #and then save mean for each day
            simIdToModelDataForDay = {}
            for the_id in members.current_ids:
                simIdToModelDataForDay[the_id] = []

            for year in xrange(start_year, end_year + 1):
                the_date = datetime(year, the_day.month, the_day.day, the_day.hour, the_day.minute, the_day.second)
                if continuous_station_data.has_key(the_date):
                    model_data_for_day.append(continuous_model_data[the_date])
                    station_data_for_day.append(continuous_station_data[the_date])
                    for the_id in members.current_ids:
                        simIdToModelDataForDay[the_id].append(simIdToContData[the_id][the_date])

            assert len(station_data_for_day) > 0
            mean_data_model.append(np.mean(model_data_for_day))
            mean_data_station.append(np.mean(station_data_for_day))
            for the_id in members.current_ids:
                simIdToMeanModelData[the_id].append(np.mean(simIdToModelDataForDay[the_id]))


         #skip stations with small discharge
        #if np.max(mean_data_station) < 300:
        #    continue

        row = current_subplot// ncols
        col = current_subplot % ncols
        ax = fig.add_subplot(gs[row, col])
        assert isinstance(ax, Axes)
        current_subplot += 1

        #put "Streamflow label on the y-axis"
        if row == 0 and col == 0:
            ax.annotate("Streamflow (${\\rm m^3/s}$)", (0.025, 0.7) , xycoords = "figure fraction",
                rotation = 90, va = "top", ha = "center")

        selected_dates = sorted( continuous_station_data.keys() )
        unrouted_stfl = get_unrouted_streamflow_for(selected_dates = selected_dates,
            all_dates=times, tot_runoff=tot_rof, cell_indices=prev_cell_indices[index])

        unrouted_daily_normals = data_select.get_means_for_stamp_dates(stamp_dates, all_dates= selected_dates,
            all_data=unrouted_stfl)

        #Calculate Nash-Sutcliff coefficient
        mean_data_model = np.array(mean_data_model)
        mean_data_station = np.array( mean_data_station )

        #mod = _get_monthly_means(stamp_dates, mean_data_model)
        #sta = _get_monthly_means(stamp_dates, mean_data_station)

        month_dates = [ datetime(stamp_year, m, 1) for m in xrange(1,13) ]


        line1, = ax.plot(stamp_dates, mean_data_model, linewidth = 3, color = "b")
        #line1, = ax.plot(month_dates, mod, linewidth = 3, color = "b")
        upper_model = np.max(mean_data_model)

        line2, = ax.plot(stamp_dates, mean_data_station, linewidth = 3, color = "r")
        #line2, = ax.plot(month_dates, sta, linewidth = 3, color = "r")

        #line3, = ax.plot(stamp_dates, unrouted_daily_normals, linewidth = 3, color = "y")


        mod = mean_data_model
        sta = mean_data_station

        ns = 1.0 - np.sum((mod - sta) ** 2) / np.sum((sta - np.mean(sta)) ** 2)

        if np.abs(ns) < 0.001:
            ns = 0

        corr_coef = np.corrcoef([mod, sta])[0,1]
        ns_unr = 1.0 - np.sum((unrouted_daily_normals - sta) ** 2) / np.sum((sta - np.mean(sta)) ** 2 )
        corr_unr = np.corrcoef([unrouted_daily_normals, sta])[0, 1]

        da_diff = (da_2d[i, j] - station.drainage_km2) / station.drainage_km2 * 100
        ax.annotate("ns = %.2f\nr = %.2f"
                  % (ns, corr_coef), (0.95, 0.90), xycoords = "axes fraction",
            va = "top", ha = "right",
            font_properties = FontProperties(size = 9)
        )



        #plot member simulation data
        lines_for_mems = []
        labels_for_mems = []

        #lines_for_mems.append(line3)
        #labels_for_mems.append("Unrouted total runoff")


        for the_id in members.current_ids:
            the_line, = ax.plot(stamp_dates, simIdToMeanModelData[the_id], "--", linewidth = 3)
            lines_for_mems.append(the_line)
            labels_for_mems.append(the_id)


        ##calculate mean error
        means_for_members = []
        for the_id in members.current_ids:
            means_for_members.append(np.mean(simIdToMeanModelData[the_id]))





        upper_station = np.max(mean_data_station)
        upper_unr = np.max(unrouted_daily_normals)

        upper = np.max([upper_model, upper_station])
        upper = round(upper / 100 ) * 100
        half = round( 0.5 * upper / 100 ) * 100
        if upper <= 100:
            upper = 100
            half = upper / 2

        print half, upper
        print 10 * '='

        ax.set_yticks([0, half , upper])
        assert isinstance(station, Station)

        print("i = {0}, j = {1}".format(indexObj.i, indexObj.j))
        print(lons[i,j], lats[i,j])
        print("id = {0}, da_sta = {1}, da_mod = {2}, diff = {3} %".format(station.id ,station.drainage_km2, da_2d[i,j], da_diff))

        grid_drainages.append(da_2d[i, j])
        grid_lons.append(lons[i, j])
        grid_lats.append(lats[i, j])

        selected_station_values.append(mean_data_station)
        selected_model_values.append(mean_data_model)



        #plot_swe_for_upstream(i_index = i, j_index = j, station_id = station.id)




        #plt.ylabel("${\\rm m^3/s}$")
        west_east = 'W' if station.longitude < 0 else 'E'
        north_south = 'N' if station.latitude > 0 else 'S'
        title_data = (station.id, np.abs(station.longitude), west_east,
                                  np.abs(station.latitude), north_south)
        ax.set_title('%s: (%3.1f%s, %3.1f%s)' % title_data)


        date_ticks = []
        for month in xrange(1,13):
            the_date = datetime(stamp_year, month, 1)
            date_ticks.append(the_date)
            date_ticks.append(the_date + timedelta(days = 15))
        ax.xaxis.set_ticks(date_ticks)



        major_ticks = ax.xaxis.get_major_ticks()


        for imtl, mtl in enumerate(major_ticks):
            mtl.tick1line.set_visible(imtl % 2 == 0)
            mtl.tick2line.set_visible(imtl % 2 == 0)
            mtl.label1On = (imtl % 4 == 1)

#        ax.xaxis.set_major_locator(
#            mpl.dates.MonthLocator(bymonth = range(2,13,2))
#        )


        ax.xaxis.set_major_formatter(
            mpl.dates.DateFormatter('%b')
        )





    lines = [line1]
    lines.extend(lines_for_mems)
    lines.append(line2)
    lines = tuple( lines )


    labels = [label1]
    labels.extend(labels_for_mems)
    labels.append(label2)
    labels = tuple(labels)

    fig.legend(lines, labels, 'lower right', ncol = 1)
#    fig.text(0.05, 0.5, "Streamflow (${\\rm m^3/s}$)",
#                  rotation=90,
#                  ha = 'center', va = 'center'
#                  )


    fig.tight_layout(pad = 2)
    fig.savefig('performance_error.png')




    
   # assert len(selected_dates_with_gw[0]) == len(selected_station_dates[0])

    do_skill_calculation = True
    if do_skill_calculation:
        calculate_skills(selected_stations,
                        stamp_dates, selected_station_values,
                        selected_model_values,
                        grid_drainages,
                        grid_lons, grid_lats)


    do_plot_selected_stations = True
    if do_plot_selected_stations:
        plot_selected_stations(selected_stations, use_warpimage=False, plot_ts = False,
                               i_list = i_list, j_list = j_list)
    #plot_drainage_scatter(selected_stations, grid_drainages)


def plot_station_positions(id_list = None, save_to_file = True, use_warpimage = True, the_basemap = None):
    """
    Plots station positions on map
    """
    stations = read_station_data()
    selected_stations = []
    for s in stations:
        if s.id in id_list:
            selected_stations.append(s)
    plot_selected_stations(selected_stations, plot_ts = False, save_to_file=save_to_file,
                           use_warpimage=use_warpimage, basemap=the_basemap)


from diagnose_ccc import bedrock_depth
def plot_selected_stations(selected_stations, plot_ts = True, save_to_file = True,
                           use_warpimage = True, basemap = None,
                           i_list = None, j_list = None
                           ):
    """
    document me
    """
    fig = plt.figure()
    if basemap is None:
        the_basemap = polar_stereographic.basemap
    else:
        the_basemap = basemap

#    basemap.warpimage()
    if use_warpimage:
        map_url = 'http://earthobservatory.nasa.gov/Features/BlueMarble/images_bmng/8km/world.topo.200407.3x5400x2700.jpg'
        the_basemap.warpimage(image = map_url)
        the_basemap.drawcoastlines()

    # draw a land-sea mask for a map background.
    # lakes=True means plot inland lakes with ocean color.
    #basemap.drawlsmask(land_color='none', ocean_color='aqua',lakes=True)

#   basemap.drawrivers(color = 'blue')


    the_basemap.drawcoastlines()
    the_xs = []
    the_ys = []

    xs = polar_stereographic.xs
    ys = polar_stereographic.ys


    dx = 0.01 * ( xs[i_list, j_list].max() - xs[i_list, j_list].min() )
    dy = 0.01 * ( ys[i_list, j_list].max() - ys[i_list, j_list].min() )


    for station in selected_stations:
        x, y = the_basemap(station.longitude, station.latitude)


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


        the_xs.append(x)
        the_ys.append(y)

        plt.annotate(station.id, xy = (x, y), xytext = (xtext, ytext),
                    font_properties = FontProperties(size = 18),
                     bbox = dict(facecolor = 'white'), weight = "bold"
                     #arrowprops=dict(facecolor='black', shrink=0.001)
                     )

    plot_basin_boundaries_from_shape(the_basemap, plotter=plt, linewidth=2.1)
    the_basemap.scatter(the_xs,the_ys, c = 'r', s = 400, marker='^', linewidth = 0.5, alpha = 1, zorder = 5)



    #plot the field of bedrock depth in meters
#    x, y = the_basemap(polar_stereographic.lons, polar_stereographic.lats)
#    bedrock_field = bedrock_depth.get_bedrock_depth_amno_180x172()
#    bedrock_field = np.ma.masked_where(bedrock_field < 0, bedrock_field)
#    #the_basemap.contourf(x, y, bedrock_field)
#    the_basemap.pcolormesh(x, y, bedrock_field)
#    cb = plt.colorbar(shrink = 0.7)
    #cb.set_label("m")





    if save_to_file:
        x_interest = polar_stereographic.xs[i_list, j_list]
        y_interest = polar_stereographic.ys[i_list, j_list]
        x_min, x_max = x_interest.min(), x_interest.max()
        y_min, y_max = y_interest.min(), y_interest.max()
        dx = 0.1 * ( x_max - x_min )
        dy = 0.1 * ( y_max - y_min )

        plt.xlim(x_min - dx, x_max + dx)
        plt.ylim(y_min - dy, y_max + dy)

        fig.tight_layout()
        fig.savefig('selected_stations.png')

    if plot_ts:
        plt.figure()
        nstations = len(selected_stations)
        plt.subplots_adjust(hspace = 0.2)
        for i, station in enumerate( selected_stations ):
            plt.subplot(nstations / 2 + nstations % 2, 2, i + 1)
            x = map(lambda date: date.year, station.dates)
            plt.plot(x, station.values)
            plt.title(station.id)

        if save_to_file:
            plt.savefig('station_time_series.png')



#####
def plot_drainage_scatter(stations, grid_drainage):
    plt.figure()

    plt.title('Drainage area ${\\rm km^2}$')

    plt.xlabel('Station')
    plt.ylabel('Model')


    s_area = []
    for station in stations:
            # @type station Station
            s_area.append(station.drainage_km2)

    plt.scatter(s_area, grid_drainage, linewidth = 0)

    
    x = plt.xlim()
    plt.plot(x,x, color = 'k')

    plt.savefig('drainage_area_scatter.png')
    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    print os.getcwd()
    #get_station_and_corresponding_model_data(path = 'data/streamflows/hydrosheds_euler10_spinup100yrs/aex_discharge_1970_01_01_00_00.nc')
    #main()

    plot_station_positions(id_list=["104001", "093806", "092715", "061502", "040830", "093801"])
    print "Hello World"
