
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

import data.data_select as data_select
import util.plot_utils as plot_utils

from index_object import IndexObject
import netCDF4 as nc

from data.cell import Cell


import diagnose_ccc.compare_precip as compare_precip

MAXIMUM_DISTANCE_METERS = 45000.0 #m

TIME_FORMAT = '%Y_%m_%d_%H_%M'

lons = polar_stereographic.lons
lats = polar_stereographic.lats
xs = polar_stereographic.xs
ys = polar_stereographic.ys


inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5.0) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 800 * inches_per_pt          # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, 2 * fig_height]

params = {
        'axes.labelsize': 25,
        'font.size': 25,
        'text.fontsize': 25,
        'legend.fontsize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'figure.figsize': fig_size
        }



#needed for calculation of mean of data for each day
def create_dates_of_year(date_list = [], year = 2000):
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
    alpha = 1
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
        
    
    if objective <= 0.7:
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
        
 
def calculate_skills(selected_stations=None, dates=[], selected_station_values=[], selected_model_values=[],
                     grid_drainages=[], grid_lons=[], grid_lats=[]):



    if not selected_stations: selected_stations = []
    for i in range(len(selected_stations)):
         model_values = selected_model_values[i]
         station = selected_stations[i]
         station_values = selected_station_values[i]
         the_dates = dates if len(dates) <= len(dates) else dates

         model_values_list = []
         station_values_list = []
         for index, the_date in enumerate(the_dates):
            if the_dates == dates:
                station_values_list.append(station_values[index])
                index_model = dates.index(the_date)
                model_values_list.append(model_values[index_model])
            else:
                index_station = dates.index(the_date)
                station_values_list.append(station_values[index_station])
                model_values_list.append(model_values[index])
            



         model_values = np.array(model_values_list)
         station_values = np.array(station_values_list)

         grid_drainage  = grid_drainages[i]
         grid_lon = grid_lons[i]
         grid_lat = grid_lats[i]

#         values_without_gw = selected_values_without_gw[i]


         #calculate skill coefficient
         std1 = np.std(model_values)
         std2 = np.std(station_values)
         R = np.corrcoef(model_values, station_values)
         S = 2 * std1 * std2 / (std1 ** 2 + std2 ** 2) * R[0,1]


#         print 'Station:'
#         print 'Id,Lon, Lat, DA: %s, %f, %f, %f' % ( station.id, station.longitude, station.latitude, station.drainage_km2 )
#         print 'Grid point'
#         print 'Lon, Lat, DA, Skill, dDA/DA : %f, %f, %f, %f, %f' % (grid_lon, grid_lat, grid_drainage,
#                                    S, (grid_drainage - station.drainage_km2) / station.drainage_km2 )

         print '%s & %.2f & %.2f & %.2f ' % ( station.id, station.longitude, station.latitude, station.drainage_km2 )
         print '& %.2f &  %.2f & %.2f & %.2f & %.2f \\\\' % (grid_lon, grid_lat, grid_drainage,
                                    S, (grid_drainage - station.drainage_km2) / station.drainage_km2 * 100 )
         print '\\hline'
 #        print '=============='


##returns a dictionary {station: modelpoint}
def get_station_and_corresponding_model_data(path = 'data/streamflows/hydrosheds_euler10_spinup100yrs/aex_discharge_1970_01_01_00_00.nc'):
    result = {}
    saved_selected_stations_file = 'selected_stations_and_model_data.bin'
    if os.path.isfile(saved_selected_stations_file):
        result = pickle.load(open(saved_selected_stations_file))
    else:
        print 'getting data from file ', path


        data, times, i_list, j_list = data_select.get_data_from_file(path)
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


    nx, ny = next_i.shape

    cells = [[Cell(ix = i, jy = j) for j in xrange(ny)]
                for i in xrange(nx)]
    
    #connect cells
    for i in xrange(nx):
        for j in xrange(ny):
            iNext = next_i[i, j]
            jNext = next_j[i, j]
            theCell = cells[i][j]
            # @type theCell Cell
            if iNext >= 0 and jNext >= 0:
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

    the_mask = the_mask * domain_mask
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


def main():

    """

    """
    skip_ids = ['081007', '081002']

    pylab.rcParams.update(params)
    path = 'data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc'
    #path = 'data/streamflows/na/discharge_1990_01_01_00_00_na_fix.nc'

    old = True #in the old version drainage and lon,lats in the file are 1D

    data, times, i_list, j_list = data_select.get_data_from_file(path)
    if not old:
        da_2d = data_select.get_field_from_file(path, 'accumulation_area')
        lons = data_select.get_field_from_file(path, field_name = 'longitude')
        lats = data_select.get_field_from_file(path, field_name = 'latitude')
    else:
        lons = polar_stereographic.lons
        lats = polar_stereographic.lats
        da_2d = np.zeros(lons.shape)
        drainage = data_select.get_field_from_file(path, 'drainage')
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

   
    reload(sys)
    sys.setdefaultencoding('iso-8859-1')


    selected_stations = []
    selected_model_values = []
    selected_station_values = []

    grid_drainages = []
    grid_lons = []
    grid_lats = []

    plot_utils.apply_plot_params(font_size=25, width_pt=1200, aspect_ratio=2)
    fig = plt.figure()

    current_subplot = 1

    label1 = 'model'
    label2 = 'observation'
    override = {'fontsize': 24}
    fig.subplots_adjust(hspace = 0.9, wspace = 0.4, top = 0.9)

    

    index_objects = []
    for index, i, j in zip( range(len(i_list)) , i_list, j_list):
        index_objects.append(IndexObject(positionIndex = index, i = i, j = j))

    #sort by latitude
    index_objects.sort( key = lambda x: x.j, reverse = True)

    for indexObj in index_objects:
        i = indexObj.i
        j = indexObj.j
        # @type indexObj IndexObject
        index = indexObj.positionIndex
        station = get_corresponding_station(lons[i, j], lats[i, j], da_2d[i, j], stations)

        if station is None or station in selected_stations:
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


        start_date = max( np.min(times), np.min(station.dates) )
        end_date = min( np.max(times),  np.max(station.dates) )

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

        #if there is no continuous observations for the period
        if not len(continuous_station_data): continue

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
        for day_number in xrange(365):
            the_day = start_day + day_number * data_step
            stamp_dates.append(the_day)

            model_data_for_day = []
            station_data_for_day = []

            for year in xrange(start_year, end_year + 1):
                the_date = datetime(year, the_day.month, the_day.day, the_day.hour, the_day.minute, the_day.second)
                if continuous_station_data.has_key(the_date):
                    model_data_for_day.append(continuous_model_data[the_date])
                    station_data_for_day.append(continuous_station_data[the_date])

            assert len(station_data_for_day) > 0
            mean_data_model.append(np.mean(model_data_for_day))
            mean_data_station.append(np.mean(station_data_for_day))




        ax = fig.add_subplot(5, 2, current_subplot)
        current_subplot += 1


        line1, = ax.plot(stamp_dates, mean_data_model,color = "blue", linewidth = 3)

        upper_model = np.max(mean_data_model)

        line2, = ax.plot(stamp_dates, mean_data_station, color = 'r', linewidth = 3)

        upper_station = np.max(mean_data_station)


        upper = max(upper_model, upper_station)
        upper = round(upper / 100 ) * 100
        half = round( 0.5 * upper / 100 ) * 100
        if upper <= 100:
            upper = 100
            half = upper / 2

        print half, upper
        print 10 * '='

        ax.set_yticks([0, half , upper])

        grid_drainages.append(da_2d[i, j])
        grid_lons.append(lons[i, j])
        grid_lats.append(lats[i, j])

        selected_station_values.append(mean_data_station)
        selected_model_values.append(mean_data_model)



#        plot_swe_for_upstream(i_index = i, j_index = j, station_id = station.id)




        #plt.ylabel("${\\rm m^3/s}$")
        west_east = 'W' if station.longitude < 0 else 'E'
        north_south = 'N' if station.latitude > 0 else 'S'
        title_data = (station.id, np.abs(station.longitude), west_east,
                                  np.abs(station.latitude), north_south)
        ax.set_title('%s: (%3.1f%s, %3.1f%s)' % title_data, override)

        ax.xaxis.set_major_locator(
            mpl.dates.MonthLocator(bymonth = range(2,13,2))
        )


        ax.xaxis.set_major_formatter(
            mpl.dates.DateFormatter('%b')
        )


    lines = (line1, line2)

    fig.legend(lines, (label1, label2), 'upper center')
    fig.text(0.05, 0.5, 'STREAMFLOW (${\\rm m^3/s}$)',
                  rotation=90,
                  ha = 'center', va = 'center'
                  )

    fig.savefig('performance_error.pdf')



    
   # assert len(selected_dates_with_gw[0]) == len(selected_station_dates[0])

    do_skill_calculation = False
    if do_skill_calculation:
        calculate_skills(selected_stations,
                        stamp_dates, selected_station_values,
                        selected_model_values,
                        grid_drainages,
                        grid_lons, grid_lats)


    do_plot_selected_stations = False
    if do_plot_selected_stations:
        plot_selected_stations(selected_stations)
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


def plot_selected_stations(selected_stations, plot_ts = True, save_to_file = True,
                           use_warpimage = True, basemap = None):
    """
    document me
    """
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
#    plot_basin_boundaries_from_shape(basemap, 1)
    the_xs = []
    the_ys = []
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

        if station.id in ['093801']:
            ytext = 0.97 * y
            xtext = 0.97 * x

        the_xs.append(x)
        the_ys.append(y)

        plt.annotate(station.id, xy = (x, y), xytext = (xtext, ytext),
                     bbox = dict(facecolor = 'white')
                     #arrowprops=dict(facecolor='black', shrink=0.001)
                     )
    the_basemap.scatter(the_xs,the_ys, c = 'r', s = 100, marker='^', linewidth = 0.5, alpha = 1)





    if save_to_file:
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        xmin = (xmin + xmax) / 2.0
        ymax = (ymin + ymax) / 2.0

        ymin = 0.7 * ymin + 0.3 * ymax

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        plt.savefig('selected_stations.pdf',  bbox_inches='tight')

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


    plt.savefig('drainage_area_scatter.png', bbox_inches = 'tight')
    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    print os.getcwd()
    #get_station_and_corresponding_model_data(path = 'data/streamflows/hydrosheds_euler10_spinup100yrs/aex_discharge_1970_01_01_00_00.nc')
    main()

    #plot_station_positions(id_list=["104001", "093806", "092715", "061502", "040830", "093801"])
    print "Hello World"
