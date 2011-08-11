
__author__="huziy"
__date__ ="$8 dec. 2010 10:20:08$"

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

from data import data_select
from plot2D import plot_utils



MAXIMUM_DISTANCE_METERS = 45000.0 #m

TIME_FORMAT = '%Y_%m_%d_%H_%M'

lons = polar_stereographic.lons
lats = polar_stereographic.lats
xs = polar_stereographic.xs
ys = polar_stereographic.ys


inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5.0) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 1500 * inches_per_pt          # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]

params = {
        'axes.labelsize': 20,
        'font.size': 20,
        'text.fontsize': 20,
        'legend.fontsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
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
    alpha = 1
    return distance / MAXIMUM_DISTANCE_METERS + np.abs(da2 - da1) / da1 * alpha


def get_closest_station(lon, lat, cell_drain_area_km2, station_list):
    '''
    returns the closest station to (lon, lat),
    or None if it was not found
    '''
    #find distance to the closest station
    objective = None
    result = None
    for station in station_list:
        station_drainage = station.drainage_km2
        distance = get_distance_in_meters(lon, lat, station.longitude, station.latitude)
        objective1 = objective_function(distance, station_drainage, cell_drain_area_km2)
        
        if objective == None:
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
    values = {}
    counts = {}

    assert len(times) == len(data)

    dates = create_dates_of_year(times, year = year)
    result_values = []
    result_dates = []

    for date, time, the_value in zip(dates, times, data):

        if start_date != None and end_date != None:
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
        
 
def calculate_skills(selected_stations = [], 
                    dates = [], selected_station_values = [],
                    selected_model_values = [],
                    grid_drainages = [],
                    grid_lons = [], grid_lats = []):



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
        drainage_area = data_select.get_field_from_file(path, field_name = 'drainage')


        #da_2d = drainage_area

        da_2d = np.zeros(lons.shape)
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
            station = get_closest_station(lons[i, j], lats[i, j], da_2d[i, j], stations)
            if station == None or station in selected_stations:
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



def main():

    pylab.rcParams.update(params)
    #path = 'data/streamflows/output_2d/data1/aex_discharge_1961_01_01_00_00.nc'
    path = 'data/streamflows/hydrosheds_euler10_spinup100yrs/aex_discharge_1970_01_01_00_00.nc'

    data, times, i_list, j_list = data_select.get_data_from_file(path)

    da_2d = data_select.get_from_file(path, 'accumulation_area')

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

#    da_2d = np.zeros((lons.shape))
#    for index, i, j in zip( range(len(i_list)) , i_list, j_list):
#        da_2d[i, j] = drainage_area[index]
       

    selected_stations = []
    selected_model_values = []
    selected_station_values = []

    grid_drainages = []
    grid_lons = []
    grid_lats = []


    plt.figure()
    current_subplot = 1

    label1 = 'model'
    label2 = 'observation'
    override = {'fontsize': 20}
    plt.subplots_adjust(hspace = 1.5, wspace = 0.4, top = 0.9)
    

    for index, i, j in zip( range(len(i_list)) , i_list, j_list):
        station = get_closest_station(lons[i, j], lats[i, j], da_2d[i, j], stations)


        if station == None or station in selected_stations:
            continue

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

        print 'Number of continuous years for station %s is %d ' % (station.id, num_of_continuous_years)

        #skip stations with less than 20 years of usable data
        #if num_of_continuous_years < 2:
        #    continue



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




        plt.subplot(10, 3, current_subplot)
        current_subplot += 1


        line1, = plt.plot(stamp_dates, mean_data_model, linewidth = 3)

        upper_model = np.max(mean_data_model)

        line2, = plt.plot(stamp_dates, mean_data_station, color = 'r', linewidth = 3)

        upper_station = np.max(mean_data_station)


        upper = max(upper_model, upper_station)
        upper = round(upper / 100 ) * 100
        half = round( 0.5 * upper / 100 ) * 100
        if upper <= 100:
            upper = 100
            half = upper / 2

        print half, upper
        print 10 * '='

        plt.yticks([0, half , upper])

        grid_drainages.append(da_2d[i, j])
        grid_lons.append(lons[i, j])
        grid_lats.append(lats[i, j])

        selected_station_values.append(mean_data_station)
        selected_model_values.append(mean_data_model)
        selected_stations.append(station)

        plt.ylabel("${\\rm m^3/s}$")
        plt.title(station.id, override)
        ax = plt.gca()
        ax.xaxis.set_major_locator(
            mpl.dates.MonthLocator(bymonth = range(2,13,2))
        )


        ax.xaxis.set_major_formatter(
            mpl.dates.DateFormatter('%b')
        )


    lines = (line1, line2)
    plt.figlegend(lines, (label1, label2), 'upper center')
    plt.savefig('performance_error.png')

    
   # assert len(selected_dates_with_gw[0]) == len(selected_station_dates[0])

    calculate_skills(selected_stations,
                    stamp_dates, selected_station_values,
                    selected_model_values,
                    grid_drainages,
                    grid_lons, grid_lats)
    plot_selected_stations(selected_stations)
    plot_drainage_scatter(selected_stations, grid_drainages)



def plot_selected_stations(selected_stations):
    plt.clf()
    basemap = polar_stereographic.basemap

#    basemap.warpimage()
#    basemap.warpimage(image = 'http://earthobservatory.nasa.gov/Features/BlueMarble/images_bmng/2km/world.200406.3x21600x10800.jpg')
    basemap.drawcoastlines()

    # draw a land-sea mask for a map background.
    # lakes=True means plot inland lakes with ocean color.
#    basemap.drawlsmask(land_color='none', ocean_color='aqua',lakes=True)

#    basemap.drawrivers(color = 'blue')
#    plot_basin_boundaries_from_shape(basemap, 1)
    for station in selected_stations:
        x, y = basemap(station.longitude, station.latitude)

        xtext = 1.005 * x
        ytext = y
        if station.id in ['061906']:
            xtext = 1.00 * x
            ytext = 0.97 * y

        if station.id in ['103603', '081002']:
            ytext = 0.98 * y

        if station.id in ['081007']:
            xtext = 0.97 * x


        plt.annotate(station.id, xy = (x, y), xytext = (xtext, ytext),
                     bbox = dict(facecolor = 'white')
                     #arrowprops=dict(facecolor='black', shrink=0.001)
                     )
        basemap.scatter(x,y, c = 'r', s = 100, marker='^', linewidth = 0, alpha = 1)

    plot_utils.zoom_to_qc(plt)

    plt.savefig('selected_stations.png',  bbox_inches='tight')


    plt.figure()
    nstations = len(selected_stations)
    plt.subplots_adjust(hspace = 0.2)
    for i, station in enumerate( selected_stations ):
        plt.subplot(nstations / 2 + nstations % 2, 2, i + 1)
        x = map(lambda date: date.year, station.dates)
        plt.plot(x, station.values,'-o')
        plt.title(station.id)
        

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
    get_station_and_corresponding_model_data(path = 'data/streamflows/hydrosheds_euler10_spinup100yrs/aex_discharge_1970_01_01_00_00.nc')
    #main()
    print "Hello World"
