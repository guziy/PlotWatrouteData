from data.station import Station
from util.lat_lon_holder import get_distance_km
import os.path

__author__="huziy"
__date__ ="$26 mai 2010 12:07:46$"

import os
import re

from data.station import *
from util.lat_lon_holder import *
from datetime import *

import application_properties
application_properties.set_current_directory() 

PATH_TO_STATION_DATA_BASE_RIVDIS = 'data/measurements'
PATH_TO_STATION_DATA_BASE_HYDAT = 'data/hydat_measure_data'
DATA_FILE_NAME = 'TABLE.DAT'


LONGITUDE = 'LONGITUDE'
LATITUDE = 'LATITUDE'
HYDAT_DATE_FORMAT = '%m--%Y'


INFLUENCE_DISTANCE_KM = 100.0

class StationManager():
    def __init__(self):
        self.stations = []

    def add_station(self, new_station):
        self.stations.append( new_station )



    def get_station_by_id(self, id):
        for station in self.stations:
            # @type station Station
            if station.id == id:
                return station
        return None

    def read_stations_from_files_rivdis(self):
        '''
        read measurements from rivdis provider
        '''
        for folder_name in self._get_folder_list(PATH_TO_STATION_DATA_BASE_RIVDIS):
            file_path = PATH_TO_STATION_DATA_BASE_RIVDIS + os.sep + folder_name + os.sep + DATA_FILE_NAME
            
            ##check whether it is a proper folder containing files TABLE.DAT
            if not os.path.isfile(file_path):
                continue

            file = open( file_path )
            first_line = file.readline()
            fields = first_line.split(',')
            lon_str = fields[2]
            lat_str = fields[1]

            multiplier = 1
            if lon_str.upper().strip().endswith('W'):
                multiplier = -1

            lon_str = re.sub('[a-zA-Z]', '', lon_str)
            lon = multiplier * float(lon_str)


            multiplier = 1
            if lat_str.upper().strip().endswith('S'):
                multiplier = -1
            lat_str = re.sub('[a-zA-Z]', '', lat_str)
            lat = multiplier * float(lat_str)

            the_station = Station(id = folder_name, longitude = lon, latitude = lat)
            self._parse_data_for_station_rivdis(file, the_station)
            self.add_station(the_station)

    def read_data_from_files_hydat(self):
        '''
        read measuremments from hydat provider
        '''
        for folder_name in self._get_folder_list(PATH_TO_STATION_DATA_BASE_HYDAT):
            data_file_path = PATH_TO_STATION_DATA_BASE_HYDAT + os.sep + folder_name + os.sep + folder_name + '.txt'
            info_file_path = PATH_TO_STATION_DATA_BASE_HYDAT + os.sep + folder_name + os.sep + 'info.txt'

            for line in open(info_file_path):
                if line.strip().upper().startswith(LONGITUDE):
                    longitude = self._get_decimal_lon_lat(line)
                if line.strip().upper().startswith(LATITUDE):
                    latitude = self._get_decimal_lon_lat(line)

            station = Station(id = folder_name, longitude = longitude, latitude = latitude )
            self._parse_data_for_station_hydat(open(data_file_path), station)
            self.add_station(station)




    def _get_folder_list(self, base_path):
        '''
        get list of folders in the base_path
        '''
        result = []
        for f in os.listdir(base_path):
            if f.startswith('.'):
                continue
            if not os.path.isdir(base_path + os.sep + f):
                continue
            result.append(f)
        return result


    def _get_decimal_lon_lat(self, line):
        if not ':' in line:
            return None

        fields = re.findall('\d+', line) 
        multiplier = 1

        line = line.strip().upper()
        if line.endswith('W') or line.endswith('S'):
           multiplier = -1

        deg = float(fields[0])
        min = float(fields[1])
        sec = float(fields[2])

        result = deg + min / 60.0 + sec / 3600.0
        return result * multiplier



    def _parse_data_for_station_rivdis(self, file_obj, the_station):
        '''
        parse data file from rivdis site 
        '''
        #skip header
        file_obj.readline()
        file_obj.readline()

        #read data
        for line in file_obj:
            if not ('|' in line):
                continue

            if line.strip().endswith('|'):
                continue
            
            fields = line.split('|')

            date = datetime(int(fields[1]), int(fields[2]),1, 0, 0, 0)
            value = float(fields[3])
            the_station.add_timeseries_data(date, value)

    def _parse_data_for_station_hydat(self, file_obj, station):
        '''
        parse data file from hydat site
        '''
        #skip header
        file_obj.readline()

        for line in file_obj:
            if not ',' in line:
                continue
            fields = line.split(',')
            date = datetime.strptime(fields[2], HYDAT_DATE_FORMAT )
            value = float( fields[3] )
            # @type station Station
            station.add_timeseries_data(date, value)






    def get_station_closest_to(self, longitude, latitude):
        r_min = -1
        the_station = None
        for station in self.stations:
            if r_min < 0:
                r_min = get_distance_km(longitude, latitude, station.longitude, station.latitude)
                r = r_min
            else:
                r = get_distance_km(longitude, latitude, station.longitude, station.latitude)
                if r_min > r:
                    the_station = station
                    r_min = r

        if r_min < INFLUENCE_DISTANCE_KM:
            return the_station
        else:
            return None


if __name__ == "__main__":
    manager = StationManager()
    manager.read_stations_from_files_rivdis()
    for station in manager.stations:
        print station
    print "Hello World"
