
__author__="huziy"
__date__ ="$25 mai 2010 16:45:32$"


from station_manager import StationManager
from util.lat_lon_holder import LatLonHolder
from util.convert import *
from util.lat_lon_holder import *

def main():
    model = LatLonHolder()
    station_manager = StationManager()
    station_manager.read_stations_from_files_rivdis()

    selected_stations = []
    ij_list = []

    composite_index = 0
    for longitude, latitude in zip( model.longitudes, model.latitudes ):
        the_station = station_manager.get_station_closest_to(longitude, latitude)
        if the_station != None:
            selected_stations.append(the_station)
            ix = model.get_ix(composite_index)
            iy = model.get_iy(composite_index)
            ij_list.append([ix, iy])
            
        composite_index += 1

    for station in selected_stations:
        print station.longitude, station.latitude, station.id
    print ij_list






if __name__ == "__main__":
    main()
