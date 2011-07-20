__author__="huziy"
__date__ ="$29 juil. 2010 16:13:51$"

import Nio # for writing to netcdf
import numpy as np
import os

from util.geo.GeoPoint import GeoPoint
from binary.read_graham_binary import get_data_subset
from binary.read_graham_binary import E, NE, N, NW, W, SW, S, SE
import util.geo.lat_lon as lat_lon


from wrf_static_data.read_variable import get_topography_interpolated_to_graham_grid


import application_properties
application_properties.set_current_directory()

MINIMUM_ALLOWED_SLOPE = 1.0e-3

def get_next_indices(i,j, flow_direction):
    the_dict = {
        E : (i+1, j),
        NE : (i + 1, j + 1),
        N : (i, j + 1),
        NW : (i - 1, j + 1 ),
        W : (i - 1, j ),
        SW : (i -1, j - 1),
        S : (i, j - 1 ),
        SE : (i + 1, j)
    }

    if not the_dict.has_key(flow_direction):
        return -1, -1

    return the_dict[flow_direction]



def save_channel_slopes_to_netcdf(file_name = 'slopes.nc', 
                                  lower_left_point = GeoPoint(-80, 45),
                                  the_shape = (300, 300),
                                  resolution = 5.0 / 60.0):


    #check the sanity of input  arguments
    max_longitude = lower_left_point.longitude + resolution * (the_shape[0] - 1 )
    max_latitude = lower_left_point.latitude + resolution * (the_shape[1] - 1 )
    assert max_longitude >= -180 and max_longitude <= 180
    assert max_latitude >= -90.0 and max_latitude <= 90.0


    flow_directions = get_data_subset(lower_left_point = lower_left_point,
                       the_shape = the_shape,
                       resolution = resolution,
                       file_name = '5minfdr.img')

    longitudes, latitudes, elevations = get_topography_interpolated_to_graham_grid(grid_shape = the_shape,
                                                            grid_lower_left = lower_left_point)

    slopes = np.zeros(the_shape)

    lon0 = lower_left_point.longitude
    lat0 = lower_left_point.latitude

    for i in range(the_shape[0]):
        for j in range(the_shape[1]):
            fdir = flow_directions[i, j]
            i1, j1 = get_next_indices(i, j, fdir)

            if i1 < 0 and j1 < 0:
                slopes[i,j] = -1
                continue

            #do not route water outside the clculation domain
            if i1 >= the_shape[0] or j1 >= the_shape[1] or i1 < 0 or j1 < 0:
                slopes[i,j] = -1
                continue


            lon, lat = lon0 + i * resolution, lat0 + j * resolution
            lon1, lat1 = lon0 + i1 * resolution, lat0 + j1 * resolution
            distance = lat_lon.get_distance_in_meters(lon, lat, lon1, lat1)


            slope = (elevations[i, j] - elevations[i1, j1]) / distance
            if slope <= 0:
                print elevations[i, j], elevations[i1, j1], distance
                slope = MINIMUM_ALLOWED_SLOPE

            slopes[i, j] = slope


    opt = 'c' #create netcdf file
    if os.path.isfile(file_name):
        os.remove(file_name)

    file = Nio.open_file( file_name, opt )
    WEST_EAST = 'west_east'
    NORTH_SOUTH = 'north_south'

    file.create_dimension(WEST_EAST, the_shape[0])
    file.create_dimension(NORTH_SOUTH, the_shape[1])

    file.create_variable('channel slopes', 'd', (WEST_EAST, NORTH_SOUTH))
    file.close()





if __name__ == "__main__":
    save_channel_slopes_to_netcdf()
