import os.path

__author__="huziy"
__date__ ="$6 juil. 2010 12:37:02$"


from util.geo.lat_lon import get_nvector
import os
import numpy as np
from struct import Struct
import matplotlib.pyplot as plt



#from binary.read_graham_binary import get_graham_grid

import application_properties
application_properties.set_current_directory() 


from util.geo.GeoPoint import GeoPoint
from util.geo.LonLatGrid import LonLatGrid
from util.geo.lat_lon import *

from mpl_toolkits.basemap import Basemap


import time

#read variable data from the wrf static data specific format

TILE_X = 'tile_x'
TILE_Y = 'tile_y'
TILE_Z = 'tile_z'

DX = 'dx'
DY = 'dy'

KNOWN_X = 'known_x'
KNOWN_Y = 'known_y'
KNOWN_LAT = 'known_lat'
KNOWN_LON = 'known_lon'

WORD_SIZE = 'wordsize'

#if tile_bdr = 3, this means that we have halo part of size 3 in the file
#so the number of data point in such file will be (tile_x + 3) * (tile_y + 3)
TILE_BORDER ='tile_bdr'

SCALE_FACTOR = 'scale_factor'
INDEX = 'index'

KEYS = [
         TILE_X, TILE_Y, TILE_Z,
         WORD_SIZE, TILE_BORDER, SCALE_FACTOR,
         KNOWN_X, KNOWN_Y, KNOWN_LON, KNOWN_LAT,
         DX, DY
]

MINUTES_PER_DEGREE = 60.0


def read_data(path_to_folder):
    data_properties = _read_header(path_to_folder + os.path.sep + INDEX)

    file_names = os.listdir(path_to_folder)
    file_names.remove(INDEX) #skip header file

    print path_to_folder

    nlongitudes, nlatitudes = _get_nx_ny(file_names)
    tile_z = data_properties[TILE_Z]
    values = np.zeros((nlongitudes, nlatitudes, tile_z))

    record_format = _get_record_format(data_properties[WORD_SIZE])

    print record_format
    
    for file_name in file_names:
        if file_name.startswith('.'): continue
        read_file(file_name, path_to_folder, values, data_properties, record_format) 

    scale_factor = data_properties[SCALE_FACTOR]
    return values * scale_factor


def read_file(file_name, path_to_folder, values, data_properties,record_format):
    starti, endi, startj, endj = _get_indices_from_name(file_name)
    tile_bdr = data_properties[TILE_BORDER]
    tile_x = data_properties[TILE_X]
    word_size = data_properties[WORD_SIZE]
    file = open(path_to_folder + os.path.sep + file_name, 'rb')

    tile_z = data_properties[TILE_Z]


    #skip upper tile border
    #tile_bdr ......
    #tile_bdr data
    #tile_bdr ......
    
    structObj = Struct(record_format)

    for k in range(tile_z):
        file.read(tile_bdr * ( tile_x + 2 * tile_bdr) * word_size)
        for j in range(startj, endj + 1):
            file.read( tile_bdr * word_size ) #skip starting tile border
            for i in range(starti, endi + 1):
                x, = structObj.unpack(file.read(word_size))
                values[i, j, k] = x
            file.read( tile_bdr * word_size ) #skip ending tile border
        file.read(tile_bdr * ( tile_x + 2 * tile_bdr) * word_size)

    file.close()


#starti, endi, startj, endj
#returns 4 element list
def _get_indices_from_name(file_name):
    result = []
    for part in file_name.split('.'):
        for part1 in part.split('-'):
            result.append(int(part1) - 1)
    return result

#reads index file
#returns dictionary with relevant properties
def _read_header(path):
    file = open(path)
    result = {}
    for line in file:
        if not '=' in line:
            continue
        the_value = _get_value(line)
        for key in KEYS:
            if key in line:
                result[key] = the_value
    if not result.has_key(SCALE_FACTOR):
        result[SCALE_FACTOR] = 1.0
    if not result.has_key(TILE_BORDER):
        result[TILE_BORDER] = 0
        print 'warning: setting tile border to default: 0'
    return result

#gets the value from a line of type
# key = value
def _get_value(line, separator = '='):
    try:
        return int(line.split(separator)[1].strip())
    except ValueError:
        pass

    try:
        return float(line.split(separator)[1].strip())
    except ValueError:
        pass

    return line.split(separator)[1].strip()


#get number of longitudes and number of latitudes
#from the list of file names
def _get_nx_ny(file_names):
    result = [0, 0]
    for file_name in file_names:
        print file_name
        if '-' not in file_name:
            continue
        if '.' not in file_name:
            continue
        for i, current in enumerate(file_name.split('.')):
            print current
            current = current.split('-')[1]
            result[i] = max(result[i], int(current))
    return result


def _get_record_format(word_size, endian = '>'):
    if word_size == 2:
        return endian + 'h'

    if word_size == 4:
        return endian + 'i'

    if word_size == 1:
        return endian + 'b'


def get_wrf_grid(path_to_index_file, resolution_minutes = 5.0):
    data_properties = _read_header(path_to_index_file)

    known_lon = data_properties[KNOWN_LON]
    known_lat = data_properties[KNOWN_LAT]

    print 'known reference: ', known_lon, known_lat 


    delta = resolution_minutes / MINUTES_PER_DEGREE

    #find longitudes
    longitude = known_lon
    nlongitudes = 0
    while longitude >= -180.0:
        longitude -= delta
    longitude += delta
    lon_min = longitude

    while longitude <= 180.0:
        longitude += delta
        nlongitudes += 1
    longitude -= delta
    lon_max = longitude
    
    longitudes = np.linspace(lon_min, lon_max, nlongitudes)
    
    latitude = known_lat
    while latitude >= -90:
        latitude -= delta
    latitude += delta
    lat_min = latitude

    nlatitudes = 0
    while latitude <= 90:
        latitude += delta
        nlatitudes += 1
    lat_max = latitude - delta


    latitudes = np.linspace(lat_min, lat_max, nlatitudes) 
    print lat_max
    print latitudes

    lower_left = GeoPoint(longitude = longitudes[0], latitude = latitudes[0])
    nlon = len(longitudes)
    nlat = len(latitudes)

    return LonLatGrid(lower_left_point = lower_left, grid_shape = (nlon, nlat)) 


#IDW interpoltation method 
#interpolated = sum(alpha_i * data_i), alpha_i = (1/dist_i ** 2) / sum(1/dist_i ** 2)
def get_topography_interpolated_to_graham_grid(grid_shape = (600, 100), grid_lower_left = GeoPoint(-108.0, 57.0) ):
    graham_grid = get_graham_grid(the_shape = grid_shape, lower_left_point = grid_lower_left)

    print 'Interpolating to the grid: '
    print graham_grid

    longitudes = graham_grid.get_2d_longitudes()
    latitudes = graham_grid.get_2d_latitudes()


    data = read_data('data/wrf/geog/topo_5m')
    
    wrf_grid = get_wrf_grid('data/wrf/geog/topo_5m/index', resolution_minutes = 5)

    yield longitudes
    yield latitudes

    result = (
        wrf_grid.interpolate_from_grid_to_point(data, the_longitude, the_latitude)
            for the_latitude, the_longitude in zip( latitudes.ravel(), longitudes.ravel() )
    )
    yield np.fromiter(result,'float').reshape(grid_shape)


        

def test_read_data():
    data = read_data('data/wrf/geog/hslop')
    print 'finished reading data ... '
    print 'plotting data'
    print 'data shape ', data.shape
    data = data[:, :, :]
 
    data = data[700:1000, 800:1000, 0]
    print 'slope minimum: %f ' % np.min(data.flatten())
    print 'slope maximum: %f ' % np.max(data.flatten())

    plt.contourf(data.transpose())
    plt.colorbar()
    plt.savefig('topo_5min_wrf.png')
    print 'finished'

def test_get_lon_lat():
    wrf_grid = get_wrf_grid('data/wrf/geog/topo_5m/index')
    lons = wrf_grid.get_2d_longitudes()
    lats = wrf_grid.get_2d_latitudes()
    n_wrf = get_nvector(np.radians(lons), np.radians(lats))

    n_wrf = np.array(n_wrf)
    print n_wrf.shape

    graham_grid = get_graham_grid()
    print graham_grid
    








from util.plot_utils import draw_meridians_and_parallels


def get_indices_from_lon_lat(longitude, latitude,
                             ref_longitude, ref_latitude,
                             ref_i, ref_j, dlon, dlat):

    i = ref_i + int( (longitude - ref_longitude) / dlon )
    j = ref_j + int( (latitude - ref_latitude) / dlat )
    return [i - 1 , j - 1 ] #-1 since the ref_indices are counted starting from 1

def get_slopes_interpolated_to_amno_grid(polar_stereographic):
    lons = polar_stereographic.lons
    lats = polar_stereographic.lats


    folder = 'data/wrf/geog/hslop'

    data_properties = _read_header(os.path.join(folder, 'index'))
    dlon = data_properties[DX]
    dlat = data_properties[DY]

    ref_lon = data_properties[KNOWN_LON]
    ref_lat = data_properties[KNOWN_LAT]

    print 'known reference: ', ref_lon, ref_lat

    ref_i = data_properties[KNOWN_X]
    ref_j = data_properties[KNOWN_Y]


    tile_x = data_properties[TILE_X]
    tile_y = data_properties[TILE_Y]
    
    data = read_data(folder)

    result = np.zeros(lons.shape)
    nx, ny = lons.shape
    for i in range(nx):
        for j in range(ny):
            lon = lons[i, j]
            lat = lats[i, j]
            i_source, j_source = get_indices_from_lon_lat(lon, lat, ref_lon, ref_lat, ref_i, ref_j, dlon, dlat)
            count = 0
            for di in range(-1,2):
                for dj in range(-1, 2):
                    if i_source >= tile_x or i_source < 0 or j_source < 0 or j_source >= tile_y:
                        continue
                    result[i, j] += data[i_source + di, j_source + dj]
                    count += 1.0

            result[i, j] /= float(count)
    return result


def test_interpolation():
    lons, lats, interpolated = get_topography_interpolated_to_graham_grid(
                                        grid_lower_left = GeoPoint(-85.0, 40.0),
                                        grid_shape = (380, 280)
                                        )

    print lons.shape
    print lats.shape
    print interpolated.shape
    print 'done interpolating'


    m = Basemap(llcrnrlon=-85, llcrnrlat=40,
                urcrnrlon=-53, urcrnrlat=65)
    lons, lats = m(lons, lats)
    m.contourf(lons, lats, interpolated)
    m.drawcoastlines()
    draw_meridians_and_parallels(m)
    plt.colorbar()

    plt.savefig('interpolated_topo.png')

    pass

if __name__ == "__main__":
    t0 = time.clock()
    #test_get_lon_lat()
#    test_get_lon_lat()
    test_read_data()

    print 'execution time is %f seconds' % (time.clock() - t0)
