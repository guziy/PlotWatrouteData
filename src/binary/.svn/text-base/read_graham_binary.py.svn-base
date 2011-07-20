__author__="huziy"
__date__ ="$22 juin 2010 15:14:20$"


#external libraries
import os
import numpy as np
import pylab
import struct
import time
import Nio
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from math import *

#set work directory
import application_properties
application_properties.set_current_directory()

from util.geo.LonLatGrid import LonLatGrid
from util.geo.GeoPoint import GeoPoint


BASE_PATH = 'data/GRAHAM_5_min_data'

WEST_EAST = 'west_east'
NORTH_SOUTH = 'north_south'
LONGITUDE = 'longitude'
LATITUDE = 'latitude'

ROWS = 'rows'
COLUMNS = 'columns'
DATA_TYPE = 'data type'
INTEGER = 'integer'
UNITS = 'units'
DESCRIPTION = 'description'

MIN_X = 'min. X'
MAX_X = 'max. X'
MIN_Y = 'min. Y'
MAX_Y = 'max. Y'

FLOW_DIRECTION = 'flow direction'
DRAINAGE_AREA = 'drainage area'
RIVER_MASK = 'river mask'

VARIABLE_TO_FILENAME = {
    FLOW_DIRECTION : '5minfdr.img',
    DRAINAGE_AREA : '5minfac.img',
    RIVER_MASK : '5minriv.img'
}

VARIABLE_TO_DESCRIPTION = {
    LONGITUDE : 'longitudes, from -180 to 180 degrees with 5 minute resolution, 2D field',
    LATITUDE : 'longitudes, from 90 to -90 degrees with 5 minute resolution, 2D field',
    FLOW_DIRECTION : 'flow directions, E = 1; SE = 2; S = 4; SW = 8; W = 16; NW = 32; N = 64; NE = 128',
    DRAINAGE_AREA : 'drainage area, is the area from which water flows into current cell',
    RIVER_MASK : 'river mask'
}

VARIABLE_TO_UNITS = {
    LONGITUDE : 'degrees',
    LATITUDE : 'degrees',
    FLOW_DIRECTION : '-',
    DRAINAGE_AREA : 'km ** 2',
    RIVER_MASK : '-'
}


VARIABLE_TO_TYPE = {
    LONGITUDE : 'd',
    LATITUDE : 'd',
    FLOW_DIRECTION : 'd',
    DRAINAGE_AREA : 'd',
    RIVER_MASK : 'd'
}

DOUBLE_FILL_VALUE = -1.0
MINUTES_PER_DEGREE = 60.0

def _get_value_string(line):
    if ':' in line:
        return line.split(':')[1].strip()


def _read_header(path):
    '''
    returns the list [nrows, ncols, data_type]
    '''
    f = open(path)
    result = [None] * 7
    for line in f:
        if ROWS in line:
            result[0] = int(_get_value_string(line)) #nlatitudes
        if COLUMNS in line:
            result[1] = int(_get_value_string(line)) #nlongitudes
        if DATA_TYPE in line:
            result[2] = _get_value_string(line)
        if MIN_X in line:
            result[3] = float( _get_value_string(line) )
        if MAX_X in line:
            result[4] = float( _get_value_string(line) )
        if MIN_Y in line:
            result[5] = float( _get_value_string(line) )
        if MAX_Y in line:
            result[6] = float( _get_value_string(line) )
    return result

def _get_path_to_header(path_to_binary):
    return path_to_binary.replace('.img', '.txt')

#read binary data
def read_binary(path):
    '''
    returns 2D field read from the path
    '''
    header_path = _get_path_to_header(path)
    nrows, ncols, data_type, min_x, max_x, min_y, max_y = _read_header(header_path) 
    print nrows, ncols, data_type, min_x, max_x, min_y, max_y 

    #word_size is the number of bytes for one data value in the file
    if data_type == INTEGER:
        word_size = 2
        format = '<h' #2 byte integer, little endian
        the_dtype = 'i'
    else:
        word_size = 4 #
        format = '<f' #4 byte float, little endian
        the_dtype = np.float32

    bin_file_obj = open(path, 'rb')
    values = np.zeros(( ncols, nrows), dtype = the_dtype)
    for row in range(nrows):
        for col in range(ncols):
            #make (0,0) - point to be the lower left point
            x, = struct.unpack(format, bin_file_obj.read(word_size))
            if x >= 128:
                print x
            values[col, nrows - row - 1] = x
    return values.astype(the_dtype)



def get_graham_grid(the_shape = (4320, 2160), lower_left_point = GeoPoint(-180.0, -90.0)):
    return LonLatGrid(lower_left_point = lower_left_point,
                      grid_shape = the_shape,
                      resolution_seconds = 300.0)



def save_graham_data_to_obj(file_obj, the_shape = (4320, 2160),
                lower_left_point = GeoPoint(-180.0, -90.0),  resolution_min = 5.0):


    nlongitudes, nlatitudes = the_shape
    graham_grid = LonLatGrid(lower_left_point = lower_left_point, grid_shape = the_shape)


    file_obj.create_dimension(WEST_EAST, nlongitudes)
    file_obj.create_dimension(NORTH_SOUTH, nlatitudes)

    #create longitude and latitude variables
    longitude_var = file_obj.create_variable(LONGITUDE, VARIABLE_TO_TYPE[LONGITUDE], (WEST_EAST, NORTH_SOUTH))
    latitude_var = file_obj.create_variable(LATITUDE, VARIABLE_TO_TYPE[LATITUDE] , (WEST_EAST , NORTH_SOUTH))
    #write data to longitude and latitude variables


    longitudes = graham_grid.get_2d_longitudes()
    latitudes = graham_grid.get_2d_latitudes()




    longitude_var[:], latitude_var[:] = longitudes, latitudes
    setattr(longitude_var, UNITS, 'degrees')
    setattr(longitude_var, DESCRIPTION, VARIABLE_TO_DESCRIPTION[LONGITUDE] )

    setattr(latitude_var, UNITS, 'degrees')
    setattr(latitude_var, DESCRIPTION, VARIABLE_TO_DESCRIPTION[LATITUDE] )

    for key, value in VARIABLE_TO_FILENAME.iteritems():
        print key
        data = read_binary( BASE_PATH + os.path.sep + value)
        the_var = file_obj.create_variable(key, VARIABLE_TO_TYPE[key], (WEST_EAST, NORTH_SOUTH))
        setattr(the_var, UNITS, VARIABLE_TO_UNITS[key])
        setattr(the_var, DESCRIPTION, VARIABLE_TO_DESCRIPTION[key])
        setattr(the_var, '_FillValue', DOUBLE_FILL_VALUE)
        the_var[:] = data
    pass

##saves all the necessary data for the model to the netcdf file
def save_graham_data_to_netcdf(netcdf_file_path, resolution_min = 5, shape = (4320, 2160),
                               lower_left_point = GeoPoint(-180.0, -90.0)):
    opt = 'c'
    if os.path.isfile(netcdf_file_path):
        os.remove(netcdf_file_path)

    file = Nio.open_file( netcdf_file_path, opt )
    save_graham_data_to_obj(file, resolution_min = resolution_min, the_shape = shape, lower_left_point = lower_left_point)
    file.close()


E = 1; SE = 2; S = 4; SW = 8; W = 16; NW = 32; N = 64; NE = 128
def get_u_coordinate(value, scale = 1.0):
    frac_1_over_root_2 = 1.0 / 2.0 ** 0.5 * scale
    if value == E:
        return -1 * scale
    elif value == SE:
        return -frac_1_over_root_2
    elif value == S:
        return 0
    elif value == SW:
        return frac_1_over_root_2
    elif value == W:
        return 1.0 * scale
    elif value == NW:
        return frac_1_over_root_2
    elif value == N:
        return 0
    elif value == NE:
        return -frac_1_over_root_2
    else:
        return None

def get_v_coordinate(value, scale = 1.0):
    frac_1_over_root_2 = 1.0 / 2.0 ** 0.5 * scale
    if value == E:
        return 0.0
    elif value == SE:
        return -frac_1_over_root_2
    elif value == S:
        return -1.0 * scale
    elif value == SW:
        return -frac_1_over_root_2
    elif value == W:
        return 0.0
    elif value == NW:
        return frac_1_over_root_2
    elif value == N:
        return 1.0 * scale
    elif value == NE:
        return frac_1_over_root_2
    else:
        return None




def get_data_subset(lower_left_point = GeoPoint(-180.0, -90.0),
                       the_shape = (4320, 2160),
                       resolution = 5.0 / 60.0,
                       file_name = '5minfdr.img'):
    start_lon_index = int( (lower_left_point.longitude + 180) / resolution )
    start_lat_index = int( (lower_left_point.latitude + 90) / resolution )

    data = read_binary(BASE_PATH + os.path.sep + file_name)
    data = data[start_lon_index: start_lon_index + the_shape[0], start_lat_index: start_lat_index + the_shape[1]]
    return data



def test_read_one_graham_binary_and_plot():
    data = read_binary(BASE_PATH + os.path.sep + '5minfdr.img')
#    plot_flow_directions(data)
    lons, lats = get_graham_grid(the_shape = (4320, 2160), lower_left_point = GeoPoint(-180.0,-90.0)).get_lons_lats_2d()
    print lons.shape
    print lats.shape
    print data.shape
    plt.contourf(lons, lats, data)
#    plt.contourf(data)
    plt.colorbar()
    plt.grid()
    plt.savefig('fdr5min.png')
    pass


def plot_flow_directions():
    data = read_binary(BASE_PATH + os.path.sep + '5minfdr.img')

    print data[:,0]

    return
    data = data[::100,::100]

    print data.shape

#    plot_flow_directions(data)
    lons, lats = get_graham_grid(the_shape = data.shape, lower_left_point = GeoPoint(-80, 45)).get_lons_lats_2d()
#    plt.contourf(lons, lats, data)



    m = Basemap(llcrnrlon=-80.0, llcrnrlat=45,
                urcrnrlon=0, urcrnrlat=90)
    lons, lats = m(lons, lats)

    scale = 1.0


    get_u_coord_vec = np.vectorize(get_u_coordinate, otypes = ['float'])
    get_v_coord_vec = np.vectorize(get_v_coordinate, otypes = ['float'])

    flat_data = data.ravel()

    u = get_u_coord_vec(flat_data, scale)
    v = get_v_coord_vec(flat_data, scale)

    u, v = u.reshape(data.shape), v.reshape(data.shape)



    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (sqrt(5) - 1.0) / 2.0       # Aesthetic ratio
    fig_width = 700 * inches_per_pt          # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, fig_height]

    params = {'axes.labelsize': 14,
        'text.fontsize': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.figsize': fig_size}

    pylab.rcParams.update(params)


    m.quiver(lons, lats, u, v, width = 0.002, scale = 50)
    m.drawcoastlines()

    plt.savefig('fdr5min.png')


def test_write_netcdf():
    save_graham_data_to_netcdf('test.nc')



if __name__ == "__main__":
    t0 = time.clock()
    #plt.imshow(data, interpolation = 'bicubic')
    #test_write_netcdf()
    test_read_one_graham_binary_and_plot()
#    plot_flow_directions()
#    save_graham_and_wrf_data_to_netcdf('static_data.nc')
    print get_graham_grid()
    #plot_flow_directions()
    print 'Execution time is: %f seconds' % (time.clock() - t0)
    pass
    