__author__="huziy"
__date__ ="$13 fevr. 2011 11:32:11$"


import application_properties
application_properties.set_current_directory()
import numpy as np


def get_directions_for_amno45grid(lons, lats):
    the_data = np.zeros(lons.shape)
    nx, ny = lons.shape

    #top left point corresponds to ddm[1,1]
    known_lon = 0.25
    known_lat = 89.75
    dlon = 0.5
    dlat = 0.5

    source = read_ddm_data()

    for i in range(nx):
        for j in range(ny):
            lon = (lons[i, j] + 360) % 360.0
            lat = lats[i, j]

            i_source = int( (lon - known_lon) / dlon )
            j_source = int( (-lat + known_lat) / dlat )
            the_data[i, j] = source[j_source, i_source]

    return the_data



def read_ddm_data(path = 'data/trip/trip05.asc'):
    f = open(path)
    data = []
    for line in f:
        fields = line.split()
        data.append(map(int, fields))
    data = np.array(data)
    print data.shape
    return data


def main():
    read_ddm_data()

if __name__ == "__main__":
    main()
    print "Hello World"
