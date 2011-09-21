__author__="huziy"
__date__ ="$22 mai 2010 16:27:42$"


from station_manager import StationManager
from readers.read_infocell import get_cell_lats_lons
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pylab
from math import *
import numpy as np
from readers.read_infocell import *

import application_properties

application_properties.set_current_directory()


path_to_longitudes = 'data/longitudes_182x174_amno_crcm37.txt'
path_to_latitudes = 'data/latitudes_182x174_amno_crcm37.txt'

#path_to_infocell = 'data/HQ2_infocell.txt'
path_to_infocell = 'data/infocellA2009_alloutlets.txt'
import os

def read_lon_lat(path, subtract_360 = False):
    """
    todo: document
    """
    result = []
    print os.getcwd()
    f = open(path)
    f.readline()
    for line in f:
        if line.strip() == '':
            continue

        start = 1
        fields = line.split()
        if fields[0].endswith('='):
            start += 1
        
        for field in fields[start:]:
            lat_lon = float(field)
            if subtract_360:
                lat_lon -= 360
            result.append(lat_lon)
    return result



def main():
    '''
    plots 2D domain map
    '''

    lons = read_lon_lat(path_to_longitudes, subtract_360 = True)
    lats = read_lon_lat(path_to_latitudes)

    cell_lat_lons = get_cell_lats_lons(path_to_infocell)
    cell_lons = []
    cell_lats = []
    for pair in cell_lat_lons:
        cell_lons.append(pair[0])
        cell_lats.append(pair[1])

    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = 1500 * inches_per_pt           # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, fig_height]

    params = {'axes.labelsize': 14,
        'text.fontsize': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.figsize': fig_size}

    pylab.rcParams.update(params)


    lat_min = min(lats)
    lat_max = max(lats)

    lon_max = max(lons)
    lon_min = min(lons)


    print lon_min, lon_max
    print lat_min, lat_max

    m =  Basemap(projection = 'npstere',
                 resolution = 'i',
                 lon_0 = (lon_min + lon_max) / 2.0, lat_0 = (lat_min + lat_max) / 2.0,
                 boundinglat = 40,
                 lat_ts = 60,
                 llcrnrlon = lon_min, llcrnrlat = lat_min,
                 urcrnrlon = lon_max, urcrnrlat = lat_max)

    


    projected_lons, projected_lats = m(lons, lats)
   # m.scatter(projected_lons, projected_lats, color = "red", s = 0.1)

    projected_lons, projected_lats = m(cell_lons, cell_lats)
    m.scatter(projected_lons, projected_lats, color="green", s = 45, marker = 's' )


    #show station positions
    manager = StationManager()
    manager.read_stations_from_files_rivdis()
    station_lons = []
    station_lats = []

    for station in manager.stations:
        station_lons.append(station.longitude)
        station_lats.append(station.latitude)


    print len(station_lons)

    projected_lons, projected_lats = m(station_lons, station_lats)
    m.scatter(projected_lons, projected_lats, color = 'red', s = 45, marker = 'o')

    m.drawcoastlines()
    m.drawcountries()

    #m.fillcontinents(color='white',lake_color='blue', zorder=0)

    m.drawmeridians(np.arange(lon_min, lon_max, 10), labels=[0,0,0,0])
    m.drawparallels(np.arange(lat_min, lat_max, 10), labels=[0,0,0,0])

    y_min, y_max = plt.ylim()
    x_min, x_max = plt.xlim()

    plt.ylim(y_min , y_max / 2.0)
    plt.xlim(x_max / 4.0, 0.9 * x_max)


    plt.savefig('domain.png')

if __name__ == "__main__":
    main()
