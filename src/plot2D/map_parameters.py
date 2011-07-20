__author__="huziy"
__date__ ="$15 nov. 2010 23:28:35$"

import numpy as np
from mpl_toolkits.basemap import Basemap
from util.geo.ps_and_latlon import psxy2latlon, latlon2psxy


class MapParameters():
    def __init__(self):
        self.n_cols = 180
        self.n_rows = 172
        self.xs, self.ys, self.basemap = self.init_map()

    def get_longitudes_and_latitudes(self, nx = 180, ny = 172, lat_center = 49.65698, lon_center = -96.99443, dx = 45000):
        xc, yc = latlon2psxy(lat_center, lon_center)

        xmin = xc - (nx - 1) / 2.0
        ymin = yc - (ny - 1) / 2.0

        print 'These coordinates can be verified with cccma site points (2,2) and (181, 173) respectively'
        print 'lower left: ', psxy2latlon(xmin, ymin)
        print 'upper right: ', psxy2latlon(xmin + nx - 1 , ymin + ny - 1)

        longitudes = np.zeros((nx, ny))
        latitudes = np.zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                latitudes[i,j], longitudes[i, j] = psxy2latlon(xmin + i, ymin + j)

        return longitudes, latitudes


    def init_map(self):
        """initializes longitudes and latitudes of grid"""
        self.lons, self.lats = self.get_longitudes_and_latitudes( self.n_cols, self.n_rows)
#        m = Basemap(projection = 'npstere', #area_thresh = 10000,
#                        lat_ts = 60, lat_0 = 60, lon_0 = -115, boundinglat = 40, resolution='i')
#
        m = Basemap(projection = 'npstere',
                        lat_ts = 60, lat_0 = 60, lon_0 = -90, boundinglat = 40, resolution='i')


        xs, ys = m(self.lons, self.lats)
        return xs, ys, m

    def get_resolution_meters(self):
        return 45000.0


def zoom_on_quebec(plt):
    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.12 * (ymax - ymin), ymax * 0.32)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.65, 0.85*xmax)


polar_stereographic = MapParameters()

if __name__ == "__main__":
    print "Hello World"
