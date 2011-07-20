# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="huziy"
__date__ ="$4 juil. 2010 16:08:46$"

from plot2D.plot_utils import draw_meridians_and_parallels
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

import pylab

from plot2D.plot_utils import *


MINUTES_PER_DEGREE = 60.0


def default_basemap_scatter():
#    m = Basemap()
#    m.drawcoastlines(linewidth = 0.5)
#    longitudes_array  = np.arange(-85, -55, 5.0 / MINUTES_PER_DEGREE)
#    latitudes_array = np.arange(70, 40, - 5.0 / MINUTES_PER_DEGREE)
#    X, Y = pylab.meshgrid(longitudes_array, latitudes_array)
#    longitudes_array, latitudes_array = m(X, Y)
#    m.contourf(longitudes_array, latitudes_array, longitudes_array)
    

#    plt.figure()
#    m = Basemap(boundinglat = 40, projection = 'npstere', lat_0=60, lon_0=0)
#    m.drawcoastlines()
#    draw_meridians_and_parallels(m, 25)

    plt.figure()
    m = Basemap(projection = 'npstere',
                        lat_ts = 60, lat_0 = 60, lon_0 = -115, boundinglat = 40, resolution='i')
    m.drawcoastlines()

    m.drawrivers(color = 'blue', linewidth = 1)


    dx = 45000.0

    px, py = m(-61.5*dx, -179.8*dx)
    m.scatter(px, py , c="red")
    draw_meridians_and_parallels(m, 25)

    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.12 * (ymax - ymin), ymax * 0.32)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.65, 0.85*xmax)


    plt.show()



if __name__ == "__main__":
    default_basemap_scatter()
    print "Hello World"
