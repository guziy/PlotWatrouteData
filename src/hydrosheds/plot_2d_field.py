
__author__="huziy"
__date__ ="$Apr 1, 2011 11:54:33 AM$"



from mpl_toolkits.basemap import Basemap, NetCDFFile

import numpy as np

import matplotlib.pyplot as plt

import application_properties
application_properties.set_current_directory()

import readers.read_infocell as infocell
from math import *

from plot2D.map_parameters import polar_stereographic

xs = polar_stereographic.xs
ys = polar_stereographic.ys

lons = polar_stereographic.lons
lats = polar_stereographic.lats


inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 600 * inches_per_pt          # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]


font_size = 16
params = {
        'axes.labelsize': font_size,
        'font.size': font_size,
        'text.fontsize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize': fig_size
}


title_font_size = font_size

import pylab
pylab.rcParams.update(params)


def plot_source_flow_accumulation(sub = -1, vmin = 0, vmax = 10):
    path = 'data/hydrosheds/corresponding_DA.nc'
    ds = NetCDFFile(path)
    data = ds.variables['DA_source'][:,:]

    lons = ds.variables['longitude'][:,:]
    lats = ds.variables['latitude'][:,:]

    basemap = polar_stereographic.basemap


    data = np.ma.masked_where(data <= 0, data)
    lons = np.ma.masked_where(lons == 0, lons )

    if sub < 0:
        plt.figure()
    x, y = basemap(lons, lats)
#    print np.log(data).shape
    print np.min(data), np.max(data)
#    print np.min(np.log(data)), np.max(np.log(data))
    ax = basemap.pcolormesh(x, y, np.ma.log10(data), vmin = vmin, vmax = vmax)
    basemap.drawcoastlines()
    
    plt.colorbar()
    return ax


def plot_target_flow_accumulations(sub = -1, vmin = 0, vmax = 10):
    path = 'data/hydrosheds/corresponding_DA.nc'
    ds = NetCDFFile(path)
    data = ds.variables['DA_target'][:,:]

    lons = ds.variables['longitude'][:,:]
    lats = ds.variables['latitude'][:,:]

    basemap = polar_stereographic.basemap


    data = np.ma.masked_where(data <= 0, data)
    lons = np.ma.masked_where(lons == 0, lons )
    if sub < 0:
        plt.figure()
    x, y = basemap(lons, lats)
 #   print np.log(data).shape
    print np.min(data), np.max(data)
 #   print np.min(np.log(data)), np.max(np.log(data))
    ax = basemap.pcolormesh(x, y, np.ma.log10(data), vmin = vmin, vmax = vmax)
    basemap.drawcoastlines()
    
    plt.colorbar()

    return ax
    pass

def plot_scatter(path = 'data/hydrosheds/corresponding_DA.nc'):
    nc_da = NetCDFFile(path)
    v1 = nc_da.variables['DA_source'][:,:]
    v2 = nc_da.variables['DA_target'][:,:]

    basin_path = 'data/infocell/amno180x172_basins.nc'

    nc = NetCDFFile(basin_path)
    mask = None
    for k, v in nc.variables.iteritems():
        if mask == None:
            mask = v.data[:,:].copy()
        else:
            mask += v.data

    condition = (v1 > 0) & (v2 > 0) & ((v1 / v2) < 3) & (mask == 1)
    v1 = v1[condition]
    v2 = v2[condition]

    print len(v1), v1.shape
    print len(v2), v2.shape

    plt.figure()
    plt.grid(True)
    plt.scatter(np.log10(v1), np.log10(v2), linewidth = 0, s = 10)
    plt.xlabel('hydrosheds, $\\log_{10}(DA_{max})$ ')
    plt.ylabel('upscaled, $\\log_{10}(DA_{sim})$')


    min_x = np.min(np.log10(v1))

    x = plt.xlim()
    plt.xlim(min_x , x[1])
    plt.ylim(min_x, x[1])
    plt.plot([min_x , x[1]], [min_x, x[1]], color = 'k')

    me = 1 - np.sum( np.power(v1 - v2, 2) ) / np.sum(np.power( v1 - np.mean(v1), 2 ))
    plt.title('ME = {0}'.format(me))
    plt.savefig('da_scatter.png')


    #######Compare areas from CRCM4 and Upscale
    plt.figure()
    plt.title('compare areas computed by CRCM4 and Upscale module')

    v2 = nc_da.variables['DA_target'][:,:]
    basins = infocell.get_basins_with_cells_connected_using_hydrosheds_data()

    crcm4 = []
    upscaler = []

    for basin in basins:
        for the_cell in basin.cells:
            # @type the_cell Cell
            crcm4.append(the_cell.drainage_area)
            upscaler.append(v2[the_cell.x, the_cell.y])

            if v2[the_cell.x, the_cell.y] > 5 * the_cell.drainage_area:
                print v2[the_cell.x, the_cell.y], the_cell.drainage_area
                print lons[the_cell.x, the_cell.y], lats[the_cell.x, the_cell.y]
                print the_cell.x, the_cell.y
                print 20 * '='

    plt.scatter(crcm4, upscaler)
    plt.xlabel('CRCM4')
    plt.ylabel('Upscaler')

    x = plt.xlim()
    plt.xlim(min_x , x[1])
    plt.ylim(min_x, x[1])
    plt.plot([min_x , x[1]], [min_x, x[1]], color = 'k')

    plt.savefig('CRCM4vsUpscale.png')




def compare_drainages_2d():
    
    ####plot
    plt.figure()
    plt.subplot(1,2,1)
    plot_source_flow_accumulation(sub = 1)

    plt.subplot(1,2,2)
    plot_target_flow_accumulations(sub = 1)


    


def main():
#    compare_drainages_2d()
    plot_scatter()
    plt.show()
    pass

if __name__ == "__main__":
    main()
    print "Hello World"
